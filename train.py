import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
import time
import sys
import logging

import opts
from dataset import VISTDataset
import models
from log_utils import Logger
import misc.utils as utils

from eval_utils import Evaluator
import criterion
from misc.yellowfin import YFOptimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setup_optimizer(opt, model):
    """
    设置优化器
    """
    if opt.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=opt.learning_rate,
                               betas=(opt.optim_alpha, opt.optim_beta),
                               eps=opt.optim_epsilon,
                               weight_decay=opt.weight_decay)
    elif opt.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=opt.learning_rate,
                              momentum=opt.momentum)
    elif opt.optim == "momSGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=opt.learning_rate,
                                    momentum=opt.momentum)
    elif opt.optim == 'Adadelta':
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=opt.learning_rate,
                                   weight_decay=opt.weight_decay)
    elif opt.optim == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    else:
        logging.error("Unknown optimizer: {}".format(opt.optim))
        raise Exception("Unknown optimizer: {}".format(opt.optim))

    # Load the optimizer
    if opt.resume_from is not None and opt.model == 'BaseModel':
        optim_path = os.path.join(opt.resume_from, "optimizer.pth")
        if os.path.isfile(optim_path):
            logging.info("Load optimizer from {}".format(optim_path))
            optimizer.load_state_dict(torch.load(optim_path))
            opt.learning_rate = optimizer.param_groups[0]['lr']
            logging.info("Loaded learning rate is {}".format(opt.learning_rate))

    return optimizer


def train(opt):
    """
    模型训练函数
    """
    # 自定义的类，日志记录
    logger = Logger(opt)

    # 获取数据
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()
    # print(dataset.get_word2id()['the'])
    dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': True}) # 若不使用caption数据，则将其设为False
    dataset.train()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    dataset.test() # 改为valid
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    # m = dataset.word2id

    # 记录上升的 valid_loss 次数
    bad_valid = 0

    # 创建Evaluator
    evaluator = Evaluator(opt, 'val')
    # 损失
    crit = criterion.LanguageModelCriterion()
    # 是否使用强化学习，默认为-1
    if opt.start_rl >= 0:
        rl_crit = criterion.ReinforceCriterion(opt, dataset)

    # set up model，函数在init文件中，若有原来模型，则加载模型参数
    model = models.setup(opt)
    model.cuda()
    optimizer = setup_optimizer(opt, model)
    dataset.train()
    model.train()
    for epoch in range(logger.epoch_start, opt.max_epochs): # 默认为 0-20
        # scheduled_sampling_start表示在第几个epoch，衰减gt使用概率，最大到0.25，5个epoch之内还是0
        if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every # 后者默认值为5，//为向下取整除
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob) # 0.05、0.25
            model.ss_prob = opt.ss_prob
        # 对数据进行一个batch一个batch的迭代
        for iter, batch in enumerate(train_loader):
            start = time.time()
            logger.iteration += 1
            torch.cuda.synchronize()

            # 获取batch中的数据，图像特征、caption、以及target
            features = Variable(batch['feature_fc']).cuda() # 64*5*2048
            caption = None
            if opt.caption:
                caption = Variable(batch['caption']).cuda() # 64*5*20         
            target = Variable(batch['split_story']).cuda() # 64*5*30
            index = batch['index']

            optimizer.zero_grad()

            # 模型运行，返回一个概率分布，然后计算交叉熵损失
            output = model(features, target, caption)
            loss = crit(output, target)

            if opt.start_rl >= 0 and epoch >= opt.start_rl:  # reinforcement learning
                # 获取 sample 数据和 baseline 数据
                seq, seq_log_probs, baseline = model.sample(features, caption=caption, sample_max=False, rl_training=True)
                rl_loss, avg_score = rl_crit(seq, seq_log_probs, baseline, index)
                print(rl_loss.data[0] / loss.data[0])
                loss = opt.rl_weight * rl_loss + (1 - opt.rl_weight) * loss
                logging.info("average {} score: {}".format(opt.reward_type, avg_score))
            # 反向传播
            loss.backward()
            train_loss = loss.item()
            # 梯度裁剪，第二个参数为梯度最大范数，大于该值则进行裁剪
            nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
            optimizer.step()
            torch.cuda.synchronize()
            # 日志记录时间以及损失
            logging.info("Epoch {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, iter,
                                                                                              len(train_loader),
                                                                                              train_loss,
                                                                                              time.time() - start))
            # Write the training loss summary，tensorboard记录
            if logger.iteration % opt.losses_log_every == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)
            # validation验证，每迭代save_checkpoint_every轮评测一次
            if logger.iteration % opt.save_checkpoint_every == 0:
                val_loss, predictions, metrics = evaluator.eval_story(model, crit, dataset, val_loader, opt)
                if opt.metric == 'XE':
                    score = -val_loss
                else:
                    score = metrics[opt.metric]
                logger.log_checkpoint(epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer)
                # halve the learning rate if not improving for a long time
                if logger.best_val_score > score:
                    bad_valid += 1
                    if bad_valid >= 4:
                        opt.learning_rate = opt.learning_rate / 2.0
                        logging.info("halve learning rate to {}".format(opt.learning_rate))
                        checkpoint_path = os.path.join(logger.log_dir, 'model-best.pth')
                        model.load_state_dict(torch.load(checkpoint_path))
                        utils.set_lr(optimizer, opt.learning_rate)  # set the decayed rate
                        bad_valid = 0
                        logging.info("bad valid : {}".format(bad_valid))
                else:
                    logging.info("achieving best {} score: {}".format(opt.metric, score))
                    bad_valid = 0


def test(opt):
    logger = Logger(opt)
    dataset = VISTDataset(opt)
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()

    dataset.test()
    test_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    evaluator = Evaluator(opt, 'test')
    model = models.setup(opt)
    model.cuda()
    predictions, metrics = evaluator.test_story(model, dataset, test_loader, opt)


if __name__ == "__main__":

    opt = opts.parse_opt()
    opt.GPU_ids = 2

    # 设置 GPU id
    torch.cuda.set_device(opt.GPU_ids)
    opt.batch_size = 64
    opt.save_checkpoint_every = 1000
    opt.caption = False

    opt.self_att = False
    opt.cnn_cap = False
    opt.bi = False
    opt.dec = True
    opt.mem = True
    # opt.context_dec = True
    opt.trick = True
    opt.att = True
    opt.option = 'train'
    opt.with_position = True
    # opt.option = 'test'
    
    # opt.id = 'test'
    # opt.resume_from = "./data/hlst/"
    # 训练模式还是测试模式
    if opt.option == 'train':
        print('Begin training:')
        train(opt)
    else:
        # opt.resume_from = './data/' + opt.id + '/'
        print(opt.resume_from)
        print('Begin testing:')
        test(opt)
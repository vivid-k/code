import copy
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
from criterion import to_contiguous
from misc.yellowfin import YFOptimizer
from train import setup_optimizer

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Flag:
    def __init__(self, D_iters, G_iters, always=None):
        # D_iters 和 G_iters 在 opt 设置
        self.D_iters = D_iters
        self.G_iters = G_iters

        self.flag = "Disc" # 初始默认为 disc
        self.iters = self.D_iters
        self.curr = 0
        self.always = always # 默认为 None

    def inc(self):
        # 更改当前标签，训练判别器(reward_model)还是生成器(base_model)
        self.curr += 1
        if self.curr >= self.iters and self.always is None: # 如果当前迭代次数大于该模型迭代次数，则更换
            if self.flag == "Disc":
                self.flag = "Gen"
                self.iters = self.G_iters
            elif self.flag == "Gen":
                self.flag = "Disc"
                self.iters = self.D_iters
            self.curr = 0


def train(opt):
    logger = Logger(opt) # 定义 logger
    flag = Flag(D_iters=opt.D_iter, G_iters=opt.G_iter, always=opt.always) # 初始化训练标签

    dataset = VISTDataset(opt) # 加载数据
    opt.vocab_size = dataset.get_vocab_size()
    opt.seq_length = dataset.get_story_length()
    dataset.set_option(data_type={'whole_story': False, 'split_story': True, 'caption': False})
    dataset.train()
    train_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle)
    dataset.val()
    val_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    bad_valid = 0

    evaluator = Evaluator(opt, 'val')
    crit = criterion.LanguageModelCriterion()
    rl_crit = criterion.ReinforceCriterion(opt, dataset) # 强化学习的损失函数

    # set up model
    model = models.setup(opt)
    model.cuda()
    disc_opt = copy.copy(opt)
    disc_opt.model = 'RewardModel' # 加入model属性
    disc = models.setup(disc_opt) # 判别器模型，实例化哪个模型的类
    if os.path.exists(os.path.join('./data/save/', 'disc-model.pth')): # 若存在，则加载模型参数
        logging.info("loading pretrained RewardModel")
        disc.load_state_dict(torch.load(os.path.join(logger.log_dir, 'disc-model.pth')))
    disc.cuda()
    # 两个优化器，完全独立的两个模型
    optimizer = setup_optimizer(opt, model)
    disc_optimizer = setup_optimizer(disc_opt, disc) # fix

    dataset.train()
    model.train()
    disc.train()
    ############################## training ##################################
    for epoch in range(logger.epoch_start, opt.max_epochs): # 最大轮数为 50
        start = time.time()
        for iter, batch in enumerate(train_loader): # 开始迭代
            logger.iteration += 1 # 记录迭代次数
            torch.cuda.synchronize()
            # 获取数据
            feature_fc = Variable(batch['feature_fc']).cuda()
            target = Variable(batch['split_story']).cuda()
            index = batch['index']

            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            if flag.flag == "Disc": 
                model.eval() # policy model参数不更新
                disc.train() # 更新判别器参数
                if opt.decoding_method_DISC == 'sample': # True，返回 sample 的序列，根据概率分布 sample
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True,
                                                                pad=True)
                elif opt.decoding_method_DISC == 'greedy':
                    seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=True, rl_training=True,
                                                                pad=True)
            else:
                model.train() # 更新模型
                disc.eval() # 判别器不更新
                seq, seq_log_probs, baseline = model.sample(feature_fc, sample_max=False, rl_training=True, pad=True)

            seq = Variable(seq).cuda()
            mask = (seq > 0).float() # 64,5,30
            mask = to_contiguous(torch.cat([Variable(mask.data.new(mask.size(0), mask.size(1), 1).fill_(1)), mask[:, :, :-1]], 2))
            normed_seq_log_probs = (seq_log_probs * mask).sum(-1) / mask.sum(-1) # 64,5，得到整个序列的概率
            gen_score = disc(seq.view(-1, seq.size(2)), feature_fc.view(-1, feature_fc.size(2))) # 计算sample序列的reward分数

            if flag.flag == "Disc": # 先训练判别器，生成器已经预训练好。训练该判别器参数，使其能对标签和生成数据进行打分。
                gt_score = disc(target.view(-1, target.size(2)), feature_fc.view(-1, feature_fc.size(2))) # 计算真实序列的reward
                loss = -torch.sum(gt_score) + torch.sum(gen_score) # 计算损失，loss为负很正常
                # 计算平均 reward，训练判别器希望能尽可能pos高
                avg_pos_score = torch.mean(gt_score) 
                avg_neg_score = torch.mean(gen_score)

                if logger.iteration % 5 == 0:
                    logging.info("pos reward {} neg reward {}".format(avg_pos_score.item(), avg_neg_score.item()))
                    # print("PREDICTION: ", utils.decode_story(dataset.get_vocab(), seq[:1].data)[0])
                    # print("GROUND TRUTH: ", utils.decode_story(dataset.get_vocab(), target[:1].data)[0])
            else:
                rewards = Variable(gen_score.data - 0 * normed_seq_log_probs.view(-1).data)
                #with open("/tmp/reward.txt", "a") as f:
                #    print(" ".join(map(str, rewards.data.cpu().numpy())), file=f)
                loss, avg_score = rl_crit(seq.data, seq_log_probs, baseline, index, rewards.view(-1, seq.size(1)))
                # if logger.iteration % opt.losses_log_every == 0:
                avg_pos_score = torch.mean(gen_score)
                # logging.info("average reward: {} average IRL score: {}".format(avg_score.item(), avg_pos_score.item()))

            if flag.flag == "Disc":
                loss.backward()
                nn.utils.clip_grad_norm(disc.parameters(), opt.grad_clip, norm_type=2)
                disc_optimizer.step()
            else:
                tf_loss = crit(model(feature_fc, target), target)
                # print("rl_loss / tf_loss = ", loss.item() / tf_loss.item())
                loss = opt.rl_weight * loss + (1 - opt.rl_weight) * tf_loss
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip, norm_type=2)
                optimizer.step()

            train_loss = loss.item()
            torch.cuda.synchronize()

            # Write the training loss summary
            if logger.iteration % opt.losses_log_every == 0:
                logger.log_training(epoch, iter, train_loss, opt.learning_rate, model.ss_prob)
                logging.info(
                    "Epoch {} Train {} - Iter {} / {}, loss = {:.5f}, time used = {:.3f}s".format(epoch, flag.flag,
                                                                                                  iter,
                                                                                                  len(train_loader),
                                                                                                  train_loss,
                                                                                                  time.time() - start))
                start = time.time()

            if logger.iteration % opt.save_checkpoint_every == 0:
                if opt.always is None:
                    # Evaluate on validation dataset and save model for every epoch
                    val_loss, predictions, metrics = evaluator.eval_story(model, crit, dataset, val_loader, opt)
                    if opt.metric == 'XE':
                        score = -val_loss
                    else:
                        score = metrics[opt.metric]
                    logger.log_checkpoint(epoch, val_loss, metrics, predictions, opt, model, dataset, optimizer)
                    # halve the learning rate if not improving for a long time
                    if logger.best_val_score > score:
                        bad_valid += 1
                        if bad_valid >= 10:
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
                else:
                    torch.save(disc.state_dict(), os.path.join(logger.log_dir, 'disc-model.pth'))
            flag.inc()


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
    # opt.always = "always"
    opt.GPU_ids = 1
    # 设置 GPU id
    torch.cuda.set_device(opt.GPU_ids)
    opt.max_epochs = 100
    opt.id = "adver3"
    opt.resume_from = './data/save/' + opt.id + '/'
    opt.option = 'train'
    if opt.option == 'train':
        print('Begin training:')
        train(opt)
    else:
        print('Begin testing:')
        test(opt)

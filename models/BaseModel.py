import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import logging
import time
import numpy as np
from models.model_utils import AttentionLayer, VisualEncoder, _smallest, CaptionEncoder
from models.attention import luong_gate_attention

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.story_size = opt.story_size
        self.word_embed_dim = opt.word_embed_dim
        self.hidden_dim = opt.hidden_dim
        self.num_layers = opt.num_layers
        self.rnn_type = opt.rnn_type
        self.dropout = opt.dropout
        self.seq_length = opt.seq_length
        self.feat_size = opt.feat_size
        self.decoder_input_dim = self.word_embed_dim + self.word_embed_dim
        self.ss_prob = 0.0  # Schedule sampling probability
        
        # Visual Encoder
        self.encoder = VisualEncoder(opt)
        if opt.caption:
            self.caption_encoder = CaptionEncoder(opt)
            self.project_d3 = nn.Linear(self.decoder_input_dim + self.hidden_dim, self.word_embed_dim)
            self.attention = luong_gate_attention(self.hidden_dim, self.word_embed_dim)
        # Decoder LSTM
        self.project_d = nn.Linear(self.decoder_input_dim, self.word_embed_dim)
        if self.rnn_type == 'gru':
            self.decoder = nn.GRU(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.decoder = nn.LSTM(input_size=self.word_embed_dim, hidden_size=self.hidden_dim, batch_first=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # word embedding layer
        self.embed = nn.Embedding(self.vocab_size, self.word_embed_dim)

        # last linear layer
        self.logit = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                   nn.Tanh(),
                                   nn.Dropout(p=self.dropout),
                                   nn.Linear(self.hidden_dim // 2, self.vocab_size))

        self.init_s_proj = nn.Linear(self.feat_size, self.hidden_dim)
        self.init_c_proj = nn.Linear(self.feat_size, self.hidden_dim)
        self.state_linear = nn.Linear(self.hidden_dim*2, self.hidden_dim)

        self.baseline_estimator = nn.Linear(self.hidden_dim, 1)

        self.init_weights(0.1)
        

    def init_weights(self, init_range):
        logging.info("Initialize the parameters of the model")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-init_range, init_range)
                if not m.bias is None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size, bi, dim):
        # the first parameter from the class
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return Variable(weight.new(self.num_layers * times, batch_size, dim).zero_())
        else:
            return (Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()),
                    Variable(weight.new(self.num_layers * times, batch_size, dim).zero_()))

    def init_hidden_with_feature(self, feature):
        if self.rnn_type == 'gru':
            output = self.init_s_proj(feature)
            return output.view(1, -1, output.size(-1))
        else:
            output1 = self.init_s_proj(feature)
            output2 = self.init_c_proj(feature)
            return (output1.view(1, -1, output1.size(-1)), \
                    output2.view(1, -1, output2.size(-1)))
    def init_hidden_with_feature_fix(self, feature):
        if self.rnn_type == 'gru':
            output = self.init_s_proj(feature)
            return output
        else:
            output1 = self.init_s_proj(feature)
            output2 = self.init_c_proj(feature)
            return (output1, output2)

    def decode(self, imgs, last_word, state_d, caption=None, penalize_previous=False):
        # imgs：320*512；last_word：320；state_d：1*320*512
        word_emb = self.embed(last_word) # 对上一个词进行embding
        word_emb = torch.unsqueeze(word_emb, 1) # 320*1*512
        if self.opt.caption:
            input_d = torch.cat([word_emb, imgs.unsqueeze(1), caption.unsqueeze(1)], 2)
            input_d = self.project_d3(input_d)
        else:
            input_d = torch.cat([word_emb, imgs.unsqueeze(1)], 2)  # batch_size * 1 * dim
            input_d = self.project_d(input_d)
        # input_d 为上一个词与图像特征结合的结果
        out_d, state_d = self.decoder(input_d.contiguous(), state_d.contiguous())
        # softmax 计算概率分布
        log_probs = F.log_softmax(self.logit(out_d.squeeze()))

        if penalize_previous: # test 时使用，对上一个词进行惩罚
            last_word_onehot = torch.FloatTensor(last_word.size(0), self.vocab_size).zero_().cuda() # lastword 的onhot形式
            penalize_value = (last_word > 0).data.float() * -100 # 惩罚值
            mask = Variable(last_word_onehot.scatter_(1, last_word.data[:, None], 1.) * penalize_value[:, None])
            log_probs = log_probs + mask

        return log_probs, out_d, state_d

    def forward(self, features, story_t, caption=None):
        """
        :param features: (batch_size, 5, feat_size)
        :param caption: (batch_size, 5, seq_length)
        """
        if self.opt.caption: # 引入caption
            out_caption, enc_state = self.caption_encoder(caption, self.embed) # 320*20*512
        # 对图像特征编码
        out_e, enc_state, mem = self.encoder(features) # out_e：64*5*512
        out_e = out_e.contiguous()
        out_e = out_e.view(-1, out_e.size(2)) # 320*512
        story_t = story_t.view(-1, story_t.size(2)) # 320*30
        batch_size = out_e.size(0) #320

        state_d = self.init_hidden_with_feature(features) # 返回 1*320*512，使用原始的feature进行初始化
        last_word = torch.FloatTensor(batch_size).long().zero_().cuda() # last_word则使用0初始化，即<EOS>
        outputs = [] # 记录输出的概率分布
        # 循环30次，将5个句子并行处理
        for i in range(self.seq_length):
            # 使用caption，对当前state做attention
            cap_att_state = None
            if self.opt.caption:
                self.attention.init_context(out_caption.transpose(0, 1))
                cap_att_state, _ = self.attention(state_d.squeeze())
            log_probs, out, state_d = self.decode(out_e, last_word, state_d, caption=cap_att_state)
            outputs.append(log_probs)
            # 选择last_word
            if self.ss_prob > 0.0: # 我觉得这块是为了增加模型鲁棒性
                sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda() # 生成随机数 64 | 320
                sample_mask = sample_prob < self.ss_prob # 小于ss_prob的为1，大于的为0
                if sample_mask.sum() == 0: # 无需 mask，直接使用上一个词
                    last_word = story_t[:, i].clone()
                else: # 使用mask
                    sample_ind = sample_mask.nonzero().view(-1) # 返回非零的下标
                    last_word = story_t[:, i].data.clone()
                    prob_prev = torch.exp(log_probs.data) # 保证每个概率都大于0，因为需要按该分布采样
                    last_word.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind)) # 按照概率分布采样，只有sample的才取采样的word作为last_word
                    last_word = Variable(last_word)
            else: # 训练时直接使用上一个词作为标签
                last_word = story_t[:, i].clone()
            # break condition，若所有均为0
            if i >= 1 and story_t[:, i].data.sum() == 0:
                break

        outputs = torch.cat([_.unsqueeze(1) for _ in outputs], 1)  # batch_size * 5, -1, vocab_size
        return outputs.view(-1, self.story_size, outputs.size(1), self.vocab_size)

    def predict(self, features, caption=None, beam_size=5):
        # beamsearch计算最终结果
        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample(features, sample_max=True, rl_training=False)
        
        out_e, enc_state, mem = self.encoder(features) # encode the visual features 64*5*512
        out_e = out_e.contiguous()
        out_e = out_e.view(-1, out_e.size(2)) # 320*512
        if self.opt.caption:
            out_caption, enc_state = self.caption_encoder(caption, self.embed) # 320*20*512   
        batch_size = out_e.size(0)
        state_d = self.init_hidden_with_feature(features) # 1*320*512
        # 记录最终结果
        seq = torch.LongTensor(self.seq_length, batch_size).zero_() # 30*320
        seq_log_probs = torch.FloatTensor(self.seq_length, batch_size)

        # window = self.opt.window
        # 一句话一句话的处理
        for k in range(batch_size):
            # beamsize作为batchsize进行后续处理 # 
            num = k % 5 # 记录处理到哪一句
            # his_set = set()
            if num == 0:
                bag = np.zeros(self.vocab_size, int)
            else:
                his = seq[:, k-1].view(-1)
                for h in his:
                    if h > 2:
                        bag[h] += 1
            # if num != 0: # 记录一个story的生成历史        
            #     his = seq[:, k-num:k]
            #     his_mask = his > 0
            #     his_len = his_mask.sum(dim=0)
            #     for m in range(num):
            #         for n in range(his_len[m]-window+1):
            #             his_set.add(his[n:n+window, m])
            out_e_k = out_e[k].unsqueeze(0).expand(beam_size, out_e.size(1)).contiguous() #3*512
            state_d_k = state_d[:, k, :].unsqueeze(1).expand(state_d.size(0), beam_size, state_d.size(2)).contiguous()#1*3*512
            out_caption_k = None
            if self.opt.caption:
                self.attention.init_context(out_caption[k].unsqueeze(0).transpose(0, 1)) # 1*20*512
                caption_context = out_caption[k].expand(beam_size, out_caption[k].size(0), out_caption[k].size(1)).contiguous()
                cap_att_state, _ = self.attention(state_d[:, k, :]) # 1*512
                out_caption_k = cap_att_state.expand(beam_size, cap_att_state.size(1)).contiguous() # 3*20*512      

            last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda())  # <BOS>

            log_probs, out, state_d_k = self.decode(out_e_k, last_word, state_d_k, caption=out_caption_k, penalize_previous=True)
            log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token，不使其生成unk
            neg_log_probs = -log_probs # 3*vocab
            # beamsearch要使用的三个向量，大小均为beamsize
            all_outputs = np.ones((1, beam_size), dtype='int32') # 全为1
            all_masks = np.ones_like(all_outputs, dtype="float32") # 全为0则生成结束，使用1初始化
            all_costs = np.zeros_like(all_outputs, dtype="float32") # 记录总的损失
            for i in range(self.seq_length): # 生成每一个词
                if all_masks[-1].sum() == 0: # 初始时全部为1
                    break
                # 之前的cost+概率分布*mask（mask标记是否生成结束，结束后为0，不再累计损失）
                next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None]) # 计算cost，all_mask一个数乘以前面一行，3*9837
                next_costs = next_costs[:] + bag * self.opt.scale
                (finished,) = np.where(all_masks[-1] == 0) # 找出最后一行为0的位置，finished记录了所有mask为0的索引

                next_costs[finished, 1:] = np.inf # 如果已经结束，则除了结束符外的cost设置为无穷大
                
                # 返回当前最小的三个概率值chosen_costs，以及outputs为选择的词表中的词位置，indexes为哪一个维度
                (indexes, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

                new_state_d = state_d_k.data.cpu().numpy()[:, indexes, :] # 1*3*512 需要根据index的维度信息取state

                all_outputs = all_outputs[:, indexes] # [len,beam]
                all_masks = all_masks[:, indexes]
                all_costs = all_costs[:, indexes]
                # if i >= window and num != 0: 
                #     current = torch.tensor(all_outputs[-window:]).transpose(0,1) # window,beam
                #     for m, cur in enumerate(current):
                #         if cur[-1] != 0:
                #             for n in his_set:
                #                 if torch.equal(n, cur):
                #                     chosen_costs[m] += 1
                #                     break
                # if i >= window and num != 0:
                #     current = all_outputs[-1]
                #     current_window = torch.tensor(all_outputs[-window:]) # [4,beam]
                #     for j,cur in enumerate(current):
                #         if cur != 0: # 当前句子还未生成结束
                #             index = np.where(his == cur) # 返回二维坐标
                #             if len(index[0]) != 0: # 有相同值，则遍历
                #                 for i in range(len(index[0])):
                #                     if index[0][i] >= window: # 获取句子位置，长度大于window才进行比较
                #                         tmp = his[index[0][i]+1-window:index[0][i]+1, index[1][i]]
                #                         if torch.equal(tmp, current_window[:, j]):
                #                             chosen_costs[j] = np.inf
                # 记录处理过的lastword和state
                last_word = Variable(torch.from_numpy(outputs)).cuda()
                state_d_k = Variable(torch.from_numpy(new_state_d)).cuda()
                cap_att_state = None
                if self.opt.caption:
                    self.attention.init_context(caption_context.transpose(0, 1))
                    cap_att_state, _ = self.attention(state_d_k.squeeze()) # 3*512

                log_probs, out, state_d_k = self.decode(out_e_k, last_word, state_d_k, cap_att_state, True)

                log_probs[:, 1] = log_probs[:, 1] - 1000  # 不生成unk
                neg_log_probs = -log_probs

                all_outputs = np.vstack([all_outputs, outputs[None, :]]) # 将输出合并近最后结果
                all_costs = np.vstack([all_costs, chosen_costs[None, :]]) # 记录所有的costs
                mask = outputs != 0
                all_masks = np.vstack([all_masks, mask[None, :]])

            all_outputs = all_outputs[1:]
            
            all_masks = all_masks[:-1]
            # costs = all_costs.sum(axis=0)
            lengths = all_masks.sum(axis=0)
            costs = 0
            for i, len in enumerate(lengths):
                costs += all_costs[int(len-1)][i]
            all_costs = all_costs[1:] - all_costs[:-1]
            normalized_cost = costs / lengths
            best_idx = np.argmin(normalized_cost)
            seq[:all_outputs.shape[0], k] = torch.from_numpy(all_outputs[:, best_idx])
            seq_log_probs[:all_costs.shape[0], k] = torch.from_numpy(all_costs[:, best_idx])

        # return the samples and their log likelihoods
        seq = seq.transpose(0, 1).contiguous()
        seq_log_probs = seq_log_probs.transpose(0, 1).contiguous()
        seq = seq.view(-1, self.story_size, seq.size(1))
        seq_log_probs = seq_log_probs.view(-1, self.story_size, seq_log_probs.size(1))
        return seq, seq_log_probs

    # def forward(self, features, story_t):
    #     """
    #     :param features: (batch_size, 5, feat_size)
    #     :param story_t: (batch_size, 5, seq_length)
    #     :return:
    #     """
    #     # encode the visual features features：64*5*2048 out_e：64*5*512
    #     out_e, _ = self.encoder(features)
    #     batch_size = out_e.size(0) # 64

    #     state_d = self.init_hidden_with_feature_fix(features) # 只是过了一个线性层 64*5*512
    #     for j in range(self.story_size): # 循环生成每个句子
    #         if j == 0: # 第一个句子
    #             state = state_d[:, j, :].unsqueeze(0)# 1*64*512，初始化state为图像特征
    #             outputs = [] # 存储所有生成的句子
    #         else: # 非第一个句子，hi与上一句话输出做att

    #             state = self.state_linear(torch.cat((state_d[:, j, :].unsqueeze(0), state), 2))
    #         output = []
    #         # last_word = Variable(torch.FloatTensor(batch_size).long().zero_()).cuda()
    #         fill_value = 3
    #         last_word = torch.full((batch_size, 1), fill_value).squeeze().long().cuda()
            
    #         for i in range(self.seq_length):
    #             log_probs, out, state = self.decode(out_e[:, j, :], last_word, state) # 每次的解码输入都相同，前两个cat
    #             output.append(log_probs)

    #             # choose the word
    #             if self.ss_prob > 0.0:
    #                 sample_prob = torch.FloatTensor(batch_size).uniform_(0, 1).cuda()
    #                 sample_mask = sample_prob < self.ss_prob
    #                 if sample_mask.sum() == 0:
    #                     last_word = story_t[:, j, i].clone()
    #                 else:
    #                     sample_ind = sample_mask.nonzero().view(-1)
    #                     last_word = story_t[:, j, i].data.clone()
    #                     # fetch prev distribution: shape Nx(M+1)
    #                     prob_prev = torch.exp(log_probs.data)
    #                     last_word.index_copy_(0, sample_ind,
    #                                         torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
    #                     last_word = Variable(last_word)
    #             else: # 训练时直接使用上一个词作为标签
    #                 last_word = story_t[:, j, i].clone()

    #             # break condition
    #             # if i >= 1 and story_t[:, j, i].data.sum() == 0:
    #             #     break
    #         outputs.append(torch.stack(output))
    #     outputs = torch.stack(outputs).permute(2,0,1,3).contiguous()

    #     return outputs

    def sample(self, features, sample_max, caption=None, rl_training=False, pad=False):
        # 强化学习中使用，sample 数据以及获取 baseline. 也可以作为 beam_search=1 使用
        out_e, _, mem = self.encoder(features)
        out_e = out_e.view(-1, out_e.size(2))
        batch_size = out_e.size(0)
        state_d = self.init_hidden_with_feature(features)

        seq = []
        seq_log_probs = []
        if rl_training:
            baseline = [] # 若为强化学习，则需计算 baseline

        last_word = torch.FloatTensor(batch_size).long().zero_().cuda() 
        for t in range(self.seq_length): # 遍历生成句子
            last_word = Variable(last_word)
            log_probs, out, state_d = self.decode(out_e, last_word, state_d, True) # 最后一个True表示对lastword进行惩罚
            if t < 6: # 保证 t<6 时不生成结束符
                mask = np.zeros((batch_size, log_probs.size(-1)), 'float32')
                mask[:, 0] = -1000 # 结束符位置给与 mask
                mask = Variable(torch.from_numpy(mask)).cuda()
                log_probs = log_probs + mask
            if sample_max: # 取概率分布里最大的词
                sample_log_prob, last_word = torch.max(log_probs, 1)
                last_word = last_word.data.view(-1).long()
            else:
                prob_prev = torch.exp(log_probs.data).cpu()
                last_word = torch.multinomial(prob_prev, 1).cuda() # 按概率分布随机取样，返回采样的位置
                sample_log_prob = log_probs.gather(1, Variable(last_word)) # 取出采样相应位置的概率
                last_word = last_word.view(-1).long()
            if t == 0: # unfinished 记录是否 batch 内所有句子均生成结束
                unfinished = last_word > 0
            else:
                unfinished = unfinished * (last_word > 0)
            if unfinished.sum() == 0 and t >= 1 and not pad:
                break
            last_word = last_word * unfinished.type_as(last_word) # 若之前已经生成了结束符，则后面的lastword均为结束符
            # 记录生成的句子以及其概率分布
            seq.append(last_word)  
            seq_log_probs.append(sample_log_prob.view(-1))
            if rl_training:
                value = self.baseline_estimator(state_d[0].detach()) # 一个线性层
                baseline.append(value)
        # concatenate output lists
        seq = torch.cat([_.unsqueeze(1) for _ in seq], 1)  # batch_size * 5, seq_length
        seq_log_probs = torch.cat([_.unsqueeze(1) for _ in seq_log_probs], 1)
        seq = seq.view(-1, self.story_size, seq.size(1))
        seq_log_probs = seq_log_probs.view(-1, self.story_size, seq_log_probs.size(1))
        if rl_training:
            baseline = torch.cat([_.unsqueeze(1) for _ in baseline], 1)  # batch_size * 5, seq_length
            baseline = baseline.view(-1, self.story_size, baseline.size(1))
            return seq, seq_log_probs, baseline
        else:
            return seq, seq_log_probs

    def topK(self, features, beam_size=5):
        assert beam_size <= self.vocab_size and beam_size > 0
        if beam_size == 1:  # if beam_size is 1, then do greedy decoding, otherwise use beam search
            return self.sample(features, sample_max=True, rl_training=False)

        # encode the visual features
        out_e, _ = self.encoder(features)

        # reshape the inputs, making the sentence generation separately
        out_e = out_e.view(-1, out_e.size(2))

        ####################### decoding stage ##################################
        batch_size = out_e.size(0)

        # initialize decoder's state
        state_d = self.init_hidden_with_feature(features)

        topK = []
        for k in range(batch_size):
            out_e_k = out_e[k].unsqueeze(0).expand(beam_size, out_e.size(1)).contiguous()
            state_d_k = state_d[:, k, :].unsqueeze(1).expand(state_d.size(0), beam_size, state_d.size(2)).contiguous()

            last_word = Variable(torch.FloatTensor(beam_size).long().zero_().cuda())  # <BOS>
            log_probs, out, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)
            log_probs[:, 1] = log_probs[:, 1] - 1000  # never produce <UNK> token
            neg_log_probs = -log_probs

            all_outputs = np.ones((1, beam_size), dtype='int32')
            all_masks = np.ones_like(all_outputs, dtype="float32")
            all_costs = np.zeros_like(all_outputs, dtype="float32")
            for i in range(self.seq_length):
                next_costs = (all_costs[-1, :, None] + neg_log_probs.data.cpu().numpy() * all_masks[-1, :, None])
                (finished,) = np.where(all_masks[-1] == 0)
                next_costs[finished, 1:] = np.inf

                (indexes, outputs), chosen_costs = _smallest(next_costs, beam_size, only_first_row=i == 0)

                new_state_d = state_d_k.data.cpu().numpy()[:, indexes, :]

                all_outputs = all_outputs[:, indexes]
                all_masks = all_masks[:, indexes]
                all_costs = all_costs[:, indexes]

                last_word = Variable(torch.from_numpy(outputs)).cuda()
                state_d_k = Variable(torch.from_numpy(new_state_d)).cuda()

                log_probs, out, state_d_k = self.decode(out_e_k, last_word, state_d_k, True)

                log_probs[:, 1] = log_probs[:, 1] - 1000
                neg_log_probs = -log_probs

                all_outputs = np.vstack([all_outputs, outputs[None, :]])
                all_costs = np.vstack([all_costs, chosen_costs[None, :]])
                mask = outputs != 0
                all_masks = np.vstack([all_masks, mask[None, :]])
            topK.append(all_outputs[1:].transpose())

        topK = np.asarray(topK, 'int64')
        topK = topK.reshape(-1, 5, topK.shape[1], topK.shape[2])

        return topK
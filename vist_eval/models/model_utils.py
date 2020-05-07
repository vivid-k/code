import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from models.attention import luong_gate_attention, MultiHeadAttention

def get_sinusoid_encoding_table(d_hid, n_position=5):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim_en, hidden_dim_de, projected_size):
        super(AttentionLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim_en, projected_size)
        self.linear2 = nn.Linear(hidden_dim_de, projected_size)
        self.linear3 = nn.Linear(projected_size, 1, False)

    def forward(self, out_e, h):
        '''
        out_e: batch_size * num_frames * en_hidden_dim
        h : batch_size * de_hidden_dim
        '''
        assert out_e.size(0) == h.size(0)
        batch_size, num_frames, _ = out_e.size()
        hidden_dim = h.size(1)

        h_att = h.unsqueeze(1).expand(batch_size, num_frames, hidden_dim)
        x = F.tanh(F.dropout(self.linear1(out_e)) + F.dropout(self.linear2(h_att)))
        x = F.dropout(self.linear3(x))
        a = F.softmax(x.squeeze(2))

        return a

def graph_attn(alpha, cen_state, adj_state, max_len):
    """
    graph attention. calculate the graph attention score for cen_state
    Args:
        alpha: float hyper parameters
        cen_state: tensor acts as central node batch_size * 1 * hidden_dim
        other_state: tensor acts as adjacent node batch_size * max_len * hidden_dim
        M: tensor learned param matrix hidden_dim * hidden_dim
        max_len: int maximum number of adjacent node
    Returns:
        socre: tensor batch_size * max_len
    """
    batch_size = cen_state.shape[0]
    hidden_dim = cen_state.shape[-1]
    # concatenate 将解码节点与编码节点拼接，构成图
    state = torch.cat((cen_state.unsqueeze(1), adj_state), dim=1) # batch_size * max_len + 1 * hidden_dim
    
    M = nn.Linear(hidden_dim, hidden_dim).cuda() 
    W = M(state) # batch_size * max_len + 1 * hidden_dim
    W = torch.matmul(state, W.transpose(1, 2)) # batch_size * max_len + 1 * max_len + 1
    
    W_sum = torch.sum(W, dim=2) # batch_size * max_len + 1
    W_sum = torch.unsqueeze(W_sum, -1) # batch_size * max_len + 1 * 1
    W_sum = W_sum.repeat((1, 1, max_len + 1)) # batch_size * max_len + 1 * max_len + 1
    
    D = torch.eye(max_len + 1).cuda() # max_len + 1 * max_len + 1
    D = torch.unsqueeze(D, 0) # 1 * max_len + 1 * max_len + 1
    D = D.repeat((batch_size, 1, 1)) * W_sum # batch_size * max_len + 1 * max_len + 1 点乘
    P = alpha * torch.matmul(W, torch.inverse(D[:])) # batch_size * max_len + 1 * max_len + 1

    I = torch.unsqueeze(torch.eye(max_len + 1), 0).cuda() # 1 * max_len + 1 * max_len + 1
    I = I.repeat(batch_size, 1, 1) - P # batch_size * max_len + 1 * max_len + 1
    Q = torch.inverse(I[:]) # batch_size * max_len + 1 * max_len + 1
    
    Y = torch.cat((torch.ones((batch_size, 1)), torch.zeros(batch_size, max_len)), 1).cuda() # batch_size * max_len + 1
    Y = torch.unsqueeze(Y, -1)
    score = (1 - alpha) * torch.matmul(Q, Y)
    score = F.softmax(score, dim=1)
    # score_mask = (score.squeeze() - last) > 0
    # score_mask = score_mask.float()
    # score = (score.squeeze() - last) * score_mask
    # score_sum = score.sum(1).unsqueeze(1)
    # score = score[:] / score_sum[:]
    state = torch.matmul(state.transpose(1, 2), score).squeeze() # 64*512*1

    return state

def _smallest(matrix, k, only_first_row=False):
    # matrix ： beam*vocab（记录了到当前步骤的总cost）  k：beam
    # 选取beam个最小的
    if only_first_row: # 是否为第一个词，第一个词概率都相同，取第一行即可
        flatten = matrix[:1, :].flatten() # 取出第一行概率分布，9837
    else:
        flatten = matrix.flatten()
    args = np.argpartition(flatten, k)[:k] # 比第三名好的放在数组前面，差的放在后面，无序，返回索引
    args = args[np.argsort(flatten[args])] # 取出相应的值并排序，argsort返回下标，args取出相应索引值
    # 返回值：前面返回matrix中的位置，最后一个返回概率最大的三个值
    return np.unravel_index(args, matrix.shape), flatten[args] # 前面函数计算args在matrix维度的矩阵中位置

class VisualEncoder(nn.Module):

    def __init__(self, opt):
        super(VisualEncoder, self).__init__()
        self.feat_size = opt.feat_size # 2048
        self.embed_dim = opt.word_embed_dim # 512

        self.rnn_type = opt.rnn_type # gru
        self.num_layers = opt.num_layers # 1
        self.hidden_dim = opt.hidden_dim # 512
        self.dropout = opt.visual_dropout # 0.2
        self.story_size = opt.story_size # 5
        self.with_position = opt.with_position # False
        self.opt = opt
        # visual embedding layer
        self.visual_emb = nn.Sequential(nn.Linear(self.feat_size, self.embed_dim),
                                        nn.BatchNorm1d(self.embed_dim),
                                        nn.ReLU(True))
        self.hin_dropout_layer = nn.Dropout(self.dropout)

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        if self.opt.context_dec:
            self.rnn_dec = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                                 dropout=self.dropout, batch_first=True, bidirectional=False)
            # self.linear_fun = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            #                             nn.BatchNorm1d(self.hidden_dim),
            #                             nn.ReLU(True))
            # 线性层 + 门控
            # self.attention = MultiHeadAttention(8, self.hidden_dim, 64, 64)
            # self.pos_ffn = PositionwiseFeedForward(self.hidden_dim, 2048)
            self.attention = luong_gate_attention(self.hidden_dim, self.embed_dim)
            self.linear_read = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid())
            self.linear_write = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())
            self.linear_mem = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            # self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(self.hidden_dim), freeze=True)
        self.project_layer = nn.Linear(self.hidden_dim * 2, self.embed_dim)
        self.relu = nn.ReLU()
        if self.with_position: # 是否可以改为transformer那样的position
            self.position_embed = nn.Embedding(self.story_size, self.embed_dim)

    def init_hidden(self, batch_size, bi, dim):
        # LSTM的初始隐状态，默认为0
        weight = next(self.parameters()).data
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return weight.new(self.num_layers * times, batch_size, dim).zero_()
        else:
            return (weight.new(self.num_layers * times, batch_size, dim).zero_(),
                    weight.new(self.num_layers * times, batch_size, dim).zero_())

    def forward(self, input, hidden=None):

        batch_size, story_size = input.size(0), input.size(1) # (batch_size, 5, feat_size)
        emb = self.visual_emb(input.view(-1, self.feat_size)) # 过一个线性层，2048-512
        emb = emb.view(batch_size, story_size, -1)  # view变回三维 64*5*512
        rnn_input = self.hin_dropout_layer(emb)  # apply dropout
        # if hidden is None:
        #     hidden = self.init_hidden(batch_size, bi=True, dim=self.hidden_dim // 2) # 最后一个维度为512/2=256
        houts, hidden = self.rnn(rnn_input) #  hidden [2,64,512]
        
        out = emb + self.project_layer(houts) # 原始的 visual_emb + rnn输出的结果, 即残差连接， 改为concat？
        out = self.relu(out)  # (batch_size, 5, embed_dim)

        if self.with_position:
            for i in range(self.story_size):
                position = torch.tensor(input.data.new(batch_size).long().fill_(i))
                out[:, i, :] = out[:, i, :] + self.position_embed(position)
        
        if self.opt.context_dec:            
            state = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            # self.attention.init_context(out)
            # mem, weights = self.attention(out, selfatt=True)
            # pos_inp = torch.tensor([0,1,2,3,4]).unsqueeze(0).cuda()
            # pos_inp = pos_inp.repeat(batch_size, 1)
            # pos = self.position_enc(pos_inp)
            
            # T_input = out + pos
            # mem, _ = self.attention(T_input, T_input, T_input)
            # mem = self.pos_ffn(mem)
            mem = out
            mem = mem.sum(dim=1) # 64*512
            result = []
            self.attention.init_context(out)
            for i in range(self.story_size):
                # graph_res = graph_attn(self.opt.alpha, state[0].squeeze(), out, self.story_size) # 64*6*1
                # graph_res = torch.matmul(out.transpose(1, 2), weights).squeeze()
                att, _ = self.attention(state[0].squeeze())
                g_r = self.linear_read(torch.cat([state[0].squeeze(), att.squeeze()], dim=-1))
                # g_r = self.linear_read(state[0].squeeze())
                mem_inp = g_r * mem
                inp = torch.cat((out[:, i, :], mem_inp, att), 1)
                inp = self.linear_mem(inp).unsqueeze(1) # 64*1*512
                output, state = self.rnn_dec(inp, state)
                g_w = self.linear_write(state[0].squeeze())
                mem = g_w * mem
                result.append(output.squeeze())
            out = torch.stack(result).transpose(0, 1)
        return out, state, mem

class CaptionEncoder(nn.Module):

    def __init__(self, opt):
        super(CaptionEncoder, self).__init__()
        # embedding (input) layer options
        self.opt = opt
        self.embed_dim = opt.word_embed_dim
        
        # rnn layer options
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.hidden_dim = opt.hidden_dim
        self.dropout = opt.visual_dropout
        self.story_size = opt.story_size
        if self.opt.cnn_cap:
            self.sw1 = nn.Sequential(nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=1, padding=0), nn.BatchNorm1d(opt.hidden_dim), nn.ReLU())
            self.sw3 = nn.Sequential(nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim),
                                        nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim))
            self.sw33 = nn.Sequential(nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim),
                                        nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim),
                                        nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim))
            self.linear = nn.Sequential(nn.Linear(2*opt.hidden_dim, 2*opt.hidden_dim), nn.GLU(), nn.Dropout(opt.dropout))
            self.filter_linear = nn.Linear(3*opt.hidden_dim, opt.hidden_dim)
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
            self.cnn = nn.Sequential(nn.Conv1d(opt.hidden_dim, opt.hidden_dim, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(opt.hidden_dim))
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_dim, bidirectional=opt.bi, batch_first=True)
            if opt.bi:
                self.out_linear = nn.Sequential(nn.Linear(2*opt.hidden_dim, 2*opt.hidden_dim), nn.GLU(), nn.Dropout(opt.dropout))
            self.sigmoid = nn.Sigmoid()
        self.attention = luong_gate_attention(self.hidden_dim, self.embed_dim)

    def forward(self, input, embed):
        # input: 64*5*20，分别对每句话进行卷积，提取句子特征
        batch = input.size(0)
        input = input.view(batch*5, -1)
        mask = torch.zeros_like(input)
        mask = input > 0 # batch*5,20
        src_len = torch.sum(mask, dim=-1) # =batch*5
        input = embed(input)
        state = None
        if self.opt.cnn_cap:
            input = input.transpose(1, 2) # 320*512*20,卷积在最后一个维度扫

            # outputs = self.cnn(input)
            # outputs = outputs.transpose(1, 2)

            conv1 = self.sw1(input)
            conv3 = self.sw3(input)
            conv33 = self.sw33(input)
            conv = torch.cat((conv1, conv3, conv33), 1).transpose(1, 2)
            conv = self.filter_linear(conv) # 320*20*512
            outputs = conv
            if self.opt.self_att: # 对句子内部进行自注意力，提取关键词特征
                self.attention.init_context(input.transpose(1, 2).transpose(0, 1))
                att_out, weights = self.attention(input.transpose(1, 2), selfatt=True)
                gate = self.sigmoid(att_out.transpose(0, 1))
                outputs = gate * conv
        else: 
            ## gru+self-att
            # lengths, indices = torch.sort(src_len, dim=0, descending=True)
            # input = torch.index_select(input, dim=0, index=indices) # batch*5,20,512
            # embs = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True)
            # outputs, state = self.rnn(embs)
            # outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs)[0] # 19,batch*5,512
            # outputs = outputs.transpose(0,1)
            # # 排列为之前的顺序
            # _, ind = torch.sort(indices)
            # outputs = torch.index_select(outputs, dim=0, index=ind) # batch*5,seq_len,512
            # state = torch.index_select(state.squeeze(), dim=0, index=ind)
            # if self.opt.self_att:
            #     self.attention.init_context(outputs.transpose(0, 1))
            #     outputs, weights = self.attention(outputs, selfatt=True)
            #     outputs = outputs.transpose(0, 1)

            ## 先self——att，再gru
            if self.opt.self_att:
                self.attention.init_context(input.transpose(0, 1))
                outputs, weights = self.attention(input, selfatt=True)
                outputs = outputs.transpose(0, 1)
            else:
                outputs = input
            lengths, indices = torch.sort(src_len, dim=0, descending=True)
            outputs = torch.index_select(outputs, dim=0, index=indices) # batch*5,20,512
            embs = torch.nn.utils.rnn.pack_padded_sequence(outputs, lengths, batch_first=True)
            outputs, state = self.rnn(embs) # state(2,320,512)
            outputs = torch.nn.utils.rnn.pad_packed_sequence(outputs)[0] # 19,batch*5,512
            if self.opt.bi:
                outputs = self.out_linear(outputs.transpose(0,1))
            else:
                outputs = outputs.transpose(0,1)
            # 排列为之前的顺序
            _, ind = torch.sort(indices)
            outputs = torch.index_select(outputs, dim=0, index=ind) # batch*5,seq_len,512
            state = torch.index_select(state[0].squeeze(), dim=0, index=ind)

        return outputs, state


def graph_attention(alpha, cen_state, adj_state, max_len):
    """
    graph attention. calculate the graph attention score for cen_state

    Args:
        alpha: float hyper parameters
        cen_state: tensor acts as central node batch_size * 1 * hidden_dim
        other_state: tensor acts as adjacent node batch_size * max_len * hidden_dim
        M: tensor learned param matrix hidden_dim * hidden_dim
        max_len: int maximum number of adjacent node
    Returns:
        socre: tensor batch_size * max_len
    """
    batch_size = cen_state.shape[0]
    hidden_dim = cen_state.shape[-1]
    # concatenate 将解码节点与编码节点拼接，构成图
    state = torch.cat((cen_state.unsqueeze(1), adj_state), dim=1) # batch_size * max_len + 1 * hidden_dim
    
    M = nn.Linear(hidden_dim, hidden_dim).cuda() 
    W = M(state) # batch_size * max_len + 1 * hidden_dim
    W = torch.matmul(state, W.transpose(1, 2)) # batch_size * max_len + 1 * max_len + 1
    
    W_sum = torch.sum(W, dim=2) # batch_size * max_len + 1
    W_sum = torch.unsqueeze(W_sum, -1) # batch_size * max_len + 1 * 1
    W_sum = W_sum.repeat((1, 1, max_len + 1)) # batch_size * max_len + 1 * max_len + 1
    
    D = torch.eye(max_len + 1).cuda() # max_len + 1 * max_len + 1
    D = torch.unsqueeze(D, 0) # 1 * max_len + 1 * max_len + 1
    D = D.repeat((batch_size, 1, 1)) * W_sum # batch_size * max_len + 1 * max_len + 1 点乘
    P = alpha * torch.matmul(W, torch.inverse(D[:])) # batch_size * max_len + 1 * max_len + 1

    I = torch.unsqueeze(torch.eye(max_len + 1), 0).cuda() # 1 * max_len + 1 * max_len + 1
    I = I.repeat(batch_size, 1, 1) - P # batch_size * max_len + 1 * max_len + 1
    Q = torch.inverse(I[:]) # batch_size * max_len + 1 * max_len + 1
    
    Y = torch.cat((torch.ones((batch_size, 1)), torch.zeros(batch_size, max_len)), 1).cuda() # batch_size * max_len + 1
    Y = torch.unsqueeze(Y, -1)
    score = (1 - alpha) * torch.matmul(Q, Y)
    score = F.softmax(score[:, 1:], dim=1) # 64*6*1

    return score


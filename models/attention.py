import math
import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = nn.Linear(d_model, n_head * d_k)
#         self.w_ks = nn.Linear(d_model, n_head * d_k)
#         self.w_vs = nn.Linear(d_model, n_head * d_v)
#         nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

#         self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
#         self.layer_norm = nn.LayerNorm(d_model)

#         self.fc = nn.Linear(n_head * d_v, d_model)
#         nn.init.xavier_normal_(self.fc.weight)

#         self.dropout = nn.Dropout(dropout)


#     def forward(self, q, k, v, mask=None):

#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

#         sz_b, len_q, _ = q.size()
#         sz_b, len_k, _ = k.size()
#         sz_b, len_v, _ = v.size()

#         residual = q

#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

#         q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
#         k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
#         v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

#         # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
#         output, attn = self.attention(q, k, v, mask=mask)

#         output = output.view(n_head, sz_b, len_q, d_v)
#         output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

#         output = self.dropout(self.fc(output))
#         output = self.layer_norm(output + residual)

#         return output, attn
class Attention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, att_type, dim, dropout=0.0):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        if att_type == 'additive':
            self.W_additive = nn.Linear(in_features=2*dim, out_features=2*dim)
            self.V_additive = nn.Linear(in_features=2*dim, out_features=1)

        self.att_type = att_type

    def calc_att_dist(self, q, k, scale, attn_mask):

        if self.att_type == 'additive':
            _, seq_len, _ = k.size()
            q = q.repeat(1, seq_len, 1)  # bsz x seq_len x dim
            att_features = torch.cat([q, k], dim=2)  # bsz x seq_len x dim
            attention = self.V_additive(self.W_additive(att_features))  # bsz x seq_len x 1

        else:
            attention = torch.bmm(k, q.transpose(1, 2))
            if scale is not None:
                attention = attention * scale
            if attn_mask is not None:
                # 给需要mask的地方设置一个负无穷
                attention = attention.masked_fill_(attn_mask, -np.inf)
            # 计算softmax
            attention = self.softmax(attention)
            # 添加dropout
            attention = self.dropout(attention)

        return attention
    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries tensor, bsz x 1 x q_dim
            k: Keys tensor, bsz x seq_len x k_dim
            v: Values tensor, bsz x seq_len x v_dim
            scale: scale factor
            attn_mask: Masking tensor, bsz x seq_len x 1
        Returns:
            context vector and attention dist
        """
        # attention = torch.bmm(q, k.transpose(1, 2))
        # attention = torch.bmm(q, k).squeeze(1).unsqueeze(2)
        attention = torch.bmm(k, q.transpose(1, 2))
        if scale is not None:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention.transpose(1, 2), v).squeeze(1)
        # context = attention * v
        return context, attention
class MultiHeadAttention(nn.Module):

    def __init__(self, n_hidden_size, num_heads=4, att_type='scaled-dot', dropout=0.5):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = n_hidden_size // num_heads
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.zeros(n_hidden_size)).float()

        self.linear_q = nn.Linear(n_hidden_size, self.dim_per_head*num_heads)
        self.linear_k = nn.Linear(n_hidden_size, self.dim_per_head*num_heads)
        self.linear_v = nn.Linear(n_hidden_size, self.dim_per_head*num_heads)

        dim = None
        if att_type == 'additive':
            dim = self.dim_per_head
        self.dot_product_attention = Attention(att_type=att_type, dim=dim, dropout=dropout)

        self.linear_final = nn.Linear(self.dim_per_head*num_heads, n_hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(n_hidden_size)

        self.att_type = att_type

    def forward(self, key, value, attn_mask=None):
        """

        :param key: bsz x seq_len x n_hsz
        :param value: bsz x seq_len x n_hsz
        :param attn_mask: bsz x seq_len x 1
        :return:
        """
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads

        bsz, seq_len, _ = key.size()

        # construct the query
        query = self.query.repeat(bsz, 1, 1)  # bsz x 1 x n_hidden_size

        # linear projection
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # split by heads
        query = query.view(bsz*num_heads, -1, dim_per_head)
        key = key.view(bsz*num_heads, -1, dim_per_head)
        value = value.view(bsz*num_heads, -1 ,dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(2)
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        scale = None
        if self.att_type == 'scaled-dot':
            scale = math.sqrt(dim_per_head)

        context, att_dist = self.dot_product_attention(query, key, value, scale, attn_mask)
        heads = context.view(bsz, num_heads, -1)
        # concat heads
        context = context.view(bsz, num_heads*dim_per_head)

        # final output projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add norm layer
        # output = self.layer_norm(output)

        return output, query.view(bsz, num_heads*dim_per_head), heads

class luong_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(luong_attention, self).__init__()
        self.hidden_size, self.emb_size, self.pool_size = hidden_size, emb_size, pool_size
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        if pool_size > 0:
            self.linear_out = maxout(2*hidden_size + emb_size, hidden_size, pool_size)
        else:
            self.linear_out = nn.Sequential(nn.Linear(2*hidden_size + emb_size, hidden_size), nn.Tanh())
        self.softmax = nn.Softmax(dim=1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_h = self.linear_in(h).unsqueeze(2)    # batch * size * 1
        weights = torch.bmm(self.context, gamma_h).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1) # batch * size
        output = self.linear_out(torch.cat([c_t, h, x], 1))

        return output, weights


class luong_gate_attention(nn.Module):
    
    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob), 
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob), 
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2*hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob), 
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context

    def forward(self, h, selfatt=False):
        if selfatt:
            gamma_enc = self.linear_enc(self.context) # Batch_size * Length * Hidden_size
            gamma_h = gamma_enc.transpose(1, 2) # Batch_size * Hidden_size * Length
            weights = torch.bmm(gamma_enc, gamma_h) # Batch_size * Length * Length
            weights = self.softmax(weights/math.sqrt(512))
            c_t = torch.bmm(weights, gamma_enc) # Batch_size * Length * Hidden_size
            output = self.linear_out(torch.cat([gamma_enc, c_t], 2)) + self.context
            output = output.transpose(0, 1) # Length * Batch_size * Hidden_size
        else:
            gamma_h = self.linear_in(h).unsqueeze(2)
            weights = torch.bmm(self.context, gamma_h).squeeze(2) # batch*len
            weights = self.softmax(weights)
            c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights


class bahdanau_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, pool_size=0):
        super(bahdanau_attention, self).__init__()
        self.linear_encoder = nn.Linear(hidden_size, hidden_size)
        self.linear_decoder = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, 1)
        self.linear_r = nn.Linear(hidden_size*2+emb_size, hidden_size*2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, x):
        gamma_encoder = self.linear_encoder(self.context)           # batch * time * size
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)    # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder+gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1) # batch * size
        r_t = self.linear_r(torch.cat([c_t, h, x], dim=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights


class maxout(nn.Module):

    def __init__(self, in_feature, out_feature, pool_size):
        super(maxout, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.pool_size = pool_size
        self.linear = nn.Linear(in_feature, out_feature*pool_size)

    def forward(self, x):
        output = self.linear(x)
        output = output.view(-1, self.out_feature, self.pool_size)
        output = output.max(2)[0]

        return output


class bigru_attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, hidden_size):
        super(bigru_attention, self).__init__()
        self.linear_enc = nn.Linear(enc_dim, hidden_size)
        self.linear_dec = nn.Linear(dec_dim, hidden_size)
        self.linear_w = nn.Linear(hidden_size, 1)
        self.enc_dim = self.enc_dim
        self.dec_dim = self.dec_dim
        self.hidden_size = hidden_size
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc, dec, context):
        """
        bahdanau attention

        Args:
            enc: tensor batch_size * seq_len * enc_dim 编码时 RNN 的全部时刻的输出
            dec: tensor batch_size * dec_dim 解码时上一步隐藏状态
            context: tensor batch_size * 
        """
        seq_len = enc.size(1)
        enc = self.linear_enc(enc) # batch_size * seq_len * hidden_size
        dec = self.linear_dec(dec).unsqueeze(1) # batch_size * 1 * hidden_size
        dec = dec.repeat(1, seq_len, 1)
        weights = self.linear_w(self.activation(enc + dec)).squeeze(2) # batch_size * seq_len
        weights = self.softmax(weights)
        attn_out = torch.bmm(weights.unsqueeze(1)).squeeze(1)
        return attn_out, weights

class Overparam(nn.Module):
    def __init__(self, n_hidden_size):
        super().__init__()
        self.l1 = nn.Linear(n_hidden_size, 2 * n_hidden_size)
        # self.l2 = nn.Linear(2 * nhid, 2 * nhid)
        self.inner_act = torch.tanh # GELU()
        self.hidden_size = n_hidden_size

    def forward(self, x):
        c, f = self.l1(x).split(self.hidden_size, dim=-1)
        # c, f = self.l2(self.inner_act(self.l1(x))).split(self.nhid, dim=-1)
        return torch.sigmoid(f) * torch.tanh(c)

class FocusAttention(nn.Module):

    def __init__(self, n_hidden_size):
        super(FocusAttention, self).__init__()

        self.query = nn.Parameter(torch.zeros(1, n_hidden_size)).float()

        # self.query_projection = nn.Linear(in_features=n_hidden_size, out_features=n_hidden_size)
        self.key_projection = nn.Linear(in_features=n_hidden_size, out_features=n_hidden_size)
        # self.value_projection = nn.Linear(in_features=n_hidden_size, out_features=n_hidden_size)

        self.over_param = Overparam(n_hidden_size=n_hidden_size)

        self.n_hidden_size = n_hidden_size

    def attention(self, query, key, value, mask=None):
        # query: 1 x n_hsz
        # key: bsz x seq_len x n_hsz
        # value: bsz x seq_len x n_hsz
        query = query.unsqueeze(1)

        att = torch.matmul(query, key.transpose(1, 2)).squeeze(1)  # bsz x seq_len
        att = att / math.sqrt(self.n_hidden_size)

        if mask is not None:
            att = att.masked_fill(mask=mask, value=-np.inf)
        att = torch.softmax(att, dim=1)

        context = torch.matmul(att.unsqueeze(1), value).squeeze(1)  # bsz x n_hsz

        return att, context

    def forward(self, key, value, key_features=None, mask=None, using_over_param=False):
        # key: bsz x seq_len x n_hidden_size
        if key_features is None:
            key_features = self.key_projection(key)

        query = self.query
        # query_feature = self.query_projection(query)
        query_feature = query
        if using_over_param:
            query_feature = self.over_param(query_feature)

        # value = self.value_projection(key)
        value = key
        att, context = self.attention(query=query_feature, key=key_features, value=value)

        return context, query.transpose(0,2)

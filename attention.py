import torch
import torch.nn as nn
import numpy as np
import math


class LuongAttention(nn.Module):
    """
    calculating the context vector by the current hidden state of decoder s_t
    there are there different ways to calculate similar between s_t and h_i which
    is the hidden state of encoder:dot, general, concat
    """
    def __init__(self, hidden_size, align_type, is_coverage):
        super(LuongAttention, self).__init__()
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.decode_projection = nn.Linear(hidden_size*2, hidden_size * 2)

        self.V = nn.Linear(hidden_size * 2, 1, bias=False)
        self.W_c = nn.Linear(1, hidden_size * 2, bias=False)

        self.align_type = align_type
        self.is_coverage = is_coverage

    def calculate_scores(self, encoder_hs_features, s_t, coverage):
        # hs_encoder: bsz x seq_len x 2*hsz
        # s_t: bsz * hsz

        if self.align_type is None or self.is_coverage:

            s_t_expand = s_t.unsqueeze(1).expand(encoder_hs_features.size())  # bsz x seq_len x 2*hsz
            attention_features = encoder_hs_features + s_t_expand
            if self.is_coverage:

                coverage_input = coverage.unsqueeze(2)  # bsz x seq_len x 1
                coverage_feature = self.W_c(coverage_input)  # bsz x seq_len x 2*hsz
                attention_features = attention_features + coverage_feature

            attention_features = torch.tanh(attention_features)  # bsz x seq_len x 2*hsz
            scores = self.V(attention_features)  # bsz x seq_len x 1

        elif self.align_type.startsWith('general'):
            s_t = s_t.unsqueeze(2)  # bsz x 2*hsz x 1
            scores = torch.bmm(self.W(encoder_hs_features), s_t)  # bsz x seq_len x 1
        else:
            # dot
            s_t = s_t.unsqueeze(2)  # bsz x 2*hsz x 1
            scores = torch.bmm(encoder_hs_features, s_t)  # bsz x seq_len x 1

        # bsz x seq_len
        return scores.squeeze(2)

    def forward(self, s_t, encoder_out, encoder_hs_features,  enc_padding_mask, coverage=None):
        # s_t: bsz x hsz, is the hidden state of current timestep
        # y: bsz x emb_dim, is the output of previous timestep
        # hs_encoder: bsz x seq_len x 2*hsz
        # enc_padding_mask: bsz x seq_len

        decoder_feature = self.decode_projection(s_t)  # bsz x att_input_dim -> batch_size x 2*hsz

        # coverage_feature = None
        # if self.is_coverage:
        #     coverage_input = coverage.unsqueeze(2)  # b x seq_len x 1
        #     coverage_feature = self.W_c(coverage_input)  # b x seq_len x 2*hidden_size

        # batch_size x seq_len
        scores = self.calculate_scores(encoder_hs_features=encoder_hs_features, s_t=decoder_feature,
                                       coverage=coverage)

        # mask
        scores = scores.masked_fill(mask=enc_padding_mask, value=-np.inf)
        att_dist = torch.softmax(scores, dim=1)  # bsz x seq_len
        # normalization_factor = weights.sum(1, keepdim=True)
        # att_dist = weights / normalization_factor

        # context[bsz, 2*hsz] = weights[bsz, 1, seq_len] x [bsz, seq_len, 2*hsz]
        context = torch.bmm(att_dist.unsqueeze(1), encoder_out).squeeze(1)

        if self.is_coverage:
            coverage = coverage + att_dist

        return context, att_dist, coverage


class FocusDecoderAttention(nn.Module):
    """
        calculating the context vector by the current hidden state of decoder s_t
        there are there different ways to calculate similar between s_t and h_i which
        is the hidden state of encoder:dot, general, concat
        """

    def __init__(self, hidden_size, align_type, is_coverage, is_focus):
        super(FocusDecoderAttention, self).__init__()
        self.encoder_projection = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

        self.decode_projection = nn.Linear(hidden_size * 2, hidden_size * 2, bias=True)  # ???

        self.V = nn.Linear(hidden_size * 2, 1, bias=False)
        self.W_c = nn.Linear(1, hidden_size * 2, bias=False)
        # for focus
        self.W_f = nn.Linear(hidden_size*2, hidden_size*2, bias=False)

        self.align_type = align_type
        self.is_coverage = is_coverage
        self.is_focus = is_focus

    def calculate_scores(self, encoder_hs_features, s_t, coverage, this_focus):
        # hs_encoder: bsz x seq_len x 2*hsz
        # s_t: bsz * hsz

        if self.align_type is None or self.is_coverage or self.is_focus:

            s_t_expand = s_t.unsqueeze(1).expand(encoder_hs_features.size())  # bsz x seq_len x 2*hsz
            attention_features = encoder_hs_features + s_t_expand
            if self.is_coverage:
                coverage_input = coverage.unsqueeze(2)  # bsz x seq_len x 1
                coverage_feature = self.W_c(coverage_input)  # bsz x seq_len x 2*hsz
                attention_features = attention_features + coverage_feature

            if self.is_focus:
                focus_features = self.W_f(this_focus)
                focus_features = focus_features.unsqueeze(1)  # bsz x 1 x 2*hsz
                attention_features = attention_features + focus_features  # bsz x seq_len x 2*hsz

            attention_features = torch.tanh(attention_features)  # bsz x seq_len x 2*hsz
            scores = self.V(attention_features)  # bsz x seq_len x 1

        elif self.align_type.startsWith('general'):
            s_t = s_t.unsqueeze(2)  # bsz x 2*hsz x 1
            scores = torch.bmm(self.W(encoder_hs_features), s_t)  # bsz x seq_len x 1
        else:
            # dot
            s_t = s_t.unsqueeze(2)  # bsz x 2*hsz x 1
            scores = torch.bmm(encoder_hs_features, s_t)  # bsz x seq_len x 1

        # bsz x seq_len
        return scores.squeeze(2)

    def forward(self, s_t, encoder_out, encoder_hs_features, enc_padding_mask, coverage=None, this_focus=None):
        # s_t: bsz x hsz, is the hidden state of current timestep
        # y: bsz x emb_dim, is the output of previous timestep
        # hs_encoder: bsz x seq_len x 2*hsz
        # enc_padding_mask: bsz x seq_len

        decoder_feature = self.decode_projection(s_t)  # bsz x att_input_dim -> batch_size x 2*hsz

        # coverage_feature = None
        # if self.is_coverage:
        #     coverage_input = coverage.unsqueeze(2)  # b x seq_len x 1
        #     coverage_feature = self.W_c(coverage_input)  # b x seq_len x 2*hidden_size

        # batch_size x seq_len
        scores = self.calculate_scores(encoder_hs_features=encoder_hs_features, s_t=decoder_feature,
                                       coverage=coverage, this_focus=this_focus)

        # mask
        scores = scores.masked_fill(mask=enc_padding_mask, value=-np.inf)
        att_dist = torch.softmax(scores, dim=1)  # bsz x seq_len
        # normalization_factor = weights.sum(1, keepdim=True)
        # att_dist = weights / normalization_factor

        # context[bsz, 2*hsz] = weights[bsz, 1, seq_len] x [bsz, seq_len, 2*hsz]
        context = torch.bmm(att_dist.unsqueeze(1), encoder_out).squeeze(1)

        if self.is_coverage:
            coverage = coverage + att_dist

        return context, att_dist, coverage

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

    def forward(self, key, key_features=None, mask=None, using_over_param=False):
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

        return context


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

        # concat heads
        context = context.view(bsz, num_heads*dim_per_head)

        # final output projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add norm layer
        # output = self.layer_norm(output)

        return output







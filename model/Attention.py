import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    # FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, hidden_dim=768, dropout_rate=0.2):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        # residual connection
        return x + self.dropout(sublayer(self.layer_norm(x)))


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout_rate=0.1, single_attention_hidden_dim=64):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        # single_attention_head hidden size
        self.single_attention_hidden_dim = single_attention_hidden_dim

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, num_att_head, seq_len_q, single_attention_hidden_dim)
        # seq_len_k = seq_len_v
        # key: (batch_size, num_att_head, seq_len_k, single_attention_hidden_dim)
        # value: (batch_size, num_att_head, seq_len_v, single_attention_hidden_dim)
        # mask: (batch_size, 1, 1, seq_len_k)

        # attention_scores: (batch_size, num_head, seq_len_q, seq_len_k)
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / np.sqrt(self.single_attention_hidden_dim)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)

        # attention_probs: (batch_size, num_head, seq_len_q, seq_len_k)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # context: (batch_size, num_head, seq_len_q, single_attention_hidden_dim)
        context = torch.matmul(attention_probs, value)

        return context, attention_probs


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_attention_head=8, hidden_dim=768, all_attention_hidden_dim=768,
                 single_attention_hidden_dim=64, dropout_rate=0.1):
        super(MultiHeadedAttention, self).__init__()

        self.num_attention_head = num_attention_head
        self.hidden_dim = hidden_dim
        self.all_attention_hidden_dim = all_attention_hidden_dim
        self.single_attention_hidden_dim = single_attention_hidden_dim

        self.attention_layer = ScaleDotProductAttention(dropout_rate=dropout_rate,
                                                        single_attention_hidden_dim=single_attention_hidden_dim)
        self.projection_query = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)
        self.projection_key = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)
        self.projection_value = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)

        self.projection_final = nn.Linear(self.all_attention_hidden_dim, self.hidden_dim)

    def forward(self, hidden_states, mask=None, src=True):
        # hidden_states[0, 1, 2]: (batch_size, seq_len, hidden_dim)
        # mask: src (batch_size, 1, 1, seq_len) or tar (batch_size, 1, seq_len, seq_len)
        batch_size = hidden_states[0].shape[0]

        if mask is not None:
            if src:
                mask = mask.unsqueeze(1).unsqueeze(1)
            else:
                mask = mask.unsqueeze(1)

        # Projection (batch_size, num_attention_head, seq_len, single_attention_hidden_dim))
        queries = self.projection_query(hidden_states[0]).view(batch_size, self.num_attention_head, -1,
                                                               self.single_attention_hidden_dim)
        keys = self.projection_key(hidden_states[1]).view(batch_size, self.num_attention_head, -1,
                                                          self.single_attention_hidden_dim)
        values = self.projection_value(hidden_states[2]).view(batch_size, self.num_attention_head, -1,
                                                              self.single_attention_hidden_dim)

        # Self-Attention, context: (batch_size, num_head, seq_len, single_attention_hidden_dim)
        context, _ = self.attention_layer(queries, keys, values, mask)

        # Final projection
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.all_attention_hidden_dim)
        context = self.projection_final(context)

        # context: (batch_size, seq_len, hidden_dim)
        return context


class EncoderLayer(nn.Module):
    def __init__(self, num_attention_head=8, hidden_dim=768, all_attention_hidden_dim=768,
                 single_attention_hidden_dim=64, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadedAttention(num_attention_head=num_attention_head, hidden_dim=hidden_dim,
                                                   all_attention_hidden_dim=all_attention_hidden_dim,
                                                   single_attention_hidden_dim=single_attention_hidden_dim,
                                                   dropout_rate=dropout_rate)
        self.feed_forward_nn = PositionwiseFeedForward(hidden_dim, hidden_dim * 4)
        self.sublayer = clones(SublayerConnection(), 2)

    def forward(self, hidden_states, mask):
        # hidden_states: (batch_size, seq_len, hidden_dim)
        # mask: (batch_size, seq_len)

        # hidden_states: (batch_size, seq_len, hidden_dim)
        hidden_states = self.sublayer[0](hidden_states, lambda x: self.self_attention([x, x, x], mask))
        hidden_states = self.sublayer[1](hidden_states, self.feed_forward_nn)

        return hidden_states


class CrossAttention(nn.Module):
    def __init__(self, num_attention_head=12, hidden_dim=768, dropout_rate=0.1):
        super(CrossAttention, self).__init__()

        self.num_attention_head = num_attention_head
        self.hidden_dim = hidden_dim
        self.single_attention_hidden_dim = hidden_dim // num_attention_head
        self.all_attention_hidden_dim = hidden_dim // num_attention_head * num_attention_head

        self.attention_layer = ScaleDotProductAttention(dropout_rate=dropout_rate,
                                                        single_attention_hidden_dim=self.single_attention_hidden_dim)
        self.projection_query = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)
        self.projection_key = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)
        self.projection_value = nn.Linear(self.hidden_dim, self.all_attention_hidden_dim)
        self.projection_final = nn.Linear(self.all_attention_hidden_dim, self.hidden_dim)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.02)

        self.projection_query.apply(init_weights)
        self.projection_key.apply(init_weights)
        self.projection_value.apply(init_weights)
        self.projection_final.apply(init_weights)

    def forward(self, query, key, value, mask):

        batch_size = query.shape[0]

        if mask is not None:
            # mask: (batch_size, 1, seq_len_q, 1)
            mask = mask.unsqueeze(1).unsqueeze(-1)

        # Projection (batch_size, num_attention_head, seq_len_q, single_attention_hidden_dim))
        queries = self.projection_query(query).view(batch_size, self.num_attention_head, -1,
                                                               self.single_attention_hidden_dim)

        # Projection (batch_size, num_attention_head, seq_len_k, single_attention_hidden_dim))
        keys = self.projection_key(key).view(batch_size, self.num_attention_head, -1,
                                                          self.single_attention_hidden_dim)

        # Projection (batch_size, num_attention_head, seq_len_v, single_attention_hidden_dim))
        values = self.projection_value(value).view(batch_size, self.num_attention_head, -1,
                                                              self.single_attention_hidden_dim)

        # Self-Attention, context: (batch_size, num_head, seq_len_q, single_attention_hidden_dim)
        context, _ = self.attention_layer(queries, keys, values, mask)

        # Final projection
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.all_attention_hidden_dim)
        context = self.projection_final(context)

        # context: (batch_size, seq_len_q, hidden_dim)
        return context


class AttentionOverAttention(nn.Module):
    def __init__(self, hidden_size=768):
        super(AttentionOverAttention, self).__init__()

        self.question_projection_start = nn.Linear(hidden_size, hidden_size)
        self.context_projection_start = nn.Linear(hidden_size, hidden_size)
        self.question_projection_end = nn.Linear(hidden_size, hidden_size)
        self.context_projection_end = nn.Linear(hidden_size, hidden_size)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, std=0.02)

        self.question_projection_start.apply(init_weights)
        self.context_projection_start.apply(init_weights)
        self.question_projection_end.apply(init_weights)
        self.context_projection_end.apply(init_weights)

    def forward(self, fuse_hidden_question, fuse_hidden_context, attention_mask):
        # fuse_hidden_question: (batch_size, seq_len_q, hidden_size)
        # fuse_hidden_context: (batch_size, seq_len_c, hidden_size)
        # attention_mask: (batch_size, seq_len_c)
        question_start = self.question_projection_start(fuse_hidden_question).permute(0, 2, 1)
        question_end = self.question_projection_end(fuse_hidden_question).permute(0, 2, 1)

        # bs, context seq len, hidden size
        context_start = self.context_projection_start(fuse_hidden_context)
        context_end = self.context_projection_end(fuse_hidden_context)

        # Attention-over-Attention, bs, context seq len, question seq len
        aoa_M_start = context_start.bmm(question_start)
        aoa_M_end = context_end.bmm(question_end)

        attention_mask = attention_mask.unsqueeze(-1)
        aoa_M_start = aoa_M_start.masked_fill(attention_mask == 0, -1e4)
        aoa_M_end = aoa_M_end.masked_fill(attention_mask == 0, -1e4)

        # document level attention, aoa alpha, (bs, context seq len, question seq len), over document
        aoa_alpha_start = torch.softmax(aoa_M_start, dim=1)
        aoa_alpha_end = torch.softmax(aoa_M_end, dim=1)

        # query level attention, aoa beta, (bs, 1, question seq len), over query then take mean on document
        aoa_beta_start = torch.softmax(aoa_M_start, dim=2).mean(1).unsqueeze(1)
        aoa_beta_end = torch.softmax(aoa_M_end, dim=2).mean(1).unsqueeze(1)

        # aos s, bs, context seq len
        aoa_s_start = aoa_alpha_start.bmm(aoa_beta_start.permute(0, 2, 1)).squeeze(-1)
        aoa_s_end = aoa_alpha_end.bmm(aoa_beta_end.permute(0, 2, 1)).squeeze(-1)

        return aoa_s_start, aoa_s_end

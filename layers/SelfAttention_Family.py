import mindspore as ms
import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention  

class FullAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale or 1. / np.sqrt(factor)
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)

    def construct(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale

        scores = ops.BatchMatMul()(queries, keys.swapaxes(-2, -1))

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores = ops.MaskedFill()(-np.inf, attn_mask.bool(), scores)

        A = self.dropout(ops.Softmax(axis=-1)(scale * scores))
        V = ops.BatchMatMul()(A, values)

        if self.output_attention:
            return (V, A)
        else:
            return (V, None)
        
class ProbAttention(nn.Cell):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(1 - attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = ops.Tile()(K.unsqueeze(-3), (1, 1, L_Q, 1, 1))
        index_sample = ms.Tensor(np.random.randint(L_K, size=(L_Q, sample_k)))
        K_sample = K_expand[:, :, np.arange(L_Q)[:, None], index_sample, :]
        Q_K_sample = ops.BatchMatMul()(Q.unsqueeze(-2), K_sample.swapaxes(-2, -1)).squeeze()

        M = ops.ReduceMax(axis=-1, keep_dims=False)(Q_K_sample) - ops.ReduceMean(axis=-1, keep_dims=False)(Q_K_sample)
        M_top = ops.TopK(sorted=False)(M, n_top)[1]

        Q_reduce = Q[np.arange(B)[:, None, None], np.arange(H)[None, :, None], M_top, :]
        Q_K = ops.BatchMatMul()(Q_reduce, K.swapaxes(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = ops.ReduceMean(axis=-2, keep_dims=False)(V)
            context = ops.Tile()(V_sum.unsqueeze(-2), (1, 1, L_Q, 1))
        else:
            context = ops.CumSum(axis=-2)(V)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores = ops.MaskedFill()(-np.inf, attn_mask.bool(), scores)

        attn = ops.Softmax(axis=-1)(scores)
        context_in = context_in.copy()
        context_in[np.arange(B)[:, None, None], np.arange(H)[None, :, None], index, :] = ops.BatchMatMul()(attn, V)
        if self.output_attention:
            attns = (ms.Tensor(np.ones([B, H, L_V, L_V])) / L_V).astype(attn.dtype).to(attn.device)
            attns[np.arange(B)[:, None, None], np.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def construct(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context, attn
    
class AttentionLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Dense(d_model, d_keys * n_heads)
        self.key_projection = nn.Dense(d_model, d_keys * n_heads)
        self.value_projection = nn.Dense(d_model, d_values * n_heads)
        self.out_projection = nn.Dense(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def construct(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).reshape(B, L, H, -1)
        keys = self.key_projection(keys).reshape(B, S, H, -1)
        values = self.value_projection(values).reshape(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn
    

class ReformerLayer(nn.Cell):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super(ReformerLayer, self).__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return ops.Concat(axis=1)((queries, ms.Tensor(np.zeros([B, fill_len, C]), queries.dtype)))

    def construct(self, queries, keys, values, attn_mask):
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

# import torch
# import torch.nn as nn

# import numpy as np
# from math import sqrt
# from utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention


# class FullAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(FullAttention, self).__init__()
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("blhe,bshe->bhls", queries, keys)

#         if self.mask_flag:
#             if attn_mask is None:
#                 attn_mask = TriangularCausalMask(B, L, device=queries.device)

#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         V = torch.einsum("bhls,bshd->blhd", A, values)

#         if self.output_attention:
#             return (V.contiguous(), A)
#         else:
#             return (V.contiguous(), None)


# class ProbAttention(nn.Module):
#     def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
#         super(ProbAttention, self).__init__()
#         self.factor = factor
#         self.scale = scale
#         self.mask_flag = mask_flag
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)

#     def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
#         # Q [B, H, L, D]
#         B, H, L_K, E = K.shape
#         _, _, L_Q, _ = Q.shape

#         # calculate the sampled Q_K
#         K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
#         index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
#         K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
#         Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

#         # find the Top_k query with sparisty measurement
#         M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
#         M_top = M.topk(n_top, sorted=False)[1]

#         # use the reduced Q to calculate Q_K
#         Q_reduce = Q[torch.arange(B)[:, None, None],
#                    torch.arange(H)[None, :, None],
#                    M_top, :]  # factor*ln(L_q)
#         Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

#         return Q_K, M_top

#     def _get_initial_context(self, V, L_Q):
#         B, H, L_V, D = V.shape
#         if not self.mask_flag:
#             # V_sum = V.sum(dim=-2)
#             V_sum = V.mean(dim=-2)
#             contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
#         else:  # use mask
#             assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
#             contex = V.cumsum(dim=-2)
#         return contex

#     def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
#         B, H, L_V, D = V.shape

#         if self.mask_flag:
#             attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
#             scores.masked_fill_(attn_mask.mask, -np.inf)

#         attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

#         context_in[torch.arange(B)[:, None, None],
#         torch.arange(H)[None, :, None],
#         index, :] = torch.matmul(attn, V).type_as(context_in)
#         if self.output_attention:
#             attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
#             attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
#             return (context_in, attns)
#         else:
#             return (context_in, None)

#     def forward(self, queries, keys, values, attn_mask):
#         B, L_Q, H, D = queries.shape
#         _, L_K, _, _ = keys.shape

#         queries = queries.transpose(2, 1)
#         keys = keys.transpose(2, 1)
#         values = values.transpose(2, 1)

#         U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
#         u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

#         U_part = U_part if U_part < L_K else L_K
#         u = u if u < L_Q else L_Q

#         scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

#         # add scale factor
#         scale = self.scale or 1. / sqrt(D)
#         if scale is not None:
#             scores_top = scores_top * scale
#         # get the context
#         context = self._get_initial_context(values, L_Q)
#         # update the context with selected top_k queries
#         context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

#         return context.contiguous(), attn


# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         queries = self.query_projection(queries).view(B, L, H, -1)
#         keys = self.key_projection(keys).view(B, S, H, -1)
#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask
#         )
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn


# class ReformerLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None, causal=False, bucket_size=4, n_hashes=4):
#         super().__init__()
#         self.bucket_size = bucket_size
#         self.attn = LSHSelfAttention(
#             dim=d_model,
#             heads=n_heads,
#             bucket_size=bucket_size,
#             n_hashes=n_hashes,
#             causal=causal
#         )

#     def fit_length(self, queries):
#         # inside reformer: assert N % (bucket_size * 2) == 0
#         B, N, C = queries.shape
#         if N % (self.bucket_size * 2) == 0:
#             return queries
#         else:
#             # fill the time series
#             fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
#             return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

#     def forward(self, queries, keys, values, attn_mask):
#         # in Reformer: defalut queries=keys
#         B, N, C = queries.shape
#         queries = self.attn(self.fit_length(queries))[:, :N, :]
#         return queries, None

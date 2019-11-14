import sys
import math
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits


def get_sinusoid_encoding_table(n_src_vocab, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])

    # sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    # sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]

        if mems is not None:
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            ##### layer normalization
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, sharing_phase_weight,dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.qkv_net1 = self.qkv_net if sharing_phase_weight else nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.o_net1 = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)
        self.d_net = nn.Linear(self.d_model + self.d_model, self.d_model, bias=False)

    def forward(self, w_real, w_phase, r, r_w_bias, r_r_bias, attn_mask=None, mems=None,mems_phase=None):
        qlen, rlen, bsz = w_real.size(0), r.size(0), w_real.size(1)

        if mems is not None:
            cat_real = torch.cat([mems, w_real], 0)
            cat_phase = torch.cat([mems_phase, w_phase], 0)
            if self.pre_lnorm:
                w_heads_real = self.qkv_net(self.layer_norm(cat_real))
                w_heads_phase = self.qkv_net1(self.layer_norm(cat_phase))
            else:
                w_heads_real = self.qkv_net(cat_real)
                w_heads_phase = self.qkv_net1(cat_phase)
            r_head_k = self.r_net(r)
            # print('ok')
            # exit()
            w_head_q_real, w_head_k_real, w_head_v_real = torch.chunk(w_heads_real, 3, dim=-1)
            w_head_q_phase, w_head_k_phase, w_head_v_phase = torch.chunk(w_heads_phase, 3, dim=-1)
            w_head_q_real = w_head_q_real[-qlen:]
            w_head_q_phase = w_head_q_phase[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads_real = self.qkv_net(self.layer_norm(w_real))
                w_heads_phase = self.qkv_net1(self.layer_norm(w_phase))
            else:
                w_heads_real = self.qkv_net(w_real)
                w_heads_phase = self.qkv_net1(w_phase)
            r_head_k = self.r_net(r)
            w_head_q_real, w_head_k_real, w_head_v_real = torch.chunk(w_heads_real, 3, dim=-1)
            w_head_q_phase, w_head_k_phase, w_head_v_phase = torch.chunk(w_heads_phase, 3, dim=-1)

        klen = w_head_k_real.size(0)

        w_head_q_real = w_head_q_real.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_q_phase = w_head_q_phase.view(qlen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        w_head_k_real = w_head_k_real.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_k_phase = w_head_k_phase.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        w_head_v_real = w_head_v_real.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head
        w_head_v_phase = w_head_v_phase.view(klen, bsz, self.n_head, self.d_head)  # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)  # qlen x n_head x d_head

        #### compute attention score
        rw_head_q_real = w_head_q_real + r_w_bias  # qlen x bsz x n_head x d_head
        rw_head_q_phase = w_head_q_phase + r_w_bias  # qlen x bsz x n_head x d_head

        AC_real = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q_real, w_head_k_real)) - torch.einsum('ibnd,jbnd->ijbn', (
        rw_head_q_phase, w_head_k_phase))  # qlen x klen x bsz x n_head
        AC_phase = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q_real, w_head_k_phase)) + torch.einsum('ibnd,jbnd->ijbn', (
        rw_head_q_real, w_head_k_phase))  # qlen x klen x bsz x n_head

        rr_head_q_real = w_head_q_real + r_r_bias
        rr_head_q_phase = w_head_q_phase + r_r_bias

        BD_real = torch.einsum('ibnd,jnd->ijbn', (rr_head_q_real, r_head_k))  # qlen x klen x bsz x n_head
        BD_phase = torch.einsum('ibnd,jnd->ijbn', (rr_head_q_phase, r_head_k))  # qlen x klen x bsz x n_head

        BD_real = self._rel_shift(BD_real)
        BD_phase = self._rel_shift(BD_phase)

        # [qlen x klen x bsz x n_head]
        AC = AC_real * AC_real + AC_phase * AC_phase
        AC = torch.sqrt(AC)

        BD = BD_real * BD_real + BD_phase * BD_phase
        BD = torch.sqrt(BD)

        attn_score = AC + BD
        #

        # ACBD_real = AC_real + BD_real
        # ACBD_phase = AC_phase + BD_phase
        # ACBD_norm = ACBD_real * ACBD_real + ACBD_phase * ACBD_phase
        # attn_score = torch.sqrt(ACBD_norm)


        attn_score.mul_(self.scale)
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:, :, :, None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec_real = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v_real))
        attn_vec_phase = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v_phase))

        # [qlen x bsz x n_head x d_head]

        attn_vec_real = attn_vec_real.contiguous().view(attn_vec_real.size(0), attn_vec_real.size(1),
                                                        self.n_head * self.d_head)
        attn_vec_phase = attn_vec_phase.contiguous().view(attn_vec_phase.size(0), attn_vec_phase.size(1),
                                                          self.n_head * self.d_head)

        ##### linear projection
        attn_out_real = self.o_net(attn_vec_real)
        attn_out_phase = self.o_net1(attn_vec_phase)
        attn_out_real = self.drop(attn_out_real)
        attn_out_phase = self.drop(attn_out_phase)

        if self.pre_lnorm:
            ##### residual connection
            output_real = attn_out_real
            output_phase = attn_out_phase
        else:
            ##### residual connection + layer normalization
            # print(attn_out_real.size())
            output_real = self.layer_norm(w_real + attn_out_real)
            output_phase = self.layer_norm(w_phase + attn_out_phase)
            # output=torch.cat([output_real,output_phase],2)

        # output = torch.cat([output_real, output_phase], 2)
        # output = self.d_net(output)

        return output_real, output_phase


class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]  # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output_real = self.pos_ff(output)

        return output_real, output_phase


class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                                  **kwargs)
        self.pos_ff_real = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))
        self.pos_ff_imag =  PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):
        output_real, output_phase = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                                                  attn_mask=dec_attn_mask,
                                                  mems=mems)
        output_real = self.pos_ff_real(output_real)
        output_phase = self.pos_ff_imag(output_phase)

        return output_real, output_phase


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,sharing_phase_weight,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                                                         d_head, dropout,sharing_phase_weight, **kwargs)
        self.pos_ff_real = PositionwiseFF(d_model, d_inner, dropout,
                                          pre_lnorm=kwargs.get('pre_lnorm'))
        self.pos_ff_imag = self.pos_ff_real if sharing_phase_weight else PositionwiseFF(d_model, d_inner, dropout,
                                          pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp_real, dec_inp_phase, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None, mems_phase = None):
        output_real, output_phase = self.dec_attn(dec_inp_real, dec_inp_phase, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems,mems_phase=mems_phase)
        # output_real = self.pos_ff(output_real)
        output_real = self.pos_ff_real(output_real)
        output_phase = self.pos_ff_imag(output_phase)

        return output_real, output_phase


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = cutoffs + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                                   dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt,sharing_phase_weight,tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1,schema = 0):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token
        self.schema = schema

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.sharing_phase_weight=sharing_phase_weight
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type

        self.layers = nn.ModuleList()
        if attn_type == 0:  # the default attention

            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, self.sharing_phase_weight,tgt_len=tgt_len,
                                                    ext_len=ext_len, mem_len=mem_len, dropatt=dropatt,
                                                    pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1:  # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]:  # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length
        self.clamp_len = clamp_len

        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):

        self.pos_emb = PositionalEmbedding(self.d_model)
        if self.attn_type == 0:  # default attention
            if self.schema == 0:
                weight = get_sinusoid_encoding_table(self.n_token, self.d_model, padding_idx=0)
                self.pos_emb_phase = nn.Embedding.from_pretrained(weight,freeze=False)  # , freeze=True
                self.pos_emb_phase.weight.requires_grad = True
            elif self.schema == 1:
                self.period = torch.nn.Embedding(self.n_token,1)

            elif self.schema == 2 :
                self.period = nn.Parameter(torch.Tensor(self.d_model))  # , freeze=True


            else:
                exit()

            # self.pos_emb_phase.weight.requires_grad = True

            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1:  # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2:  # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3:  # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None,mems_phase=None):
        qlen, bsz = dec_inp.size()

        word_emb = self.word_emb(dec_inp)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :,
                            None]  # -1
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        hids = []
        hids_phase = []
        if self.attn_type == 0:  # default


            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)

            pos_emb = self.pos_emb(pos_seq)


            pos_emb = self.drop(pos_emb)

            pos_seq = torch.arange(1, qlen + 1, 1.0, device=word_emb.device, dtype=word_emb.dtype)
            pos_seq = torch.unsqueeze(pos_seq, -1)
            pos_seq = torch.unsqueeze(pos_seq, -1)

            pos_seq = pos_seq.repeat([1, bsz, word_emb.size()[-1]])  # no neccesary to run this line

            if self.schema == 0:
                emb_peroid = self.pos_emb_phase(dec_inp)

            elif self.schema == 1: # vocab_size

                emb_peroid = self.period(dec_inp).repeat([1, 1, word_emb.size()[-1]])

            elif self.schema == 2:
                dimension_multiplier = torch.unsqueeze(self.period, -2)
                dimension_multiplier = torch.unsqueeze(dimension_multiplier, -2)
                emb_peroid = dimension_multiplier.repeat([qlen, bsz, 1])




            else:
                print("wrong schema")
                exit(1)
            emb_phase = torch.mul(pos_seq, emb_peroid)

            cos = torch.cos(emb_phase)
            sin = torch.sin(emb_phase)

            core_out = word_emb * cos
            pos_emb_phase = word_emb * sin

            core_out = self.drop(core_out)
            core_out_phase = self.drop(pos_emb_phase)
            hids.append(core_out)
            hids_phase.append(core_out_phase)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                mems_phase_i = None if mems_phase is None else mems_phase[i]
                core_out,core_out_phase = layer(core_out, core_out_phase, pos_emb, self.r_w_bias, self.r_r_bias,
                                 dec_attn_mask=dec_attn_mask, mems=mems_i,mems_phase=mems_phase_i)

                # print(core_out.size())
                # exit()
                # core_out=(core_out+core_out_phase)/2

                # core_out=torch.sqrt(core_out)

                # core_out=torch.sqrt((1/2)*core_out)
                hids.append(core_out)
                hids_phase.append(core_out_phase)
        # elif self.attn_type == 1: # learnable
        #     core_out = self.drop(word_emb)
        #     hids.append(core_out)
        #     for i, layer in enumerate(self.layers):
        #         if self.clamp_len > 0:
        #             r_emb = self.r_emb[i][-self.clamp_len :]
        #             r_bias = self.r_bias[i][-self.clamp_len :]
        #         else:
        #             r_emb, r_bias = self.r_emb[i], self.r_bias[i]

        #         mems_i = None if mems is None else mems[i]
        #         core_out = layer(core_out, r_emb, self.r_w_bias[i],
        #                 r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
        #         hids.append(core_out)
        # elif self.attn_type == 2: # absolute
        #     pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
        #                            dtype=word_emb.dtype)
        #     if self.clamp_len > 0:
        #         pos_seq.clamp_(max=self.clamp_len)
        #     pos_emb = self.pos_emb(pos_seq)

        #     core_out = self.drop(word_emb + pos_emb[-qlen:])

        #     hids.append(core_out)
        #     for i, layer in enumerate(self.layers):
        #         mems_i = None if mems is None else mems[i]
        #         if mems_i is not None and i == 0:
        #             mems_i += pos_emb[:mlen]
        #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
        #                          mems=mems_i)
        #         hids.append(core_out)
        # elif self.attn_type == 3:
        #     core_out = self.drop(word_emb)

        #     hids.append(core_out)
        #     for i, layer in enumerate(self.layers):
        #         mems_i = None if mems is None else mems[i]
        #         if mems_i is not None and mlen > 0:
        #             cur_emb = self.r_emb[i][:-qlen]
        #             cur_size = cur_emb.size(0)
        #             if cur_size < mlen:
        #                 cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
        #                 cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
        #             else:
        #                 cur_emb = cur_emb[-mlen:]
        #             mems_i += cur_emb.view(mlen, 1, -1)
        #         core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

        #         core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
        #                          mems=mems_i)
        #         hids.append(core_out)



        new_mems = self._update_mems(hids, mems, mlen, qlen)
        new_mems_phase = self._update_mems(hids_phase, mems_phase, mlen, qlen)

        return core_out,core_out_phase, new_mems,new_mems_phase

    def forward(self, data, target, mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        if  mems[0] == ():
            mems_real = self.init_mems()
            mems_phase = self.init_mems()
        else:
            mems_real,mems_phase = mems

        tgt_len = target.size(0)
        hidden, hidden_phase, new_mems,new_mems_phase = self._forward(data, mems=mems_real,mems_phase =mems_phase)
        norms = (torch.sqrt( torch.mul(hidden,hidden) + torch.mul(hidden_phase,hidden_phase)))/1.5
        pred_hid = norms[-tgt_len:]
        if self.sample_softmax > 0 and self.training:
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                                  self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return loss, new_mems , new_mems_phase


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    n_token = 10000

    n_layer = 3
    n_head = 2
    d_model = 100
    d_head = 100
    dropout = 0.1
    d_inner= 100
    import data_utils

    data = torch.LongTensor(data_len * B).random_(0, n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            sharing_phase_weight = True
            model = MemTransformerLM(n_token, n_layer, n_head,
                                     d_model, d_head, d_inner, dropout,
                                     dropout,sharing_phase_weight, tie_weight=True,
                                     d_embed=d_embed, div_val=div_val,
                                     tie_projs=tie_projs, pre_lnorm=True,
                                     tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                                     cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = (tuple(),tuple())
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, mems)
                mems = out[1:]

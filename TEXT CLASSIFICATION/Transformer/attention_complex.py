# attention.py

import torch
from torch import nn
import math
import torch.nn.functional as F
from train_utils import clones

def attention(query_real, key_real, value_real,query_phase, key_phase, value_phase, mask=None, dropout=None):
    "Implementation of Scaled dot product attention"


    d_k = query_real.size(-1)
    scores_real = torch.matmul(query_real, key_real.transpose(-2, -1))-torch.matmul(query_phase, key_phase.transpose(-2, -1))
    scores_phase = torch.matmul(query_real, key_phase.transpose(-2, -1))+torch.matmul(query_phase, key_real.transpose(-2, -1))
    
    scores = scores_real*scores_real+scores_phase*scores_phase
    scores = torch.sqrt(scores)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value_real), torch.matmul(p_attn, value_phase),p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1,tgt_emb_prj_weight_sharing=True):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.real = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.linears_real = clones(self.real, 4)
        self.linears_phase = self.linears_real
        # self.dropout = nn.Dropout(p=dropout)s
        # self.attn = None
        
    def forward(self, query_real, key_real, value_real,query_phase, key_phase, value_phase, mask=None):
        "Implements Multi-head attention"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query_real.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query_real, key_real, value_real = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears_real, (query_real, key_real, value_real))]

        query_phase, key_phase, value_phase = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears_phase, (query_phase, key_phase, value_phase))]

        
        # 2) Apply attention on all the projected vectors in batch. 
        x_real,x_phase,self.attn = attention(query_real, key_real, value_real,query_phase, key_phase, value_phase, mask=mask, 
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear. 
        x_real = x_real.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x_phase = x_phase.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        

        return self.linears_real[-1](x_real),self.linears_phase[-1](x_phase)

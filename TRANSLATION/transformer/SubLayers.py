''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1_real = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_1_phase = nn.Conv1d(d_in, d_hid, 1) # position-wise
        

        self.w_2_real = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.w_2_phase = nn.Conv1d(d_hid, d_in, 1) # position-wise
        
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_real,x_phase):
        residual_real = x_real
        residual_phase = x_phase
        cnn_real = x_real.transpose(1, 2)
        cnn_phase = x_phase.transpose(1, 2)
        

        w1_real = F.relu(self.w_1_real(cnn_real)-self.w_1_phase(cnn_phase))
        w1_phase = F.relu(self.w_1_real(cnn_phase)+self.w_1_phase(cnn_real))


        output_real = self.w_2_real(w1_real)-self.w_2_phase(w1_phase)
        output_phase = self.w_2_real(w1_phase)+self.w_2_phase(w1_real)

        
        output_real = output_real.transpose(1, 2)
        output_phase = output_phase.transpose(1, 2)
        
        output_real = self.dropout(output_real)
        output_phase = self.dropout(output_phase)
        

        output_real = self.layer_norm(output_real + residual_real)
        output_phase = self.layer_norm(output_phase + residual_phase)
        return output_real,output_phase



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k*2, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q_real, k_real, v_real,q_phase, k_phase, v_phase, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b_real, len_q_real, _ = q_real.size()
        sz_b_real, len_k_real, _ = k_real.size()
        sz_b_real, len_v_real, _ = v_real.size()
        
        sz_b_phase, len_q_phase, _ = q_phase.size()
        sz_b_phase, len_k_phase, _ = k_phase.size()
        sz_b_phase, len_v_phase, _ = v_phase.size()

        residual_real = q_real
        residual_phase = q_phase

        q_real = self.w_qs(q_real).view(sz_b_real, len_q_real, n_head, d_k)
        k_real = self.w_ks(k_real).view(sz_b_real, len_k_real, n_head, d_k)
        v_real = self.w_vs(v_real).view(sz_b_real, len_v_real, n_head, d_v)

        q_phase= self.w_qs(q_phase).view(sz_b_phase, len_q_phase, n_head, d_k)
        k_phase = self.w_ks(k_phase).view(sz_b_phase, len_k_phase, n_head, d_k)
        v_phase = self.w_vs(v_phase).view(sz_b_phase, len_v_phase, n_head, d_v)

        

        q_real = q_real.permute(2, 0, 1, 3).contiguous().view(-1, len_q_real, d_k) # (n*b) x lq x dk
        k_real = k_real.permute(2, 0, 1, 3).contiguous().view(-1, len_k_real, d_k) # (n*b) x lk x dk
        v_real = v_real.permute(2, 0, 1, 3).contiguous().view(-1, len_v_real, d_v) # (n*b) x lv x dv


        q_phase = q_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_q_phase, d_k) # (n*b) x lq x dk
        k_phase = k_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_k_phase, d_k) # (n*b) x lk x dk
        v_phase = v_phase.permute(2, 0, 1, 3).contiguous().view(-1, len_v_phase, d_v) # (n*b) x lv x dv



        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        

        output_real,output_phase, attn = self.attention(q_real, k_real, v_real,q_phase,k_phase,v_phase, mask=mask,continue_complex=False)

        output_real = output_real.view(n_head, sz_b_real, len_q_real, d_v)
        output_real = output_real.permute(1, 2, 0, 3).contiguous().view(sz_b_real, len_q_real, -1) # b x lq x (n*dv)


        output_phase = output_phase.view(n_head, sz_b_phase, len_q_phase, d_v)
        output_phase = output_phase.permute(1, 2, 0, 3).contiguous().view(sz_b_phase, len_q_phase, -1) # b x lq x (n*dv)

        output_real = self.dropout(self.fc(output_real))
        output_real = self.layer_norm(output_real + residual_real)

        output_phase = self.dropout(self.fc(output_phase))
        output_phase = self.layer_norm(output_phase + residual_phase)

        return output_real,output_phase, attn

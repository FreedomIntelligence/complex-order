import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q_real, k_real, v_real,q_phase, k_phase, v_phase, mask=None,continue_complex=True):

        attn_real = torch.bmm(q_real, k_real.transpose(1, 2))-torch.bmm(q_phase, k_phase.transpose(1, 2))             

        attn_phase = torch.bmm(q_real, k_phase.transpose(1, 2))+torch.bmm(q_phase, k_real.transpose(1, 2))              

        if(continue_complex):
            attn_real = attn_real / self.temperature
            attn_phase = attn_phase / self.temperature
            if mask is not None:
                attn_real = attn_real.masked_fill(mask, -np.inf)
                attn_phase = attn_phase.masked_fill(mask, -np.inf)

            attn_real = self.softmax(attn_real)
            attn_real = self.dropout(attn_real)

            attn_phase = self.softmax(attn_phase)
            attn_phase = self.dropout(attn_phase)
            

            output_real = torch.bmm(attn_real, v_real)- torch.bmm(attn_phase, v_phase)

            output_phase = torch.bmm(attn_real, v_phase)+torch.bmm(attn_phase, v_real)

        else:

            attn=attn_real*attn_real+attn_phase*attn_phase
            # attn=attn/self.temperature
            attn=torch.sqrt(attn)
            attn = attn / self.temperature

            if mask is not None:
                attn = attn.masked_fill(mask, -np.inf)

            attn= self.softmax(attn)
                        

            output_real = torch.bmm(attn, v_real)

            output_phase = torch.bmm(attn, v_phase)



        return output_real,output_phase, attn_real

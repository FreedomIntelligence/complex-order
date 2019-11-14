''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward



class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input_real, enc_input_phase, non_pad_mask=None, slf_attn_mask=None):
        enc_output_real,enc_output_phase, enc_slf_attn = self.slf_attn(
            enc_input_real, enc_input_real, enc_input_real, enc_input_phase,enc_input_phase,enc_input_phase,mask=slf_attn_mask)
        
        enc_output_real *= non_pad_mask
        enc_output_phase *= non_pad_mask

        enc_output_real,enc_output_phase= self.pos_ffn(enc_output_real,enc_output_phase)

        enc_output_real *= non_pad_mask
        enc_output_phase *= non_pad_mask


        return enc_output_real,enc_output_phase, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input_real,dec_input_phase, enc_output_real,enc_output_phase, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output_real,dec_output_phase, dec_slf_attn = self.slf_attn(
            dec_input_real, dec_input_real, dec_input_real,dec_input_phase,dec_input_phase,dec_input_phase, mask=slf_attn_mask)
        
        dec_output_real *= non_pad_mask
        dec_output_phase*= non_pad_mask

        dec_output_real,dec_output_phase, dec_enc_attn = self.enc_attn(dec_output_real, enc_output_real, enc_output_real,dec_output_phase, enc_output_phase, enc_output_phase, mask=dec_enc_attn_mask)
        
        dec_output_real *= non_pad_mask
        dec_output_phase*= non_pad_mask

        dec_output_real,dec_output_phase = self.pos_ffn(dec_output_real,dec_output_phase)
        dec_output_real *= non_pad_mask
        dec_output_phase*= non_pad_mask
        
        return dec_output_real,dec_output_phase, dec_slf_attn, dec_enc_attn

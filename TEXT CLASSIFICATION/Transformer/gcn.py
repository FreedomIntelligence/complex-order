# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_rnn import DynamicLSTM
import numpy as np

def get_sinusoid_encoding_table(n_src_vocab, d_hid, padding_idx=None):

    def cal_angle(position, hid_idx):
        return 1 / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_src_vocab)])


    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ASGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(ASGCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(torch.tensor(embedding_matrix, dtype=torch.float).size()[0], torch.tensor(embedding_matrix, dtype=torch.float).size()[1], padding_idx=0),freeze=False)
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(4*opt.hidden_dim, opt.polarities_dim)
        self.softmax = nn.Softmax()
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
        return weight*x

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x

    def forward(self, inputs):
        text_indices, adj = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        # aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        # left_len = torch.sum(left_indices != 0, dim=-1)
        # aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        enc_output_real = self.embed(text_indices)
        enc_output_phase= self.position_enc(text_indices)


        pos_seq = torch.arange(text_len.max()-1, -1, -1.0)
        pos_seq=torch.unsqueeze(pos_seq,-1)

        enc_output_phase=torch.mul(pos_seq,enc_output_phase)
        enc_output = self.text_embed_dropout(enc_output_real)
        enc_output_phase = self.text_embed_dropout(enc_output_phase)
        cos = torch.cos(enc_output_phase)
        sin = torch.sin(enc_output_phase)

        enc_output_real=enc_output*cos
        enc_output_phase=enc_output*sin

        text_out_real, (_, _) = self.text_lstm(enc_output_real, text_len)
        x_real = F.relu(self.gc1(text_out_real, adj))
        alpha_real = F.softmax(x_real.sum(1, keepdim=True), dim=2)
        x_real = alpha_real.squeeze(1)

        text_out_phase, (_, _) = self.text_lstm(enc_output_phase, text_len)
        x_phase = F.relu(self.gc1(text_out_phase, adj))
        alpha_phase = F.softmax(x_phase.sum(1, keepdim=True), dim=2)
        x_phase = alpha_phase.squeeze(1)


        text_out_phase, (_, _) = self.text_lstm(enc_output_phase, text_len)
        x_phase = F.relu(self.gc1(text_out_phase, adj))
        alpha_phase = F.softmax(x_phase.sum(1, keepdim=True), dim=2)
        x_phase = alpha_phase.squeeze(1)

        text = torch.cat([x_real,x_phase],1)
        output = self.fc(text)
        output=self.softmax(output)
        return output
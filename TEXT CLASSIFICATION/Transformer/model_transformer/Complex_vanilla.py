# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from attention_complex import MultiHeadedAttention
from encoder_complex import EncoderLayer, Encoder
from feed_forward_complex import PositionwiseFeedForward
import numpy as np
from torch.autograd import Variable
from utils import *
import math
from torch.nn.parameter import Parameter

class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        self.src_vocab=src_vocab
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder_layer=EncoderLayer(config.d_model,deepcopy(attn), deepcopy(ff), dropout)
        self.encoder =Encoder(self.encoder_layer, N)
        
        self.src_word_emb = nn.Embedding(src_vocab, config.d_model, padding_idx=0)
        self.position_enc = nn.Embedding(src_vocab, config.d_model, padding_idx=0)        

        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.config.d_model+self.config.d_model,self.config.output_size)
        
        self.softmax = nn.Softmax()

    def forward(self, x):


        enc_output_real = self.src_word_emb(x.permute(1,0))
        enc_output_phase= self.position_enc(x.permute(1,0))

        klen=enc_output_phase.size(1)
        pos_seq = torch.arange(klen-1, -1, -1.0, device=enc_output_real.device,dtype=enc_output_real.dtype)
        os_seq=torch.unsqueeze(pos_seq,-1)
        pos_seq=torch.unsqueeze(pos_seq,-1)
        enc_output_phase=torch.mul(pos_seq,enc_output_phase)

        enc_output = self.drop(enc_output_real)
        enc_output_phase = self.drop(enc_output_phase)

        cos = torch.cos(enc_output_phase)
        sin = torch.sin(enc_output_phase)

        enc_output_real=enc_output*cos
        enc_output_phase=enc_output*sin

        encoded_sents_real,encoded_sents_phase = self.encoder_layer(enc_output_real,enc_output_phase)
        encoded_sents_real,encoded_sents_phase = self.encoder(encoded_sents_phase,encoded_sents_real)
        final_feature_map_real = encoded_sents_real[:,-1,:]
        final_feature_map_phase = encoded_sents_phase[:,-1,:]
        encoded_sents=torch.cat([encoded_sents_real,encoded_sents_phase],2)
        encoded_sents = encoded_sents[:,-1,:] 
        final_out= self.fc(encoded_sents)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()
            
        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label - 1).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label - 1).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
            self.train()
                 
                
        return train_losses, val_accuracies
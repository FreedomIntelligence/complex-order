# train.py

from utils import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import numpy as np
from model_transformer import *
# from model_transformer.transformer_wo import Transformer
# from model_transformer.PE_reduce import Transformer
# from model_transformer.TPE_reduce import Transformer
# from model_transformer.Complex_vanilla import Transformer
from model_transformer.Complex_order import Transformer

acc_flod=[]

config = Config()
def dev_point_wise():
    train_file = "../data/" + config.data
    dataset = Dataset(config)
    dataset.load_data(train_file,config.data)
    
    model = Transformer(config, len(dataset.vocab))
    n_all_param = sum([p.nelement() for p in model.parameters()])
    print('#params = {}'.format(n_all_param))
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    NLLLoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    
    train_losses = []
    val_accuracies = []
    max_score=0.1
    acc_max = 0.0000
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        train_acc = evaluate_model(model, dataset.train_iterator)
        val_acc = evaluate_model(model, dataset.val_iterator)
        print ('Final Training Accuracy: {:.4f}'.format(train_acc))
        print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
        if val_acc> acc_max:
            torch.save(model, 'model_save/model')
            acc_max = val_acc
    acc_flod.append(acc_max)


if __name__ == '__main__':
    if config.data=='TREC_transformer' or config.data=='sst2_transformer':
        dev_point_wise()
        train_file = "../data/" + config.data
        dataset = Dataset(config)
        dataset.load_data(train_file,config.data)
        model = torch.load('model_save/model')
        test_acc = evaluate_model(model, dataset.test_iterator)
        print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    else:
        for i in range(1,config.n_fold+1):
            print("{} cross validation ".format(i))
            dev_point_wise()
        print("the average acc {}".format(np.mean(acc_flod)))
import sys
import os
import subprocess
import time

dropouts=[0.1]

n_layerss= [6]

batch_sizes = [64]

for dropout in dropouts:
    for n_layers in n_layerss:
        for batch_size in batch_sizes:
            subprocess.call('python3 train.py -data data/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing -dropout %f -n_layers %d -batch_size %d' % (dropout,n_layers,batch_size), shell = True)
            subprocess.call('python3 translate.py -model trained.chkpt -vocab data/multi30k.atok.low.pt -src data/multi30k/test.en.atok -no_cuda', shell = True)
            subprocess.call('perl multi-bleu.perl test.de < pred.txt', shell = True)



class Config(object):
    N = 1 #6 in Transformer Paper
    d_model = 256 #512 in Transformer Paper
    d_ff = 512 #2048 in Transformer Paper
    data='TREC_transformer'
    model='Complex_order'
    h = 8
    dropout = 0.1
    output_size = 6
    lr = 0.00001
    max_epochs = 100
    n_fold=10
    batch_size = 32
    max_sen_len = 53

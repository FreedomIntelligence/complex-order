# utils.py

import torch
from torchtext import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os 
import re

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):

        return int(label.strip()[0])

    def get_pandas_df(self, filename):

        full_df = pd.read_csv(filename, header=None, sep="\t", names=["text", "label"],encoding='gbk', quoting=3).fillna('N')
        return full_df
    
    def load_data(self, train_file,dataset, val_file=None):
        
        if dataset=='sst2_transformer' or dataset=='TREC_transformer':
            print('no n fold cross validation')
        else:
            process(train_file)

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        train_df = self.get_pandas_df(train_file+'/train.csv')
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        val_df = self.get_pandas_df(train_file+'/dev.csv')
        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)


        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator= data.BucketIterator(
             val_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        if dataset=='sst2_transformer' or dataset=='TREC_transformer':
            test_df = self.get_pandas_df(train_file+'/test.csv')
            test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
            test_data = data.Dataset(test_examples, datafields)
            self.test_iterator= data.BucketIterator(
                test_data,
                batch_size=self.config.batch_size,
                sort_key=lambda x: len(x.text),
                repeat=False,
                shuffle=False)
        else:
            print('no test')
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} val examples".format(len(val_data)))

def clean_str_sst(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def process(dataset):
    data_dir = "../data/" + dataset
    root = os.path.join(data_dir,"rt-polaritydata")
    saved_path=data_dir
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    datas=[]
    for polarity in  ("neg","pos"):
        filename = os.path.join(root,polarity) 
        records=[]
        with open(filename,encoding="utf-8",errors="replace") as f:
            for i,line in enumerate(f):
                records.append({"text":clean_str(line).strip(),"label": 1 if polarity == "pos" else 2})
        datas.append(pd.DataFrame(records))
    df = pd.concat(datas)
    from sklearn.utils import shuffle  
    df = shuffle(df).reset_index()
    split_index = [True] * int (len(df) *0.9) + [False] *(len(df)-int (len(df) *0.9))
    train = df[split_index]
    dev = df[~np.array(split_index)]
    train_filename=os.path.join(saved_path,"train.csv")
    test_filename = os.path.join(saved_path,"dev.csv")
    train[["text","label"]].to_csv(train_filename,encoding="utf-8",sep="\t",index=False,header=None)
    dev[["text","label"]].to_csv(test_filename,encoding="utf-8",sep="\t",index=False,header=None)
    print("processing into formated files over")

def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

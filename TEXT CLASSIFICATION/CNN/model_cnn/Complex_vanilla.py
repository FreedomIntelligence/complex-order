#!/usr/bin/env python
#encoding=utf-8

import tensorflow as tf
import numpy as np
from multiply import ComplexMultiply
import math
from scipy import linalg
from numpy.random import RandomState
rng = np.random.RandomState(23455)
from keras import initializers
from keras import backend as K
import math

class CNN(object):
    def __init__(
      self, max_input_left,embeddings,vocab_size,embedding_size,batch_size,filter_sizes,
      num_filters,dataset,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10):

        self.num_filters = num_filters
        self.embeddings=embeddings
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.filter_sizes = filter_sizes
        self.batch_size = batch_size
        self.dataset=dataset
        self.l2_reg_lambda = l2_reg_lambda
        self.para = []
        self.max_input_left = max_input_left
        self.extend_feature_dim = extend_feature_dim
        self.is_Embedding_Needed = is_Embedding_Needed
        self.rng = 23455
    def create_placeholder(self):
        self.question = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'input_question')
        if self.dataset=='TREC':
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,6], name = "input_y")
        else:
            self.input_y = tf.placeholder(tf.float32, [self.batch_size,2], name = "input_y")
        self.q_position = tf.placeholder(tf.int32,[self.batch_size,self.max_input_left],name = 'q_position')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    def add_embeddings(self):
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
                W_pos=tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size], 0, 2*math.pi),name="W",trainable = self.trainable)
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.embedding_W_pos=W_pos
        self.M_qa_real,self.M_qa_imag = self.concat_embedding(self.question,self.q_position)

    def feed_neural_work(self):
        self.h_drop_out = self.narrow_convolutionandpool_real_imag(tf.expand_dims(self.M_qa_real,-1),tf.expand_dims(self.M_qa_imag,-1))
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            if self.dataset=='TREC':
                W = tf.get_variable("W",shape=[2*self.num_filters_total, 6],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[6]), name="b")
            else:
                W = tf.get_variable("W",shape=[2*self.num_filters_total, 2],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop_out, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss


        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



    def concat_embedding(self,words_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        embedding_chars_q_phase=tf.nn.embedding_lookup(self.embedding_W_pos,words_indice)
        [embedded_chars_q, embedding_chars_q_phase] = ComplexMultiply()([embedding_chars_q_phase,embedded_chars_q])
        return embedded_chars_q,embedding_chars_q_phase

    def narrow_convolutionandpool_real_imag(self,embedding_real,embedding_imag):
        pooled_outputs_real=[]
        pooled_outputs_imag=[]
        for i,filter_size in enumerate(self.filter_sizes):
            filter_shape = [filter_size,self.embedding_size,1,self.num_filters]
            input_dim=2
            fan_in = np.prod(filter_shape[:-1])
            fan_out = (filter_shape[-1] * np.prod(filter_shape[:2]))
            s=1./fan_in
            rng=RandomState(23455)
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            modulus=rng.rayleigh(scale=s,size=filter_shape)
            phase=rng.uniform(low=-np.pi,high=np.pi,size=filter_shape)
            W_real=modulus*np.cos(phase)
            W_imag=modulus*np.sin(phase)
            W_real = tf.Variable(W_real,dtype = 'float32')
            W_imag = tf.Variable(W_imag,dtype = 'float32')
            conv_real = tf.nn.conv2d(embedding_real,W_real,strides=[1, 1, 1, 1],padding='VALID',name="conv-1")
            cov_imag=tf.nn.conv2d(embedding_imag,W_imag,strides=[1, 1, 1, 1],padding='VALID',name="conv-1")
            cov_real_imag=tf.nn.conv2d(embedding_imag,W_real,strides=[1, 1, 1, 1],padding='VALID',name="conv-1")
            cov_imag_real=tf.nn.conv2d(embedding_real,W_imag,strides=[1, 1, 1, 1],padding='VALID',name="conv-1")
            qa_real=conv_real-cov_imag
            qa_imag=cov_real_imag+cov_imag_real
            h_real = tf.nn.relu(tf.nn.bias_add(qa_real, b), name="relu")
            h_imag = tf.nn.relu(tf.nn.bias_add(qa_imag, b), name="relu")
            pooled_real = tf.nn.max_pool(
                    h_real,
                    ksize=[1, self.max_input_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            pooled_imag = tf.nn.max_pool(
                    h_imag,
                    ksize=[1, self.max_input_left - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
            pooled_outputs_real.append(pooled_real)
            pooled_outputs_imag.append(pooled_imag)
        self.h_pool_real = tf.concat(pooled_outputs_real, 3)
        self.h_pool_imag = tf.concat(pooled_outputs_imag, 3)
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool_real=tf.reshape(self.h_pool_real, [-1, self.num_filters_total])
        self.h_pool_imag=tf.reshape(self.h_pool_imag, [-1, self.num_filters_total])
        h_drop_real = tf.nn.dropout(self.h_pool_real, self.dropout_keep_prob)
        h_drop_imag = tf.nn.dropout(self.h_pool_imag, self.dropout_keep_prob)
        h_drop=tf.concat([h_drop_real,h_drop_imag],1)
        return h_drop

    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.feed_neural_work()

if __name__ == '__main__':
    cnn = Fasttext(max_input_left = 33,
                max_input_right = 40,
                vocab_size = 5000,
                embedding_size = 50,
                batch_size = 3,
                embeddings = None,
                embeddings_complex=None,
                dropout_keep_prob = 1,
                filter_sizes = [40],
                num_filters = 65,
                l2_reg_lambda = 0.0,
                is_Embedding_Needed = False,
                trainable = True,
                overlap_needed = False,
                pooling = 'max',
                position_needed = False)
    cnn.build_graph()
    input_x_1 = np.reshape(np.arange(3*33),[3,33])
    input_x_2 = np.reshape(np.arange(3 * 40),[3,40])
    input_y = np.ones((3,2))

    input_overlap_q = np.ones((3,33))
    input_overlap_a = np.ones((3,40))
    q_posi = np.ones((3,33))
    a_posi = np.ones((3,40))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            cnn.question:input_x_1,
            cnn.input_y:input_y,
            cnn.q_position:q_posi,
        }

        see,question,scores = sess.run([cnn.embedded_chars_q,cnn.question,cnn.scores],feed_dict)
        print (see)


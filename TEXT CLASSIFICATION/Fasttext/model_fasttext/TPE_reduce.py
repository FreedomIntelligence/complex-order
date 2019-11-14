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

class Fasttext(object):
    def __init__(
      self, max_input_left,embeddings,vocab_size,embedding_size,batch_size,dataset,hidden_num,l2_reg_lambda = 0.0, is_Embedding_Needed = False,trainable = True,extend_feature_dim = 10):

        self.embeddings=embeddings
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.trainable = trainable
        self.batch_size = batch_size
        self.dataset=dataset
        self.hidden_num=hidden_num
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
    def Position_Embedding(self,position_size):
        batch_size=self.batch_size
        seq_len =500
        position_j = 1. / tf.pow(10000., 2 * tf.range(position_size/2, dtype=tf.float32) / position_size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position_i = tf.expand_dims(position_i, 1)
        position_ij = tf.matmul(position_i, position_j)
        position_embedding = tf.concat([math.pi*tf.cos(position_ij),math.pi*tf.sin(position_ij)],1)

        return position_embedding
    def add_embeddings(self):
        with tf.name_scope("embedding"):
            if self.is_Embedding_Needed:
                W = tf.Variable(np.array(self.embeddings),name="W" ,dtype="float32",trainable = self.trainable )
                W_pos=tf.Variable(self.Position_Embedding(self.embedding_size),name = 'W',trainable = False)
            else:
                W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),name="W",trainable = self.trainable)
            self.embedding_W = W
            self.embedding_W_pos=W_pos
        self.embedded_chars_q,self.embedded_chars_q_pos = self.concat_embedding(self.question,self.q_position)
        self.embedded_chars_q=tf.reduce_sum([self.embedded_chars_q,self.embedded_chars_q_pos],0)
        print(self.embedded_chars_q)
        self.represent=tf.reduce_mean(self.embedded_chars_q,1)
        print(self.represent)
    def feed_neural_work(self):
        self.l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            if self.dataset=='TREC':
                W = tf.get_variable("W",shape=[self.embedding_size, 6],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[6]), name="b")
            else:
                W = tf.get_variable("W",shape=[self.embedding_size, 2],initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.represent, W, b, name = "logits")
            self.scores = tf.nn.softmax(self.logits, name = "scores")
            self.predictions = tf.argmax(self.scores, 1, name = "predictions")
    def create_loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
            self.loss = tf.reduce_mean(losses)+self.l2_reg_lambda*self.l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    def concat_embedding(self,words_indice,position_indice):
        embedded_chars_q = tf.nn.embedding_lookup(self.embedding_W,words_indice)
        embedding_chars_q_pos=tf.nn.embedding_lookup(self.embedding_W_pos,position_indice)
        return embedded_chars_q,embedding_chars_q_pos
    def build_graph(self):
        self.create_placeholder()
        self.add_embeddings()
        self.feed_neural_work()
        self.create_loss()

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


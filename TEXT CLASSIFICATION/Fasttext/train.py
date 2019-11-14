# coding=utf-8
#! /usr/bin/env python3.4
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper import batch_gen_with_point_wise, load, load_trec_sst2, prepare, batch_gen_with_single
import operator
from model_fasttext import *
# from model_fasttext.Fasttext_origin import Fasttext
# from model_fasttext.PE_reduce import Fasttext
# from model_fasttext.TPE_reduce import Fasttext
# from model_fasttext.Complex_vanilla import Fasttext
from model_fasttext.Complex_order import Fasttext
import random
import pickle
import config
from functools import wraps
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
now = int(time.time())
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print (timeStamp)
FLAGS = config.flags.FLAGS
FLAGS._parse_flags()
log_dir = 'wiki_log/' + timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/dev_' + FLAGS.data + timeStamp
para_file = log_dir + '/dev_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'
pickle.dump(FLAGS.__flags, open(para_file, 'wb+'))
acc_flod=[]


def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds" % (func.__name__, delta))
        return ret
    return _deco


def predict(sess, cnn, dev, alphabet, batch_size, q_len):
    scores = []
    for data in batch_gen_with_single(dev, alphabet, batch_size, q_len):
        feed_dict = {
            cnn.question: data[0],
            cnn.q_position: data[1],
            cnn.dropout_keep_prob: 1.0
        }
        score = sess.run(cnn.scores, feed_dict)
        scores.extend(score)
    return np.array(scores[:len(dev)])


@log_time_delta
def dev_point_wise():
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        train,dev,test=load_trec_sst2(FLAGS.data)
    else:
        train, dev = load(FLAGS.data)
    q_max_sent_length = max(
        map(lambda x: len(x), train['question'].str.split()))
    print(q_max_sent_length)
    print(len(train))
    print ('train question unique:{}'.format(len(train['question'].unique())))
    print ('train length', len(train))
    print ('dev length', len(dev))
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        alphabet,embeddings = prepare([train, dev,test], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    else:
        alphabet,embeddings = prepare([train, dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)
    print ('alphabet:', len(alphabet))
    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto()
            session_conf.allow_soft_placement = FLAGS.allow_soft_placement
            session_conf.log_device_placement = FLAGS.log_device_placement
            session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        now = int(time.time())
        timeArray = time.localtime(now)
        timeStamp1 = time.strftime("%Y%m%d%H%M%S", timeArray)
        timeDay = time.strftime("%Y%m%d", timeArray)
        print (timeStamp1)
        with sess.as_default(), open(precision, "w") as log:
            log.write(str(FLAGS.__flags) + '\n')
            cnn = Fasttext(
                max_input_left=q_max_sent_length,
                vocab_size=len(alphabet),
                embeddings=embeddings,
                embedding_size=FLAGS.embedding_dim,
                batch_size=FLAGS.batch_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                is_Embedding_Needed=True,
                hidden_num=FLAGS.hidden_num,
                trainable=FLAGS.trainable,
                dataset=FLAGS.data,
                extend_feature_dim=FLAGS.extend_feature_dim)
            cnn.build_graph()
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            acc_max = 0.0000
            for i in range(FLAGS.num_epochs):
                datas = batch_gen_with_point_wise(
                    train, alphabet, FLAGS.batch_size, q_len=q_max_sent_length)
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.input_y: data[1],
                        cnn.q_position: data[2],
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}  ".format(time_str, step, loss, accuracy))
                predicted = predict(
                    sess, cnn, train, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted, 1)
                acc_train= accuracy_score(predicted_label,train['flag'])
                predicted_dev = predict(
                    sess, cnn, dev, alphabet, FLAGS.batch_size, q_max_sent_length)
                predicted_label = np.argmax(predicted_dev, 1)
                acc_dev= accuracy_score(predicted_label,dev['flag'])
                if acc_dev> acc_max:
                    tf.train.Saver().save(sess, "model_save/model",write_meta_graph=True)
                    acc_max = acc_dev
                print ("{}:train epoch:acc {}".format(i, acc_train))
                print ("{}:dev epoch:acc {}".format(i, acc_dev))
                line2 = " {}:epoch: map_dev{}".format(i, acc_dev)
                log.write(line2 + '\n')
                log.flush()
            acc_flod.append(acc_max)
            log.close()

if __name__ == '__main__':
    if FLAGS.data=='TREC' or FLAGS.data=='sst2':
        for attr, value in sorted(FLAGS.__flags.items()):
            print(("{}={}".format(attr.upper(), value)))
        dev_point_wise()
        ckpt = tf.train.get_checkpoint_state("model_save" + '/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        train,dev,test=load_trec_sst2(FLAGS.data)
        q_max_sent_length = max(map(lambda x: len(x), train['question'].str.split()))
        alphabet,embeddings = prepare([train, test,dev], max_sent_length=q_max_sent_length, dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=True)    
        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            scores=[]
            question = graph.get_operation_by_name('input_question').outputs[0]
            q_position = graph.get_operation_by_name('q_position').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            for data in batch_gen_with_single(test, alphabet, FLAGS.batch_size, q_max_sent_length):
                feed_dict = {question.name: data[0],q_position.name: data[1],dropout_keep_prob.name: 1.0}
                score = sess.run("output/scores:0", feed_dict)
                scores.extend(score)
            scores=np.array(scores[:len(test)])
            predicted_label = np.argmax(scores, 1)
            acc_test = accuracy_score(predicted_label,test['flag'])
            print ("test epoch:acc{}".format(acc_test))
    else:
        for i in range(1,FLAGS.n_fold+1):
            print("{} cross validation ".format(i))
            for attr, value in sorted(FLAGS.__flags.items()):
                print(("{}={}".format(attr.upper(), value)))
            dev_point_wise()
        print("the average acc {}".format(np.mean(acc_flod)))

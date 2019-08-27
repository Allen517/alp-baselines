# -*- coding:utf8 -*-

from __future__ import print_function

import random
import tensorflow as tf
import numpy as np
import sys,os
from collections import defaultdict

from utils.LogHandler import LogHandler
from utils.utils import load_train_valid_labels, batch_iter, valid_iter, read_embeddings

class PALE(object):

    def __init__(self, learning_rate, batch_size, n_input, n_hidden, n_layer, type_model
                    , is_valid, device, files, log_file):
        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        self.device = device

        # Parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.valid = is_valid
        self.valid_prop = .9 if self.valid else 1.
        self.valid_sample_size = 9

        self.cur_epoch = 1

        # Network Parameters
        self.n_hidden = n_hidden if type_model=='mlp' else n_input # number of neurons in hidden layer
        self.n_input = n_input # size of node embeddings
        self.n_layer = n_layer # number of layer

        # Set Train Data
        if not isinstance(files, list) and len(files)<3:
            self.logger.info('The alogrihtm needs files like [First Graph File, Second Graph File, Label File]')
            return

        # tf Graph input
        self.lookup = defaultdict(dict)
        self.look_back = defaultdict(list)
        self._read_train_dat(files[0], files[1], files[2]) # douban, weibo, label files
        self.valid_sample_size = min(min(self.valid_sample_size, len(self.look_back['f'])-1), len(self.look_back['g'])-1)

        # TF Graph Building
        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
        with tf.device(self.device):
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                self._init_weights(type_model)
                self.build_train_graph(type_model)
                self.build_valid_graph(type_model)
            self.sess.run(tf.global_variables_initializer())

    def _read_labels(self, label_file):
        labels = list()
        with open(label_file, 'r') as lb_handler:
            for ln in lb_handler:
                ln = ln.strip()
                if not ln:
                    break
                labels.append(ln.split())
        return labels

    def _read_train_dat(self, embed1_file, embed2_file, label_file):
        self.X, self.lookup['f'], self.look_back['f'] = read_embeddings(embed1_file)
        self.Y, self.lookup['g'], self.look_back['g'] = read_embeddings(embed2_file)

        self.L = load_train_valid_labels(label_file, self.lookup, self.valid_prop)

    def _init_weights(self, type_code_graph):
        # Store layers weight & bias
        self.weights = dict()
        self.biases = dict()
        if type_code_graph=='mlp':
            self.weights['h0'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
            self.biases['b0'] = tf.Variable(tf.zeros([self.n_hidden]))
            for i in range(1,self.n_layer):
                self.weights['h{}'.format(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
                self.biases['b{}'.format(i)] = tf.Variable(tf.zeros([self.n_hidden]))
        self.weights['out'] = tf.Variable(tf.random_normal([self.n_hidden, self.n_input]))
        self.biases['b_out'] = tf.Variable(tf.zeros([self.n_input]))

    def build_mlp_code_graph(self, inputs):

        # Input layer
        layer = tf.nn.sigmoid(tf.add(tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['h0'])
                                    , self.biases['b0']))
        for i in range(1,self.n_layer):
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights['h{}'.format(i)])
                                    , self.biases['b{}'.format(i)]))
        # Output fully connected layer with a neuron
        code = tf.nn.tanh(tf.matmul(layer, self.weights['out']) + self.biases['b_out'])

        return code

    def build_lin_code_graph(self, inputs):

        # Output fully connected layer with a neuron
        code = tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['out']) + self.biases['b_out']

        return code

    def build_train_graph(self, type_code_graph):

        if type_code_graph=='lin':
            code_graph = self.build_lin_code_graph
        elif type_code_graph=='mlp':
            code_graph = self.build_mlp_code_graph

        self.cur_batch_size = tf.placeholder('float32', name='batch_size')

        self.pos_inputs = {
            'f': tf.placeholder('float32', [None, self.n_input]),
            'g': tf.placeholder('float32', [None, self.n_input])
        }

        self.PF = code_graph(self.pos_inputs['f']) # batch_size*n_input

        # train loss
        self.loss = tf.reduce_mean(.5*tf.square(self.PF-self.pos_inputs['g']))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

    def build_valid_graph(self, type_code_graph):

        if type_code_graph=='lin':
            code_graph = self.build_lin_code_graph
        elif type_code_graph=='mlp':
            code_graph = self.build_mlp_code_graph

        # validation
        self.valid_inputs = {
            'f': tf.placeholder('float32', [None, self.valid_sample_size, self.n_input]),
            'g': tf.placeholder('float32', [None, self.valid_sample_size, self.n_input])
        }

        valid = tf.reshape(
                code_graph(self.valid_inputs['f'])
                , [-1, self.valid_sample_size, self.n_input]
                ) # batch_size*neg_ratio*n_input
        self.dot_dist = tf.reduce_sum(tf.pow(valid-self.valid_inputs['g'],2.),axis=2)

    def train_one_epoch(self):
        sum_loss = 0.0
        mrr = 0.0

        # train process
        batches = batch_iter(self.L, self.batch_size, 0, self.lookup, 'f', 'g')
        batch_id = 0
        for batch in batches:
            pos, neg = batch
            if not len(pos['f'])==len(pos['g']) and not len(neg['f'])==len(neg['g']):
                self.logger.info('The input label file goes wrong as the file format.')
                continue
            batch_size = len(pos['f'])
            feed_dict = {
                self.pos_inputs['f']:self.X[pos['f'],:],
                self.pos_inputs['g']:self.Y[pos['g'],:],
                self.cur_batch_size:batch_size
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss],feed_dict)

            sum_loss += cur_loss
            batch_id += 1

        # valid process
        if self.valid:
            valid = valid_iter(self.L, self.valid_sample_size, self.lookup, 'f', 'g')
            if not len(valid['f'])==len(valid['g']):
                self.logger.info('The input label file goes wrong as the file format.')
                return
            valid_size = len(valid['f'])
            feed_dict = {
                self.valid_inputs['f']:self.X[valid['f'],:],
                self.valid_inputs['g']:self.Y[valid['g'],:]
            }
            valid_dist = self.sess.run(self.dot_dist,feed_dict)

            mrr = .0
            for i in range(valid_size):
                fst_dist = valid_dist[i][0]
                pos = 1
                for k in range(1,len(valid_dist[i])):
                    if fst_dist>=valid_dist[i][k]:
                        pos+=1
                mrr += 1./pos
            self.logger.info('Epoch={}, sum of loss={!s}, mrr in validation={}'
                                .format(self.cur_epoch, sum_loss/(batch_id+1e-8), mrr/(valid_size+1e-8)))
        else:
            self.logger.info('Epoch={}, sum of loss={!s}'
                                .format(self.cur_epoch, sum_loss/batch_id))
        self.cur_epoch += 1

        return sum_loss/(batch_id+1e-8), mrr/(valid_size+1e-8)

    def _write_in_file(self, filename, vec, tag):
        with open(filename, 'a+') as res_handler:
            if len(vec.shape)>1:
                column_size = vec.shape[1]
            else:
                column_size = 1
            reshape_vec = vec.reshape(-1)
            vec_size = len(reshape_vec)
            res_handler.write(tag+'\n')
            for i in range(0,vec_size,column_size):
                res_handler.write('{}\n'.format(' '.join([str(reshape_vec[i+k]) for k in range(column_size)])))

    def save_model(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        for k,v in self.weights.items():
            self._write_in_file(filename, v.eval(self.sess), k)
        for k,v in self.biases.items():
            self._write_in_file(filename, v.eval(self.sess), k)

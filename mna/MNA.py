from __future__ import print_function

import os
from collections import defaultdict
import numpy as np
import random

from sklearn import svm
from sklearn.externals import joblib
from utils.utils import *
from utils.LogHandler import LogHandler

class _MNA(object):

    def __init__(self, graph, anchorfile, valid_prop, neg_ratio, log_file):
        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        if not isinstance(graph, dict):
            self.logger.error('The graph must contain src and target graphs.')
            return

        self.graph = graph
        self.lookup = dict()
        self.lookup['f'] = self.graph['f'].look_up_dict
        self.lookup['g'] = self.graph['g'].look_up_dict
        self.look_back = dict()
        self.look_back['f'] = self.graph['f'].look_back_list
        self.look_back['g'] = self.graph['g'].look_back_list
        self.L = load_train_valid_labels(anchorfile, self.lookup, valid_prop)

        self.neg_ratio = neg_ratio
        self.batch_size = 1024

        self.clf = svm.SVC(probability=True)

    def __get_pair_features(self, src_nds, target_nds):
        pair_features = list()
        if len(src_nds)!=len(target_nds):
            self.logger.warn('The size of sampling in processing __get_pair_features is not equal.')
            yield pair_features
        for i in range(len(src_nds)):
            src_nd, target_nd = src_nds[i],target_nds[i]

            src_neighbor_anchors = set()
            for src_nd_to in self.graph['f'].G[self.look_back['f'][src_nd]]:
                if src_nd_to in self.L['f2g']['train']:
                    src_neighbor_anchors.add(src_nd_to)

            target_neighbor_anchors = set()
            for target_nd_to in self.graph['g'].G[self.look_back['g'][target_nd]]:
                if target_nd_to in self.L['g2f']['train']:
                    target_neighbor_anchors.add(target_nd_to)

            cnt_common_neighbors = .0
            AA_measure = .0
            for sna in src_neighbor_anchors:
                for k in range(len(self.L['f2g']['train'][sna])):
                    target_anchor_nd = self.L['f2g']['train'][sna][k]
                    if target_anchor_nd in target_neighbor_anchors:
                        cnt_common_neighbors += 1.
                        AA_measure += 1./np.log((len(self.graph['f'].G[sna])\
                                                +len(self.graph['g'].G[self.L['f2g']['train'][sna][k]]))/2.)
            jaccard = cnt_common_neighbors/(len(self.graph['f'].G[self.look_back['f'][src_nd]])\
                                            +len(self.graph['g'].G[self.look_back['g'][target_nd]])\
                                            -cnt_common_neighbors+1e-6)

            yield [cnt_common_neighbors, jaccard, AA_measure]

    def train(self):

        batches_f2g = batch_iter(self.L, self.batch_size, self.neg_ratio, self.lookup, 'f', 'g')

        X = list()
        Y = list()
        for batch in batches_f2g:
            pos,neg = batch
            if not len(pos['f'])==len(pos['g']) and not len(neg['f'])==len(neg['g']):
                self.logger.info('The input label file goes wrong as the file format.')
                continue
            pos_features = list(self.__get_pair_features(pos['f'], pos['g']))
            X.extend(pos_features)
            Y.extend([1 for m in range(len(pos_features))])

            for k in range(self.neg_ratio):
                neg_features = list(self.__get_pair_features(neg['f'][k], neg['g'][k]))
                X.extend(neg_features)
                Y.extend([-1 for m in range(len(neg_features))])

            self.logger.info('Training Model...')
            self.clf.fit(X,Y)
            self.logger.info('Training score: %f'%self.clf.score(X,Y))
            self.logger.info('Complete Training process...')

class MNA(object):

    def __init__(self, graph, anchorfile, valid_prop, neg_ratio, log_file):
        # training
        self.model = _MNA(graph, anchorfile, valid_prop, neg_ratio, log_file)
        self.model.train()

    def save_model(self, modelfile):
        modelfile = modelfile+'.pkl'
        joblib.dump(self.model.clf, modelfile)
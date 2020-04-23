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

    def __init__(self, graph, attr_file, anchorfile, use_net, valid_prop, neg_ratio, log_file):
        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        if not isinstance(graph, dict):
            self.logger.error('The graph must contain src and target graphs.')
            return

        self.use_net=use_net
        # self.use_net=False
        # print(self.use_net)
        self.graph = graph
        self.lookup = dict()
        self.lookup['f'] = self.graph['f'].look_up_dict
        self.lookup['g'] = self.graph['g'].look_up_dict
        self.look_back = dict()
        self.look_back['f'] = self.graph['f'].look_back_list
        self.look_back['g'] = self.graph['g'].look_back_list
        self.L = load_train_valid_labels(anchorfile, self.lookup, valid_prop)

        self.attributes = dict()
        if attr_file:
            self.attributes['f']=self._set_node_attributes(attr_file[0])
            self.attributes['g']=self._set_node_attributes(attr_file[1])

        self.neg_ratio = neg_ratio
        self.batch_size = 1024

        self.clf = svm.SVC(probability=True)

    def _set_node_attributes(self, attr_file):
        node_attributes = defaultdict(list)
        if not attr_file:
            return None
        with open(attr_file, 'r') as fin:
            for ln in fin:
                elems = ln.strip().split(',')
                node_attributes[elems[0]]=list(map(float,elems[1:]))
        return node_attributes

    def _get_pair_features(self, src_nds, target_nds):
        pair_features = list()
        if len(src_nds)!=len(target_nds):
            self.logger.warn('The size of sampling in processing _get_pair_features is not equal.')
            yield pair_features
        for i in range(len(src_nds)):
            feat_net = []
            feat_attr = []

            src_nd_idx, target_nd_idx = src_nds[i],target_nds[i]
            src_nd = self.look_back['f'][src_nd_idx]
            target_nd = self.look_back['g'][target_nd_idx]

            if self.use_net:
                src_neighbor_anchors = set()
                for src_nd_to in self.graph['f'].G[src_nd]:
                    if src_nd_to in self.L['f2g']['train']:
                        src_neighbor_anchors.add(src_nd_to)

                target_neighbor_anchors = set()
                for target_nd_to in self.graph['g'].G[target_nd]:
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
                jaccard = cnt_common_neighbors/(len(self.graph['f'].G[src_nd])\
                                                +len(self.graph['g'].G[target_nd])\
                                                -cnt_common_neighbors+1e-6)

                # print(self.attributes['f'][src_nd], self.attributes['g'][target_nd])
                feat_net = [cnt_common_neighbors, jaccard, AA_measure]
            if len(self.attributes)>0:
                if (len(self.attributes['f'][src_nd])-len(self.attributes['g'][target_nd]))!=0:
                    continue
                # print('src_nd:',self.attributes['f'][src_nd],len(self.attributes['f'][src_nd]))
                # print('target_nd:',self.attributes['g'][target_nd],len(self.attributes['g'][target_nd]))
                feat_len = len(self.attributes['f'][src_nd])
                feat_attr = [1-self.attributes['f'][src_nd][k]\
                                +self.attributes['g'][target_nd][k] for k in range(feat_len)]
            if len(feat_net+feat_attr)<1:
                continue

            # print(len(feat_net), len(feat_attr))
            yield feat_net+feat_attr

    def _find_empty_idx(self, features):
        empty_idxes = []
        max_f_len = 0
        for k in range(len(features)):
            if len(features)<1 or len(features[k])<max_f_len:
                empty_idxes.append(k)
            if max_f_len<len(features[k]):
                max_f_len=len(features[k])
            # if len(features[k])<12:
            #     print(features[k])
        return empty_idxes

    def _remove_empty_idx(self, features, idxes):
        for idx in idxes[::-1]:
            features.pop(idx)
        return features

    def train(self):

        batches_f2g = batch_iter(self.L, self.batch_size, self.neg_ratio, self.lookup, 'f', 'g')

        X = list()
        Y = list()
        for batch in batches_f2g:
            pos,neg = batch
            if not len(pos['f'])==len(pos['g']) and not len(neg['f'])==len(neg['g']):
                self.logger.info('The input label file goes wrong as the file format.')
                continue
            pos_features = list(self._get_pair_features(pos['f'], pos['g']))
            empty_idxes = self._find_empty_idx(pos_features)
            # print('feat_len (pos):',len(pos_features[0]))
            X.extend(pos_features)
            Y.extend([1 for m in range(len(pos_features))])

            for k in range(self.neg_ratio):
                neg_features = list(self._get_pair_features(neg['f'][k], neg['g'][k]))
                X.extend(neg_features)
                # print('feat_len (neg):',len(neg_features[0]))
                Y.extend([-1 for m in range(len(neg_features))])

            empty_idxes = self._find_empty_idx(X)
            X = self._remove_empty_idx(X, empty_idxes)
            Y = self._remove_empty_idx(Y, empty_idxes)
            self.logger.info('Training Model...')
            # print(X)
            # print(len(X), len(X[0]), len(Y))
            self.clf.fit(X,Y)
            # print(self.clf)
            self.logger.info('Training score: %f'%self.clf.score(X,Y))
            self.logger.info('Complete Training process...')

class MNA(object):

    def __init__(self, graph, attr_file, anchorfile, use_net, valid_prop, neg_ratio, log_file):
        # training
        self.model = _MNA(graph, attr_file, anchorfile, use_net, valid_prop, neg_ratio, log_file)
        self.model.train()

    def save_model(self, modelfile):
        modelfile = modelfile+'.pkl'
        joblib.dump(self.model.clf, modelfile)
        print("Save model in %s"%(modelfile))
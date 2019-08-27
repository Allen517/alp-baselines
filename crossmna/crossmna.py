# -*- coding:utf8 -*-

from __future__ import print_function

import random
import math
import numpy as np
from collections import defaultdict
from utils.LogHandler import LogHandler
from utils.utils import *
import os
import re

class _CROSSMNA(object):

    def __init__(self, layer_graphs, anchor_file, lr=.001, nd_rep_size=16, layer_rep_size=16
                    , batch_size=100, negative_ratio=5
                    , table_size=1e8, log_file='log', last_emb_file=None):

        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        self.epsilon = 1e-7
        self.table_size = table_size
        self.sigmoid_table = {}
        self.sigmoid_table_size = 1000
        self.SIGMOID_BOUND = 6

        self._init_simgoid_table()

        self.anchors, num_anchors = self._read_anchors(anchor_file, ',')
        self.logger.info('Number of anchors:%d'%num_anchors)

        self.num_layers = len(layer_graphs) # number of calculated networks
        self.layer_graphs = layer_graphs # graphs in different layers
        self.nd_rep_size = nd_rep_size # representation size of node
        self.layer_rep_size = layer_rep_size # representation size of layer

        self.idx = 0 # for speeding up calculation

        # self.node_size = 0
        # for i in range(self.num_layers):
        #     self.node_size += layer_graphs[i].node_size
        # self.node_size -= num_anchors
        # print(self.node_size)
        # may need to be revised
        self.update_dict = defaultdict(int)
        self.update_look_back = list()
        self._build_dict(layer_graphs, self.anchors)
        self.logger.info('Number of nodes:%d'%len(self.look_back))

        self.node_size = len(self.look_back)
        
        self._init_params(self.node_size, self.num_layers, nd_rep_size, layer_rep_size, last_emb_file)

        self.lr = lr
        self.cur_epoch = 0
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self._gen_sampling_table()

    def _build_dict(self, layer_graphs, anchors):
        self.look_up = defaultdict(int)
        self.look_back = list()
        idx = 0
        for i in range(self.num_layers):
            for nd in layer_graphs[i].G.nodes():
                if nd in self.look_up:
                    continue
                if nd in self.anchors:
                    for ac_nd in self.anchors[nd]:
                        self.look_up[ac_nd] = idx
                self.look_up[nd] = idx
                self.look_back.append(nd)
                idx += 1

    def _init_params(self, node_size, n_layers, nd_rep_size, layer_rep_size, last_emb_file):
        self.params = dict()
        self.params['node'] = np.random.normal(0,1,(node_size,nd_rep_size))
        self.params['layer'] = np.random.normal(0,1,(n_layers,layer_rep_size))
        self.params['W'] = np.random.normal(0,1,(nd_rep_size,layer_rep_size))
        if last_emb_file:
            self.params['node'] = self._init_emb_matrix(self.params['node']\
                        , '{}.node'.format(last_emb_file))
            self.params['layer'] = self._init_emb_matrix(self.params['layer']\
                        , '{}.layer'.format(last_emb_file))
            self.params['W'] = self._init_emb_matrix(self.params['W'], '{}.W'.format(last_emb_file))
        # adagrad
        self.h_delta = dict()
        self.h_delta['node'] = np.zeros((node_size,nd_rep_size))
        self.h_delta['layer'] = np.zeros((n_layers,layer_rep_size))
        self.h_delta['W'] = np.zeros((nd_rep_size,layer_rep_size))
        # adam
        self.m = dict()
        self.m['node'] = np.zeros((node_size,nd_rep_size))
        self.m['layer'] = np.zeros((n_layers,layer_rep_size))
        self.m['W'] = np.zeros((nd_rep_size,layer_rep_size))
        self.v = dict()
        self.v['node'] = np.zeros((node_size,nd_rep_size))
        self.v['layer'] = np.zeros((n_layers,layer_rep_size))
        self.v['W'] = np.zeros((nd_rep_size,layer_rep_size))
        self.t = 1

    def _init_emb_matrix(self, emb, emb_file):
        with open(emb_file, 'r') as embed_handler:
            for ln in embed_handler:
                elems = ln.strip().split()
                if len(elems)<=2:
                    continue
                emb[self.look_up[elems[0]]] = map(float, elems[1:])
        return emb

    def _read_anchors(self, anchor_file, delimiter):
        anchors = dict()
        num_anchors = 0
        with open(anchor_file, 'r') as anchor_handler:
            for ln in anchor_handler:
                elems = ln.strip().split(delimiter)
                for i in range(len(elems)):
                    elems[i] = '{}-{}'.format(i,elems[i])
                num_anchors += len(elems)-1
                for k in range(len(elems)):
                    anchors[elems[k]] = elems[:k]+elems[k+1:]
        return anchors, num_anchors

    def _init_simgoid_table(self):
        for k in range(self.sigmoid_table_size):
            x = 2*self.SIGMOID_BOUND*k/self.sigmoid_table_size-self.SIGMOID_BOUND
            self.sigmoid_table[k] = 1./(1+np.exp(-x))

    def _fast_sigmoid(self, val):
        if val>self.SIGMOID_BOUND:
            return 1-self.epsilon
        elif val<-self.SIGMOID_BOUND:
            return self.epsilon
        k = int((val+self.SIGMOID_BOUND)*self.sigmoid_table_size/self.SIGMOID_BOUND/2)
        return self.sigmoid_table[k]
        # return 1./(1+np.exp(-val))

    def _calc_delta_vec(self, nd, delta, opt_vec):
        if nd not in self.update_dict:
            cur_idx = self.idx
            self.update_dict[nd] = cur_idx
            self.update_look_back.append(nd)
            self.idx += 1
        else:
            cur_idx = self.update_dict[nd]
        if cur_idx>=len(delta):
            for i in range(cur_idx-len(delta)):
                delta.append(np.zeros(opt_vec.shape))
            delta.append(opt_vec)
        else:
            delta[cur_idx] += opt_vec
        return delta

    def _update_intra_vec(self, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos, neg = batch
        batch_size = len(pos['h'])

        # order 1
        pos_u = np.dot(self.params['node'][pos['h'],:],self.params['W'])+self.params['layer'][pos['h_layer'],:]
        pos_v = np.dot(self.params['node'][pos['t'],:],self.params['W'])+self.params['layer'][pos['t_layer'],:]
        neg_u = np.dot(self.params['node'][neg['h'],:],self.params['W'])+self.params['layer'][neg['h_layer'],:]
        neg_v = np.dot(self.params['node'][neg['t'],:],self.params['W'])+self.params['layer'][neg['t_layer'],:]

        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # delta calculation
        delta_eh = list()
        delta_l = np.zeros((self.num_layers, self.layer_rep_size)) ### problem in here ###

        idx = 0
        for i in range(len(pos['t'])):
            u,v = pos['h'][i],pos['t'][i]
            u_layer,v_layer = pos['h_layer'][i],pos['t_layer'][i]
            delta_eh = self._calc_delta_vec(v, delta_eh, (sigmoid_pos_e[i]-1)*pos_u[i,:])
            delta_eh = self._calc_delta_vec(u, delta_eh, (sigmoid_pos_e[i]-1)*pos_v[i,:])
            delta_l[v_layer] = (sigmoid_pos_e[i]-1)*pos_u[i,:]
            delta_l[u_layer] = (sigmoid_pos_e[i]-1)*pos_v[i,:]
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = neg['h'][i][j],neg['t'][i][j]
                u_layer,v_layer = neg['h_layer'][i][j],neg['t_layer'][i][j]
                delta_eh = self._calc_delta_vec(v, delta_eh, sigmoid_neg_e[i,j]*neg_u[i,j,:])
                delta_eh = self._calc_delta_vec(u, delta_eh, sigmoid_neg_e[i,j]*neg_v[i,j,:])
                delta_l[v_layer] = sigmoid_neg_e[i,j]*neg_u[i,j,:]
                delta_l[u_layer] = sigmoid_neg_e[i,j]*neg_v[i,j,:]

        delta_eh = np.array(delta_eh)
        delta_l = np.array(delta_l)

        # delta node, delta W, delta layer
        # print(self.params['node'][self.update_look_back,:].shape, delta_eh.shape)
        return np.dot(delta_eh, self.params['W'].T)/(batch_size*(1+self.negative_ratio))/2 \
                , np.dot(self.params['node'][self.update_look_back,:].T, delta_eh/batch_size) \
                , np.sum(delta_l, axis=0)/(batch_size*(1+self.negative_ratio))*self.num_layers

    def _get_loss(self, batch):
        pos, neg = batch
        batch_size = len(pos['h'])

        # order 1
        pos_u = np.dot(self.params['node'][pos['h'],:],self.params['W'])+self.params['layer'][pos['h_layer'],:]
        pos_v = np.dot(self.params['node'][pos['t'],:],self.params['W'])+self.params['layer'][pos['t_layer'],:]
        neg_u = np.dot(self.params['node'][neg['h'],:],self.params['W'])+self.params['layer'][neg['h_layer'],:]
        neg_v = np.dot(self.params['node'][neg['t'],:],self.params['W'])+self.params['layer'][neg['t_layer'],:]

        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_cur_batch_loss(self, t, batch):
        loss = self._get_loss(batch)

        self.logger.info('Finish processing batch {} and loss:{}'.format(t, loss))
        return loss

    def update_node_vec(self, h_delta, delta, embeddings, len_delta):
        h_delta[self.update_look_back[:len_delta],:] += delta**2
        # print 'original embedding:',embeddings[self.update_look_back[:len_delta]]
        embeddings[self.update_look_back[:len_delta],:] -= \
                                self.lr/np.sqrt(h_delta[self.update_look_back[:len_delta],:])*delta
        # print 'delta:',delta
        # print 'h_delta:',h_delta[self.update_look_back[:len_delta]]
        # print 'embeddings:',embeddings[self.update_look_back[:len_delta]]
        # print 'lmd_rda:',elem_lbd
        return h_delta, embeddings

    def update_vec(self, h_delta, delta, embeddings):
        h_delta += delta**2
        # print 'original embedding:',embeddings[self.update_look_back[:len_delta]]
        embeddings -= self.lr/np.sqrt(h_delta)*delta
        # print 'delta:',delta
        # print 'h_delta:',h_delta[self.update_look_back[:len_delta]]
        # print 'embeddings:',embeddings[self.update_look_back[:len_delta]]
        # print 'lmd_rda:',elem_lbd
        return h_delta, embeddings

    def update_node_vec_by_adam(self, m, v, delta, embeddings, t):
        self.beta1 = .9
        self.beta2 = .999
        m[self.update_look_back,:] = \
            self.beta1*m[self.update_look_back,:]+(1-self.beta1)*delta
        v[self.update_look_back,:] = \
            self.beta2*v[self.update_look_back,:]+(1-self.beta2)*(delta**2)
        m_ = m[self.update_look_back,:]/(1-self.beta1**t)
        v_ = v[self.update_look_back,:]/(1-self.beta2**t)

        embeddings[self.update_look_back,:] -= self.lr*m_/(np.sqrt(v_)+self.epsilon)

        return m,v,embeddings

    def update_vec_by_adam(self, m, v, delta, embeddings, t):
        self.beta1 = .9
        self.beta2 = .999
        m = self.beta1*m+(1-self.beta1)*delta
        v = self.beta2*v+(1-self.beta2)*(delta**2)
        m_ = m/(1-self.beta1**t)
        v_ = v/(1-self.beta2**t)

        embeddings -= self.lr*m_/(np.sqrt(v_)+self.epsilon)

        return m,v,embeddings

    def train_one_epoch(self):
        DISPLAY_EPOCH=1000

        opt_type = 'adam'
        loss = 0
        batches = self.batch_iter()
        for batch in batches:
            self.idx = 0
            self.update_look_back = list()
            self.update_dict = defaultdict(int)

            # delta node, delta W, delta layer
            delta_node, delta_W, delta_layer = self._update_intra_vec(batch)
            if opt_type=='adagrad':
                self.h_delta['node'] = \
                        self.update_node_vec(self.h_delta['node'], delta_node, self.params['node'], len(delta_node))
                self.h_delta['W'] = self.update_vec(self.h_delta['W'], delta_W, self.params['W'])
                self.h_delta['layer'] = self.update_vec(self.h_delta['layer'], delta_layer, self.params['layer'])
            if opt_type=='adam':
                self.m['node'], self.v['node'], self.params['node'] = \
                        self.update_node_vec_by_adam(self.m['node'], self.v['node'], delta_node
                            , self.params['node'], self.t)
                self.m['W'], self.v['W'], self.params['W'] = \
                        self.update_vec_by_adam(self.m['W'], self.v['W'], delta_W
                            , self.params['W'], self.t)
                self.m['layer'], self.v['layer'], self.params['layer'] = \
                        self.update_vec_by_adam(self.m['layer'], self.v['layer'], delta_layer
                            , self.params['layer'], self.t)

            if (self.t-1)%DISPLAY_EPOCH==0:
                loss += self.get_cur_batch_loss(self.t,batch)

            # print self.t, DISPLAY_EPOCH
            self.t += 1
        self.cur_epoch += 1

    def _get_nd_layer(self, idx):
        nd = self.look_back[idx]
        p=re.compile(r'(^\d+)-.*?')
        m=p.match(nd)
        if m:
            return int(m.group(1))
        return -1

    def layer_adjust(self, h_idx, t_idx):
        return self._get_nd_layer(h_idx) \
                if self._get_nd_layer(h_idx)>self._get_nd_layer(t_idx) \
                else self._get_nd_layer(t_idx)

    def get_random_node_pairs(self, i, shuffle_indices, edges, edge_set, numNodes):
        # balance the appearance of edges according to edge_prob
        if not random.random() < self.edge_prob[shuffle_indices[i]]:
            shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
        pos=dict()
        pos['h'] = edges[shuffle_indices[i]][0]
        pos['t'] = edges[shuffle_indices[i]][1]
        pos['h_layer'] = self.layer_adjust(pos['h'], pos['t'])
        pos['t_layer'] = self.layer_adjust(pos['h'], pos['t'])
        head = pos['h']*numNodes
        neg=defaultdict(list)
        # print(self.negative_ratio)
        for j in range(self.negative_ratio):
            rn = self.sampling_table[random.randint(0, self.table_size-1)]
            # print(self.sampling_table)
            # print('rn:',rn)
            while head+rn in edge_set or pos['h'] == rn or rn in neg['t']:
                rn = self.sampling_table[random.randint(0, self.table_size-1)]
                # print('rn in iteration:',rn)
            # print(rn)
            neg['h'].append(pos['h'])
            neg['t'].append(rn)
            neg['h_layer'].append(self.layer_adjust(pos['h'], rn))
            neg['t_layer'].append(self.layer_adjust(pos['h'], rn))
        return pos, neg

    def batch_iter(self):

        edges = []
        for k in range(self.num_layers):
            g = self.layer_graphs[k]
            edges += [(self.look_up[x[0]], self.look_up[x[1]]) for x in g.G.edges()]

        data_size = len(edges)
        edge_set = set([x[0]*self.node_size+x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            ret = {}
            pos = defaultdict(list)
            neg = defaultdict(list)
            for i in range(start_index, end_index):
                cur_pos, cur_neg = self.get_random_node_pairs(i, shuffle_indices, edges, edge_set, self.node_size)
                pos['h'].append(cur_pos['h'])
                pos['h_layer'].append(cur_pos['h_layer'])
                pos['t'].append(cur_pos['t'])
                pos['t_layer'].append(cur_pos['t_layer'])
                neg['h'].append(cur_neg['h'])
                neg['h_layer'].append(cur_neg['h_layer'])
                neg['t'].append(cur_neg['t'])
                neg['t_layer'].append(cur_neg['t_layer'])
            ret = (pos, neg)

            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)

            yield ret

    def _gen_sampling_table(self):
        table_size = self.table_size
        power = 0.75
        look_up = self.look_up
        numNodes = self.node_size

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes) # out degree
        edges = []
        for k in range(self.num_layers):
            g = self.layer_graphs[k]
            edges += [(look_up[x[0]], look_up[x[1]], g.G[x[0]][x[1]]['weight']) for x in g.G.edges()]
        # print(g.G.edges())
        # print('look_up',look_up)
        for edge in edges:
            node_degree[edge[0]] += edge[2]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        # print(numNodes)
        # print(node_degree)
        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1
        # print(self.sampling_table)

        data_size = len(edges)
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([edge[2] for edge in edges])
        norm_prob = [edge[2]*data_size/total_sum for edge in edges]
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        for k in range(data_size-1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
                num_small_block += 1
            else:
                large_block[num_large_block] = k
                num_large_block += 1
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
            self.edge_prob[cur_small_block] = norm_prob[cur_small_block]
            self.edge_alias[cur_small_block] = cur_large_block
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            self.edge_prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            self.edge_prob[small_block[num_small_block]] = 1

    def get_one_embeddings(self, params):
        vectors = dict()
        look_back = self.look_back
        for i, param in enumerate(params):
            vectors[look_back[i]] = param
        return vectors

    def get_vectors(self):
        ret = dict()
        ret['node']=self.get_one_embeddings(self.params['node'])
        ret['W']=self.params['W']
        ret['layer']=self.params['layer']

        return ret

class CROSSMNA(object):

    def __init__(self, layer_graphs, anchor_file, lr=.001, nd_rep_size=16,
                    layer_rep_size=16, batch_size=1000, epoch=10, 
                    negative_ratio=5, table_size=1e8, outfile='test',
                    last_emb_file=None, log_file='log'):
        SAVING_EPOCH=1
        # paramter initialization
        self.layer_graphs = layer_graphs
        self.nd_rep_size = nd_rep_size
        self.best_result = 0
        self.vectors = {}
        # training
        self.model = _CROSSMNA(layer_graphs, anchor_file, lr=lr, nd_rep_size=nd_rep_size
                            , layer_rep_size=layer_rep_size
                            , batch_size=batch_size, negative_ratio=negative_ratio
                            , table_size=table_size, log_file=log_file
                            , last_emb_file=last_emb_file)
        for i in range(1,epoch+1):
            self.model.train_one_epoch()
            if i%SAVING_EPOCH==0:
                self.get_params()
                self.save_params('{}.epoch{}'.format(outfile,i))
        self.get_params()

    def get_params(self):
        self.last_vectors = self.vectors
        self.vectors = self.model.get_vectors()

    def save_params(self, outfile):
        vectors = self.vectors
        for c in vectors.keys():
            fout = open('{}.{}'.format(outfile,c), 'w') 
            if 'node' in c:
                # outfile-[node_embeddings/content-embeddings]-[src/obj]
                node_num = len(vectors[c].keys())
                fout.write("{},{}\n".format(node_num, self.nd_rep_size))
                for node, vec in vectors[c].items():
                    fout.write("{},{}\n".format(node,','.join([str(x) for x in vec])))
            else:
                write_in_file(fout, vectors[c], c)
            fout.close()

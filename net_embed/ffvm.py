# -*- coding:utf8 -*-

from __future__ import print_function

import random
import math
import numpy as np
from collections import defaultdict
from utils.LogHandler import LogHandler
import os

class _FFVM(object):

    def __init__(self, graph, lr=.001, rep_size=128, batch_size=100, negative_ratio=5, order=3, table_size=1e8,
                    log_file='log', last_emb_file=None):

        if os.path.exists('log/'+log_file+'.log'):
            os.remove('log/'+log_file+'.log')
        self.logger = LogHandler(log_file)

        self.epsilon = 1e-7
        self.table_size = table_size
        self.sigmoid_table = {}
        self.sigmoid_table_size = 1000
        self.SIGMOID_BOUND = 6

        self._init_simgoid_table()

        self.g = graph
        self.look_up = self.g.look_up_dict
        self.idx = defaultdict(int)
        self.update_dict = defaultdict(dict)
        self.update_look_back = defaultdict(list)

        self.node_size = self.g.node_size
        self.rep_size = rep_size
        
        self._init_params(self.node_size, rep_size, last_emb_file)

        self.order = order
        self.lr = lr
        self.cur_epoch = 0
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

    def _init_params(self, node_size, rep_size, last_emb_file):
        self.embeddings = dict()
        self.embeddings['node'] = np.random.normal(0,1,(node_size,rep_size))
        self.embeddings['content'] = np.random.normal(0,1,(node_size,rep_size))
        if last_emb_file:
            self.embeddings['node'] = self._init_emb_matrix(self.embeddings['node']\
                        , '{}.node_embeddings'.format(last_emb_file))
            self.embeddings['content'] = self._init_emb_matrix(self.embeddings['content']\
                        , '{}.content_embeddings'.format(last_emb_file))
        self.embeddings['node'] = np.vstack((self.embeddings['node'], np.zeros(rep_size))) # for "-1" nodes
        # adagrad
        self.h_delta = dict()
        self.h_delta['node'] = np.zeros((node_size,rep_size))
        self.h_delta['content'] = np.zeros((node_size,rep_size))
        # adam
        self.m = dict()
        self.m['node'] = np.zeros((node_size,rep_size))
        self.m['content'] = np.zeros((node_size,rep_size))
        self.v = dict()
        self.v['node'] = np.zeros((node_size,rep_size))
        self.v['content'] = np.zeros((node_size,rep_size))
        self.t = 1

    def _init_emb_matrix(self, emb, emb_file):
        with open(emb_file, 'r') as embed_handler:
            for ln in embed_handler:
                elems = ln.strip().split()
                if len(elems)<=2:
                    continue
                emb[self.look_up[elems[0]]] = map(float, elems[1:])
        return emb

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

    def _format_vec(self, cal_type, vec):
        len_gap = len(vec)-self.idx[cal_type]
        if len_gap>0:
            for i in range(len_gap):
                vec.append(np.zeros(vec[0].shape))
        return np.array(vec)

    def _calc_delta_vec(self, cal_type, nd, delta, opt_vec):
        if nd not in self.update_dict[cal_type]:
            cur_idx = self.idx[cal_type]
            self.update_dict[cal_type][nd] = cur_idx
            self.update_look_back[cal_type].append(nd)
            self.idx[cal_type] += 1
        else:
            cur_idx = self.update_dict[cal_type][nd]
        if cur_idx>=len(delta):
            for i in range(cur_idx-len(delta)):
                delta.append(np.zeros(opt_vec.shape))
            delta.append(opt_vec)
        else:
            delta[cur_idx] += opt_vec
        return delta

    def _update_graph(self, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        sp_nds, sp_neighbors = batch
        batch_size = len(sp_nds)

        # order 1
        pos_q = self.embeddings['content'][sp_nds,:]
        pos_c = np.sum(self.embeddings['node'][sp_neighbors,:], axis=1)
        neg_q = self.embeddings['content'][sp_neighbors,:]

        neg_c = list()
        for c in pos_c:
            neg_c.append(np.tile(c, (self.negative_ratio, 1)))
        neg_c = np.array(neg_c)

        pos_e = np.sum(pos_q*pos_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_q*neg_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # delta calculation
        delta_q = list()
        delta_f = list()

        idx = 0
        for i in range(len(sp_nds)):
            u,neighbors = sp_nds[i],sp_neighbors[i]
            delta_q = self._calc_delta_vec('content', u, delta_q, (sigmoid_pos_e[i]-1)*pos_c[i,:])
            for v in neighbors:
                if v!=-1:
                    delta_f = self._calc_delta_vec('node', v, delta_f, (sigmoid_pos_e[i]-1)*pos_q[i,:])

        for i in range(len(sp_neighbors)):
            neighbors = sp_neighbors[i]
            for j in range(len(neighbors)):
                u = sp_neighbors[i][j]
                if u!=-1:
                    delta_q = self._calc_delta_vec('content', u, delta_q, sigmoid_neg_e[i,j]*neg_c[i,j,:])
                for v in neighbors:
                    delta_f = self._calc_delta_vec('node', v, delta_f, sigmoid_neg_e[i,j]*neg_q[i,j,:])

        delta_q = self._format_vec('content', delta_q)
        delta_f = self._format_vec('node', delta_f)

        return delta_q/batch_size, delta_f/batch_size

    def get_graph_loss(self, batch):
        sp_nds, sp_neighbors = batch
        batch_size = len(sp_nds)

        # order 1
        pos_q = self.embeddings['content'][sp_nds,:]
        pos_c = np.sum(self.embeddings['node'][sp_neighbors,:], axis=1)
        neg_q = self.embeddings['content'][sp_neighbors,:]

        neg_c = list()
        for c in pos_c:
            neg_c.append(np.tile(c, (self.negative_ratio, 1)))
        neg_c = np.array(neg_c)

        pos_e = np.sum(pos_q*pos_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_q*neg_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_cur_batch_loss(self, t, batch):
        loss = self.get_graph_loss(batch)
        self.logger.info('Finish processing batch {} and loss:{}'
                            .format(t, loss))

    def update_vec(self, cal_type, h_delta, delta, embeddings, len_delta, t):
        h_delta[self.update_look_back[cal_type][:len_delta],:] += delta**2
        # print 'original embedding:',embeddings[self.update_look_back[cal_type][:len_delta]]
        embeddings[self.update_look_back[cal_type][:len_delta],:] -= \
                                self.lr/np.sqrt(h_delta[self.update_look_back[cal_type][:len_delta],:])*delta
        # print 'delta:',delta
        # print 'h_delta:',h_delta[self.update_look_back[cal_type][:len_delta]]
        # print 'embeddings:',embeddings[self.update_look_back[cal_type][:len_delta]]
        # print 'lmd_rda:',elem_lbd
        return h_delta, embeddings

    def update_vec_by_adam(self, cal_type, m, v, delta, embeddings, len_delta, t):
        self.beta1 = .9
        self.beta2 = .999
        m[self.update_look_back[cal_type][:len_delta],:] = \
            self.beta1*m[self.update_look_back[cal_type][:len_delta],:]+(1-self.beta1)*delta
        v[self.update_look_back[cal_type][:len_delta],:] = \
            self.beta2*v[self.update_look_back[cal_type][:len_delta],:]+(1-self.beta2)*(delta**2)
        m_ = m[self.update_look_back[cal_type][:len_delta],:]/(1-self.beta1**t)
        v_ = v[self.update_look_back[cal_type][:len_delta],:]/(1-self.beta2**t)

        embeddings[self.update_look_back[cal_type][:len_delta],:] -= self.lr*m_/(np.sqrt(v_)+self.epsilon)

        return m,v,embeddings

    def train_one_epoch(self):
        DISPLAY_EPOCH=100
        batches = self.batch_iter()
        # opt_type = 'adagrad'
        opt_type = 'adam'
        for batch in batches:
            self.idx = defaultdict(int)
            self.update_look_back = defaultdict(list)
            self.update_dict = defaultdict(dict)
            delta_q, delta_f = self._update_graph(batch)
            len_delta_f = len(delta_f)
            # print 'order2, nd'
            if opt_type=='adagrad':
                self.h_delta['node'], self.embeddings['node'] = \
                                    self.update_vec('node', self.h_delta['node'], delta_f
                                                    , self.embeddings['node'], len_delta_f, self.t)
            if opt_type=='adam':
                self.m['node'], self.v['node'], self.embeddings['node'] = \
                                self.update_vec_by_adam('node', self.m['node'], self.v['node'], delta_f
                                                , self.embeddings['node'], len_delta_f, self.t)
            len_delta_q = len(delta_q)
            # print 'order2, content'
            if opt_type=='adagrad':
                self.h_delta['content'], self.embeddings['content'] = \
                                    self.update_vec('content', self.h_delta['content'], delta_q
                                                    , self.embeddings['content'], len_delta_q, self.t)
            if opt_type=='adam':
                self.m['content'], self.v['content'], self.embeddings['content'] = \
                                self.update_vec_by_adam('content', self.m['content'], self.v['content'], delta_q
                                                , self.embeddings['content'], len_delta_q, self.t)
            if (self.t-1)%DISPLAY_EPOCH==0:
                self.get_cur_batch_loss(self.t,batch)
            self.t += 1
        self.cur_epoch += 1

    def get_random_neighbor_nodes(self, nd_idx):
        graph = self.g.G
        look_up = self.g.look_up_dict
        look_back = self.g.look_back_list

        nd = self.g.look_back_list[nd_idx]
        neigh_nds = np.array([self.look_up[vid] for vid in graph[nd].keys()])
        shuffle_idx = np.random.permutation(np.arange(len(neigh_nds)))

        end_idx = self.negative_ratio if len(neigh_nds)>self.negative_ratio else len(neigh_nds)

        return neigh_nds[shuffle_idx[:end_idx]]

    def batch_iter(self):

        numNodes = self.node_size

        data_size = numNodes
        shuffle_indices = np.random.permutation(np.arange(data_size))

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            ret = {}
            sp_nds = shuffle_indices[start_index:end_index]
            sp_neighbors = []
            for idx in sp_nds:
                neighbors = self.get_random_neighbor_nodes(idx)
                if len(neighbors)<self.negative_ratio:
                    neighbors = np.hstack((neighbors, -np.ones(self.negative_ratio-len(neighbors)))).astype(int)
                sp_neighbors.append(neighbors)
            ret = sp_nds, sp_neighbors

            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)

            yield ret

    def get_one_embeddings(self, embeddings):
        vectors = dict()
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            if i==len(embeddings)-1:
                continue
            vectors[look_back[i]] = embedding
        return vectors

    def get_vectors(self):
        order = self.order
        ret = dict()
        node_embeddings=self.get_one_embeddings(self.embeddings['node'])
        ret['node']=node_embeddings
        content_embeddings=self.get_one_embeddings(self.embeddings['content'])
        ret['content']=content_embeddings

        return ret

class FFVM(object):

    def __init__(self, graph, lr=.001, rep_size=128, batch_size=1000, epoch=10, 
                    negative_ratio=5, table_size=1e8, outfile='test',
                    last_emb_file=None, log_file='log', auto_stop=True):
        SAVING_EPOCH=1
        # paramter initialization
        self.g = graph
        self.rep_size = rep_size
        self.vectors = {}
        # training
        self.model = _FFVM(graph, lr=lr, rep_size=rep_size
                            , batch_size=batch_size, negative_ratio=negative_ratio
                            , log_file=log_file, last_emb_file=last_emb_file)
        for i in range(1,epoch+1):
            self.model.train_one_epoch()
            if i%SAVING_EPOCH==0:
                self.get_embeddings()
                self.save_embeddings('{}.epoch{}'.format(outfile,i))
        self.get_embeddings()

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = self.model.get_vectors()

    def save_embeddings(self, outfile):
        vectors = self.vectors
        for c in vectors.keys():
            if 'node' in c or 'content' in c:
                # outfile-[node_embeddings/content-embeddings]-[src/obj]
                fout = open('{}.{}'.format(outfile,c), 'w') 
                node_num = len(vectors[c].keys())
                fout.write("{},{}\n".format(node_num, self.rep_size))
                for node, vec in vectors[c].items():
                    fout.write("{},{}\n".format(node,','.join([str(x) for x in vec])))
                fout.close()
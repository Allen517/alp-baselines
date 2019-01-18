# -*- coding:utf8 -*-

import random
import math
import numpy as np
from collections import defaultdict
from utils.LogHandler import LogHandler
import os

class _LINE(object):

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

        self.node_size = graph.G.number_of_nodes()
        self.rep_size = rep_size
        
        self._init_params(self.node_size, rep_size, last_emb_file)

        self.order = order
        self.lr = lr
        self.cur_epoch = 0
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self._gen_sampling_table()

    def _init_params(self, node_size, rep_size, last_emb_file):
        self.embeddings = dict()
        self.embeddings['order1'] = np.random.normal(0,1,(node_size,rep_size))
        self.embeddings['order2'] = np.random.normal(0,1,(node_size,rep_size))
        self.embeddings['content'] = np.random.normal(0,1,(node_size,rep_size))
        if last_emb_file:
            self.embeddings['order1'] = self._init_emb_matrix(self.embeddings['order1']\
                        , '{}.node_embeddings_order1'.format(last_emb_file))
            self.embeddings['order2'] = self._init_emb_matrix(self.embeddings['order2']\
                        , '{}.node_embeddings_order2'.format(last_emb_file))
            self.embeddings['content'] = self._init_emb_matrix(self.embeddings['content']\
                        , '{}.content_embeddings'.format(last_emb_file))
        # adagrad
        self.h_delta = dict()
        self.h_delta['order1'] = np.zeros((node_size,rep_size))
        self.h_delta['order2'] = np.zeros((node_size,rep_size))
        self.h_delta['content'] = np.zeros((node_size,rep_size))
        # adam
        self.m = dict()
        self.m['order1'] = np.zeros((node_size,rep_size))
        self.m['order2'] = np.zeros((node_size,rep_size))
        self.m['content'] = np.zeros((node_size,rep_size))
        self.v = dict()
        self.v['order1'] = np.zeros((node_size,rep_size))
        self.v['order2'] = np.zeros((node_size,rep_size))
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

    def _update_graph_by_order2(self, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos_h, pos_t, pos_h_v, neg_t = batch
        batch_size = len(pos_h)

        # order 2
        pos_u = self.embeddings['order2'][pos_h,:]
        pos_v_c = self.embeddings['content'][pos_t,:]
        neg_u = self.embeddings['order2'][pos_h_v,:]
        neg_v_c = self.embeddings['content'][neg_t,:]

        pos_e = np.sum(pos_u*pos_v_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # temporal delta
        delta_eh = list()
        delta_c = list()

        idx = 0
        for i in range(len(pos_t)):
            u,v = pos_h[i],pos_t[i]
            delta_c = self._calc_delta_vec('cnt_order2', v, delta_c, (sigmoid_pos_e[i]-1)*pos_u[i,:])
            delta_eh = self._calc_delta_vec('nd_order2', u, delta_eh, (sigmoid_pos_e[i]-1)*pos_v_c[i,:])
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = pos_h_v[i][j],neg_t[i][j]
                delta_c = self._calc_delta_vec('cnt_order2', v, delta_c, sigmoid_neg_e[i,j]*neg_u[i,j,:])
                delta_eh = self._calc_delta_vec('nd_order2', u, delta_eh, sigmoid_neg_e[i,j]*neg_v_c[i,j,:])

        # delta x & delta codebook
        delta_eh = self._format_vec('nd_order2', delta_eh)
        delta_c = self._format_vec('cnt_order2', delta_c)

        return delta_c/batch_size, delta_eh/batch_size

    def _update_graph_by_order1(self, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos_h, pos_t, pos_h_v, neg_t = batch
        batch_size = len(pos_h)

        # order 1
        pos_u = self.embeddings['order1'][pos_h,:]
        pos_v = self.embeddings['order1'][pos_t,:]
        neg_u = self.embeddings['order1'][pos_h_v,:]
        neg_v = self.embeddings['order1'][neg_t,:]

        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # delta calculation
        delta_eh = list()

        idx = 0
        for i in range(len(pos_t)):
            u,v = pos_h[i],pos_t[i]
            delta_eh = self._calc_delta_vec('nd_order1', v, delta_eh, (sigmoid_pos_e[i]-1)*pos_u[i,:])
            delta_eh = self._calc_delta_vec('nd_order1', u, delta_eh, (sigmoid_pos_e[i]-1)*pos_v[i,:])
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = pos_h_v[i][j],neg_t[i][j]
                delta_eh = self._calc_delta_vec('nd_order1', v, delta_eh, sigmoid_neg_e[i,j]*neg_u[i,j,:])
                delta_eh = self._calc_delta_vec('nd_order1', u, delta_eh, sigmoid_neg_e[i,j]*neg_v[i,j,:])

        # delta x & delta codebook
        delta_eh = self._format_vec('nd_order1', delta_eh)

        return delta_eh/batch_size

    def _mat_add(self, mat1, mat2):
        len_gap = len(mat1)-len(mat2)
        if len_gap>0:
            for i in range(len_gap):
                mat2 = np.vstack((mat2, np.zeros(mat2[0,:].shape)))
        else:
            for i in range(-len_gap):
                mat1 = np.vstack((mat1, np.zeros(mat1[0,:].shape)))

        return mat1+mat2

    def get_graph_loss_by_order2(self, batch):
        pos_h, pos_t, pos_h_v, neg_t = batch

        # order 2
        pos_u = self.embeddings['order2'][pos_h,:]
        pos_v_c = self.embeddings['content'][pos_t,:]
        neg_u = self.embeddings['order2'][pos_h_v,:]
        neg_v_c = self.embeddings['content'][neg_t,:]

        pos_e = np.sum(pos_u*pos_v_c, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_graph_loss_by_order1(self, batch):
        pos_h, pos_t, pos_h_v, neg_t = batch

        # order 2
        pos_u = self.embeddings['order1'][pos_h,:]
        pos_v = self.embeddings['order1'][pos_t,:]
        neg_u = self.embeddings['order1'][pos_h_v,:]
        neg_v = self.embeddings['order1'][neg_t,:]

        # pos_e_1 = np.sum(pos_u*pos_v, axis=1)+np.sum(self.b_e[key][0][pos_t,:], axis=1) # pos_e.shape = batch_size
        # neg_e_1 = np.sum(neg_u*neg_v, axis=2)+np.sum(self.b_e[key][0][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio
        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_cur_batch_loss(self, t, batch):
        loss_order_1 = 0.0
        loss_order_2 = 0.0
        if self.order==1 or self.order==3:
            loss_order_1 += self.get_graph_loss_by_order1(batch)
        if self.order==2 or self.order==3:
            loss_order_2 += self.get_graph_loss_by_order2(batch)
        if self.order==1:
            self.logger.info('Finish processing batch {} and loss from order 1:{}'
                            .format(t, loss_order_1))
        elif self.order==2:
            self.logger.info('Finish processing batch {} and loss from order 2:{}'
                            .format(t, loss_order_2))
        elif self.order==3:
            self.logger.info('Finish processing batch {} and loss from order 3:{}'
                            .format(t, loss_order_1+loss_order_2))

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
        DISPLAY_EPOCH=1000
        order = self.order
        batches = self.batch_iter()
        opt_type = 'adam'
        for batch in batches:
            self.idx = defaultdict(int)
            self.update_look_back = defaultdict(list)
            self.update_dict = defaultdict(dict)
            if order==1 or order==3:
                delta_eh_o1 = self._update_graph_by_order1(batch)
                len_delta = len(delta_eh_o1)
                # print 'order1 nd'
                if opt_type=='adagrad':
                    self.h_delta['order1'], self.embeddings['order1'] = \
                                    self.update_vec('nd_order1', self.h_delta['order1'], delta_eh_o1
                                                    , self.embeddings['order1'], len_delta, self.t)
                if opt_type=='adam':
                    self.m['order1'], self.v['order1'], self.embeddings['order1'] = \
                                    self.update_vec_by_adam('nd_order1', self.m['order1'], self.v['order1'], delta_eh_o1
                                                    , self.embeddings['order1'], len_delta, self.t)
            if order==2 or order==3:
                delta_c, delta_eh_o2 = self._update_graph_by_order2(batch)
                len_delta = len(delta_eh_o2)
                # print 'order2, nd'
                if opt_type=='adagrad':
                    self.h_delta['order2'], self.embeddings['order2'] = \
                                        self.update_vec('nd_order2', self.h_delta['order2'], delta_eh_o2
                                                        , self.embeddings['order2'], len_delta, self.t)
                if opt_type=='adam':
                    self.m['order2'], self.v['order2'], self.embeddings['order2'] = \
                                    self.update_vec_by_adam('nd_order2', self.m['order2'], self.v['order2'], delta_eh_o2
                                                    , self.embeddings['order2'], len_delta, self.t)
                len_content = len(delta_c)
                # print 'order2, content'
                if opt_type=='adagrad':
                    self.h_delta['content'], self.embeddings['content'] = \
                                        self.update_vec('cnt_order2', self.h_delta['content'], delta_c
                                                        , self.embeddings['content'], len_content, self.t)
                if opt_type=='adam':
                    self.m['content'], self.v['content'], self.embeddings['content'] = \
                                    self.update_vec_by_adam('cnt_order2', self.m['content'], self.v['content'], delta_c
                                                    , self.embeddings['content'], len_content, self.t)
                # self.embeddings_order2[self.update_look_back[:len_de],:] -= self.lr*delta_eh
                # len_content = len(delta_c)
                # self.content_embeddings[self.update_look_back[:len_content],:] -= self.lr*delta_c
                # break
            if (self.t-1)%DISPLAY_EPOCH==0:
                self.get_cur_batch_loss(self.t,batch)
            # print self.t, DISPLAY_EPOCH
            self.t += 1
        self.cur_epoch += 1

    def get_random_node_pairs(self, i, shuffle_indices, edges, edge_set, numNodes):
        # balance the appearance of edges according to edge_prob
        if not random.random() < self.edge_prob[shuffle_indices[i]]:
            shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
        cur_h = edges[shuffle_indices[i]][0]
        head = cur_h*numNodes
        cur_t = edges[shuffle_indices[i]][1]
        cur_h_v = []
        cur_neg_t = []
        for j in range(self.negative_ratio):
            rn = self.sampling_table[random.randint(0, self.table_size-1)]
            while head+rn in edge_set or cur_h == rn or rn in cur_neg_t:
                rn = self.sampling_table[random.randint(0, self.table_size-1)]
            cur_h_v.append(cur_h)
            cur_neg_t.append(rn)
        return cur_h, cur_t, cur_h_v, cur_neg_t

    def batch_iter(self):

        numNodes = self.node_size

        edges = [(self.look_up[x[0]], self.look_up[x[1]]) for x in self.g.G.edges()]
        data_size = self.g.G.number_of_edges()
        edge_set = set([x[0]*numNodes+x[1] for x in edges])
        shuffle_indices = np.random.permutation(np.arange(data_size))

        start_index = 0
        end_index = min(start_index+self.batch_size, data_size)
        while start_index < data_size:
            ret = {}
            pos_h = []
            pos_t = []
            pos_h_v = []
            neg_t = []
            for i in range(start_index, end_index):
                cur_h, cur_t, cur_h_v, cur_neg_t = self.get_random_node_pairs(i, shuffle_indices, edges, edge_set, numNodes)
                pos_h.append(cur_h)
                pos_t.append(cur_t)
                pos_h_v.append(cur_h_v)
                neg_t.append(cur_neg_t)
            ret = (pos_h, pos_t, pos_h_v, neg_t)

            start_index = end_index
            end_index = min(start_index+self.batch_size, data_size)

            yield ret

    def _gen_sampling_table(self):
        table_size = self.table_size
        power = 0.75
        numNodes = self.node_size

        print "Pre-procesing for non-uniform negative sampling!"
        self.node_degree = np.zeros(numNodes) # out degree

        look_up = self.g.look_up_dict
        for edge in self.g.G.edges():
            self.node_degree[look_up[edge[0]]] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(self.node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(self.node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1

        data_size = self.g.G.number_of_edges()
        self.edge_alias = np.zeros(data_size, dtype=np.int32)
        self.edge_prob = np.zeros(data_size, dtype=np.float32)
        large_block = np.zeros(data_size, dtype=np.int32)
        small_block = np.zeros(data_size, dtype=np.int32)

        total_sum = sum([self.g.G[edge[0]][edge[1]]["weight"] for edge in self.g.G.edges()])
        norm_prob = [self.g.G[edge[0]][edge[1]]["weight"]*data_size/total_sum for edge in self.g.G.edges()]
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

    def get_one_embeddings(self, embeddings):
        vectors = dict()
        look_back = self.g.look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

    def get_vectors(self):
        order = self.order
        ret = dict()
        node_embeddings_order1=self.get_one_embeddings(self.embeddings['order1'])
        ret['node_order1']=node_embeddings_order1
        if order==2:
            node_embeddings_order2=self.get_one_embeddings(self.embeddings['order2'])
            ret['node_order2']=node_embeddings_order2

        if order==2 or order==3:
            content_embeddings = dict()
            content_embeddings=self.get_one_embeddings(self.embeddings['content'])
            ret['content']=content_embeddings

        return ret

class LINE(object):

    def __init__(self, graph, lr=.001, rep_size=128, batch_size=1000, epoch=10, 
                    negative_ratio=5, order=3, table_size=1e8, outfile='test',
                    last_emb_file=None, log_file='log', auto_stop=True):
        SAVING_EPOCH=1
        # paramter initialization
        self.g = graph
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        # training
        self.model = _LINE(graph, lr=lr, rep_size=rep_size
                            , batch_size=batch_size, negative_ratio=negative_ratio
                            , order=self.order
                            , table_size=table_size, log_file=log_file
                            , last_emb_file=last_emb_file)
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
                fout.write("{} {}\n".format(node_num, self.rep_size))
                for node, vec in vectors[c].items():
                    fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
                fout.close()
        if self.order==3:
            fout = open('{}.node_all'.format(outfile), 'w') 
            node_num = len(vectors[c].keys())
            fout.write("{} {}\n".format(node_num, self.rep_size*2))
            for node, vec in vectors['node_order1'].items():
                fout.write("{} {} {}\n".format(node,' '.join([str(x) for x in vec])
                                , ' '.join([str(x) for x in vectors['node_order2'][node]])))
            fout.close()

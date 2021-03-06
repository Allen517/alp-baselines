from __future__ import print_function

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#存放c.py所在的绝对路径

sys.path.append(BASE_DIR)

from eval.eval import *
from utils.graphx import *
from collections import defaultdict
from sklearn import svm
from sklearn.externals import joblib

class Eval_MNA(Eval):

    def __init__(self):
        super(Eval_MNA,self).__init__()

    def _init_eval(self, **kwargs):
        allows_keys = {'net_src', 'net_end', 'feat_src', 'feat_end', 'use_net', 'linkage'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid file inputs: '+kw

        # print(kwargs['use_net'])
        self.use_net = kwargs['use_net']
        # print(self.use_net)

        self.graphs = {
                'src': GraphX(),
                'end': GraphX()
            }
        self.graphs['src'].read_adjlist(kwargs['net_src'])
        self.graphs['end'].read_adjlist(kwargs['net_end'])
        assert self.graphs['src'], 'Failed to read graph from source network'
        assert self.graphs['end'], 'Failed to read graph from end network'

        if kwargs['feat_src'] and kwargs['feat_end']:
            self.inputs['src'] = self._read_inputs(kwargs['feat_src'])
            self.inputs['end'] = self._read_inputs(kwargs['feat_end'])
            assert self.inputs['src'], 'Failed to read features from source network'
            assert self.inputs['end'], 'Failed to read features from end network'

        self.labels = self._read_labels(kwargs['linkage'])
        assert self.labels, 'Failed to read labels'

    def _read_model(self, filepath):
        print('reading model %s'%(filepath))
        assert os.path.exists(filepath), 'Model file does not exist: %s'%filepath

        return joblib.load(filepath)

    def _read_labels(self, filepath):
        print('reading inputs %s'%(filepath+'.train'))
        assert os.path.exists(filepath+'.train'), 'Label file does not exist: %s'%filepath+'.train'

        print('reading inputs %s'%(filepath+'.test'))
        assert os.path.exists(filepath+'.test'), 'Label file does not exist: %s'%filepath+'.test'

        lbs = {
            'src2end': dict(),
            'end2src': dict()
        }
        lbs['src2end'] = {
            'train': defaultdict(list),
            'test': defaultdict(list)
        }
        lbs['end2src'] = {
            'train': defaultdict(list),
            'test': defaultdict(list)
        }
        for tag in ['train', 'test']:
            with open('%s.%s'%(filepath,tag), 'r') as fin:
                for ln in fin:
                    elems = ln.strip().split(',')
                    if len(elems)!=2:
                        continue
                    nd_src,nd_end = elems
                    lbs['src2end'][tag][nd_src].append(nd_end)
                    lbs['end2src'][tag][nd_end].append(nd_src)
        return lbs

    def _calc_model_res(self, **kwargs):
        allows_keys = {'inputs'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        return self.model.predict_proba(np.array(kwargs['inputs']))

    def _get_pair_features(self, nd_from, nd_to):

        feat_net = []
        if self.use_net:
            src_neighbor_anchors = set()
            for src_nd_to in self.graphs['src'].G[nd_from]:
                if src_nd_to in self.labels['src2end']['train']:
                    src_neighbor_anchors.add(src_nd_to)

            target_neighbor_anchors = set()
            for target_nd_to in self.graphs['end'].G[nd_to]:
                if target_nd_to in self.labels['end2src']['train']:
                    target_neighbor_anchors.add(target_nd_to)

            cnt_common_neighbors = .0
            AA_measure = .0
            for sna in src_neighbor_anchors:
                for k in range(len(self.labels['src2end']['train'][sna])):
                    target_anchor_nd = self.labels['src2end']['train'][sna][k]
                    if target_anchor_nd in target_neighbor_anchors:
                        cnt_common_neighbors += 1.
                        AA_measure += 1./(np.log((len(self.graphs['src'].G[sna])\
                                        +len(self.graphs['end'].G[self.labels['src2end']['train'][sna][k]]))/2.)+1e-6)
            jaccard = cnt_common_neighbors/(len(self.graphs['src'].G[nd_from])\
                                            +len(self.graphs['end'].G[nd_to])\
                                            -cnt_common_neighbors+1e-6)

            feat_net = [cnt_common_neighbors, jaccard, AA_measure]

        feat_attr = []
        if len(self.inputs)>0:
            if nd_from not in self.inputs['src'] or nd_to not in self.inputs['end']\
                 or len(self.inputs['src'][nd_from])!=len(self.inputs['end'][nd_to]):
                return []
            feat_len = len(self.inputs['src'][nd_from])
            feat_attr = [1-self.inputs['src'][nd_from][k]\
                            +self.inputs['end'][nd_to][k] for k in range(feat_len)]

        # print(feat_len)
        # print(self.inputs['src'][nd_from], self.inputs['src'][nd_to])
        return feat_net+feat_attr

    def calc_mrr_by_dist(self, **kwargs):
        allows_keys = {'model', 'candidate_num', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        print("Read model from %s"%kwargs['model'])
        self.model = self._read_model(kwargs['model'])
        assert self.model, 'Failed to read model'

        n_cands = kwargs['candidate_num']

        mrr_list = tuple()

        with open(kwargs['out_file'], 'w') as fout:
            cnt = 0
            wrt_lns = ''
            to_keys = list(self.graphs['end'].G.keys())
            to_size = len(to_keys)

            inputs_batch = {
                'pos': list(),
                'neg': list(),
            }

            node_pairs = {
                'pos': list(),
                'neg': list()
            }

            model_res = {
                'pos': list(),
                'neg': list(),
            }

            max_feat_len = 0
            for nd_from, nds_to in self.labels['src2end']['test'].items():
                for nd_to in nds_to:

                    pos_pair_features = self._get_pair_features(nd_from, nd_to)
                    if len(pos_pair_features)<1 or len(pos_pair_features)<max_feat_len:
                        continue
                    if max_feat_len<len(pos_pair_features):
                        max_feat_len=len(pos_pair_features)
                    inputs_batch['pos'].append(pos_pair_features)
                    node_pairs['pos'].append([nd_from, nd_to])

                    rand_nds = set()
                    for k in range(n_cands):
                        rand_nd_to = to_keys[np.random.randint(0, to_size)]
                        while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                        rand_nds.add(rand_nd_to)
                        neg_pair_features = self._get_pair_features(nd_from, rand_nd_to)
                        if len(neg_pair_features)<1 or len(neg_pair_features)<max_feat_len:
                            k-=1
                            continue
                        inputs_batch['neg'].append(neg_pair_features)
                        node_pairs['neg'].append([nd_from, rand_nd_to])

                    cnt += 1
                    if not cnt%100:
                        for k in model_res.keys():
                            # print('feat_len:', len(inputs_batch[k][0]))
                            model_res[k].extend(self._calc_model_res(inputs=inputs_batch[k]))
                            inputs_batch[k] = list()
            if cnt%100:
                for k in model_res.keys():
                    model_res[k].extend(self._calc_model_res(inputs=inputs_batch[k]))

            cnt = 0
            # print(len(node_pairs['pos']),len(node_pairs['neg']), len(model_res['pos']), len(model_res['neg']))
            for i in range(len(node_pairs['pos'])):
                is_valid = True
                # print(model_res['pos'])
                anchor_dist = model_res['pos'][i][0]
                # print('anchor dist:',anchor_dist)
                pred_pos = 1
                for k in range(n_cands):
                    if i*n_cands+k>=len(model_res['neg']):
                        is_valid = False
                        continue
                    noise_dist = model_res['neg'][i*n_cands+k][0]
                    # print('noise_dist:',noise_dist)
                    if anchor_dist>=noise_dist:
                        pred_pos += 1
                if not is_valid:
                    continue
                cur_mrr = 1./pred_pos
                mrr_list += cur_mrr,
                cnt += 1
                wrt_lns += '(%s,%s):%f\n'%(node_pairs['pos'][i][0], node_pairs['pos'][i][1], cur_mrr)
                if not cnt%100:
                    fout.write(wrt_lns)
                    print('Processing %d records'%cnt)
                    wrt_lns = ''
            if cnt%100:
                fout.write(wrt_lns)
            fout.write('mean_mrr:{}, var:{}\n'
                    .format(np.mean(mrr_list), np.var(mrr_list)))

    def eval_classes(self, **kwargs):
        allows_keys = {'model', 'candidate_num', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        self.model = self._read_model(kwargs['model'])
        assert self.model, 'Failed to read model'

        n_cands = kwargs['candidate_num']

        with open(kwargs['out_file'], 'w') as fout:
            cnt = 0
            wrt_lns = ''
            to_keys = list(self.graphs['end'].G.keys())
            to_size = len(to_keys)
            fout.write('Overall: %d\n'%len(self.labels['src2end'].keys()))

            inputs_batch = {
                'pos': list(),
                'neg': list(),
            }

            node_pairs = {
                'pos': list(),
                'neg': list()
            }

            model_res = {
                'pos': list(),
                'neg': list(),
            }

            max_feat_len = 0
            for nd_from, nds_to in self.labels['src2end']['test'].items():
                for nd_to in nds_to:

                    pos_pair_features = self._get_pair_features(nd_from, nd_to)
                    if len(pos_pair_features)<1 or len(pos_pair_features)<max_feat_len:
                        continue
                    if max_feat_len<len(pos_pair_features):
                        max_feat_len=len(pos_pair_features)
                    inputs_batch['pos'].append(pos_pair_features)
                    node_pairs['pos'].append([nd_from, nd_to])

                    rand_nds = set()
                    for k in range(n_cands):
                        rand_nd_to = to_keys[np.random.randint(0, to_size)]
                        while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                        rand_nds.add(rand_nd_to)
                        neg_pair_features = self._get_pair_features(nd_from, rand_nd_to)
                        if len(neg_pair_features)<1 or len(neg_pair_features)<max_feat_len:
                            k-=1
                            continue
                        inputs_batch['neg'].append(neg_pair_features)
                        node_pairs['neg'].append([nd_from, rand_nd_to])

                    cnt += 1
                    if not cnt%100:
                        for k in model_res.keys():
                            model_res[k].extend(self._calc_model_res(inputs=inputs_batch[k]))
                            inputs_batch[k] = list()
            if cnt%100:
                for k in model_res.keys():
                    model_res[k].extend(self._calc_model_res(inputs=inputs_batch[k]))

            cnt = 0
            # print(len(node_pairs['pos']),len(node_pairs['neg']), len(model_res['pos']), len(model_res['neg']))
            for i in range(len(node_pairs['pos'])):
                is_valid = True
                # print(model_res['pos'])
                anchor_dist = model_res['pos'][i][0]
                wrt_lns += '(%s,%s),%f,%d\n'%(node_pairs['pos'][i][0],node_pairs['pos'][i][1],anchor_dist,1)
                # print('anchor dist:',anchor_dist)
                for k in range(n_cands):
                    if i*n_cands+k>=len(model_res['neg']):
                        is_valid = False
                        continue
                    noise_dist = model_res['neg'][i*n_cands+k][0]
                    wrt_lns += '(%s,%s),%f,%d\n'%(node_pairs['neg'][i*n_cands+k][0]\
                                    ,node_pairs['neg'][i*n_cands+k][1],noise_dist,0)
                if not is_valid:
                    continue
                cnt += 1
                if not cnt%100:
                    fout.write(wrt_lns)
                    print('Processing %d records'%cnt)
                    wrt_lns = ''
            if cnt%100:
                fout.write(wrt_lns)

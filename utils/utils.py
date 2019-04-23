from __future__ import print_function

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random
from collections import defaultdict
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

def load_train_valid_labels(filename, lookups, valid_prop, delimiter=','):
    lbs = dict()
    lbs['f2g'] = dict()
    lbs['f2g']['train'] = defaultdict(list)
    lbs['f2g']['valid'] = defaultdict(list)
    lbs['g2f'] = dict()
    lbs['g2f']['train'] = defaultdict(list)
    lbs['g2f']['valid'] = defaultdict(list)
    with open(filename, 'r') as fin:
        for ln in fin:
            elems = ln.strip().split(delimiter)
            if len(elems)!=2:
                continue
            nd_src,nd_end = elems
            if nd_src not in lookups['f']:
                continue
            if nd_end not in lookups['g']:
                continue
            if np.random.random()<valid_prop:
                lbs['f2g']['train'][nd_src].append(nd_end)
                lbs['g2f']['train'][nd_end].append(nd_src)
            else:
                lbs['f2g']['valid'][nd_src].append(nd_end)
                lbs['g2f']['valid'][nd_end].append(nd_src)
    assert lbs['f2g']['train'] and lbs['g2f']['train']\
                , 'Fail to read labels. The delimiter may be mal-used!'
    return lbs

def batch_iter(lbs, batch_size, neg_ratio, lookup, lb_tag_src, lb_tag_end):
    train_lbs = lbs['{}2{}'.format(lb_tag_src,lb_tag_end)]['train']
    cands_end = list(lookup[lb_tag_end].keys())

    start_index = 0
    train_size = len(train_lbs)
    end_index = min(start_index+batch_size, train_size)

    lb_keys_src = list(train_lbs.keys())
    shuffle_indices = np.random.permutation(np.arange(train_size))
    while start_index < end_index:
        pos = {lb_tag_src:list(), lb_tag_end:list()}
        neg = {lb_tag_src:list(), lb_tag_end:list()}
        for i in range(start_index,end_index):
            idx = shuffle_indices[i]
            src_lb = lb_keys_src[idx]
            if not src_lb in lookup[lb_tag_src]:
                continue
            nd_idx = {lb_tag_src:-1, lb_tag_end:-1, 'rand':-1}
            nd_idx[lb_tag_src] = lookup[lb_tag_src][src_lb] # idx in src network
            lbs_idx_end = [lookup[lb_tag_end][lb_end] for lb_end in train_lbs[src_lb]]
            for lb_idx in lbs_idx_end:
                nd_idx[lb_tag_end] = lb_idx
                neg_idx_cur = {lb_tag_src:list(), lb_tag_end:list()}
                for k in range(neg_ratio):
                    nd_idx['rand'] = -1
                    while nd_idx['rand']<0 or nd_idx['rand'] in lbs_idx_end:
                        nd_idx['rand'] = np.random.randint(0, len(cands_end))
                    neg_idx_cur[lb_tag_src].append(nd_idx[lb_tag_src])
                    neg_idx_cur[lb_tag_end].append(nd_idx['rand'])
                pos[lb_tag_src].append(nd_idx[lb_tag_src])
                pos[lb_tag_end].append(nd_idx[lb_tag_end])
                neg[lb_tag_src].append(neg_idx_cur[lb_tag_src])
                neg[lb_tag_end].append(neg_idx_cur[lb_tag_end])

        start_index = end_index
        end_index = min(start_index+batch_size, train_size)
        
        yield pos,neg

def valid_iter(lbs, valid_sample_size, lookup, lb_tag_src, lb_tag_end):
    valid_lbs = lbs['{}2{}'.format(lb_tag_src,lb_tag_end)]['valid']
    cands_end = list(lookup[lb_tag_end].keys())

    valid = {lb_tag_src:list(), lb_tag_end:list()}
    lb_keys_src = list(valid_lbs.keys())
    for lb_src in lb_keys_src:
        if not lb_src in lookup[lb_tag_src]:
            continue
        nd_idx = {lb_tag_src:-1, lb_tag_end:-1, 'rand':-1}
        nd_idx[lb_tag_src] = lookup[lb_tag_src][lb_src] # idx in src network
        lbs_idx_end = [lookup[lb_tag_end][lb_end] for lb_end in valid_lbs[lb_src]]
        for lb_idx in lbs_idx_end:
            nd_idx[lb_tag_end] = lb_idx
            cand = {lb_tag_src:list(),lb_tag_end:list()}
            cand[lb_tag_src].append(nd_idx[lb_tag_src])
            cand[lb_tag_end].append(nd_idx[lb_tag_end])
            for k in range(valid_sample_size-1):
                nd_idx['rand'] = -1
                while nd_idx['rand']<0 or nd_idx['rand'] in lbs_idx_end:
                    nd_idx['rand'] = np.random.randint(0, len(cands_end))
                cand[lb_tag_src].append(nd_idx[lb_tag_src])
                cand[lb_tag_end].append(nd_idx['rand'])
            if (cand[lb_tag_src] and cand[lb_tag_end])\
                and len(cand[lb_tag_src])==valid_sample_size\
                and len(cand[lb_tag_end])==valid_sample_size:
                valid[lb_tag_src].append(cand[lb_tag_src])
                valid[lb_tag_end].append(cand[lb_tag_end])

    return valid

def read_embeddings(embed_file):
    embedding = list()
    lookup = dict()
    look_back = list()
    with open(embed_file, 'r') as emb_handler:
        idx = 0
        for ln in emb_handler:
            ln = ln.strip()
            if ln:
                elems = ln.split()
                if len(elems)==2:
                    continue
                embedding.append(list(map(float, elems[1:])))
                lookup[elems[0]] = idx
                look_back.append(elems[0])
                idx += 1

    return np.array(embedding), lookup, look_back

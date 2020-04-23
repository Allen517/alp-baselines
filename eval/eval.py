# -*- coding=UTF-8 -*-\n
from __future__ import print_function

import numpy as np
import random
from collections import defaultdict
import json
import re
import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#存放c.py所在的绝对路径

sys.path.append(BASE_DIR)

from eval.measures import *

class Eval(object):

    def __init__(self):
        self.model = defaultdict(dict)
        self.labels = list()
        self.inputs = defaultdict(dict)

    def _read_model(self, filepath):
        raise NotImplementedError

    def _calc_model_res(self, **kwargs):
        raise NotImplementedError

    def _read_inputs(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Inputs file does not exist: %s'%filepath

        inputs = dict()
        with open(filepath, 'r') as f_handler:
            for ln in f_handler:
                ln = ln.strip()
                if ln:
                    elems = ln.split(',')
                    if len(elems)==2:
                        continue
                    inputs[elems[0]] = list(map(float, elems[1:]))
        return inputs

    def _read_labels(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Label file does not exist: %s'%filepath

        lbs = {
            'src2end': defaultdict(list),
            'end2src': defaultdict(list)
        }
        with open(filepath, 'r') as fin:
            for ln in fin:
                elems = ln.strip().split(',')
                if len(elems)!=2:
                    continue
                nd_src,nd_end = elems
                lbs['src2end'][nd_src].append(nd_end)
                lbs['end2src'][nd_end].append(nd_src)
        return lbs

    def _init_eval(self, **kwargs):
        allows_keys = {'feat_src', 'feat_end', 'linkage'}
        for k in kwargs.keys():
            assert k in allows_keys, 'Invalid file inputs: '+k

        print('processing {} and {}'.format(kwargs['feat_src'], kwargs['feat_end']))
        assert os.path.exists(kwargs['feat_src']) and os.path.exists(kwargs['feat_end'])\
                , 'Files not found: %s, %s'%(kwargs['feat_src'], kwargs['feat_end'])

        self.inputs['src'] = self._read_inputs(kwargs['feat_src'])
        self.inputs['end'] = self._read_inputs(kwargs['feat_end'])
        assert self.inputs['src'], 'Failed to read features from source network'
        assert self.inputs['end'], 'Failed to read features from end network'

        self.labels = self._read_labels(kwargs['linkage'])
        assert self.labels, 'Failed to read labels'

    def calc_mrr_by_dist(self, **kwargs):
        pass
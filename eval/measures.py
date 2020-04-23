from __future__ import print_function

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#存放c.py所在的绝对路径

sys.path.append(BASE_DIR)

import numpy as np

def hamming_distance(vec1, vec2):
    res = np.where(vec1*vec2<0, np.ones(vec1.shape), np.zeros(vec1.shape))
    return np.sum(res)

def dot_distance(vec1, vec2):
    return -np.sum(vec1*vec2)

def geo_distance(vec1, vec2):
    return .5*np.sum((vec1-vec2)**2)

def tanh(mat):
    # (1-exp(-2x))/(1+exp(-2x))
    mat = np.array(mat)
    return np.tanh(mat)

def sigmoid(mat):
    # 1/(1+exp(-mat))
    mat = np.array(mat)
    return 1./(1+np.exp(-mat))

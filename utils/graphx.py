"""Graph utilities."""

# from time import time
import os
from collections import defaultdict

__author__ = "WANG Yongqing"
__email__ = "wangyongqing@ict.ac.cn"


class Graph(object):
    def __init__(self):
        self.G = defaultdict(dict)
        self.nodes = set()
        self.num_of_edges = 0
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.nodes:
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1

    def _read_adjlist(self, filename, split_token=' ', directed=True):
        if not filename or not os.path.exists(filename):
            print 'The file {} is not exist'.format(filename)

        if directed:
            def store_in_graph(elems):
                v_nd = elems[0]
                self.nodes.add(v_nd)
                for n_nd in elems[1:]:
                    self.G[v_nd][n_nd] = 1.
                    self.nodes.add(n_nd)
                    self.num_of_edges += 1
        else:
            def store_in_graph(elems):
                v_nd = elems[0]
                self.nodes.add(v_nd)
                for n_nd in elems[1:]:
                    self.G[v_nd][n_nd] = 1.
                    self.G[n_nd][v_nd] = 1.
                    self.nodes.add(n_nd)
                    self.num_of_edges += 2

        with open(filename, 'r') as fin:
            func = store_in_graph
            for ln in fin:
                elems = ln.strip().split(split_token)
                if len(elems)<2:
                    continue
                func(elems)
            fin.close()

    def read_adjlist(self, filename, split_token=' ', directed=True):
        """ Read graph from adjacency file in which the edge must be unweighted
            the format of each line: v1 n1 n2 n3 ... nk
            :param filename: the filename of input file
        """
        self._read_adjlist(filename, split_token, directed)
        self.encode_node()

    def read_edgelist(self, filename, split_token=' ', weighted=False, directed=False):
        """ Read graph from adjacency file in which the edge could be unweighted or weighted
            the format of each line: v1 n1 w(or Blank)
            :param filename: the filename of input file
        """
        if directed:
            def read_unweighted(l, split_token):
                src, dst = l.split(split_token)
                self.nodes.add(src)
                self.nodes.add(dst)
                self.G[src][dst] = 1.
                self.num_of_edges += 1

            def read_weighted(l, split_token):
                src, dst, w = l.split(split_token)
                self.nodes.add(src)
                self.nodes.add(dst)
                self.G[src][dst] = float(w)
                self.num_of_edges += 1
        else:
            def read_unweighted(l, split_token):
                src, dst = l.split(split_token)
                self.nodes.add(src)
                self.nodes.add(dst)
                self.G[src][dst] = 1.0
                self.G[dst][src] = 1.0
                self.num_of_edges += 2

            def read_weighted(l, split_token):
                src, dst, w = l.split(split_token)
                self.nodes.add(src)
                self.nodes.add(dst)
                self.G[src][dst] = float(w)
                self.G[dst][src] = float(w)
                self.num_of_edges += 2

        with open(filename, 'r') as fin:
            func = read_unweighted
            if weighted:
                func = read_weighted
            for ln in fin:
                l = l.strip()
                if not l:
                    continue
                func(l)
            fin.close()
        self.encode_node()
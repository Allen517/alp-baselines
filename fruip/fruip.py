# -*- coding:utf8 -*-

from __future__ import print_function

import numpy as np
from collections import defaultdict
from utils.graphx import *
import sys

class FRUIP(object):

	def __init__(self, graph, embed_files, linkage_file):
		self.emb = dict()
		self.look_up = dict()
		self.look_back = dict()

		self.graph = graph
		self.look_up['f'] = self.graph['f'].look_up_dict
		self.look_up['g'] = self.graph['g'].look_up_dict
		self.look_back['f'] = self.graph['f'].look_back_list
		self.look_back['g'] = self.graph['g'].look_back_list
		self.emb['f'] = self._set_emb_matrix(embed_files[0], self.look_up['f'])
		self.emb['g'] = self._set_emb_matrix(embed_files[1], self.look_up['g'])

		if linkage_file:
			self.anchors = self._read_anchors(linkage_file)

	def _set_emb_matrix(self, emb_file, look_up):
		node_size = len(look_up)
		emb = None
		with open(emb_file, 'r') as embed_handler:
			for i,ln in enumerate(embed_handler):
				if i==0:
					_, feat_num = map(int, ln.strip().split())
					emb = np.zeros((node_size, feat_num))
					continue
				elems = ln.strip().split()
				emb[look_up[elems[0]]] = map(float, elems[1:])

		return emb

	def _read_anchors(self, anchor_file):
		anchors = list()
		with open(anchor_file, 'r') as anchor_handler:
			for ln in anchor_handler:
				elems = ln.strip().split()
				anchors.append((elems[0], elems[1]))

		return anchors

	def _func_s(self, f_idx, g_idx):
		return 1./(1+np.log(1+np.linalg.norm(self.emb['f'][f_idx]-self.emb['g'][g_idx])))

	def _heaviside(self, val, thres):
		return 1 if val>=thres else 0

	def _similarity(self, f_idx, g_idx):

		max_val = -1000000
		for nd in self.graph['g'].G.nodes:
			idx = self.look_up['g'][nd]
			if idx == g_idx:
				continue
			tmp_val = self._func_s(f_idx, idx)
			if tmp_val>max_val:
				max_val = tmp_val

		sim_res = .0
		f_nd = self.look_back['f'][f_idx]
		g_nd = self.look_back['g'][g_idx]
		f_nd_neighbors = list(self.graph['f'].G.neighbors(f_nd))
		g_nd_neighbors = list(self.graph['g'].G.neighbors(g_nd))
		for t_f in f_nd_neighbors:
			f_nb_idx = self.look_up['f'][t_f]
			for t_g in g_nd_neighbors:
				g_nb_idx = self.look_up['g'][t_g]
				s_val = self._func_s(f_nb_idx, g_nb_idx)
				if self._heaviside(s_val, max_val):
					sim_res += s_val

		return sim_res*np.log(min(len(f_nd_neighbors), len(g_nd_neighbors))+1e-8)

	def _h_similarity(self, f_idx, g_idx, thres):
		'''
		Heuristic method
		'''
		sim_res = .0
		f_nd = self.look_back['f'][f_idx]
		g_nd = self.look_back['g'][g_idx]
		f_nd_neighbors = list(self.graph['f'].G.neighbors(f_nd))
		g_nd_neighbors = list(self.graph['g'].G.neighbors(g_nd))
		for t_f in f_nd_neighbors:
			f_nb_idx = self.look_up['f'][t_f]
			for t_g in g_nd_neighbors:
				g_nb_idx = self.look_up['g'][t_g]
				s_val = self._func_s(f_nb_idx, g_nb_idx)
				if self._heaviside(s_val, thres):
					sim_res += s_val

		return sim_res*np.log(min(len(f_nd_neighbors), len(g_nd_neighbors))+1e-8)

	def _alignment(self, thres):
		size_f = len(self.look_up['f'])
		size_g = len(self.look_up['g'])

		align_res = dict()
		used_idx = np.zeros(size_g, int)
		for i in range(size_f):
			if np.sum(np.abs(self.emb['f'][i]))==.0:
				continue
			max_val = -1000000
			sec_max_val = -1000000
			max_val_idx = -1
			for j in range(size_g):
				if np.sum(np.abs(self.emb['g'][j]))==.0:
					continue
				tmp_res = self._similarity(i, j)
				if tmp_res>max_val and not used_idx[j]:
					sec_max_val = max_val
					max_val = tmp_res
					max_val_idx = j
			if (max_val-sec_max_val)>thres:
				align_res[self.look_back['f'][i]] = self.look_back['g'][max_val_idx]
				used_idx[max_val_idx] = 1
			if (i+1)%100==0:
				print('Finish {} alignments'.format(i+1))

		return align_res

	def _h_alignment(self, thres):
		'''
		Heuristic method
		'''
		size_f = len(self.look_up['f'])
		size_g = len(self.look_up['g'])

		align_res = dict()
		used_idx = np.zeros(size_g, int)
		for i in range(size_f):
			if np.sum(np.abs(self.emb['f'][i]))==.0:
				continue
			max_val = -1000000
			sec_max_val = -1000000
			max_val_idx = -1
			for j in range(size_g):
				if np.sum(np.abs(self.emb['g'][j]))==.0:
					continue
				tmp_res = self._h_similarity(i, j)
				if tmp_res>max_val and not used_idx[j]:
					sec_max_val = max_val
					max_val = tmp_res
					max_val_idx = j
			if (max_val-sec_max_val)>thres:
				align_res[self.look_back['f'][i]] = self.look_back['g'][max_val_idx]
				used_idx[max_val_idx] = 1
			if (i+1)%100==0:
				print('Finish {} alignments'.format(i+1))

		return align_res

	def main_proc(self, thres):
		self.align_res = self._alignment(thres)

	def save_model(self, output):
		with open(output, 'w') as fout:
			for k,v in self.align_res.iteritems():
				fout.write('{} {}\n'.format(k,v))

if __name__=='__main__':

	if len(sys.argv)<8:
		print('please input [graph f, embedding of graph f, graph g, embedding of graph g, linkage, threshold, output file]')
		sys.exit(1)

	graph = defaultdict(Graph)
	graph['f'].read_adjlist(filename=sys.argv[1])
	graph['g'].read_adjlist(filename=sys.argv[3])

	embed_files = [sys.argv[2], sys.argv[4]]
	linkage_file = sys.argv[5]
	thres = float(sys.argv[6])

	fruip = FRUIP(graph, embed_files, linkage_file)
	fruip.main_proc(thres)
	fruip.save_models(sys.argv[7])


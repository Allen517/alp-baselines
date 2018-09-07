import os
from collections import defaultdict
import numpy as np
import random

from sklearn import svm
from sklearn.externals import joblib
from utils.utils import load_train_valid_labels
from utils.LogHandler import LogHandler

class _MNA(object):

	def __init__(self, graph, anchorfile, valid_prop, neg_ratio, log_file):
		if os.path.exists('log/'+log_file+'.log'):
			os.remove('log/'+log_file+'.log')
		self.logger = LogHandler(log_file)

		if not isinstance(graph, dict):
			self.logger.error('The graph must contain src and target graphs.')
			return

		self.L = load_train_valid_labels(anchorfile, valid_prop)
		self.graph = graph
		self.look_up = dict()
		self.look_up['f'] = self.graph['f'].look_up_dict
		self.look_up['g'] = self.graph['g'].look_up_dict
		self.look_back = dict()
		self.look_back['f'] = self.graph['f'].look_back_list
		self.look_back['g'] = self.graph['g'].look_back_list

		self.neg_ratio = neg_ratio
		self.batch_size = 1024

		self.clf = svm.SVC()

	def __get_pair_features(self, src_nds, target_nds):
		pair_features = list()
		if len(src_nds)!=len(target_nds):
			self.logger.warn('The size of sampling in processing __get_pair_features is not equal.')
			yield pair_features
		for i in range(len(src_nds)):
			src_nd, target_nd = src_nds[i],target_nds[i]

			if not src_nd in self.graph['f'].G or not target_nd in self.graph['g'].G:
				continue

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
						AA_measure += 1./np.log((len(self.graph['f'].G[sna])+len(self.graph['g'].G[self.L['f2g']['train'][sna][k]]))/2.)
			jaccard = cnt_common_neighbors/(len(self.graph['f'].G[src_nd])\
											+len(self.graph['g'].G[target_nd])-cnt_common_neighbors+1e-6)

			yield [cnt_common_neighbors, jaccard, AA_measure]

	def __batch_iter(self, lbs, batch_size, neg_ratio, lookup_src, lookup_obj, src_lb_tag, obj_lb_tag):
		train_lb_src2obj = lbs['{}2{}'.format(src_lb_tag,obj_lb_tag)]['train']
		train_lb_obj2src = lbs['{}2{}'.format(obj_lb_tag,src_lb_tag)]['train']
		train_size = len(train_lb_src2obj)
		start_index = 0
		end_index = min(start_index+batch_size, train_size)

		src_lb_keys = train_lb_src2obj.keys()
		obj_lb_keys = train_lb_obj2src.keys()
		shuffle_indices = np.random.permutation(np.arange(train_size))
		while start_index < end_index:
			pos_src = list()
			pos_obj = list()
			neg_src = list()
			neg_obj = list()
			for i in range(start_index,end_index):
				idx = shuffle_indices[i]
				src_lb = src_lb_keys[idx]
				obj_lbs = train_lb_src2obj[src_lb]
				for obj_lb in obj_lbs:
					cur_neg_src = list()
					cur_neg_obj = list()
					for k in range(neg_ratio):
						rand_obj_lb = None
						while not rand_obj_lb or rand_obj_lb in cur_neg_obj or rand_obj_lb in obj_lbs:
							rand_obj_lb_idx = random.randint(0, len(obj_lb_keys)-1)
							rand_obj_lb = obj_lb_keys[rand_obj_lb_idx]
						cur_neg_src.append(src_lb)
						cur_neg_obj.append(rand_obj_lb)
					pos_src.append(src_lb)
					pos_obj.append(obj_lb)
					neg_src.append(cur_neg_src)
					neg_obj.append(cur_neg_obj)

			start_index = end_index
			end_index = min(start_index+batch_size, train_size)

			yield pos_src,pos_obj,neg_src,neg_obj

	def train(self):

		batches_f2g = list(self.__batch_iter(self.L, self.batch_size, self.neg_ratio\
								, self.look_up['f'], self.look_up['g'], 'f', 'g'))
		n_batches = len(batches_f2g)

		X = list()
		Y = list()
		for i in range(n_batches):
			pos_src_f2g,pos_obj_f2g,neg_src_f2g,neg_obj_f2g = batches_f2g[i]
			if not len(pos_src_f2g)==len(pos_obj_f2g) and not len(neg_src_f2g)==len(neg_obj_f2g):
				self.logger.info('The input label file goes wrong as the file format.')
				continue
			pos_features = list(self.__get_pair_features(pos_src_f2g, pos_obj_f2g))
			X.extend(pos_features)
			Y.extend([1 for m in range(len(pos_features))])

			for k in range(self.neg_ratio):
				neg_features = list(self.__get_pair_features(neg_src_f2g[k], neg_obj_f2g[k]))
				X.extend(neg_features)
				Y.extend([-1 for m in range(len(neg_features))])

			self.logger.info('Training Model...')
			self.clf.fit(X,Y)
			self.logger.info('Complete Training process...')

class MNA(object):

	def __init__(self, graph, anchorfile, valid_prop, neg_ratio, log_file):
		# training
		self.model = _MNA(graph, anchorfile, valid_prop, neg_ratio, log_file)
		self.model.train()

	def save_model(self, modelfile):
		modelfile = modelfile+'.pkl'
		joblib.dump(self.model.clf, modelfile)
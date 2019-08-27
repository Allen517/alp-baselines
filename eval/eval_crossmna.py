from __future__ import print_function

from eval.eval import *
import re

class Eval_CrossMNA(Eval):

    def __init__(self):
        super(Eval_CrossMNA, self).__init__()

    # def _read_model(self, filepath):
    #     param_W_path = filepath+'.W'
    #     param_layer_path = filepath+'.layer'
    #     print('reading model %s and %s'%(param_W_path, param_layer_path))
    #     assert os.path.exists(param_W_path), 'Model file does not exist: %s'%param_W_path
    #     assert os.path.exists(param_layer_path), 'Model file does not exist: %s'%param_layer_path
        
    #     model = defaultdict(list)
    #     for filepath in [param_W_path, param_layer_path]:
    #         with open(filepath, 'r') as f_handler:
    #             cur_key = ''
    #             for ln in f_handler:
    #                 ln = ln.strip()
    #                 if 'W' in ln or 'layer' in ln:
    #                     cur_key = ln
    #                     continue
    #                 model[cur_key].append(list(map(float, ln.split())))
    #     return model

    def _read_inputs(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Inputs file does not exist: %s'%filepath

        delimiter = ','
        inputs = dict()
        with open(filepath, 'r') as f_handler:
            for ln in f_handler:
                ln = ln.strip()
                if ln:
                    elems = ln.split(delimiter)
                    if len(elems)==2:
                        continue
                    inputs[elems[0]] = list(map(float, elems[1:]))
        return inputs

    def _read_labels(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Label file does not exist: %s'%filepath

        delimiter = ','
        lbs = dict()
        with open(filepath, 'r') as anchor_handler:
            for ln in anchor_handler:
                elems = ln.strip().split(delimiter)
                for i in range(len(elems)):
                    elems[i] = '{}-{}'.format(i,elems[i])
                for k in range(len(elems)):
                    lbs[elems[k]] = elems[:k]+elems[k+1:]
        return lbs

    def _init_eval(self, **kwargs):
        allows_keys = {'node', 'train_linkage', 'linkage'}
        for k in kwargs.keys():
            assert k in allows_keys, 'Invalid file inputs: '+k

        print('processing {}'.format(kwargs['node']))
        assert os.path.exists(kwargs['node']), 'Files not found: %s'%(kwargs['node'])

        self.t_labels = self._read_labels(kwargs['train_linkage'])
        assert self.t_labels, 'Failed to read train labels'

        self.inputs = self._read_inputs(kwargs['node'])
        assert self.inputs, 'Failed to read node embeddings'

        self.labels = self._read_labels(kwargs['linkage'])
        assert self.labels, 'Failed to read labels'

    def _get_inputs(self, nd):
        if nd in self.inputs:
            return np.array(self.inputs[nd])
        elif nd in self.t_labels:
            return np.array(self.inputs[self.t_labels[nd][0]])
        return []

    def calc_mrr_by_dist(self, **kwargs):
        allows_keys = {'candidate_num', 'dist_calc','out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        # self.model = self._read_model(kwargs['model'])
        # assert self.model, 'Failed to read model'

        mrr_list = tuple()

        with open(kwargs['out_file'], 'w') as fout:
            cnt = 0
            wrt_lns = ''
            for from_nd, to_nds in self.labels.items():
                # print('0')
                # print(from_nd)
                from_inputs = self._get_inputs(from_nd)
                if len(from_inputs)<1:
                    continue
                for to_nd in to_nds:
                    to_inputs = self._get_inputs(to_nd)
                    if len(to_inputs)<1:
                        continue

                    model_res = {
                        'from': from_inputs,
                        'to': to_inputs,
                        'rand': None
                    }

                    # print(model_res['from'], model_res['to'])
                    anchor_dist = kwargs['dist_calc'](model_res['from'], model_res['to'])

                    to_keys = list(self.inputs.keys())
                    to_size = len(to_keys)
                    rand_nds = list()
                    for k in range(kwargs['candidate_num']):
                        rand_to_nd = to_keys[np.random.randint(0, to_size)]
                        while rand_to_nd in rand_nds or rand_to_nd in to_nds \
                                or (rand_to_nd not in self.inputs or rand_to_nd not in self.t_labels):
                            rand_to_nd = to_keys[np.random.randint(0, to_size)]
                        rand_nds.append(rand_to_nd)

                    pred_pos = 1
                    for k in range(len(rand_nds)):
                        rand_to_nd = rand_nds[k]
                        rand_inputs = self._get_inputs(rand_to_nd)
                        model_res['rand'] = rand_inputs
                        noise_dist = kwargs['dist_calc'](model_res['from'], model_res['rand'])
                        if anchor_dist>=noise_dist:
                            pred_pos += 1
                        cur_mrr = 1./pred_pos
                        mrr_list += cur_mrr,
                        cnt += 1
                        wrt_lns += '(%s,%s):%f\n'%(from_nd, to_nd, cur_mrr)
                        if not cnt%100:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
            if cnt%100:
                fout.write(wrt_lns)
            fout.write('mean_mrr:{}, var:{}\n'
                    .format(np.mean(mrr_list), np.var(mrr_list)))

    def eval_classes(self, **kwargs):
        allows_keys = {'candidate_num', 'dist_calc','out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid class calculation parameter: '+kw

        # self.model = self._read_model(kwargs['model'])
        # assert self.model, 'Failed to read model'

        with open(kwargs['out_file'], 'w') as fout:
            cnt = 0
            wrt_lns = ''
            for from_nd, to_nds in self.labels.items():
                # print('0')
                # print(from_nd)
                from_inputs = self._get_inputs(from_nd)
                if len(from_inputs)<1:
                    continue
                for to_nd in to_nds:
                    to_inputs = self._get_inputs(to_nd)
                    if len(to_inputs)<1:
                        continue

                    model_res = {
                        'from': from_inputs,
                        'to': to_inputs,
                        'rand': None
                    }

                    # print(model_res['from'], model_res['to'])
                    anchor_dist = kwargs['dist_calc'](model_res['from'], model_res['to'])
                    wrt_lns += '(%s,%s),%f,%d\n'%(from_nd,to_nd,anchor_dist,1)

                    to_keys = list(self.inputs.keys())
                    to_size = len(to_keys)
                    rand_nds = list()
                    for k in range(kwargs['candidate_num']):
                        rand_to_nd = to_keys[np.random.randint(0, to_size)]
                        rand_inputs = self._get_inputs(rand_to_nd)
                        while rand_to_nd in rand_nds or rand_to_nd in to_nds \
                                or (rand_to_nd not in self.inputs or rand_to_nd not in self.t_labels):
                            rand_to_nd = to_keys[np.random.randint(0, to_size)]
                        rand_nds.append(rand_to_nd)

                    for k in range(len(rand_nds)):
                        rand_to_nd = rand_nds[k]
                        rand_inputs = self._get_inputs(rand_to_nd)
                        model_res['rand'] = rand_inputs
                        noise_dist = kwargs['dist_calc'](model_res['from'], model_res['rand'])
                        wrt_lns += '(%s,%s),%f,%d\n'%(from_nd,rand_to_nd,noise_dist,0)
                        cnt += 1
                        if not cnt%100:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
            if cnt%100:
                fout.write(wrt_lns)
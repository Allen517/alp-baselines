from __future__ import print_function

import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#存放c.py所在的绝对路径

sys.path.append(BASE_DIR)

from eval.eval import *

class Eval_PALE(Eval):

    def __init__(self, model_type):
        super(Eval_PALE, self).__init__()
        assert model_type in {'lin', 'mlp'}, 'Model type must be lin/mlp'
        self.model_type = model_type

    def _read_model(self, filepath):
        print('reading model %s'%(filepath))
        assert os.path.exists(filepath), 'Model file does not exist: %s'%filepath
        
        model = defaultdict(list)
        with open(filepath, 'r') as f_handler:
            cur_key = ''
            for ln in f_handler:
                ln = ln.strip()
                if 'h' in ln or 'b' in ln or 'out' in ln:
                    cur_key = ln
                    continue
                model[cur_key].append(list(map(float, ln.split())))
        return model

    def _calc_model_lin_res(self, **kwargs):
        allows_keys = {'inputs', 'n_layer'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        inputs = np.array(kwargs['inputs'])

        out = tanh(
                np.dot(inputs,np.array(self.model['out']))
                    +np.array(self.model['b_out']).reshape(-1)
                )

        return out

    def _calc_model_mlp_res(self, **kwargs):
        allows_keys = {'inputs', 'n_layer'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid model calculation parameter: '+kw

        inputs = np.array(kwargs['inputs'])
        n_layer = kwargs['n_layer']

        layer = sigmoid(np.dot(np.array(inputs),np.array(self.model['h0']))+np.array(self.model['b0']).reshape(1,-1))
        for i in range(1, n_layer):
            layer = sigmoid(np.dot(layer,np.array(self.model['h{}'.format(i)]))
                                +np.array(self.model['b{}'.format(i)]).reshape(1,-1))
        out = tanh(np.dot(layer,np.array(self.model['out']))+np.array(self.model['b_out']).reshape(1,-1))

        return out

    def _calc_model_res(self, **kwargs):
        if self.model_type=='lin':
            return self._calc_model_lin_res(**kwargs)

        return self._calc_model_mlp_res(**kwargs)

    def calc_mrr_by_dist(self, **kwargs):
        allows_keys = {'model', 'n_layer', 'candidate_num', 'dist_calc', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        self.model = self._read_model(kwargs['model'])
        assert self.model, 'Failed to read model'

        mrr_list = tuple()

        with open(kwargs['out_file'], 'w') as fout:
            tps = ['src', 'end']
            tps_len = len(tps)-1
            for tp_id in range(len(tps)-1):
                cnt = 0
                wrt_lns = ''
                lb_tp = '%s2%s'%(tps[tp_id], tps[tps_len-tp_id])
                fout.write('%s\n'%lb_tp)
                to_keys = list(self.inputs[tps[tps_len-tp_id]].keys())
                to_size = len(to_keys)
                fout.write('Overall: %d\n'%len(self.labels[lb_tp].keys()))
                for nd_from, nds_to in self.labels[lb_tp].items():
                    for nd_to in nds_to:
                        if nd_from not in self.inputs[tps[tp_id]] or nd_to not in self.inputs[tps[tps_len-tp_id]]:
                            continue

                        model_res = {
                            'from': self._calc_model_res(inputs=[self.inputs[tps[tp_id]][nd_from]]
                                                            , n_layer=kwargs['n_layer']),
                            'to': self.inputs[tps[tps_len-tp_id]][nd_to],
                            'rand': None
                        }

                        anchor_dist = kwargs['dist_calc'](model_res['from'][0], model_res['to'][0])

                        pred_pos = 1
                        rand_nds = set()
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                                rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            rand_nds.add(rand_nd_to)
                            model_res['rand'] = self.inputs[tps[tps_len-tp_id]][rand_nd_to]
                            noise_dist = kwargs['dist_calc'](model_res['from'][0], model_res['rand'][0])
                            if anchor_dist>=noise_dist:
                                pred_pos += 1
                        cur_mrr = 1./pred_pos
                        mrr_list += cur_mrr,
                        cnt += 1
                        wrt_lns += '(%s,%s):%f\n'%(nd_from, nd_to, cur_mrr)
                        if not cnt%10:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
                if cnt%10:
                    fout.write(wrt_lns)
                fout.write('mean_mrr:{}, var:{}\n'
                        .format(np.mean(mrr_list), np.var(mrr_list)))

    def eval_classes(self, **kwargs):
        allows_keys = {'model', 'n_layer', 'candidate_num', 'dist_calc', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        self.model = self._read_model(kwargs['model'])
        assert self.model, 'Failed to read model'

        with open(kwargs['out_file'], 'w') as fout:
            tps = ['src', 'end']
            tps_len = len(tps)-1
            for tp_id in range(len(tps)-1):
                cnt = 0
                wrt_lns = ''
                lb_tp = '%s2%s'%(tps[tp_id], tps[tps_len-tp_id])
                fout.write('%s\n'%lb_tp)
                to_keys = list(self.inputs[tps[tps_len-tp_id]].keys())
                to_size = len(to_keys)
                fout.write('Overall: %d\n'%len(self.labels[lb_tp].keys()))
                for nd_from, nds_to in self.labels[lb_tp].items():
                    for nd_to in nds_to:
                        if nd_from not in self.inputs[tps[tp_id]]:
                            continue

                        model_res = {
                            'from': self._calc_model_res(inputs=[self.inputs[tps[tp_id]][nd_from]]
                                                            , n_layer=kwargs['n_layer']),
                            'to': self.inputs[tps[tps_len-tp_id]][nd_to],
                            'rand': None
                        }

                        rand_nds = set()
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                                rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            rand_nds.add(rand_nd_to)

                        anchor_dist = kwargs['dist_calc'](model_res['from'][0], model_res['to'][0])
                        wrt_lns += '(%s,%s),%f,%d\n'%(nd_from,nd_to,anchor_dist,1)

                        rand_nds = list(rand_nds)
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = rand_nds[k]
                            model_res['rand'] = self.inputs[tps[tps_len-tp_id]][rand_nd_to]
                            noise_dist = kwargs['dist_calc'](model_res['from'][0], model_res['rand'][0])
                            wrt_lns += '(%s,%s),%f,%d\n'%(nd_from,rand_nd_to,noise_dist,0)

                        cnt += 1
                        if not cnt%10:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
                if cnt%10:
                    fout.write(wrt_lns)

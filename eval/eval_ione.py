from __future__ import print_function

from eval.eval import *

class Eval_IONE(Eval):

    def __init__(self):
        super(Eval_IONE, self).__init__()

    def _read_model(self, filepath):
        return None

    def _read_inputs(self, filepath):
        print('reading inputs %s'%(filepath))
        assert os.path.exists(filepath), 'Inputs file does not exist: %s'%filepath

        inputs = dict()
        with open(filepath, 'r') as f_handler:
            for ln in f_handler:
                ln = ln.strip()
                if ln:
                    elems = ln.split()
                    if len(elems)==2:
                        continue
                    inputs[elems[0]] = list(map(float, elems[1:]))
        return inputs

    def calc_mrr_by_dist(self, **kwargs):
        allows_keys = {'model', 'candidate_num', 'dist_calc','out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        mrr_list = tuple()

        with open(kwargs['out_file'], 'w') as fout:
            tps = ['src', 'end']
            tps_len = len(tps)-1
            for tp_id in range(len(tps)):
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
                            'from': np.array(self.inputs[tps[tp_id]][nd_from]),
                            'to': np.array(self.inputs[tps[tps_len-tp_id]][nd_to]),
                            'rand': None
                        }

                        anchor_dist = kwargs['dist_calc'](model_res['from'], model_res['to'])

                        pred_pos = 1
                        rand_nds = set()
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                                rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            rand_nds.add(rand_nd_to)
                            model_res['rand'] = np.array(self.inputs[tps[tps_len-tp_id]][rand_nd_to])
                            noise_dist = kwargs['dist_calc'](model_res['from'], model_res['rand'])
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
        allows_keys = {'model', 'candidate_num', 'dist_calc', 'out_file'}
        for kw in kwargs.keys():
            assert kw in allows_keys, 'Invalid mrr calculation parameter: '+kw

        with open(kwargs['out_file'], 'w') as fout:
            tps = ['src', 'end']
            tps_len = len(tps)-1
            for tp_id in range(len(tps)):
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
                            'from': np.array(self.inputs[tps[tp_id]][nd_from]),
                            'to': np.array(self.inputs[tps[tps_len-tp_id]][nd_to]),
                            'rand': None
                        }

                        rand_nds = set()
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            while rand_nd_to in rand_nds or rand_nd_to in nds_to:
                                rand_nd_to = to_keys[np.random.randint(0, to_size)]
                            rand_nds.add(rand_nd_to)

                        anchor_dist = kwargs['dist_calc'](model_res['from'], model_res['to'])
                        wrt_lns += '(%s,%s),%f,%d\n'%(nd_from,nd_to,anchor_dist,1)

                        rand_nds = list(rand_nds)
                        for k in range(kwargs['candidate_num']):
                            rand_nd_to = rand_nds[k]
                            model_res['rand'] = np.array(self.inputs[tps[tps_len-tp_id]][rand_nd_to])
                            noise_dist = kwargs['dist_calc'](model_res['from'], model_res['rand'])
                            wrt_lns += '(%s,%s),%f,%d\n'%(nd_from,rand_nd_to,noise_dist,0)

                        cnt += 1
                        if not cnt%10:
                            fout.write(wrt_lns)
                            print('Processing %d records'%cnt)
                            wrt_lns = ''
                if cnt%10:
                    fout.write(wrt_lns)

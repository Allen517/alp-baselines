# -*- coding=UTF-8 -*-\n
from eval.eval_mna import Eval_MNA
from eval.measures import *
import ast

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-net-src', required=True, default=None
                        , help='features from source network')
    parser.add_argument('-net-end', required=True, default=None
                        , help='features from end network')
    parser.add_argument('-use-net', required=True, default=True, type=str2bool
                        , help='If use structural information in MNA')
    parser.add_argument('-feat-src', required=False, default=None
                        , help='features from source network')
    parser.add_argument('-feat-end', required=False, default=None
                        , help='features from end network')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    parser.add_argument('-model', required=True
                        , help='Model file')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='number of candidates')
    parser.add_argument('-eval-type', default='mrr'
                        , help='mrr/ca/cls (MRR/Candidate selection/Classification)')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    # args.use_net=False
    print(args)
    eval_model = Eval_MNA()
    eval_model._init_eval(net_src=args.net_src
                        , net_end=args.net_end
                        , feat_src=args.feat_src
                        , feat_end=args.feat_end
                        , linkage=args.linkage
                        , use_net=args.use_net
                )
    if args.eval_type=='mrr':
        eval_model.calc_mrr_by_dist(model=args.model, candidate_num=args.n_cands, out_file=args.output)
    if args.eval_type=='cls':
        eval_model.eval_classes(model=args.model, candidate_num=args.n_cands, out_file=args.output)
    
if __name__=='__main__':
    main(parse_args())




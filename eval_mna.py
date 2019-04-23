# -*- coding=UTF-8 -*-\n
from eval.eval_mna import Eval_MNA
from eval.measures import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-net-src', required=True
                        , help='features from source network')
    parser.add_argument('-net-end', required=True
                        , help='features from end network')
    parser.add_argument('-feat-src', required=True
                        , help='features from source network')
    parser.add_argument('-feat-end', required=True
                        , help='features from end network')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    parser.add_argument('-model', required=True
                        , help='Model file')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='number of candidates')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    eval_model = Eval_MNA()
    eval_model._init_eval(net_src=args.net_src
                        , net_end=args.net_end
                        , linkage=args.linkage
                )
    eval_model.calc_mrr_by_dist(model=args.model, candidate_num=args.n_cands, out_file=args.output)
    
if __name__=='__main__':
    main(parse_args())




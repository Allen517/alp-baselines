# -*- coding=UTF-8 -*-\n
from eval.eval_crossmna import Eval_CrossMNA
from eval.measures import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-node', required=True
                        , help='node embeddings from crossmna')
    parser.add_argument('-train-linkage', required=True
                        , help='linkage for training')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    # parser.add_argument('-model', required=True
    #                     , help='model from crossmna (ignore postfix)')
    parser.add_argument('-eval-type', default='mrr'
                        , help='mrr/ca/cls (MRR/Candidate selection/Classification)')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='number of candidates')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    eval_model = Eval_CrossMNA()
    eval_model._init_eval(node=args.node,
                            train_linkage=args.train_linkage,
                            linkage=args.linkage
                )
    if args.eval_type=='mrr':
        eval_model.calc_mrr_by_dist(candidate_num=args.n_cands
                                , dist_calc=geo_distance, out_file=args.output)
    if args.eval_type=='cls':
        eval_model.eval_classes(candidate_num=args.n_cands
                                , dist_calc=geo_distance, out_file=args.output)

if __name__=='__main__':
    main(parse_args())

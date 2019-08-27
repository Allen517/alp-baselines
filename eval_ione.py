# -*- coding=UTF-8 -*-\n
from eval.eval_ione import Eval_IONE
from eval.measures import *

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')
    parser.add_argument('-feat-src', required=True
                        , help='features from source network')
    parser.add_argument('-feat-end', required=True
                        , help='features from end network')
    parser.add_argument('-linkage', required=True
                        , help='linkage for test')
    parser.add_argument('-eval-type', default='mrr'
                        , help='mrr/ca/cls (MRR/Candidate selection/Classification)')
    parser.add_argument('-n-cands', default=9, type=int
                        , help='number of candidates')
    parser.add_argument('-output', required=True
                        , help='Output file')

    return parser.parse_args()

def main(args):
    eval_model = Eval_IONE()
    eval_model._init_eval(feat_src=args.feat_src,
                            feat_end=args.feat_end,
                            linkage=args.linkage
                )
    if args.eval_type=='mrr':
        eval_model.calc_mrr_by_dist(candidate_num=args.n_cands, dist_calc=geo_distance, out_file=args.output)
    if args.eval_type=='cls':
        eval_model.eval_classes(candidate_num=args.n_cands, dist_calc=geo_distance, out_file=args.output)

if __name__=='__main__':
    main(parse_args())




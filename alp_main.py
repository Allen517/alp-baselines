from __future__ import print_function

import numpy as np
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pale.pale import *
from mna.MNA import *
from utils.graphx import *
from utils.graph import *
from utils.utils import *
from fruip.fruip import *
from final.final import *
from crossmna.crossmna import *
import time
import os,sys

from utils.LogHandler import LogHandler

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--gpu-id', required=False,
                        help='Set env CUDA_VISIBLE_DEVICES', default="0")
    parser.add_argument('--embeddings', nargs='+', required=False
                        , help='Embeddings of networks (used in PALE, FRUI-P) or Inputs of networks (used in MNA)')
    # parser.add_argument('--embedding1', required=False,
    #                     help='Embeddings of source  network (used in PALE, FRUI-P)')
    # parser.add_argument('--embedding2', required=False,
    #                     help='Embeddings of target network (used in PALE, FRUI-P)')
    parser.add_argument('--graphs', nargs='+', required=False,
                        help='Network (used in MNA, FRUI-P, FINAL, CrossMNA)')
    # parser.add_argument('--graph2', required=False,
    #                     help='Target network (used in MNA, FRUI-P, FINAL)')
    parser.add_argument('--graph-sizes', nargs='+', required=False, type=int,
                        help='Size of networks (used in FINAL)')
    # parser.add_argument('--graph-size2', required=False, type=int,
    #                     help='Size of target network (used in FINAL)')
    parser.add_argument('--nd-rep-size', required=False, type=int,
                        help='Size of Node Representation (used in CrossMNA)')
    parser.add_argument('--layer-rep-size', required=False, type=int,
                        help='Size of Layer Representation (used in CrossMNA)')
    parser.add_argument('--type-model', default='mlp', choices=['mlp', 'lin'],
                        help='Model type (used in PALE)')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Graph format for reading')
    parser.add_argument('--identity-linkage', required=False,
                        help='File of anchor links')
    parser.add_argument('--output', required=True,
                        help='Output file')
    parser.add_argument('--log-file', default='ALP',
                        help='logging file')
    parser.add_argument('--is-valid', default=False, type=bool,
                        help='If use validation in training')
    parser.add_argument('--use-net', default=True, type=bool,
                        help='If use structural information in MNA (used in MNA)')
    parser.add_argument('--early-stop', default=False, type=bool,
                        help='Early stop')
    parser.add_argument('--lr', default=.01, type=float,
                        help='Learning rate (used in PALE, CrossMNA)')
    parser.add_argument('--table-size', default=100000, type=int,
                        help='Size of sampling table (used in CrossMNA)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size (used in PALE)')
    parser.add_argument('--input-size', default=256, type=int,
                        help='Number of embedding (used in PALE)')
    parser.add_argument('--hidden-size', default=32, type=int,
                        help='Number of embedding (used in PALE)')
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of layers (used in PALE)')
    parser.add_argument('--saving-step', default=1, type=int,
                        help='The training epochs (used in PALE)')
    parser.add_argument('--epochs', default=21, type=int,
                        help='The training epochs (used in PALE, CrossMNA)')
    parser.add_argument('--method', required=True, choices=['pale', 'final', 'mna', 'fruip', 'crossmna'],
                        help='The learning methods')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='The negative ratio (used in PALE, MNA, CrossMNA)')
    parser.add_argument('--threshold', default=0.1, type=float,
                        help='threshold (used in FRUIP)')
    parser.add_argument('--alpha', default=0.1, type=float,
                        help='alpha (used in FINAL)')
    parser.add_argument('--tol', default=0.1, type=float,
                        help='tolerance (used in FINAL)')
    parser.add_argument('--device', default=':/gpu:0',
                        help='Running device (You can choose :/cpu:* or :/gpu:* to run your procedure)')
    args = parser.parse_args()
    return args

def main(args):
    t1 = time.time()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # args.use_net=False
    logger = LogHandler('RUN.'+time.strftime('%Y-%m-%d',time.localtime(time.time())))
    logger.info(args)

    SAVING_STEP = args.saving_step
    MAX_EPOCHS = args.epochs
    if args.method == 'pale':
        model = PALE(learning_rate=args.lr, batch_size=args.batch_size
                        , n_input=args.input_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=args.embeddings+args.identity_linkage
                        , type_model = args.type_model, is_valid=args.is_valid
                        , log_file=args.log_file, device=args.device)
        losses = np.zeros(MAX_EPOCHS)
        val_scrs = np.zeros(MAX_EPOCHS)
        best_scr = .0
        best_epoch = 0
        thres = 100
        for i in range(1,MAX_EPOCHS+1):
            losses[i-1], val_scrs[i-1] = model.train_one_epoch()
            if i>0 and i%SAVING_STEP==0:
                loss_mean = np.mean(losses[i-SAVING_STEP:i])
                scr_mean = np.mean(val_scrs[i-SAVING_STEP:i])
                logger.info('loss in last {} epoches: {}, validation in last {} epoches: {}'
                    .format(SAVING_STEP, loss_mean, SAVING_STEP, scr_mean))
                if scr_mean>best_scr:
                    best_scr = scr_mean
                    best_epoch = i
                    model.save_models(args.output)
                if args.early_stop and i>=thres*SAVING_STEP:
                    cnt = 0
                    for k in range(thres-1,-1,-1):
                        cur_val = np.mean(val_scrs[i-(k+1)*SAVING_STEP:i-k*SAVING_STEP])
                        if cur_val<best_scr:
                            cnt += 1
                    if cnt==thres and (i-best_epoch)>=thres*SAVING_STEP:
                        logger.info('*********early stop*********')
                        logger.info('The best epoch: {}\nThe validation score: {}'.format(best_epoch, best_scr))
                        break
    if args.method == 'mna' or args.method == 'fruip':
        graph = defaultdict(GraphX)
        print("Loading graph...")
        if len(args.graphs)!=2:
            logger.error('#####The input graphs must be pairwise!#####')
            sys.exit(1)
        if args.graph_format=='adjlist':
            if args.graphs[0]:
                graph['f'].read_adjlist(filename=args.graphs[0])
            if args.graphs[1]:
                graph['g'].read_adjlist(filename=args.graphs[1])
        if args.graph_format=='edgelist':
            if args.graphs[0]:
                graph['f'].read_edgelist(filename=args.graphs[0])
            if args.graphs[1]:
                graph['g'].read_edgelist(filename=args.graphs[1])

        if args.method == 'mna':
            model = MNA(graph=graph, attr_file=args.embeddings, anchorfile=args.identity_linkage, valid_prop=1.\
                        , use_net=args.use_net, neg_ratio=args.neg_ratio, log_file=args.log_file)
        if args.method == 'fruip':
            model = FRUIP(graph=graph, embed_files=args.embeddings, linkage_file=args.identity_linkage)
            model.main_proc(args.threshold)
    if args.method == 'final':
        main_proc(graph_files=args.graphs, graph_sizes=args.graph_sizes
                            , linkage_file=args.identity_linkage, alpha=args.alpha
                            , epoch=args.epochs, tol=args.tol, graph_format=args.graph_format
                            , output_file=args.output)
    if args.method == 'crossmna':
        num_graphs = len(args.graphs)
        layer_graphs = [Graph() for i in range(num_graphs)]
        for k in range(num_graphs):
            graph_path = args.graphs[k]
            format_graph_path = '{}.crossmna'.format(graph_path)
            format_crossmna_graph(graph_path, format_graph_path, k)
            if args.graph_format=='adjlist':
                layer_graphs[k].read_adjlist(filename=format_graph_path)
            if args.graph_format=='edgelist':
                layer_graphs[k].read_edgelist(filename=format_graph_path)
        model = CROSSMNA(layer_graphs=layer_graphs, anchor_file=args.identity_linkage, lr=args.lr
                        , batch_size=args.batch_size, nd_rep_size=args.nd_rep_size
                        , layer_rep_size=args.layer_rep_size
                        , epoch=args.epochs, negative_ratio=args.neg_ratio
                        , table_size=args.table_size, outfile=args.output
                        , log_file=args.log_file)
    if args.method in ['mna', 'fruip', 'pale']:
        model.save_model(args.output)
    t2 = time.time()
    print('time cost:',t2-t1)

if __name__ == "__main__":
    main(parse_args())

import numpy as np
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pale.pale import *
from mna.MNA import *
from utils.graph import *
from fruip.fruip import *
from final.final import *
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--embedding1', required=False,
                        help='Embeddings of source  network (used in PALE, FRUI-P)')
    parser.add_argument('--embedding2', required=False,
                        help='Embeddings of target network (used in PALE, FRUI-P)')
    parser.add_argument('--graph1', required=False,
                        help='Source network (used in MNA, FRUI-P, FINAL)')
    parser.add_argument('--graph2', required=False,
                        help='Target network (used in MNA, FRUI-P, FINAL)')
    parser.add_argument('--graph-size1', required=False, type=int,
                        help='Size of source network (used in FINAL)')
    parser.add_argument('--graph-size2', required=False, type=int,
                        help='Size of target network (used in FINAL)')
    parser.add_argument('--type-model', default='mlp', choices=['mlp', 'lin'],
                        help='Model type (used in PALE)')
    parser.add_argument('--identity-linkage', required=False,
                        help='File of anchor links')
    parser.add_argument('--output', required=True,
                        help='Output file')
    parser.add_argument('--log-file', default='ALP',
                        help='logging file')
    parser.add_argument('--lr', default=.01, type=float,
                        help='Learning rate (used in PALE)')
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
                        help='The training epochs (used in PALE)')
    parser.add_argument('--method', required=True, choices=['pale', 'final', 'mna', 'fruip'],
                        help='The learning methods')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='The negative ratio (used in PALE)')
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
    SAVING_STEP = args.saving_step
    MAX_EPOCHS = args.epochs
    if args.method == 'pale':
        model = PALE(learning_rate=args.lr, batch_size=args.batch_size
                        , n_input=args.input_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , type_model = args.type_model
                        , log_file=args.log_file, device=args.device)
    if args.method == 'mna' or args.method == 'fruip':
        graph = defaultdict(Graph)
        print "Loading graph..."
        if args.graph1:
            graph['f'].read_adjlist(filename=args.graph1)
        if args.graph2:
            graph['g'].read_adjlist(filename=args.graph2)

        if args.method == 'mna':
            model = MNA(graph=graph, anchorfile=args.identity_linkage, valid_prop=1., neg_ratio=3, log_file=args.log_file)
        if args.method == 'fruip':
            embed_files = [args.embedding1, args.embedding2]
            model = FRUIP(graph=graph, embed_files=embed_files, linkage_file=args.identity_linkage)
            model.main_proc(args.threshold)
    if args.method == 'final':
        main_proc([args.graph1, args.graph2], [args.graph_size1, args.graph_size2]
                            , args.identity_linkage, args.alpha, args.epochs, args.tol, args.output)

    if args.method in ['pale']:
        for i in range(1,MAX_EPOCHS+1):
            model.train_one_epoch()
            if i>0 and i%SAVING_STEP==0:
                model.save_models(args.output+'.epoch_'+str(i))
    if args.method in ['mna', 'fruip']:
        model.save_model(args.output)
    t2 = time.time()
    print 'time cost:',t2-t1

if __name__ == "__main__":
    main(parse_args())

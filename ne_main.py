from __future__ import print_function

import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.graph import *
from net_embed.line import *
from net_embed.ffvm import *
from utils.graph import Graph
import os

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--gpu-id', required=False,
                        help='Set env CUDA_VISIBLE_DEVICES', default="0")
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--input-delimiter', required=False, default=',',
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--weighted', required=False, default=False,
                        help='If the edge is weighted in edgelist format')
    parser.add_argument('--directed', required=False, default=False,
                        help='If the edge is directed in edgelist format')
    parser.add_argument('--last-emb-file', required=False,
                        help='Representation file from last training (For restoring the last training)')
    parser.add_argument('--log-file', default='NETEMB',
                        help='logging file')
    parser.add_argument('--lr', default=.001, type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Batch size')
    parser.add_argument('--table-size', default=1e8, type=int,
                        help='Table size')
    parser.add_argument('--rep-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs')
    parser.add_argument('--method', required=True, choices=['line', 'ffvm'],
                        help='The learning method')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='the negative ratio)')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    args = parser.parse_args()
    return args


def main(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    g = Graph()
    print("Reading...")
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input, delimiter=args.input_delimiter)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted
                            , directed=args.directed, delimiter=args.input_delimiter)
    if args.method == 'line':
        model = LINE(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs, 
                            rep_size=args.rep_size, table_size=args.table_size,
                            order=args.order, outfile=args.output, log_file=args.log_file, 
                            last_emb_file=args.last_emb_file, negative_ratio=args.neg_ratio)
    if args.method == 'ffvm':
        model = FFVM(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs, 
                            rep_size=args.rep_size, negative_ratio=args.neg_ratio, outfile=args.output,
                            last_emb_file=args.last_emb_file, log_file=args.log_file)

if __name__ == "__main__":
    main(parse_args())

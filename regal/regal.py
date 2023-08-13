import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
try: import cPickle as pickle 
except ImportError:
    import pickle
from scipy.sparse import csr_matrix

import xnetmf
from config import *

def parse_args():
    parser = argparse.ArgumentParser(description="Run REGAL.")

    parser.add_argument('--input', nargs='+', required=True
                        , help="Edgelist of input graphs")

    parser.add_argument('--output', required=True,
                        help='Embeddings path')

    parser.add_argument('--attributes', nargs='+', required=False,
                        help='File with node attributes')

    # parser.add_argument('--attrvals', type=int, default=2,
    #                     help='Number of attribute values. Only used if synthetic attributes are generated')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--k', type=int, default=10,
                        help='Controls of landmarks to sample. Default is 10.')

    parser.add_argument('--untillayer', type=int, default=2,
                        help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
    # parser.add_argument('--numtop', type=int, default=10,help="Number of top similarities to compute with kd-tree.  If 0, computes all pairwise similarities.")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    return parser.parse_args()

def main(args):
    # dataset_name = args.output.split("/")
    # if len(dataset_name) == 1:
    #     dataset_name = dataset_name[-1].split(".")[0]
    # else:
    #     dataset_name = dataset_name[-2]

    #Get true alignments
    # true_alignments_fname = args.input.split("_")[0] + "_edges-mapping-permutation.txt" #can be changed if desired
    # print "true alignments file: ", true_alignments_fname
    # true_alignments = None
    # if os.path.exists(true_alignments_fname):
    #     with open(true_alignments_fname, "rb") as true_alignments_file:
    #         true_alignments = pickle.load(true_alignments_file)

    # print true_alignments
    #Load in attributes if desired (assumes they are numpy array)
    # if args.attributes is not None:
    #     args.attributes = np.load(args.attributes) #load vector of attributes in from file
    #     print args.attributes.shape
    # print args.attributes

    # unify structure info from networks
    if args.input:
        format_struc_file = 'links.regal'
        if not os.path.exists(format_struc_file):
            for k in range(len(args.input)):
                format_unigraph(args.input[k], format_struc_file, k)
        args.input=format_struc_file

    # unify features from networks
    if args.attributes:
        format_attr_file = 'feats.regal'
        if not os.path.exists(format_attr_file):
            for k in range(len(args.attributes)):
                format_unifeat(args.attributes[k], format_attr_file, k)
        args.attributes=format_attr_file

    #Learn embeddings and save to output
    print("learning representations...")
    before_rep = time.time()
    learn_representations(args)
    after_rep = time.time()
    print("Learned representations in %f seconds" % (after_rep - before_rep))

    #Score alignments learned from embeddings
    # embed = np.load(args.output,allow_pickle=True)
    # emb1, emb2 = get_embeddings(embed)
    # before_align = time.time()
    # if args.numtop == 0:
    #     args.numtop = None
    # alignment_matrix = get_embedding_similarities(emb1, emb2, num_top = args.numtop)

    # #Report scoring and timing
    # after_align = time.time()
    # total_time = after_align - before_align
    # print("Align time: ", total_time)

    # if true_alignments is not None:
    #     topk_scores = [1,5,10,20,50]
    #     for k in topk_scores:
    #         score, correct_nodes = score_alignment_matrix(alignment_matrix, topk = k, true_alignments = true_alignments)
    #         print("score top%d: %f" % (k, score))

#Should take in a file with the input graph as edgelist (args.input)
#Should save representations to args.output
def learn_representations(args):
    nx_graph = nx.read_adjlist(args.input, create_using=nx.DiGraph(), delimiter=',')
    print("read in graph")
    adj = nx.adjacency_matrix(nx_graph)#.todense()
    print("got adj matrix")
    
    graph = Graph(adj, nx_graph=nx_graph, node_attributes = args.attributes)
    max_layer = args.untillayer
    if args.untillayer == 0:
        max_layer = None
    alpha = args.alpha
    num_buckets = args.buckets #BASE OF LOG FOR LOG SCALE
    if num_buckets == 1:
        num_buckets = None
    rep_method = RepMethod(max_layer = max_layer, 
                            alpha = alpha, 
                            k = args.k, 
                            num_buckets = num_buckets, 
                            normalize = True, 
                            gammastruc = args.gammastruc, 
                            gammaattr = args.gammaattr)
    rep_method.p = args.dimensions
    if max_layer is None:
        max_layer = 1000
    print("Learning representations with max layer %d and alpha = %f" % (max_layer, alpha))
    representations = xnetmf.get_representations(graph, rep_method)
    save_vectors(get_one_embeddings(graph,representations), args.output)
    # pickle.dump(representations, open(args.output, "w"))

def format_unigraph(filepath, outfile, layer_k, delimiter=','):
    with open(filepath, 'r') as fin, open(outfile, 'a+') as fout:
        cnt = 0
        wrt_ln = ''
        for ln in fin:
            elems = ln.strip().split(delimiter)
            wrt_ln += ','.join(['{}-{}'.format(layer_k, val) for val in elems])+'\n'
            cnt += 1
            if not cnt%1000:
                fout.write(wrt_ln)
                wrt_ln = ''
        if cnt%1000:
            fout.write(wrt_ln)
        print(cnt)

def format_unifeat(filepath, outfile, layer_k, delimiter=','):
    with open(filepath, 'r') as fin, open(outfile, 'a+') as fout:
        cnt = 0
        wrt_ln = ''
        for ln in fin:
            elems = ln.strip().split(delimiter)
            nd = '{}-{},'.format(layer_k,elems[0])
            wrt_ln += nd+'{}\n'.format(','.join([val for val in elems[1:]]))
            cnt += 1
            if not cnt%1000:
                fout.write(wrt_ln)
                wrt_ln = ''
        if cnt%1000:
            fout.write(wrt_ln)

def save_vectors(vectors, outfile):
    with open(outfile, 'w') as fout:
        # outfile-[node_embeddings/content-embeddings]-[src/obj]
        node_num = len(vectors.keys())
        fout.write("node num: {}\n".format(node_num))
        for node, vec in vectors.items():
            fout.write("{},{}\n".format(node,','.join([str(x) for x in vec])))

def get_one_embeddings(graph, embeddings):
    vectors = dict()
    look_back = graph.look_back_list
    for i, param in enumerate(embeddings):
        vectors[look_back[i]] = param
    return vectors

if __name__ == "__main__":
    args = parse_args()
    main(args)

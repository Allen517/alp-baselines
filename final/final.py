from __future__ import print_function

import scipy.sparse as sp
import numpy as np
import time
from scipy.sparse.linalg import norm
import sys

'''
 Description:
   The algorithm is the generalized attributed network alignment algorithm.
   The algorithm can handle the cases no matter node attributes and/or edge
   attributes are given. If no node attributes or edge attributes are given,
   then the corresponding input variable of the function is empty, e.g.,
   N1 = [], E1 = {}.
   The algorithm can handle either numerical or categorical attributes
   (feature vectors) for both edges and nodes.

   The algorithm uses cosine similarity to calculate node and edge feature
   vector similarities. E.g., sim(v1, v2) = <v1, v2>/(||v1||_2*||v2||_2).
   For categorical attributes, this is still equivalent to the indicator
   function in the original published paper.

 Input:
   A1, A2: Input adjacency matrices with n1, n2 nodes
   N1, N2: Node attributes matrices, N1 is an n1*K matrix, N2 is an n2*K
         matrix, each row is a node, and each column represents an
         attribute. If the input node attributes are categorical, we can
         use one hot encoding to represent each node feature as a vector.
         And the input N1 and N2 are still n1*K and n2*K matrices.
         E.g., for node attributes as countries, including USA, China, Canada, 
         if a user is from China, then his node feature is (0, 1, 0).
         If N1 and N2 are emtpy, i.e., N1 = [], N2 = [], then no node
         attributes are input. 

   E1, E2: a L*1 cell, where E1{i} is the n1*n1 matrix and nonzero entry is
         the i-th attribute of edges. E2{i} is same. Similarly,  if the
         input edge attributes are categorical, we can use one hot
         encoding, i.e., E1{i}(a,b)=1 if edge (a,b) has categorical
         attribute i. If E1 and E2 are empty, i.e., E1 = {} or [], E2 = {}
         or [], then no edge attributes are input.

   H: a n2*n1 prior node similarity matrix, e.g., degree similarity. H
      should be normalized, e.g., sum(sum(H)) = 1.
   alpha: decay factor 
   maxiter, tol: maximum number of iterations and difference tolerance.

 Output: 
   S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
    x in A2 is aligned to node-y in A1

 Reference:
   Zhang, Si, and Hanghang Tong. "FINAL: Fast Attributed Network Alignment." 
   Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.
'''
class FINAL(object):

    def __init__(self, A1, A2, H, alpha, maxiter, tol):
        if not sp.isspmatrix_coo(A1):
            A1 = A1.tocoo()
        if not sp.isspmatrix_coo(A2):
            A2 = A2.tocoo()
        self.A1 = A1
        self.A2 = A2
        if not sp.isspmatrix_coo(H):
            H = H.tocoo()
        self.H = H
        self.alpha = alpha
        self.maxiter = maxiter
        self.tol = tol

    def main_proc(self):
        alpha = self.alpha

        A1 = self.A1.tocsr()
        A2 = self.A2.tocsr()

        n1 = self.A1.shape[0]
        n2 = self.A2.shape[0]

        # If no node attributes input, then initialize as a vector of 1
        # so that all nodes are treated to have the save attributes which 
        # is equivalent to no given node attribute.
        N1 = np.ones((n1, 1));
        N2 = np.ones((n2, 1));

        # If no edge attributes are input, i.e., E1 and E2 are empty, then
        # initialize as a cell with 1 element, which is same as adjacency matrix
        # but the entries that are nonzero in adjacency matrix are equal to 1 so 
        # that all edges are treated as with the same edge attributes. This is 
        # equivalent to no given edge attributes.
        E1 = list()
        E2 = list()
        E1.append(A1)
        E2.append(A2)
    
        K = N1.shape[1]
        L = len(E1);
        T1 = sp.coo_matrix((n1, n1));
        T2 = sp.coo_matrix((n2, n2));

        # Normalize edge feature vectors 
        for l in range(L):
            T1 = T1 + E1[l].power(2) # calculate ||v1||_2^2
            T2 = T2 + E2[l].power(2) # calculate ||v2||_2^2

        T1.data = 1./np.sqrt(T1.data)
        T2.data = 1./np.sqrt(T2.data)

        for l in range(L):
           E1[l].data = E1[l].data * T1.data # normalize each entry by vector norm T1
           E2[l].data = E2[l].data * T2.data # normalize each entry by vector norm T2

        # Normalize node feature vectors
        K1 = np.power(np.sum(np.power(N1, 2),1), -.5)
        K1[np.isinf(K1)] = 0
        K2 = np.power(np.sum(np.power(N2, 2),1), -.5)
        K2[np.isinf(K2)] = 0

        N1 = K1 * N1 # normalize the node attribute for A1
        N2 = K2 * N2 # normalize the node attribute for A2

        # Compute node feature cosine cross-similarity
        N = sp.csr_matrix((n1*n2, 1));
        for k in range(K):
            N = N + sp.csr_matrix(np.kron(N1[:,k], N2[:,k]).reshape((-1,1))) # compute N as a kronecker similarity

        # Compute the Kronecker degree vector
        t1 = time.time()
        d = sp.csr_matrix((n1*n2, 1));
        for l in range(L):
            for k in range(K):
                d = d + sp.csr_matrix(np.kron(E1[l].multiply(A1).dot(N1[:,k])
                                , E2[l].multiply(A2).dot(N2[:,k])).reshape((-1,1)))
        print('Time for degree: {} sec\n'.format(time.time()-t1))

        D = N.multiply(d)
        DD = D.power(-.5)

        DD[D==0] = 0   # define inf to 0

        # fixed-point solution
        q = DD.multiply(N).tolil().reshape((n2, n1))
        h = self.H.tolil()
        s = h

        for i in range(self.maxiter):
            print('iteration {}\n'.format(i))
            t2 = time.time()
            prev = s
            M = q.multiply(s)
            S = sp.coo_matrix((n2, n1))

            for l in range(L):
                S = S + E2[l].multiply(A2).dot(M).dot(E1[l].multiply(A1)) # calculate the consistency part

            s = (1-alpha)*h + (alpha*q).multiply(S) # add the prior part
            diff = norm(s-prev);
            
            print('Time for iteration {}: {} sec, diff = {}\n'.format(i, time.time()-t2, 100*diff))
            if diff < self.tol: # if converge
                break

        return s

def read_adjlist(graph_file, graph_size):
    graph_dict = dict()
    graph_dict_back = list()
    idx = 0
    graph = sp.lil_matrix((graph_size, graph_size))
    with open(graph_file, 'r') as fin:
        for ln in fin:
            elems = ln.strip().split()
            src_id = None
            for gid in elems:
                if not gid in graph_dict:
                    graph_dict[gid] = idx
                    graph_dict_back.append(gid)
                    idx+=1
                if not src_id:
                    src_id = graph_dict[gid]
                    continue
                graph[src_id, graph_dict[gid]] = 1

    return graph, graph_dict, graph_dict_back

def read_edgelist(graph_file, graph_size):
    graph_dict = dict()
    graph_dict_back = list()
    idx = 0
    graph = sp.lil_matrix((graph_size, graph_size))
    with open(graph_file, 'r') as fin:
        for ln in fin:
            elems = ln.strip().split()
            if len(elems)<2:
                continue
            first_id_str = elems[0]
            sec_id_str = elems[1]
            if first_id_str not in graph_dict:
                graph_dict[first_id_str] = idx
                graph_dict_back.append(first_id_str)
                idx += 1
            if sec_id_str not in graph_dict:
                graph_dict[sec_id_str] = idx
                graph_dict_back.append(sec_id_str)
                idx += 1
            first_idx = graph_dict[first_id_str]
            sec_idx = graph_dict[sec_id_str]
            graph[first_idx, sec_idx] = 1
            graph[sec_idx, first_idx] = 1

    return graph, graph_dict, graph_dict_back

def read_linkage(linkage_file, graph_g_dict, graph_g_size, graph_f_dict, graph_f_size):
    linkage = sp.lil_matrix((graph_g_size, graph_f_size))
    with open(linkage_file, 'r') as fin:
        for ln in fin:
            elems = ln.strip().split()
            if not elems[0] in graph_g_dict:
                continue
            if not elems[1] in graph_f_dict:
                continue
            g_id = graph_g_dict[elems[0]]
            f_id = graph_f_dict[elems[1]]
            linkage[g_id, f_id] = 1

    return linkage

def main_proc(graph_files, graph_sizes, linkage_file, alpha, epoch, tol, graph_format, test_anchor_file, output_file):

    graph_f_size = graph_sizes[0]
    graph_g_size = graph_sizes[1]

    if graph_format=='adjlist':
        read_graph = read_adjlist
    if graph_format=='edgelist':
        read_graph = read_edgelist

    graph_f, graph_f_dict, graph_f_dict_back = read_graph(graph_files[0], graph_f_size)
    graph_g, graph_g_dict, graph_g_dict_back = read_graph(graph_files[1], graph_g_size)

    linkage = read_linkage(linkage_file, graph_g_dict, graph_g_size, graph_f_dict, graph_f_size)

    final = FINAL(graph_f, graph_g, linkage, alpha, epoch, tol)

    res = final.main_proc().tocoo()
    row = res.row
    col = res.col
    data = res.data
    last_row = None
    wrtLn = ''
    wrtCnt = 0
    with open(output_file, 'w') as fout:
        for i in range(len(data)):
            if row[i] != last_row:
                wrtLn = wrtLn[:-1]+'\n' if last_row else ''
                last_row = row[i]
                wrtLn += graph_g_dict_back[last_row]+','
                wrtCnt += 1
                if wrtCnt%100==0:
                    fout.write(wrtLn)
                    wrtLn = ''
            wrtLn += '{}:{},'.format(graph_f_dict_back[col[i]], data[i])
        if wrtLn:
            fout.write(wrtLn[:-1])

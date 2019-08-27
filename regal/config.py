import numpy as np

class RepMethod():
    def __init__(self, 
                align_info = None, 
                p=None, 
                k=10, 
                max_layer=None, 
                alpha = 0.1, 
                num_buckets = None, 
                normalize = True, 
                gammastruc = 1, 
                gammaattr = 1):
        self.p = p #sample p points
        self.k = k #control sample size
        self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
        self.alpha = alpha #discount factor for higher layers
        self.num_buckets = num_buckets #number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
        self.normalize = normalize #whether to normalize node embeddings
        self.gammastruc = gammastruc #parameter weighing structural similarity in node identity
        self.gammaattr = gammaattr #parameter weighing attribute similarity in node identity

class Graph():
    #Undirected, unweighted
    def __init__(self, 
                adj, 
                nx_graph=None,
                num_buckets=None, 
                node_labels = None, 
                edge_labels = None,
                graph_label = None, 
                node_attributes = None, 
                true_alignments = None):
        self.G_adj = adj #adjacency matrix
        self.N = self.G_adj.shape[0] #number of nodes
        self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
        self.max_degree = max(self.node_degrees)
        self.num_buckets = num_buckets #how many buckets to break node features into

        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.kneighbors = None #dict of k-hop neighbors for each node
        self.true_alignments = true_alignments #dict of true alignments, if this graph is a combination of multiple graphs

        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0
        self.encode_node(nx_graph)
        # self.node_attributes = node_attributes #N x A matrix, where N is # of nodes, and A is # of attributes
        self.node_attributes = self.set_node_attributes(node_attributes) #N x A matrix, where N is # of nodes, and A is # of attributes

    def set_node_attributes(self, attr_file):
        node_attributes = [list() for i in range(self.node_size)]
        if not attr_file:
            return None
        with open(attr_file, 'r') as fin:
            for ln in fin:
                elems = ln.strip().split(',')
                if elems[0] not in self.look_up_dict:
                    continue
                node_attributes[self.look_up_dict[elems[0]]].append(list(map(float,elems[1:])))
        return np.array(node_attributes)

    def encode_node(self, nx_graph):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in nx_graph.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
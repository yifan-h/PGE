from __future__ import division
from __future__ import print_function

import numpy as np

np.random.seed(123)

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as PGE

class EdgeMinibatchIterator(object):

    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx,
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs # random walk pair
        self.train_edges = self.edges = np.random.permutation(edges)
        self.train_edges = self._remove_isolated(self.train_edges) # remove test/val or isolated nodes
        self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size,
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test']
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class WeightedEdgeMinibatchIterator(EdgeMinibatchIterator):

    """ This minibatch iterator extends from EdgeMinibatchIterator to
    implement the weighted neighborhood aggregation based on the biased sampled neighboring edges.

    Ref URL: https://www.kdd.org/kdd2019/accepted-papers/view/a-representation-learning-framework-for-property-graphs
    Please check Section 3.3 for more details

    node_tags -- mapping to the cluster id for nodes after node clustering
    pweight -- assigned bias if in the same cluster (i.e., b_s, see Alg1 in our paper)
    nweight -- assigned bias if in different cluster (i.e., b_d, see Alg1 in our paper)
    directed_graph -- if directed graph or not. If so, we differentiate the in/out edges separately for training 
    adj_in -- in neighbors, will be used only when directed = True
    adj_weight -- the biased weighted matrix
    adj_weight_test -- for testing
    adj_in_weight -- the biased weighted matrix for in_neighbors, will be used only when directed = True 
    adj_in_weight_test -- for testing
    """

    def __init__(self, CModel, dif_weight=1, directed_graph=False, same_weight = 1, **kwargs):
        self.node_tags = CModel.labels_
        self.pweight = same_weight
        self.nweight = dif_weight
        self.directed_graph = directed_graph
        self.adj_in = None
        self.adj_weight = None
        self.adj_weight_test = None
        self.adj_in_weight = None
        self.adj_in_weight_test = None

        super(WeightedEdgeMinibatchIterator, self).__init__(**kwargs)

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        adj_in = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))
        mask = np.zeros(adj.shape, dtype=np.float)
        mask_in = np.zeros(adj.shape, dtype=np.float)

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            node_tag = self.node_tags[nodeid]

            if self.directed_graph:  # directed condition
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])])
                predecessors = np.array([self.id2idx[predecessor]
                    for predecessor in self.G.predecessors(nodeid)
                    if (not self.G[predecessor][nodeid]['train_removed'])]) # place need to switch
                deg[self.id2idx[nodeid]] = len(neighbors)+len(predecessors)

                # == sampling bi-adjecent table ===
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors

                # == sampling bi-adjecent table ===
                if len(predecessors) == 0:
                    continue
                if len(predecessors) > self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=False)
                elif len(predecessors) < self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=True)
                adj_in[self.id2idx[nodeid], :] = predecessors

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
                for i, nid in enumerate(predecessors):
                    mask_in[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight

            else:
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])])
                deg[self.id2idx[nodeid]] = len(neighbors)

                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight

        mask /= (np.sum(mask, axis=1, keepdims=True) + 0.0001)
        mask_in /= (np.sum(mask_in, axis=1, keepdims=True) + 0.0001)
        self.adj_in = adj_in  # get in neighbors
        self.adj_weight = mask
        self.adj_in_weight = mask_in

        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        adj_in = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        mask = np.zeros(adj.shape, dtype=np.float)
        mask_in = np.zeros(adj.shape, dtype=np.float)

        for nodeid in self.G.nodes():
            node_tag = self.node_tags[nodeid]

            if self.directed_graph:
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)])
                predecessors = np.array([self.id2idx[predecessor]
                    for predecessor in self.G.predecessors(nodeid)]) # place need to switch

                # fixed size adjecent table
                if len(neighbors) == 0:
                    continue
                # sampling
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :self.max_degree] = neighbors[:self.max_degree]

                # fixed size in-adjecent table
                if len(predecessors) == 0:
                    continue
                # sampling
                if len(predecessors) > self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=False)
                elif len(predecessors) < self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=True)
                adj_in[self.id2idx[nodeid], :self.max_degree] = predecessors[:self.max_degree]

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
                for i, nid in enumerate(predecessors):
                    mask_in[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight

            else:
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)])
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors
                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
                adj[self.id2idx[nodeid], :self.max_degree] = neighbors[:self.max_degree]

        self.test_adj_in = adj_in  # get in neighbors
        mask /= (np.sum(mask, axis=1, keepdims=True)+0.0001)
        mask_in /= (np.sum(mask_in, axis=1, keepdims=True) + 0.0001)
        self.adj_weight_test = mask
        self.adj_in_weight_test = mask_in

        return adj

class NodeMinibatchIterator(object):

    """
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx,
            placeholders, label_map, num_classes,
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']] # list of id
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])]) # neighborhood ids
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]

        labels = np.vstack([self._make_label_vec(node) for node in batch1id]) # class vector [batch size, class num]
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size,
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size,
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

class WeightedNodeMinibatchIterator(NodeMinibatchIterator):

    """ This minibatch iterator extends from NodeMinibatchIterator to
    implement the weighted neighborhood aggregation based on the biased sampled neighbors.

    Ref URL: https://www.kdd.org/kdd2019/accepted-papers/view/a-representation-learning-framework-for-property-graphs
    """

    def __init__(self, CModel, dif_weight = 1, directed_graph = False, same_weight = 1, **kwargs):
        self.node_tags = CModel.labels_
        # scalability design: weight configuration -> dict
        self.pweight = same_weight
        self.nweight = dif_weight
        self.directed_graph = directed_graph
        self.adj_in = None
        self.adj_weight = None
        self.adj_weight_test = None
        self.adj_in_weight = None
        self.adj_in_weight_test = None
        super(WeightedNodeMinibatchIterator, self).__init__(**kwargs)
        del CModel
        import gc
        gc.collect()

    def construct_adj(self):
        # construct a adj table with weight mask
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx)+1,), dtype=np.float)
        mask = np.zeros(adj.shape, dtype=np.float)


        if self.directed_graph:
            adj_in = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
            mask_in = np.zeros(adj_in.shape, dtype=np.float)
            for nodeid in self.G.nodes():
                if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                    continue
                node_tag = self.node_tags[nodeid]
                # graph links
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])])
                predecessors = np.array([self.id2idx[predecessor]
                    for predecessor in self.G.predecessors(nodeid)
                    if (not self.G[predecessor][nodeid]['train_removed'])])
                deg[self.id2idx[nodeid]] = len(neighbors)+len(predecessors)

                # == sampling bi-adjecent table ===
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) > 0 and len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                elif len(neighbors) == 0:
                    neighbors = np.random.choice([nodeid], self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors

                # == sampling in-adjecent table ===
                if len(predecessors) > self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=False)
                elif len(predecessors) > 0 and len(predecessors) < self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=True)
                elif len(predecessors) == 0:
                    predecessors = np.random.choice([nodeid], self.max_degree, replace=True)
                adj_in[self.id2idx[nodeid], :] = predecessors

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
                for i, nid in enumerate(predecessors):
                    mask_in[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
            mask_in /= (np.sum(mask_in, axis=1, keepdims=True) + 0.0001)
            self.adj_in_weight = mask_in
            self.adj_in = adj_in  # get in neighbors
        else: # no-directed graph
            for nodeid in self.G.nodes():
                if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                    continue
                node_tag = self.node_tags[nodeid]
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])]) # neighborhood ids
                deg[self.id2idx[nodeid]] = len(neighbors)

                # fixed size adjecent table
                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight

        mask /= (np.sum(mask, axis=1, keepdims=True) + 0.0001)
        self.adj_weight = mask
        return adj, deg

    def construct_test_adj(self):
        # construct a adj table with weight mask
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        mask = np.zeros(adj.shape, dtype=np.float)

        if self.directed_graph:
            adj_in = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
            mask_in = np.zeros(adj.shape, dtype=np.float)
            for nodeid in self.G.nodes():
                node_tag = self.node_tags[nodeid]
                # graph link
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)])
                predecessors = np.array([self.id2idx[predecessor]
                    for predecessor in self.G.predecessors(nodeid)]) # place need to switch

                # == adjecent table ===
                if len(neighbors) >= self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) > 0 and len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                elif len(neighbors) == 0:
                    neighbors = np.random.choice([nodeid], self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors

                if len(predecessors) >= self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=False)
                elif len(predecessors) > 0 and len(predecessors) < self.max_degree:
                    predecessors = np.random.choice(predecessors, self.max_degree, replace=True)
                elif len(predecessors) == 0:
                    predecessors = np.random.choice([nodeid], self.max_degree, replace=True)
                adj_in[self.id2idx[nodeid], :] = predecessors

                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
                for i, nid in enumerate(predecessors):
                    mask_in[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight
            self.test_adj_in = adj_in  # get in neighbors
            mask_in /= np.sum(mask_in, axis=1, keepdims=True) + 0.0001
            self.adj_in_weight_test = mask_in
        else:
            for nodeid in self.G.nodes():
                node_tag = self.node_tags[nodeid]
                neighbors = np.array([self.id2idx[neighbor]
                    for neighbor in self.G.neighbors(nodeid)])

                if len(neighbors) == 0:
                    continue
                if len(neighbors) > self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
                elif len(neighbors) < self.max_degree:
                    neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
                adj[self.id2idx[nodeid], :] = neighbors
                # process weight mask
                for i, nid in enumerate(neighbors):
                    mask[self.id2idx[nodeid], i] = self.pweight if self.node_tags[nid] == node_tag else self.nweight

        mask /= (np.sum(mask, axis=1, keepdims=True)+0.0001)
        self.adj_weight_test = mask
        return adj


from __future__ import print_function

import numpy as np
import random, time
import json
import sys, os, pickle
import concurrent.futures as futures
import sklearn
import sklearn.cluster as sklc
import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as PGE

def process_graph(G):
    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)

    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    return G

def loadG(x, d):
    return json_graph.node_link_graph(json.load(open(x+'-G.json')), d)

def loadjson(x):
    return json.load(open(x))

def convert_dict(x, conv, lconv=int):
    return {conv(k):lconv(v) for k, v in x.items()}

def load_data_with_threadpool(prefix, normalize=True, load_walks=False, directed=False):
    with futures.ProcessPoolExecutor(max_workers=5) as executor:
        # 1. read data
        start_time = time.time()
        futs = [
            executor.submit(loadG, prefix, directed),
            executor.submit(loadjson, prefix + '-id_map.json'),
            executor.submit(loadjson, prefix+'-class_map.json'),
            ]
        if os.path.exists(prefix + '-feats.npy'):
            feats = np.load(prefix + '-feats.npy')
            print("read feats in", "{:.5f}".format(time.time() - start_time), "seconds")
        else:
            feats = None
        # 2. process preparation
        start_time = time.time()
        id_map, class_map = futs[1].result(), futs[2].result()
        print("read id_map, class_map in", "{:.5f}".format(time.time() - start_time), "seconds")
        start_time = time.time()
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n : n
        else:
            lab_conversion = lambda n : int(n)

        G = futs[0].result()
        print("read G in", "{:.5f}".format(time.time() - start_time), "seconds")
        # 3. process data
        start_time = time.time()
        if isinstance(G.nodes()[0], int):
            conversion = lambda n : int(n)
        else:
            conversion = lambda n : n
        fut = executor.submit(process_graph, G)
        id_map = convert_dict(id_map, conversion)
        class_map = convert_dict(class_map, conversion, lab_conversion)

        if load_walks:
            def load_walk(x, conv):
                with open(x) as fp:
                    walks = []
                    for line in fp:
                        walks.append(map(conv, line.split()))
                return walks
            walks = load_walks(prefix+'-walks.txt', conversion)
        else:
            walks = []
        G = fut.result()
        print("process G in", "{:.5f}".format(time.time() - start_time), "seconds")
        # 4. post process
        if normalize and not feats is None:
            from sklearn.preprocessing import StandardScaler
            start_time = time.time()
            train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
            train_feats = feats[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            feats = scaler.transform(feats)
            print("post process G in", "{:.5f}".format(time.time() - start_time), "seconds")

    return G, feats, id_map, walks, class_map

def run_cluster(features, n_clusters, max_iter, random_state, eps, min_samples):
    #model = sklc.KMeans(n_clusters, max_iter = max_iter, random_state=random_state)
    model = sklc.DBSCAN(eps = eps, min_samples = min_samples, n_jobs = 4)
    # model = sklc.SpectralClustering(n_clusters)
    model.fit(features)
    print('The number of clusters is: ', len(set(model.labels_)))
    return model

def makeExists(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

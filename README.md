## A Representation Learning Framework for Property Graphs

#### Authors: Yifan Hou(yfhou@cse.cuhk.edu.hk), [Hongzhi Chen](https://yaobaiwei.github.io/)(hzchen@cse.cuhk.edu.hk), Changji Li(cjli@cse.cuhk.edu.hk)

### Overview

Representation learning on graphs, also called graph embedding, has demonstrated its significant impact on a series of machine learning applications such as classification, prediction and recommendation. 
However, existing works have largely ignored the rich information contained in the properties (or attributes) of both nodes and edges of graphs in modern applications represented by property graphs. Most of them either focus on plain graphs with only the graph topology, or consider properties on nodes only.

We propose PGE, a graph representation learning framework that incorporates both node and edge properties into the graph embedding procedure.
PGE uses node clustering to assign biases to differentiate neighbors of a node and leverages multiple data-driven matrices to aggregate the property information of neighbors sampled based on a biased strategy. PGE adopts the popular inductive model for neighborhood aggregation. 
We provide detailed analyses on the efficacy of our method and validate the performance of PGE by showing how PGE achieves better embedding results than the state-of-the-art graph embedding methods on benchmark applications such as node classification and link prediction over real-world datasets.

Please see the [paper](https://www.kdd.org/kdd2019/accepted-papers/view/a-representation-learning-framework-for-property-graphs) for more details.

*Note:* If you make use of this code or the PGE model in your work, please cite the following paper:

    @inproceedings{DBLP:conf/kdd/HouCLCY19,
      author    = {Yifan Hou and Hongzhi Chen and Changji Li and James Cheng and Ming{-}Chang Yang},
      title     = {A Representation Learning Framework for Property Graphs},
      booktitle = {SIGKDD},
      pages     = {65--73},
      year      = {2019}
    }

### Requirements

* tensorflow (1.14.0)
* numpy
* scikit-learn
* networkx (1.11)

### How to run

The link_pred_ppi.sh and node_class_ppi.sh files contain example usages of the code for applications: link prediction task and node classification task respectively, based on ppi dataset.

#### Input format

* <train_prefix>-G.json -- A networkx-specified json file describing the input graph. Nodes have 'val' and 'test' attributes specifying if they are a part of the validation and test sets, respectively.
* <train_prefix>-id_map.json -- A json-stored dictionary mapping the graph node ids to consecutive integers.
* <train_prefix>-class_map.json -- A json-stored dictionary mapping the graph node ids to classes.
* <train_prefix>-feats.npy --- A numpy-stored array of node features; ordering given by id_map.json. Can be omitted and only identity features will be used.
* <train_prefix>.skl [optional] --- A sklearn-stored format of clustering results. Can be re-caculated by removing the file.

### Academic Paper

[**SIGKDD 19**] **A Representation Learning Framework for Property Graphs**, Yifan Hou, Hongzhi Chen, Changji Li, James Cheng,  Ming-Chang Yang.

### Acknowledgement
The original version of this code base was originally forked from https://github.com/williamleif/GraphSAGE, and we owe many thanks to William L. Hamilton for making the code available. We also thank Xiaoxi Wang's contribution to this project during her senior year internship in CUHK.

### License

Copyright 2019, Husky Data Lab, The Chinese University of Hong Kong.

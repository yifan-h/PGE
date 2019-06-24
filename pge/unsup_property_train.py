
from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn

import pge.utils as utils
from pge.models import WeightedSampleAndAggregate, SampleAndAggregate, PGEInfo
from pge.minibatch import WeightedEdgeMinibatchIterator, EdgeMinibatchIterator
from pge.neigh_samplers import WeightedNeighborSampler, UniformNeighborSampler

from sklearn.externals import joblib
from sklearn import metrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as PGE

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 1, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.3, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', False, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

# preload cluster
flags.DEFINE_string('cluster_model_path', '', 'load pretrained cluster')
flags.DEFINE_integer('cluster_max_iter', 800, '')
flags.DEFINE_boolean('directed', True, 'directed graph or not')
flags.DEFINE_integer('n_clusters', 1, 'number of clusters')
flags.DEFINE_float('dif_weight', 1000, 'different cluster weight')
flags.DEFINE_float('min_samples', 10, 'min_samples for DBSCAN clustering')
flags.DEFINE_float('eps', 10, 'eps for DBSCAN clustering')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr],
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr],
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1],
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'), # edge node1
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'), # edge node2
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, CModel, test_data=None):
    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None

    placeholders = construct_placeholders()
    minibatch = WeightedEdgeMinibatchIterator(CModel,
            dif_weight = FLAGS.dif_weight,
            directed_graph = FLAGS.directed,
            G=G,
            id2idx = id_map,
            placeholders = placeholders,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs)

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
    adj_weight_ph = tf.placeholder(tf.float32, shape=minibatch.adj.shape)
    adj_weight = tf.Variable(adj_weight_ph, trainable=False, name="adj_weight")

    if FLAGS.directed:
        adj_in_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj_in.shape)
        adj_in_info = tf.Variable(adj_in_info_ph, trainable=False, name="adj_in_info")
        adj_in_weight_ph = tf.placeholder(tf.float32, shape=minibatch.adj_in.shape)
        adj_in_weight = tf.Variable(adj_in_weight_ph, trainable=False, name="adj_in_weight")
    else:
        adj_in_info_ph = None
        adj_in_info = None
        adj_in_weight_ph = None
        adj_in_weight = None

    if FLAGS.model == 'mean':
        # Create model
        sampler = WeightedNeighborSampler(adj_info, adj_weight)
        if FLAGS.directed: # directed graph condition
            sampler_in = WeightedNeighborSampler(adj_in_info, adj_in_weight)

        layer_infos = [PGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            PGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        if FLAGS.directed:
            layer_in_infos = [PGEInfo("node", sampler_in, FLAGS.samples_1, FLAGS.dim_1),
                                    PGEInfo("node", sampler_in, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_in_infos = []
        model = WeightedSampleAndAggregate(placeholders=placeholders,
                                     features=features,
                                     adj=adj_info,
                                     adj_in=adj_in_info,
                                     degrees=minibatch.deg,
                                     layer_infos=layer_infos,
                                     layer_in_infos = layer_in_infos,
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     directed_graph = FLAGS.directed,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    if FLAGS.save_embeddings:
        summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
    # Init
    if not FLAGS.directed:
        sess.run(tf.global_variables_initializer(), \
            feed_dict={adj_info_ph: minibatch.adj, adj_weight_ph: minibatch.adj_weight})
    else:
        sess.run(tf.global_variables_initializer(), \
            feed_dict={adj_info_ph: minibatch.adj, adj_weight_ph: minibatch.adj_weight,
            adj_in_info_ph: minibatch.adj_in, adj_in_weight_ph: minibatch.adj_in_weight})

    # Train model
    train_shadow_mrr = None
    shadow_mrr = None
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    train_adj_weight = tf.assign(adj_weight, minibatch.adj_weight)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    val_adj_weight = tf.assign(adj_weight, minibatch.adj_weight_test)

    if FLAGS.directed:
        train_adj_in_info = tf.assign(adj_in_info, minibatch.adj_in)
        train_adj_in_weight = tf.assign(adj_in_weight, minibatch.adj_in_weight)
        val_adj_in_info = tf.assign(adj_in_info, minibatch.test_adj_in)
        val_adj_in_weight = tf.assign(adj_in_weight, minibatch.adj_in_weight_test)

    for epoch in range(FLAGS.epochs):
        minibatch.shuffle()
        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)

        while not minibatch.end():
            # Construct feed dictionary
            feed_dict = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all,
                    model.mrr, model.outputs1], feed_dict=feed_dict)
            train_cost = outs[2]
            train_mrr = outs[5]
            if train_shadow_mrr is None:
                train_shadow_mrr = train_mrr
            else:
                train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)
            if iter % FLAGS.validate_iter == 0:
                # Validation
                if not FLAGS.directed:
                    sess.run([val_adj_info.op, val_adj_weight.op])
                else:
                    sess.run([val_adj_info.op, val_adj_weight.op, val_adj_in_info.op, val_adj_in_weight.op])
                val_cost, ranks, val_mrr, duration = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size)
                if not FLAGS.directed:
                    sess.run([train_adj_info.op, train_adj_weight.op])
                else:
                    sess.run([train_adj_info.op, train_adj_weight.op, train_adj_in_info.op, train_adj_in_weight.op])
                epoch_val_costs[-1] += val_cost
            if shadow_mrr is None:
                shadow_mrr = val_mrr
            else:
                shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)
            if FLAGS.save_embeddings:
                if total_steps % FLAGS.print_every == 0:
                    summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
            if total_steps % FLAGS.print_every == 0:
                # train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_mrr=", "{:.5f}".format(train_mrr),
                      "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_mrr=", "{:.5f}".format(val_mrr),
                      "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
                      "time=", "{:.5f}".format(avg_time))
            iter += 1
            total_steps += 1
            if total_steps > FLAGS.max_total_steps:
                break
        if total_steps > FLAGS.max_total_steps:
                break
    print("Optimization Finished!")

    if FLAGS.save_embeddings:
        if not FLAGS.directed:
            sess.run([val_adj_info.op, val_adj_weight.op])
        else:
            sess.run([val_adj_info.op, val_adj_weight.op, val_adj_in_info.op, val_adj_in_weight.op])
        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())
    val_losses, val_mrrs, _ = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)

    tf.reset_default_graph()
    return val_losses, val_mrrs


def printCModel(model):
    a = model.labels_
    print('nodes number: ', len(a))
    for i in range(model.n_clusters):
        print('label {}: {}'.format(i, len(a) - np.count_nonzero(a-i)))


def unsup_pg_main(argv=None):
    print("Loading training data..")
    train_data = utils.load_data_with_threadpool(FLAGS.train_prefix, directed=FLAGS.directed)
    utils.makeExists(FLAGS.base_log_dir)
    if FLAGS.cluster_model_path != '':
        print('Loading cluster data in {}'.format(FLAGS.cluster_model_path))
        CModel = joblib.load(FLAGS.cluster_model_path)
    else:
        path = FLAGS.train_prefix + '.skl'
        CModel = utils.run_cluster(train_data[1], FLAGS.n_clusters, FLAGS.cluster_max_iter, seed, eps = FLAGS.eps, min_samples = FLAGS.min_samples)
        joblib.dump(CModel, path)

    val_losses, val_mrrs = train(train_data, CModel)
    print('Final test results: '
                "losses=", "{:.5f}".format(val_losses),
                "mrrs=", "{:.5f}".format(val_mrrs))

if __name__ == '__main__':
    tf.app.run(unsup_pg_main)

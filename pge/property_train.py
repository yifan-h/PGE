from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.externals import joblib
from sklearn import metrics

import pge.utils as utils
from pge.supervised_models import SupervisedPGE
from pge.models import PGEInfo
from pge.minibatch import WeightedNodeMinibatchIterator, NodeMinibatchIterator
from pge.neigh_samplers import WeightedNeighborSampler,UniformNeighborSampler

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

# log setting
flags.DEFINE_boolean('log_device_placement', False, 'whether to log device placement')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_string('dir_suffix', '', 'directory suffix descript parameters')

#core params..
flags.DEFINE_string('model', 'mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 100, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_boolean('use_feature', True, 'Set to true to use feature as initial embedding')

# validate & test setting
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 10, "How often to print training info.")
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

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], (time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'), # dynamic
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'), # dynamic
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'), # scalar
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, CModel, test_data=None):
    G = train_data[0] # nx Graph
    features = train_data[1] # feature numpy array [train_node_num, feature len]
    id_map = train_data[2]
    class_map  = train_data[4]

    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    if FLAGS.random_context:
        context_pairs = train_data[3] if FLAGS.random_context else None
        # TODO: weighted random walk
    else:
        context_pairs = None

    placeholders = construct_placeholders(num_classes) # dict of placeholder
    minibatch = WeightedNodeMinibatchIterator(CModel,
            dif_weight = FLAGS.dif_weight,
            directed_graph = FLAGS.directed,  # sampling 128 neighbors randomly considering directed condition
            G=G,
            id2idx = id_map,
            placeholders = placeholders,
            label_map = class_map,
            num_classes = num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
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

    sampler = WeightedNeighborSampler(adj_info, adj_weight)
    if FLAGS.directed: # directed graph condition
        sampler_in = WeightedNeighborSampler(adj_in_info, adj_in_weight)

    if FLAGS.model == 'mean':
        # Create model
        if FLAGS.samples_3 != 0:
            layer_infos = [PGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                PGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                PGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
            if FLAGS.directed:
                layer_in_infos = [PGEInfo("node", sampler_in, FLAGS.samples_1, FLAGS.dim_1),
                                    PGEInfo("node", sampler_in, FLAGS.samples_2, FLAGS.dim_2),
                                    PGEInfo("node", sampler_in, FLAGS.samples_3, FLAGS.dim_2)]
            else:
                layer_in_infos = []
        elif FLAGS.samples_2 != 0:
            layer_infos = [PGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                PGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
            if FLAGS.directed:
                layer_in_infos = [PGEInfo("node", sampler_in, FLAGS.samples_1, FLAGS.dim_1),
                                    PGEInfo("node", sampler_in, FLAGS.samples_2, FLAGS.dim_2)]
            else:
                layer_in_infos = []
        else:
            layer_infos = [PGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]
            if FLAGS.directed:
                layer_in_infos = [PGEInfo("node", sampler_in, FLAGS.samples_1, FLAGS.dim_1)]
            else:
                layer_in_infos = []
        model = SupervisedPGE(num_classes, placeholders,
                                    features if FLAGS.use_feature else None,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss = FLAGS.sigmoid,
                                    identity_dim = FLAGS.identity_dim,
                                    directed_graph = FLAGS.directed,
                                    adj_in=adj_in_info, layer_in_infos=layer_in_infos,
                                    logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)

    # Init variables
    if not FLAGS.directed:
        sess.run(tf.global_variables_initializer(),\
            feed_dict={adj_info_ph: minibatch.adj, adj_weight_ph: minibatch.adj_weight})
    else:
        sess.run(tf.global_variables_initializer(),\
            feed_dict={adj_info_ph: minibatch.adj, adj_weight_ph: minibatch.adj_weight,\
            adj_in_info_ph: minibatch.adj_in, adj_in_weight_ph: minibatch.adj_in_weight})

    # Train model
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
            feed_dict, labels = minibatch.next_minibatch_feed_dict() # a {placeholder: value}
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            t = time.time()
            # Training step
            outs = sess.run([model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[1]
            if iter % FLAGS.validate_iter == 0:
                # Validation
                if not FLAGS.directed:
                    sess.run([val_adj_info.op, val_adj_weight.op])
                else:
                    sess.run([val_adj_info.op, val_adj_weight.op, val_adj_in_info.op, val_adj_in_weight.op])
                if FLAGS.validate_batch_size == -1:
                    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
                else:
                    val_cost, val_f1_mic, val_f1_mac, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
                if not FLAGS.directed:
                    sess.run([train_adj_info.op, train_adj_weight.op])
                else:
                    sess.run([train_adj_info.op, train_adj_weight.op, train_adj_in_info.op, train_adj_in_weight.op])
                epoch_val_costs[-1] += val_cost

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
    print("Optimization Finished!")
    if not FLAGS.directed:
        sess.run([val_adj_info.op, val_adj_weight])
    else:
        sess.run([val_adj_info.op, val_adj_weight, val_adj_in_info.op, val_adj_in_weight])

    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
    print("Full validation stats:",
                  "loss=", "{:.5f}".format(val_cost),
                  "f1_micro=", "{:.5f}".format(val_f1_mic),
                  "f1_macro=", "{:.5f}".format(val_f1_mac),
                  "time=", "{:.5f}".format(duration))

    val_cost, val_f1_mic, val_f1_mac, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)

    tf.reset_default_graph()
    return val_cost, val_f1_mic, val_f1_mac


def printCModel(model):
    a = model.labels_
    print('nodes number: ', len(a))
    for i in range(model.n_clusters):
        print('label {}: {}'.format(i, len(a) - np.count_nonzero(a-i)))


def sup_pg_main(argv=None):
    print('Loading training data...')
    train_data = utils.load_data_with_threadpool(FLAGS.train_prefix,directed=FLAGS.directed)
    utils.makeExists(FLAGS.base_log_dir)
    if FLAGS.cluster_model_path != '':
        print('Loading cluster data in {}'.format(FLAGS.cluster_model_path))
        CModel = joblib.load(FLAGS.cluster_model_path)
    else:
        path = FLAGS.train_prefix + '.skl'
        CModel = utils.run_cluster(train_data[1], FLAGS.n_clusters, FLAGS.cluster_max_iter, seed, eps = FLAGS.eps, min_samples = FLAGS.min_samples)
        joblib.dump(CModel, path)

    loss, f1_mic, f1_mac = train(train_data, CModel)
    print('Final test results: '
                "loss=", "{:.5f}".format(loss),
                "f1_micro=", "{:.5f}".format(f1_mic),
                "f1_macro=", "{:.5f}".format(f1_mac))

if __name__ == '__main__':
    tf.app.run(sup_pg_main)


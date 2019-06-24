import tensorflow as tf

import pge.models as models
import pge.layers as layers
from pge.aggregators import MeanAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/williamleif/GraphSAGE
# which is under an identical MIT license as PGE

class SupervisedPGE(models.SampleAndAggregate):
    """Implementation of supervised PGE."""

    def __init__(self, num_classes,
            placeholders, features, adj,degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False,
            identity_dim=0, directed_graph=False,
            adj_in = None, layer_in_infos = None,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of PGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See PGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs) # GeneralizedModel -> Model

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj # adjecent table
        self.adj_in_info = adj_in # adjecent table
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
			#self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim], trainable=False)
        else:
           self.embeds = None

        # === define embedding variable ===
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.constant(features, dtype=tf.float32)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.layer_in_infos = layer_in_infos
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        # === directed graph setting ===
        self.directed_graph=directed_graph  # directed condition
        self.directed_factor = 2 if self.directed_graph else 1
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        if self.directed_graph:
            self.dims.extend([layer_infos[i].output_dim + layer_in_infos[i].output_dim for i in range(len(layer_infos))])
        else:
            self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.build()

    def build(self):
        # samples1: all support nodes [Tensor(layer_support_size*batch_size)], len=num_layers
        # support_sizes1: support nodes number each layer [int]
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        if self.directed_graph: # directed capacity
            samples_in1, _ = self.sample(self.inputs1, self.layer_in_infos)
            self.outputs1, self.aggregators = self.diAggregate(samples1, samples_in1, [self.features], self.dims, num_samples,
                    support_sizes1, concat=self.concat, model_size=self.model_size)
        else:
            self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                    support_sizes1, concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        self.node_pred = layers.Dense(dim_mult*self.dims[-1]*self.directed_factor, self.num_classes,
                dropout=self.placeholders['dropout'],
                act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()


    def diAggregate(self, samples1, samples_in1, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=False, model_size="small"):
        """ at each layer, aggregate hidden representations of neighbors to compute the hidden representations
            at next layer.
        args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. length is the number of layers + 1. each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        returns:
            the hidden representation at the final layer for all nodes in batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden1 = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples1]
        hidden2 = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples_in1]
        hidden = []
        for h1,h2 in zip(hidden1,hidden2):
            hidden.append(tf.concat([h1, h2], 1))

        new_agg = aggregators is None
        if new_agg:
            aggregators = []

        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(self.directed_factor*dim_mult*dims[layer], dims[layer+1]*self.directed_factor, act=lambda x : x,
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size)
                else:
                    aggregator = self.aggregator_cls(self.directed_factor*dim_mult*dims[layer], dims[layer+1]*self.directed_factor,
                            dropout=self.placeholders['dropout'],
                            name=name, concat=concat, model_size=model_size)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1

                neigh_dims = [batch_size * support_sizes[hop],
                              num_samples[len(num_samples) - hop - 1],
                              self.directed_factor*dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0], aggregators

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))

        tf.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss:
            return tf.nn.sigmoid(self.node_preds)
        else:
            return tf.nn.softmax(self.node_preds)

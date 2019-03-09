from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf
from caloGraphNN import *
from ops.ties import *

class GravnetSegment(NetworkSegmentInterface):
    def build_network_segment(self, feat):
        x = feat
        feat_in = feat
        feat_in= high_dim_dense(feat_in, 32, activation=tf.nn.leaky_relu)
        for i in range(4):
            x = tf.layers.batch_normalization(x, momentum=0.8, training=self.training)
            x = layer_global_exchange(x)
            x = high_dim_dense(x, 96, activation=tf.nn.leaky_relu)
            x = high_dim_dense(x, 96, activation=tf.nn.leaky_relu)
            x = high_dim_dense(x, 96, activation=tf.nn.leaky_relu)

            x = layer_GravNet2(x,
                              n_neighbours=30,
                              n_dimensions=4,
                              n_filters=64,
                              n_propagate=64)

        x = tf.concat([x, feat_in], axis=-1)
        x = high_dim_dense(x, 128, activation=tf.nn.relu)
        x = high_dim_dense(x, 128, activation=tf.nn.relu)
        return x

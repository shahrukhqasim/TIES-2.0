from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf
from caloGraphNN import *
from ops.ties import *

class GarNetSegment(NetworkSegmentInterface):
    def build_network_segment(self, feat):
        x = feat
        for i in range(6):
            x = tf.layers.batch_normalization(x, momentum=0.8, training=self.training)
            x = layer_global_exchange(x)
            x = high_dim_dense(x, 128, activation=tf.nn.relu)
            x = high_dim_dense(x, 128, activation=tf.nn.relu)
            x = high_dim_dense(x, 128, activation=tf.nn.relu)

            x = layer_GarNet(x, n_aggregators=10, n_filters=128, n_propagate=128)


        x = high_dim_dense(x, 128, activation=tf.nn.relu)
        return high_dim_dense(x, 128, activation=tf.nn.relu)

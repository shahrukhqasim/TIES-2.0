from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf
from caloGraphNN import *
from ops.ties import *

class FcnnSegment(NetworkSegmentInterface):
    def build_network_segment(self, feat):
        feat = tf.layers.batch_normalization(feat, momentum=0.8, training=self.training)

        feat = high_dim_dense(feat, 196, activation=tf.nn.relu)
        feat = high_dim_dense(feat, 256, activation=tf.nn.relu)
        feat = high_dim_dense(feat, 256, activation=tf.nn.relu)
        feat = high_dim_dense(feat, 256, activation=tf.nn.relu)


        feat = high_dim_dense(feat, 128, activation=None)

        return feat

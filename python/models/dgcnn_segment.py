from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf
from caloGraphNN import *
from ops.ties import *


class DgcnnSegment(NetworkSegmentInterface):
    def build_network_segment(self, feat):
        feat = tf.layers.batch_normalization(feat, momentum=0.8, training=self.training)

        feat = high_dim_dense(feat, 64)  # global transform to 3D

        feat = edge_conv_layer(feat, 10, [64, 64, 64])
        feat_g = layer_global_exchange(feat)

        feat = tf.layers.dense(tf.concat([feat, feat_g], axis=-1),
                               64, activation=tf.nn.relu)

        feat1 = edge_conv_layer(feat, 10, [64, 64, 64])
        feat1_g = layer_global_exchange(feat1)
        feat1 = tf.layers.dense(tf.concat([feat1, feat1_g], axis=-1),
                                64, activation=tf.nn.relu)

        feat2 = edge_conv_layer(feat1, 10, [64, 64, 64])
        feat2_g = layer_global_exchange(feat2)
        feat2 = tf.layers.dense(tf.concat([feat2, feat2_g], axis=-1),
                                64, activation=tf.nn.relu)

        feat3 = edge_conv_layer(feat2, 10, [64, 64, 64])

        # global_feat = tf.layers.dense(feat2,1024,activation=tf.nn.relu)
        # global_feat = max_pool_on_last_dimensions(global_feat, skip_first_features=0, n_output_vertices=1)
        # print('global_feat',global_feat.shape)
        # global_feat = tf.tile(global_feat,[1,feat.shape[1],1])
        # print('global_feat',global_feat.shape)

        feat = tf.concat([feat, feat1, feat2, feat_g, feat1_g, feat2_g, feat3], axis=-1)
        feat = tf.layers.dense(feat, 128, activation=tf.nn.relu)
        feat = tf.layers.dense(feat, 128, activation=tf.nn.relu)
        return feat

from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf


class BasicConvSegment(NetworkSegmentInterface):
    def build_network_segment(self, inputs):
        conv_computation_graph = inputs
        conv_computation_graph = tf.image.resize_images(conv_computation_graph, size=(256,256))
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)
        conv_computation_graph = tf.layers.conv2d(conv_computation_graph, filters=10, kernel_size=3)

        return conv_computation_graph

from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf


class FastConvSegment(NetworkSegmentInterface):
    def build_network_segment(self, inputs):
        graph = inputs
        graph = tf.image.resize_images(graph, size=(256,256))
        # graph = tf.layers.conv2d(graph, filters=3, kernel_size=3, strides=3, padding='same', activation=None)
        graph = tf.layers.conv2d(graph, filters=16, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=16, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=32, kernel_size=3, activation=tf.nn.leaky_relu)

        graph = tf.layers.conv2d(graph, filters=32, kernel_size=2, strides=2, padding='same', activation=None)

        graph = tf.layers.conv2d(graph, filters=32, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=32, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=32, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=32, kernel_size=3, activation=tf.nn.leaky_relu)

        graph = tf.layers.conv2d(graph, filters=32, kernel_size=2, padding='same', activation=None)

        graph = tf.layers.conv2d(graph, filters=48, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=48, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=48, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=48, kernel_size=3, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=128, kernel_size=1, activation=tf.nn.leaky_relu)
        graph = tf.layers.conv2d(graph, filters=128, kernel_size=1, activation=tf.nn.leaky_relu)

        return graph

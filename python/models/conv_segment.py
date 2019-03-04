from models.network_segment_interface import NetworkSegmentInterface
import tensorflow as tf


class BasicConvSegment(NetworkSegmentInterface):
    def build_network_segment(self, inputs):
        _graph_from_image = inputs
        _graph_from_image = tf.image.resize_images(_graph_from_image, size=(256,256))
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)

        return _graph_from_image

import tensorflow as tf
from caloGraphNN import *



def gather_features_from_conv_head(conv_head, vertices_y, vertices_x, vertices_y2, vertices_x2, scale_y, scale_x):
    """
    Gather features from a 2D image.

    :param conv_head: The 2D conv head with shape [batch, height, width, channels]
    :param vertices_y: The y position of each of the vertex with shape [batch, max_vertices]
    :param vertices_x: The x position of each of the vertex with shape [batch, max_vertices]
    :param vertices_height: The height of each of the vertex with shape [batch, max_vertices]
    :param vertices_width: The width of each of the feature with shape [batch, max_vertices]
    :param scale_y: A scalar to show y_scale
    :param scale_x: A scalar to show x_scale
    :return: The gathered features with shape [batch, max_vertices, channels]
    """
    vertices_y = tf.cast(vertices_y, tf.float32) * scale_y
    vertices_x = tf.cast(vertices_x, tf.float32) * scale_x
    vertices_y2 = tf.cast(vertices_y2, tf.float32) * scale_y
    vertices_x2 = tf.cast(vertices_x2, tf.float32) * scale_x

    batch_size, max_vertices = vertices_y.shape
    batch_size, max_vertices = int(batch_size.value), int(max_vertices.value)

    batch_range = tf.range(0, batch_size, dtype=tf.float32)[..., tf.newaxis, tf.newaxis]
    batch_range =  tf.tile(batch_range, multiples=[1, max_vertices, 1])

    indexing_tensor = tf.concat((batch_range, ((vertices_y + vertices_y2 ) /2)[..., tf.newaxis], ((vertices_x + vertices_x2 ) /2)[..., tf.newaxis]), axis=-1)
    indexing_tensor = tf.cast(indexing_tensor, tf.int64)
    return tf.gather_nd(conv_head, indexing_tensor)


def edge_conv_layer(vertices_in, num_neighbors=30,
                          mpl_layers=[64, 64, 64],
                          aggregation_function=tf.reduce_max,
                          share_keyword=None,  # TBI,
                          edge_activation=None
                          ):
    trans_space = vertices_in
    indexing, _ = indexing_tensor(trans_space, num_neighbors)
    # change indexing to be not self-referential
    neighbour_space = tf.gather_nd(vertices_in, indexing)

    expanded_trans_space = tf.expand_dims(trans_space, axis=2)
    expanded_trans_space = tf.tile(expanded_trans_space, [1, 1, num_neighbors, 1])

    diff = expanded_trans_space - neighbour_space
    edge = tf.concat([expanded_trans_space, diff], axis=-1)

    for f in mpl_layers:
        edge = tf.layers.dense(edge, f, activation=tf.nn.relu)

    if edge_activation is not None:
        edge = edge_activation(edge)

    vertex_out = aggregation_function(edge, axis=2)

    return vertex_out
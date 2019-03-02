from models.model_interface import ModelInterface
from overrides import overrides
from libs.configuration_manager import ConfigurationManager as gconfig
import tensorflow as tf
from caloGraphNN import layer_GravNet, layer_global_exchange, layer_GarNet, high_dim_dense
from readers.image_words_reader import ImageWordsReader
from tensorflow.contrib import tpu


class BasicModel(ModelInterface):
    @overrides
    def initialize(self, training=True):
        self.max_vertices = gconfig.get_config_param("max_vertices", "int")
        self.num_vertex_features = gconfig.get_config_param("num_vertex_features", "int")
        self.image_height = gconfig.get_config_param("max_image_height", "int")
        self.image_width = gconfig.get_config_param("max_image_width", "int")
        self.max_words_len = gconfig.get_config_param("max_words_len", "int")
        self.num_batch = gconfig.get_config_param("batch_size", "int")
        self.num_global_features = gconfig.get_config_param("num_global_features", "int")
        self.image_channels = gconfig.get_config_param("image_channels", "int")
        self.dim_vertex_x_position = gconfig.get_config_param("dim_vertex_x_position", "int")
        self.dim_vertex_y_position = gconfig.get_config_param("dim_vertex_y_position", "int")
        self.dim_vertex_width = gconfig.get_config_param("dim_vertex_width", "int")
        self.dim_vertex_height = gconfig.get_config_param("dim_vertex_height", "int")

        self.dim_num_vertices = gconfig.get_config_param("dim_num_vertices", "int")
        self.samples_per_vertex = gconfig.get_config_param("samples_per_vertex", "int")
        self.variable_scope = gconfig.get_config_param("variable_scope", "str")
        self.training_files_list = gconfig.get_config_param("training_files_list", "str")
        self.validation_files_list = gconfig.get_config_param("test_files_list", "str")
        self.test_files_list = gconfig.get_config_param("validation_files_list", "str")
        self.learning_rate = gconfig.get_config_param("learning_rate", "float")

        self.training = training
        self.momentum = 0.6

        if training:
            self.validation_reader = ImageWordsReader(self.validation_files_list, self.num_global_features,
                                                      self.max_vertices, self.num_vertex_features,
                                                      self.image_height, self.image_width, self.image_channels,
                                                      self.max_words_len, self.num_batch)
            self.training_reader = ImageWordsReader(self.training_files_list, self.num_global_features,
                                                      self.max_vertices, self.num_vertex_features,
                                                      self.image_height, self.image_width, self.image_channels,
                                                      self.max_words_len, self.num_batch)
            self.training_feeds = self.training_reader.get_feeds()
            self.validation_feeds = self.validation_reader.get_feeds()
        else:
            self.testing_reader = ImageWordsReader(self.validation_files_list, self.num_global_features,
                                                      self.max_vertices, self.num_vertex_features,
                                                      self.image_height, self.image_width, self.image_channels,
                                                      self.max_words_len, self.num_batch)
            self.testing_feeds = self.testing_reader.get_feeds()

        self._make_tpu_graphs()



    def get_variable_scope(self):
        return self.variable_scope


    @staticmethod
    def _make_placeholders(feeds):
        _placeholder_image = feeds[1]
        _placeholder_vertex_features = feeds[0]
        _placeholder_global_features = feeds[2]
        _placeholder_cell_adj_matrix = feeds[3]
        _placeholder_row_adj_matrix = feeds[4]
        _placeholder_col_adj_matrix = feeds[5]

        return {
            "placeholder_image" : feeds[1],
            "placeholder_vertex_features" : feeds[0],
            "placeholder_global_features" : feeds[2],
            "placeholder_cell_adj_matrix" : feeds[3],
            "placeholder_row_adj_matrix" : feeds[4],
            "placeholder_col_adj_matrix" : feeds[5],
        }

    @staticmethod
    def _make_image_conv_net(image):
        _graph_from_image = image
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)

        return _graph_from_image

    # TODO: Move it to layers or something
    @staticmethod
    def _gather_from_image_features(image, vertices_y, vertices_x, vertices_height, vertices_width, scale_y, scale_x):
        """
        Gather features from a 2D image.

        :param image: The 2D image with shape [batch, height, width, channels]
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
        vertices_height = tf.cast(vertices_height, tf.float32) * scale_y
        vertices_width = tf.cast(vertices_width, tf.float32) * scale_x

        batch_range = tf.range(0, gconfig.get_config_param("batch_size", "int"), dtype=tf.float32)[..., tf.newaxis, tf.newaxis]
        batch_range =  tf.tile(batch_range, multiples=[1, gconfig.get_config_param("max_vertices", "int"), 1])

        indexing_tensor = tf.concat((batch_range, (vertices_y + vertices_height/2)[..., tf.newaxis], (vertices_x + vertices_width/2)[..., tf.newaxis]), axis=-1)
        indexing_tensor = tf.cast(indexing_tensor, tf.int64)
        return tf.gather_nd(image, indexing_tensor)

    @staticmethod
    def _make_the_graph_model(vertices_combined_features):
        x = vertices_combined_features
        for i in range(8):
            x = layer_global_exchange(x)
            x = high_dim_dense(x, 16, activation=tf.nn.tanh)
            x = high_dim_dense(x, 16, activation=tf.nn.tanh)
            x = high_dim_dense(x, 16, activation=tf.nn.tanh)

            x = layer_GravNet(x,
                              n_neighbours=40,
                              n_dimensions=4,
                              n_filters=42,
                              n_propagate=18)
            x = tf.layers.batch_normalization(x, momentum=0.6, training=True) # TODO: Fix training parameter

        x = high_dim_dense(x, 128, activation=tf.nn.relu)
        return high_dim_dense(x, 32, activation=tf.nn.relu)

    @staticmethod
    def _get_distribution_for_mote_carlo_sampling(self, placeholders):
        x = tf.ones(shape=(self.num_batch, self.max_vertices, self.max_vertices), dtype=tf.float32)
        x = x / placeholders['placeholder_global_features'][:, self.dim_num_vertices][..., tf.newaxis, tf.newaxis]
        mask = tf.sequence_mask(placeholders['placeholder_global_features'][:, self.dim_num_vertices], maxlen=self.max_vertices)\
                            [..., tf.newaxis] # [batch, num_vertices, 1] It will broadcast on the last dimension!

        return x * tf.cast(mask, tf.float32)

    @staticmethod
    def _do_monte_carlo_sampling(self, graph, gt_matrices):
        x = tf.distributions.Categorical(probs=BasicModel._get_distribution_for_mote_carlo_sampling(self, self.placeholders)).sample(sample_shape=self.samples_per_vertex)
        x = tf.transpose(x, perm=[1,2,0]) # [batch, max_vertices, samples_per_vertex]

        y = tf.range(0, self.num_batch)[...,tf.newaxis] # [batch, 1]
        y = tf.tile(y, multiples=[1, self.max_vertices]) # [batch, max_vertices]

        z = tf.range(0, self.max_vertices)[tf.newaxis, ..., tf.newaxis]
        z = tf.tile(z, multiples=[self.num_batch, 1, 1])

        batch_range_1 = tf.tile(y[..., tf.newaxis], multiples=[1, 1, self.samples_per_vertex])
        max_vertices_range_1 = tf.tile(z, multiples=[1, 1, self.samples_per_vertex])

        indexing_tensor_for_adj_matrices = tf.concat((batch_range_1[..., tf.newaxis],max_vertices_range_1[..., tf.newaxis],x[..., tf.newaxis]), axis=-1)
        indexing_tensor = tf.concat((batch_range_1[..., tf.newaxis], x[..., tf.newaxis]), axis=-1)

        x = tf.gather_nd(graph, indexing_tensor) # [ batch, num_vertices, samples_per_vertex, features]

        y = tf.expand_dims(graph, axis=2)
        y = tf.tile(y, multiples=[1, 1, self.samples_per_vertex, 1])

        x = tf.concat((x,y), axis=-1)

        truncated_matrices = []
        for y in gt_matrices:
            truncated_matrices.append(tf.gather_nd(y, indexing_tensor_for_adj_matrices))

        return x, truncated_matrices

    @staticmethod
    def _make_classification_model(self, classification_head):
        graph, truths = classification_head

        x = graph
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)

        x = tf.layers.dense(x, units=2, activation=tf.nn.relu)
        y = tf.layers.dense(x, units=2, activation=tf.nn.relu)
        z = tf.layers.dense(x, units=2, activation=tf.nn.relu)

        mask = tf.sequence_mask(self.placeholders['placeholder_global_features'][:, self.dim_num_vertices], maxlen=self.max_vertices)[..., tf.newaxis]
        mask = tf.cast(mask, dtype=tf.float32)
        loss_x = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[0], depth=2), logits=x) * mask
        loss_y = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[1], depth=2), logits=y) * mask
        loss_z = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[2], depth=2), logits=z) * mask

        total_loss = loss_x + loss_y + loss_z # [batch, max_vertices, samples_per_vertex]
        total_loss = tf.reduce_mean(total_loss, axis=-1)
        total_loss = tf.reduce_sum(total_loss) / tf.cast(self.placeholders['placeholder_global_features'][:, self.dim_num_vertices], tf.float32)
        total_loss = tf.reduce_mean(total_loss)
        self.loss = total_loss
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        return [self.loss, self.optimizer]

    @staticmethod
    def _make_model(arguments):
        feeds = arguments[:-1]
        self = arguments[-1]

        placeholders = BasicModel._make_placeholders(feeds)
        self.placeholders = placeholders
        _, image_height, image_width, _ = placeholders['placeholder_image'].shape
        conv_head = BasicModel._make_image_conv_net(placeholders['placeholder_image'])

        vertices_y = placeholders['placeholder_vertex_features'][:, :, gconfig.get_config_param("dim_vertex_y_position", "int")]
        vertices_x = placeholders['placeholder_vertex_features'][:, :, gconfig.get_config_param("dim_vertex_x_position", "int")]
        vertices_height = placeholders['placeholder_vertex_features'][:, :, gconfig.get_config_param("dim_vertex_height", "int")]
        vertices_width = placeholders['placeholder_vertex_features'][:, :, gconfig.get_config_param("dim_vertex_width", "int")]

        _, post_height, post_width, _ = conv_head.shape
        post_height, post_width, image_height, image_width =  float(post_height.value), float(post_width.value),\
                                                              float(image_height.value), float(image_width.value)

        scale_y = float(post_height) / float(image_height)
        scale_x = float(post_width) / float(image_width)
        gathered_image_features = BasicModel._gather_from_image_features(conv_head, vertices_y, vertices_x,
                                                                   vertices_height, vertices_width, scale_y, scale_x)
        vertices_combined_features = tf.concat((placeholders['placeholder_vertex_features'], gathered_image_features), axis=-1)
        graph_features = BasicModel._make_the_graph_model(vertices_combined_features)

        classification_head = BasicModel._do_monte_carlo_sampling(self, graph_features,
                                                                  [placeholders['placeholder_row_adj_matrix'],
                                                                   placeholders['placeholder_row_adj_matrix'],
                                                                   placeholders['placeholder_row_adj_matrix']])
        return BasicModel._make_classification_model(self, classification_head)


    def _make_tpu_graphs(self):

        # Only for testing:
        with tf.variable_scope(self.get_variable_scope()):
            x = list(self.training_feeds)
            x.append(self)
            self.tpu_graphs = BasicModel._make_model(x)

        # with tf.variable_scope(self.get_variable_scope()):
        #     self.tpu_graphs = tpu.rewrite(BasicModel._make_model, inputs=[self.training_feeds], infeed_queue=self)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.get_variable_scope()))

    @overrides
    def get_saver(self):
        return self.saver

    @overrides
    def run_training_iteration(self, sess, summary_writer, iteration_number):
        loss, _ = sess.run(self.tpu_graphs)
        print("Iteration %d - Loss %.4E", iteration_number, loss)

    @overrides
    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        # feeds = sess.run(self.validation_feeds)
        # feed_dict = {
        #     self._placeholder_vertex_features : feeds[0],
        #     self._placeholder_image : feeds[1],
        #     self._placeholder_global_features : feeds[2],
        #     self._placeholder_cell_adj_matrix : feeds[3],
        #     self._placeholder_row_adj_matrix : feeds[4],
        #     self._placeholder_col_adj_matrix : feeds[5],
        # }
        # loss = sess.run(self.tpu_graphs, feed_dict = feed_dict)
        # print("Iteration %d - Loss %.4E", iteration_number, loss)
        print("Failed to run validation iteration")
        pass

    @overrides
    def run_testing_iteration(self, sess, summary_writer, iteration_number):
        print("Failed to run testing iteration")
        pass


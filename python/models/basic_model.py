from models.model_interface import ModelInterface
from overrides import overrides
from libs.configuration_manager import ConfigurationManager as gconfig
import tensorflow as tf
from caloGraphNN import layer_GravNet, layer_global_exchange, layer_GarNet, high_dim_dense


class BasicModel(ModelInterface):
    def initialize(self, training=True):
        self._make_model()
        self.max_vertices = gconfig.get_config_param("max_vertices", "int")
        self.num_vertex_features = gconfig.get_config_param("num_vertex_features", "int")
        self.image_height = gconfig.get_config_param("image_height", "int")
        self.image_width = gconfig.get_config_param("image_width", "int")
        self.max_words_len = gconfig.get_config_param("max_words_len", "int")
        self.num_batch = gconfig.get_config_param("num_batch", "int")
        self.num_global_features = gconfig.get_config_param("num_global_features", "int")
        self.image_channels = gconfig.get_config_param("image_channels", "int")
        self.dim_vertex_x_position = gconfig.get_config_param("dim_vertex_x_position", "int")
        self.dim_vertex_y_position = gconfig.get_config_param("dim_vertex_y_position", "int")
        self.dim_vertex_width = gconfig.get_config_param("dim_vertex_width", "int")
        self.dim_vertex_height = gconfig.get_config_param("dim_vertex_height", "int")

        self.dim_num_vertices = gconfig.get_config_param("dim_num_vertices", "int")
        self.samples_per_vertex = gconfig.get_config_param("samples_per_vertex", "int")
        self.training = training
        self.momentum = 0.6


    def _make_placeholders(self):
        self._placeholder_image = tf.placeholder(dtype=tf.float32, shape=[self.num_batch, self.image_width,
                                                                            self.image_height, self.image_channels])
        self._placeholder_vertex_features = tf.placeholder(dtype=tf.float32, shape=[self.num_batch, self.max_vertices,
                                                                            self.num_vertex_features])
        self._placeholder_global_features = tf.placeholder(dtype=tf.float32, shape=[self.num_batch,
                                                                                    self.num_global_features])

    def _make_image_conv_net(self):
        _graph_from_image = self._placeholder_image
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)
        _graph_from_image = tf.layers.conv2d(_graph_from_image, filters=10, kernel_size=3)

        _, self.post_height, self.post_width, _ = _graph_from_image.shape
        self._image_features = _graph_from_image

    # TODO: Move it to layers or something
    def _gather_from_image_features(self, image, vertices_y, vertices_x, vertices_height, vertices_width, scale_y, scale_x):
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

        batch_range = tf.range(self.num_batch)[..., tf.newaxis, tf.newaxis]

        indexing_tensor = tf.concat((batch_range, (vertices_y + vertices_height/2)[..., tf.newaxis], (vertices_x + vertices_width/2)[..., tf.newaxis]), axis=-1)
        indexing_tensor = tf.cast(indexing_tensor, tf.float32)
        return tf.gather_nd(image, indexing_tensor)

    def _make_the_graph_model(self, vertices_combined_features):
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
            x = tf.layers.batch_normalization(x, momentum=self.momentum, training=self.training)

        x = high_dim_dense(x, 128, activation=tf.nn.relu)
        return high_dim_dense(x, 3, activation=tf.nn.relu)

    def _get_distribution_for_mote_carlo_sampling(self):
        x = tf.ones(shape=(self.num_batch, self.max_vertices, self.max_vertices), dtype=tf.float32)
        x = x / self._placeholder_global_features[:, self.dim_num_vertices][..., tf.newaxis, tf.newaxis]
        mask = tf.sequence_mask(self._placeholder_global_features[:, self.dim_num_vertices], maxlen=self.max_vertices)\
                            [..., tf.newaxis] # [batch, num_vertices, 1] It will broadcast on the last dimension!

        return x * tf.cast(mask, tf.float32)

    def _do_monte_carlo_sampling(self):
        x = tf.distributions.Categorical(probs=self._get_distribution_for_mote_carlo_sampling()).sample(sample_shape=self.samples_per_vertex)
        x = tf.transpose(x, perm=[1,2,0])
        return x

    def _make_model(self):
        self._make_placeholders()
        self._make_image_conv_net()
        vertices_y = self._placeholder_vertex_features[:, self.dim_vertex_x_position]
        vertices_x = self._placeholder_vertex_features[:, self.dim_vertex_y_position]
        vertices_height = self._placeholder_vertex_features[:, self.dim_vertex_height]
        vertices_width = self._placeholder_vertex_features[:, self.dim_vertex_width]
        scale_y = float(self.post_height) / float(self.image_height)
        scale_x = float(self.post_width) / float(self.image_width)
        gathered_image_features = self._gather_from_image_features(self._image_features, vertices_y, vertices_x,
                                                                   vertices_height, vertices_width, scale_y, scale_x)
        vertices_combined_features = tf.concat((self._placeholder_vertex_features, gathered_image_features), axis=-1)
        graph_features = self._make_the_graph_model(vertices_combined_features)
        # Use graph features to do the actual classification

    @overrides
    def get_saver(self):
        pass

    @overrides
    def run_training_iteration(self, sess, summary_writer, iteration_number):
        pass

    @overrides
    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        pass


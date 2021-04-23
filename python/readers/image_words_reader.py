import tensorflow as tf
import os
import glob


class ImageWordsReader:
    def __init__(self, files_list, len_global_features,num_max_vertices, num_data_dims, max_height, max_width,
                 num_image_channels, max_words_length, num_batch, repeat=True, shuffle_size=None):
        self.files_list = files_list
        self.repeat = repeat
        self.num_max_vertices = num_max_vertices
        self.num_data_dims = num_data_dims
        self.num_batch = num_batch
        self.shuffle_size = self.num_batch #* 3 if shuffle_size is None else shuffle_size
        self.max_width = max_width
        self.max_height = max_height
        self.max_word_length = max_words_length
        self.len_global_features=len_global_features
        self.num_image_channels = num_image_channels

    def _parse_function(self, example_proto):
        keys_to_features = {
            'image': tf.FixedLenFeature((self.max_height * self.max_width), tf.float32),
            'global_features': tf.FixedLenFeature((self.len_global_features), tf.float32),
            'vertex_features': tf.FixedLenFeature((self.num_max_vertices * self.num_data_dims), tf.float32),

            'adjacency_matrix_cells': tf.FixedLenFeature((self.num_max_vertices * self.num_max_vertices), tf.int64),
            'adjacency_matrix_rows': tf.FixedLenFeature((self.num_max_vertices * self.num_max_vertices), tf.int64),
            'adjacency_matrix_cols': tf.FixedLenFeature((self.num_max_vertices * self.num_max_vertices), tf.int64),
            'vertex_text': tf.FixedLenFeature((self.num_max_vertices * self.max_word_length), tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        return parsed_features['vertex_features'], parsed_features['vertex_text'], parsed_features['image'], \
               parsed_features['global_features'], parsed_features['adjacency_matrix_cells'], \
               parsed_features['adjacency_matrix_rows'], parsed_features['adjacency_matrix_cols']

    def get_feeds(self, shuffle=True):
        """
        Returns the feeds (data, num_entries)
        :param files_list:
        :param num_batch:
        :param num_max_entries:
        :param num_data_dims:
        :param repeat:
        :param shuffle_size:
        :return:
        """
        # print("Max height", self.max_height)
        # print("Max width", self.max_width)
        # print("Len global features", self.len_global_features)
        # print("Max vertices", self.num_max_vertices)
        # print("Data dims", self.num_data_dims)
        # print("Max word length", self.max_word_length)

        dataset = tf.data.TFRecordDataset(self.files_list, compression_type='GZIP')
        dataset = dataset.map(self._parse_function)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_size)
        dataset = dataset.repeat(None if self.repeat else 10000)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        vertex_features, vertex_text, image, global_features, adj_cells, adj_rows, adj_cols = iterator.get_next()

        vertex_features = tf.reshape(vertex_features, shape=(-1, self.num_max_vertices, self.num_data_dims))
        vertex_text = tf.reshape(vertex_text, shape=(self.num_max_vertices, self.max_word_length))
        image = tf.reshape(image, shape=(-1, self.max_height, self.max_width, self.num_image_channels))

        adj_cells = tf.reshape(adj_cells, shape=(-1, self.num_max_vertices, self.num_max_vertices))
        adj_cols = tf.reshape(adj_cols, shape=(-1, self.num_max_vertices, self.num_max_vertices))
        adj_rows = tf.reshape(adj_rows, shape=(-1, self.num_max_vertices, self.num_max_vertices))

        return vertex_features, image, global_features, adj_cells, adj_rows, adj_cols

import tensorflow as tf


class ImageWordsReader:
    def __init__(self, files_list, num_max_vertices, num_data_dims, max_height, max_width, max_words_length, num_batch,
                 repeat=True, shuffle_size=None):
        self.files_list = files_list
        self.repeat = repeat
        self.num_max_vertices = num_max_vertices
        self.num_data_dims = num_data_dims
        self.num_batch = num_batch
        self.shuffle_size = self.num_batch * 3 if self.shuffle_size is None else self.shuffle_size
        self.max_width = max_width
        self.max_height = max_height
        self.max_word_length = max_words_length

    def _parse_function(self, example_proto):
        keys_to_features = {
            'vertex_features': tf.FixedLenFeature((self.num_max_vertices, self.num_data_dims), tf.float32),
            'vertex_text': tf.FixedLenFeature((self.num_max_vertices, self.max_word_length), tf.uint8),
            'image': tf.FixedLenFeature((self.max_height, self.max_width), tf.uint8),
            'global_features': tf.FixedLenFeature((self.max_height, self.max_width), tf.uint8),
            'adjacency_matrix_cells': tf.FixedLenFeature((self.num_max_vertices, self.max_width), tf.int64),
            'adjacency_matrix_rows': tf.FixedLenFeature((self.num_max_vertices, self.num_max_vertices), tf.int64),
            'adjacency_matrix_cols': tf.FixedLenFeature((self.num_max_vertices, self.num_max_vertices), tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['vertex_features'], parsed_features['vertex_text'], parsed_features['image'],\
               parsed_features['global_features'], parsed_features['adjacency_matrix_cells'],\
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
        with open(self.files_list) as f:
            content = f.readlines()
        file_paths = [x.strip() for x in content]
        dataset = tf.data.TFRecordDataset(file_paths, compression_type='GZIP')
        dataset = dataset.map(self._parse_function)
        if shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.shuffle_size)
        dataset = dataset.repeat(None if self.repeat else 10000)
        dataset = dataset.batch(self.num_batch)
        iterator = dataset.make_one_shot_iterator()
        vertex_features, _, image , global_features, adj_cells, adj_rows_adj_cols = iterator.get_next()

        return vertex_features, image , global_features, adj_cells, adj_rows_adj_cols

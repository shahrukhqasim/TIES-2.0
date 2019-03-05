from models.model_interface import ModelInterface
from overrides import overrides
from libs.configuration_manager import ConfigurationManager as gconfig
import tensorflow as tf
from caloGraphNN import layer_GravNet, layer_global_exchange, layer_GarNet, high_dim_dense
from readers.image_words_reader import ImageWordsReader
from ops.ties import *
import os
import cv2
from models.conv_segment import BasicConvSegment
from libs.inference_output_streamer import InferenceOutputStreamer
from libs.visual_feedback_generator import VisualFeedbackGenerator
import random
import numpy as np

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
        self.dim_vertex_x2_position = gconfig.get_config_param("dim_vertex_x2_position", "int")
        self.dim_vertex_y2_position = gconfig.get_config_param("dim_vertex_y2_position", "int")

        self.dim_num_vertices = gconfig.get_config_param("dim_num_vertices", "int")
        self.samples_per_vertex = gconfig.get_config_param("samples_per_vertex", "int")
        self.variable_scope = gconfig.get_config_param("variable_scope", "str")
        self.training_files_list = gconfig.get_config_param("training_files_list", "str")
        self.validation_files_list = gconfig.get_config_param("test_files_list", "str")
        self.test_files_list = gconfig.get_config_param("validation_files_list", "str")
        self.learning_rate = gconfig.get_config_param("learning_rate", "float")

        self.visual_feedback_out_path = gconfig.get_config_param("visual_feedback_out_path", type="str")
        self.test_output_path = gconfig.get_config_param("test_out_path", type="str")
        self.tile_samples = True # False is not implemented


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
            self.visual_feedback_generator = VisualFeedbackGenerator(self.visual_feedback_out_path)
            self.visual_feedback_generator.start_thread()
        else:
            self.testing_reader = ImageWordsReader(self.validation_files_list, self.num_global_features,
                                                      self.max_vertices, self.num_vertex_features,
                                                      self.image_height, self.image_width, self.image_channels,
                                                      self.max_words_len, self.num_batch)
            self.testing_feeds = self.testing_reader.get_feeds()

            self.inference_output_streamer = InferenceOutputStreamer(self.test_output_path)
            self.inference_output_streamer.start_thread()

        self.conv_segment = BasicConvSegment()

        self.build_computation_graphs()



    def get_variable_scope(self):
        return self.variable_scope

    def make_placeholders(self):
        _placeholder_image = tf.placeholder(dtype=tf.float32, shape=[self.num_batch, self.image_height,
                                                                          self.image_width, self.image_channels])
        _placeholder_vertex_features = tf.placeholder(dtype=tf.float32, shape=[self.num_batch, self.max_vertices,
                                                                                    self.num_vertex_features])
        _placeholder_global_features = tf.placeholder(dtype=tf.float32, shape=[self.num_batch,
                                                                                    self.num_global_features])
        _placeholder_cell_adj_matrix = tf.placeholder(dtype=tf.int64, shape=[self.num_batch,
                                                                                  self.max_vertices,
                                                                                  self.max_vertices])
        _placeholder_row_adj_matrix = tf.placeholder(dtype=tf.int64, shape=[self.num_batch,
                                                                                 self.max_vertices,
                                                                                 self.max_vertices])
        _placeholder_col_adj_matrix = tf.placeholder(dtype=tf.int64, shape=[self.num_batch,
                                                                                 self.max_vertices,
                                                                                 self.max_vertices])

        self._placeholder_vertex_features = _placeholder_vertex_features,
        self._placeholder_image = _placeholder_image
        self._placeholder_global_features = _placeholder_global_features
        self._placeholder_cell_adj_matrix = _placeholder_cell_adj_matrix
        self._placeholder_row_adj_matrix = _placeholder_row_adj_matrix
        self._placeholder_col_adj_matrix = _placeholder_col_adj_matrix

        self.placeholders_dict =   {
            "placeholder_image" : _placeholder_image,
            "placeholder_vertex_features" : _placeholder_vertex_features,
            "placeholder_global_features" : _placeholder_global_features,
            "placeholder_cell_adj_matrix" : _placeholder_cell_adj_matrix,
            "placeholder_row_adj_matrix" : _placeholder_row_adj_matrix,
            "placeholder_col_adj_matrix" : _placeholder_col_adj_matrix,
        }

    def build_graph_segment(self, vertices_combined_features):
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

    def get_distribution_for_mote_carlo_sampling(self, placeholders):
        x = tf.ones(shape=(self.num_batch, self.max_vertices), dtype=tf.float32) # [batch, max_vertices]
        x = x / placeholders['placeholder_global_features'][:, self.dim_num_vertices][..., tf.newaxis]
        mask = tf.sequence_mask(placeholders['placeholder_global_features'][:, self.dim_num_vertices], maxlen=self.max_vertices) # [batch, num_vertices] It will broadcast on the last dimension!

        return x * tf.cast(mask, tf.float32)

    def do_monte_carlo_sampling(self, graph, gt_matrices):
        if self.training:
            if self.tile_samples:
                x = tf.distributions.Categorical(probs=self.get_distribution_for_mote_carlo_sampling(
                    self.placeholders_dict)).sample(sample_shape=(1, self.samples_per_vertex))  # [batch, max_vertices] = [1, samples, batch, max_vertices]
                x = tf.transpose(x, perm=[2,0,1]) # [batch, max_vertices, samples_per_vertex]
                x = tf.tile(x, multiples=[1, self.max_vertices, 1])
            else:
                x = tf.distributions.Categorical(probs=self.get_distribution_for_mote_carlo_sampling(self.placeholders_dict)).sample(sample_shape=(self.max_vertices, self.samples_per_vertex))
                x = tf.transpose(x, perm=[2,0,1]) # [batch, max_vertices, samples_per_vertex]
                print("Visual feedback not implemented for this method!")
                assert False
        else:
            # TODO: Could be  made faster since the subsequent gather operations can be avoided now
            x = tf.tile(tf.range(0, self.max_vertices)[tf.newaxis, :, tf.newaxis], multiples=[self.num_batch, 1,
                                                                                              self.samples_per_vertex])

        self.graph_sampled_indices = x

        y = tf.range(0, self.num_batch)[...,tf.newaxis] # [batch, 1]
        y = tf.tile(y, multiples=[1, self.max_vertices]) # [batch, max_vertices]

        z = tf.range(0, self.max_vertices)[tf.newaxis, ..., tf.newaxis] # [1, max_vertices, 1]
        z = tf.tile(z, multiples=[self.num_batch, 1, 1]) #[batch, max_vertices, 1]  # [batch, max_vertices, samples]

        batch_range_1 = tf.tile(y[..., tf.newaxis], multiples=[1, 1, self.samples_per_vertex]) # [batch, max_vertices, samples]
        max_vertices_range_1 = tf.tile(z, multiples=[1, 1, self.samples_per_vertex]) # [batch, max_vertices, samples]

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

    def reduce_mean_variable_vertices(self, x):
        x = tf.reduce_mean(x, axis=-1)
        x = tf.reduce_sum(x, axis=-1) / tf.cast(self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices], tf.float32)
        x = tf.reduce_mean(x)

        return x

    def build_classification_model(self, classification_head):
        graph, truths = classification_head


        x = graph
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
        x = tf.layers.dense(x, units=128, activation=tf.nn.relu)

        x = tf.layers.dense(x, units=2, activation=tf.nn.relu)
        y = tf.layers.dense(x, units=2, activation=tf.nn.relu)
        z = tf.layers.dense(x, units=2, activation=tf.nn.relu)

        mask = tf.sequence_mask(self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices], maxlen=self.max_vertices)[..., tf.newaxis]
        mask = tf.cast(mask, dtype=tf.float32)
        loss_x = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[0], depth=2), logits=x) * mask
        loss_y = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[1], depth=2), logits=y) * mask
        loss_z = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths[2], depth=2), logits=z) * mask

        self.predicted_cell_adj_matrix = tf.argmax(x, axis=-1)
        self.predicted_rows_adj_matrix = tf.argmax(y, axis=-1)
        self.predicted_cols_adj_matrix = tf.argmax(z, axis=-1)

        self.gt_cell_adj_matrix = truths[0]
        self.gt_rows_adj_matrix = truths[1]
        self.gt_cols_adj_matrix = truths[2]

        accuracy_x = tf.cast(tf.equal(self.predicted_cell_adj_matrix, truths[0]), tf.float32) * mask
        accuracy_y = tf.cast(tf.equal(self.predicted_rows_adj_matrix, truths[1]), tf.float32) * mask
        accuracy_z = tf.cast(tf.equal(self.predicted_cols_adj_matrix, truths[2]), tf.float32) * mask

        self.accuracy_x = self.reduce_mean_variable_vertices(accuracy_x)
        self.accuracy_y = self.reduce_mean_variable_vertices(accuracy_y)
        self.accuracy_z = self.reduce_mean_variable_vertices(accuracy_z)
        summary_stochastic_accuracy_x = tf.summary.scalar('validation_stochastic_accuracy_x', self.accuracy_x)
        summary_stochastic_accuracy_y = tf.summary.scalar('validation_stochastic_accuracy_y', self.accuracy_y)
        summary_stochastic_accuracy_z = tf.summary.scalar('validation_stochastic_accuracy_z', self.accuracy_z)

        training_summary_stochastic_accuracy_x = tf.summary.scalar('training_stochastic_accuracy_x', self.accuracy_x)
        training_summary_stochastic_accuracy_y = tf.summary.scalar('training_stochastic_accuracy_y', self.accuracy_y)
        training_summary_stochastic_accuracy_z = tf.summary.scalar('training_stochastic_accuracy_z', self.accuracy_z)


        total_loss = loss_x + loss_y + loss_z # [batch, max_vertices, samples_per_vertex]
        total_loss = tf.reduce_mean(total_loss, axis=-1)
        total_loss = tf.reduce_sum(total_loss, axis=-1) / tf.cast(self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices], tf.float32)
        total_loss = tf.reduce_mean(total_loss)


        validation_summary_loss = tf.summary.scalar('validation_loss', total_loss)
        training_summary_loss = tf.summary.scalar('training_loss', total_loss)

        self.summary_training = tf.summary.merge([training_summary_stochastic_accuracy_x,
                                                  training_summary_stochastic_accuracy_y,
                                                  training_summary_stochastic_accuracy_z,
                                                  training_summary_loss])
        self.summary_validation = tf.summary.merge([summary_stochastic_accuracy_x,
                                                  summary_stochastic_accuracy_y,
                                                  summary_stochastic_accuracy_z,
                                                  validation_summary_loss])

        loss = total_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
        return loss, optimizer


    def build_computation_graphs(self):
        with tf.variable_scope(self.get_variable_scope()):
            self.make_placeholders()
            placeholders = self.placeholders_dict

            _, image_height, image_width, _ = placeholders['placeholder_image'].shape
            conv_head = self.conv_segment.build_network_segment(placeholders['placeholder_image'])

            vertices_y = placeholders['placeholder_vertex_features'][:, :,
                         gconfig.get_config_param("dim_vertex_y_position", "int")]
            vertices_x = placeholders['placeholder_vertex_features'][:, :,
                         gconfig.get_config_param("dim_vertex_x_position", "int")]
            vertices_y2 = placeholders['placeholder_vertex_features'][:, :,
                          gconfig.get_config_param("dim_vertex_y2_position", "int")]
            vertices_x2 = placeholders['placeholder_vertex_features'][:, :,
                          gconfig.get_config_param("dim_vertex_x2_position", "int")]

            _, post_height, post_width, _ = conv_head.shape
            post_height, post_width, image_height, image_width = float(post_height.value), float(post_width.value), \
                                                                 float(image_height.value), float(image_width.value)

            scale_y = float(post_height) / float(image_height)
            scale_x = float(post_width) / float(image_width)
            gathered_image_features = gather_features_from_conv_head(conv_head, vertices_y, vertices_x,
                                                                     vertices_y2, vertices_x2, scale_y, scale_x)
            vertices_combined_features = tf.concat(
                (placeholders['placeholder_vertex_features'], gathered_image_features), axis=-1)
            graph_features = self.build_graph_segment(vertices_combined_features)

            classification_head = self.do_monte_carlo_sampling(graph_features,
                                                               [placeholders['placeholder_cell_adj_matrix'],
                                                                placeholders['placeholder_row_adj_matrix'],
                                                                placeholders['placeholder_col_adj_matrix']])
            self.loss, self.optimizer = self.build_classification_model(classification_head)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.get_variable_scope()))

    @overrides
    def get_saver(self):
        return self.saver

    @overrides
    def run_training_iteration(self, sess, summary_writer, iteration_number):
        feeds = sess.run(self.training_feeds)
        feed_dict = {
            self._placeholder_vertex_features : feeds[0],
            self._placeholder_image : feeds[1],
            self._placeholder_global_features : feeds[2],
            self._placeholder_cell_adj_matrix : feeds[3],
            self._placeholder_row_adj_matrix : feeds[4],
            self._placeholder_col_adj_matrix : feeds[5],
        }
        loss,_, summary_result = sess.run([self.loss, self.optimizer, self.summary_training], feed_dict = feed_dict)
        summary_writer.add_summary(summary_result, iteration_number)

        print("Iteration %d - Loss %.4E" % (iteration_number, loss))

    @overrides
    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        feeds = sess.run(self.validation_feeds)
        feed_dict = {
            self._placeholder_vertex_features : feeds[0],
            self._placeholder_image : feeds[1],
            self._placeholder_global_features : feeds[2],
            self._placeholder_cell_adj_matrix : feeds[3],
            self._placeholder_row_adj_matrix : feeds[4],
            self._placeholder_col_adj_matrix : feeds[5],
        }

        loss, summary_result, pcells, prows, pcols, sampled_indices, tcells, trows, tcols = sess.run([self.loss, self.summary_validation, self.predicted_cell_adj_matrix,
                                         self.predicted_rows_adj_matrix, self.predicted_cols_adj_matrix, self.graph_sampled_indices,
                                                                                self.gt_cell_adj_matrix,
                                                                                self.gt_rows_adj_matrix,
                                                                                self.gt_cols_adj_matrix
                                                                                ],
                                        feed_dict = feed_dict)
        summary_writer.add_summary(summary_result, iteration_number)


        data = {
            'image' : feeds[1][0],
            'sampled_ground_truths' : [tcells[0], trows[0], tcols[0]],
            'sampled_predictions' : [pcells[0], prows[0], pcols[0]],
            'sampled_indices' : sampled_indices[0],
            'global_features' : feeds[2][0],
            'vertex_features' : feeds[0][0],
        }
        self.visual_feedback_generator.add(iteration_number,  data)

        print("VALIDATION Iteration %d - Loss %.4E"  % (iteration_number, loss))

    @overrides
    def run_testing_iteration(self, sess, summary_writer, iteration_number):

        feeds = sess.run(self.testing_feeds)
        feed_dict = {
            self._placeholder_vertex_features : feeds[0],
            self._placeholder_image : feeds[1],
            self._placeholder_global_features : feeds[2],
            self._placeholder_cell_adj_matrix : feeds[3],
            self._placeholder_row_adj_matrix : feeds[4],
            self._placeholder_col_adj_matrix : feeds[5],
        }
        loss, pcells, prows, pcols, ax, ay, az = sess.run([self.loss, self.predicted_cell_adj_matrix,
                                       self.predicted_rows_adj_matrix, self.predicted_cols_adj_matrix, self.accuracy_x,
                                       self.accuracy_y, self.accuracy_z], feed_dict = feed_dict)


        # [0] is for test
        feed_output = {
            "vertex_features" : feeds[0][0],
            "image" : feeds[1][0],
            "global_features" : feeds[2][0],
            "gt_cells_adj_matrix" : feeds[3][0],
            "gt_rows_adj_matrix" : feeds[4][0],
            "gt_cols_adj_matrix" : feeds[5][0],
            "predicted_cells_adj_matrix" : pcells[0],
            "predicted_rows_adj_matrix" : prows[0],
            "predicted_cols_adj_matrix" : pcols[0]
        }

        print("Iteration %d - Loss %.4E %.3f %.3f %.3f" % (iteration_number, loss, ax, ay, az))

        self.inference_output_streamer.add(feed_output)

        print("TESTING Iteration %d - Loss %.4E"  % (iteration_number, loss))

    def sanity_preplot(self, sess, summary_writer):
        feeds = sess.run(self.validation_feeds)
        feeds = [x[0] for x in feeds]  # Pick the first back element
        image = feeds[1]
        vertex_features = feeds[0]
        global_features = feeds[2]
        cell_adj = feeds[3]
        row_adj = feeds[4]
        col_adj = feeds[5]
        num_vertices = int(global_features[self.dim_num_vertices])

        if self.image_channels==1:
            image = image[:,:,0]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        _image = image.copy()

        for i in range(num_vertices):
            x = vertex_features[i, 0]
            y = vertex_features[i, 1]
            x2 = vertex_features[i, 2]
            y2 = vertex_features[i, 3]

            cv2.rectangle(image, (x,y), (x2, y2), color=(255,0,0))

        num_sanity_per_matrix = 10

        def draw_sanity_adj(matrix, name):
            for i in range(num_sanity_per_matrix):
                image_x = _image.copy()
                x = random.randint(0, num_vertices)
                loc_me = (vertex_features[x, 0],vertex_features[x, 1]), (vertex_features[x, 2],vertex_features[x, 3])
                neighbors = np.argwhere(matrix[x]==1)

                for y in neighbors:
                    loc_him = (vertex_features[y, 0], vertex_features[y, 1]), (vertex_features[y, 2], vertex_features[y, 3])
                    cv2.rectangle(image_x, loc_him[0], loc_him[1], color=(255, 0, 0))

                cv2.rectangle(image_x, loc_me[0], loc_me[1], color=(0, 255, 0))
                cv2.imwrite(os.path.join(self.visual_feedback_out_path, 'sanity_%s_adj_%d.png' % (name, i)), image_x)

        draw_sanity_adj(cell_adj, 'cells')
        draw_sanity_adj(row_adj, 'rows')
        draw_sanity_adj(col_adj, 'cols')

        cv2.imwrite(os.path.join(self.visual_feedback_out_path, 'sanity_boxes.png'), image)




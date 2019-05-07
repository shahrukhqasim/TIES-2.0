from models.model_interface import ModelInterface
from overrides import overrides
from libs.configuration_manager import ConfigurationManager as gconfig
import tensorflow as tf
from caloGraphNN import *
from readers.image_words_reader import ImageWordsReader
from ops.ties import *
import os
import cv2
from models.conv_segment import BasicConvSegment
from models.dgcnn_segment import DgcnnSegment
from libs.inference_output_streamer import InferenceOutputStreamer
from libs.visual_feedback_generator import VisualFeedbackGenerator
import random
import numpy as np
from libs.helpers import *
import time


class BasicModel(ModelInterface):
    def set_conv_segment(self, conv_segment):
        self.conv_segment = conv_segment

    def set_graph_segment(self, dgcnn_segment):
        self.graph_segment = dgcnn_segment

    @overrides
    def initialize(self, training=True):
        self.max_vertices = gconfig.get_config_param("max_vertices", "int")
        if not training:
            gconfig.set_config_param("samples_per_vertex", str(self.max_vertices))

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

        self.is_sampling_balanced = gconfig.get_config_param("is_sampling_balanced", "bool")

        self.visual_feedback_out_path = gconfig.get_config_param("visual_feedback_out_path", type="str")
        self.test_output_path = gconfig.get_config_param("test_out_path", type="str")
        self.tile_samples = True # False is not implemented

        self.visualize_validation_results_after = gconfig.get_config_param("visualize_validation_results_after", type="int")


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
            self.testing_feeds = self.testing_reader.get_feeds(shuffle=False)

            self.inference_output_streamer = InferenceOutputStreamer(self.test_output_path)
            self.inference_output_streamer.start_thread()

        if not hasattr(self, 'conv_segment'):
            self.conv_segment = BasicConvSegment()

        if not hasattr(self, 'graph_segment'):
            self.graph_segment = DgcnnSegment()

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


    def get_distribution_for_mote_carlo_sampling(self, placeholders):
        x = tf.ones(shape=(self.num_batch, self.max_vertices), dtype=tf.float32) # [batch, max_vertices]
        x = x / placeholders['placeholder_global_features'][:, self.dim_num_vertices][..., tf.newaxis]
        mask = tf.sequence_mask(placeholders['placeholder_global_features'][:, self.dim_num_vertices], maxlen=self.max_vertices) # [batch, num_vertices] It will broadcast on the last dimension!

        return x * tf.cast(mask, tf.float32)

    def get_balanced_distribution_for_mote_carlo_sampling(self, ground_truth):
        N = self._placeholder_global_features[:, self.dim_num_vertices] # [b]
        NN = tf.tile(N[..., tf.newaxis], multiples=[1, self.max_vertices]) # [b]

        M = tf.sequence_mask(tf.cast(N, dtype=tf.int64), maxlen=self.max_vertices) # [b, v]
        M = tf.cast(M, dtype=tf.float32)
        MM = tf.cast(tf.sequence_mask(tf.cast(NN, dtype=tf.int64), maxlen=self.max_vertices), tf.float32)* M[...,tf.newaxis] #[b, v, v]

        P = tf.cast(ground_truth, dtype=tf.float32)
        X = tf.reduce_sum(P, axis=2)
        Y = tf.reduce_sum(P, axis=2)

        G_0 = tf.cast(tf.equal(ground_truth,0), tf.float32)
        G_1 = tf.cast(tf.equal(ground_truth,1), tf.float32)

        X = tf.reduce_sum(G_0*MM, axis=2)
        Y = tf.reduce_sum(G_1*MM, axis=2)

        P_0 = G_0 * 0.5 * ((X+Y)/X)[..., tf.newaxis] * MM
        P_1 = G_1 * 0.5 * ((X+Y)/Y)[..., tf.newaxis] * MM

        P = P_0 + P_1

        return P

    def do_monte_carlo_sampling(self, graph, gt_matrix):
        if self.training:
            if self.is_sampling_balanced:
                distribution = self.get_balanced_distribution_for_mote_carlo_sampling(gt_matrix)
                x = tf.distributions.Categorical(probs=distribution).sample(sample_shape=(self.samples_per_vertex))  # [batch, max_vertices] = [1, samples, batch, max_vertices]
                x = tf.transpose(x, perm=[1,2,0]) # [batch, max_vertices, samples_per_vertex]
            else:
                x = tf.distributions.Categorical(probs=self.get_distribution_for_mote_carlo_sampling(
                    self.placeholders_dict)).sample(sample_shape=(1, self.samples_per_vertex))  # [batch, max_vertices] = [1, samples, batch, max_vertices]
                x = tf.transpose(x, perm=[2,0,1]) # [batch, max_vertices, samples_per_vertex]
                x = tf.tile(x, multiples=[1, self.max_vertices, 1])
        else:
            # TODO: Could be  made faster since the subsequent gather operations can be avoided now
            x = tf.tile(tf.range(0, self.max_vertices)[tf.newaxis, tf.newaxis, :], multiples=[self.num_batch,
                                                                                              self.samples_per_vertex, 1])
        samples = x

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

        return samples, x, tf.gather_nd(gt_matrix, indexing_tensor_for_adj_matrices)

    def reduce_mean_variable_vertices(self, x):
        if self.training:
            x = tf.reduce_mean(x, axis=-1)
        else:
            placeholder_num_features = self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices]
            vv = tf.cast(tf.sequence_mask(placeholder_num_features, maxlen=self.max_vertices)[:, tf.newaxis, :], tf.float32)
            x =  x * vv
            x = tf.reduce_sum(x, axis=-1)
            x = x / tf.cast(self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices], tf.float32)
            # print(x.shape)
            # 0/0

        x = tf.reduce_sum(x, axis=-1) / tf.cast(self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices], tf.float32)
        x = tf.reduce_mean(x)
        return x

    def build_classification_model(self, classification_head):
        graph, truths = classification_head

        placeholder_num_features = self.placeholders_dict['placeholder_global_features'][:, self.dim_num_vertices]

        graph = tf.layers.batch_normalization(graph, momentum=0.8, training=self.training)
        net = graph
        net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
        net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
        net = tf.layers.dense(net, units=256, activation=tf.nn.relu)

        net = tf.layers.dense(net, units=2, activation=tf.nn.relu)

        mask = tf.sequence_mask(placeholder_num_features, maxlen=self.max_vertices)[..., tf.newaxis]
        mask = tf.cast(mask, dtype=tf.float32)

        predicted_adj_matrix = tf.argmax(net, axis=-1)

        classes_fraction =  tf.cast(truths, dtype=tf.float32)
        classes_fraction = tf.reduce_sum(tf.reduce_mean(classes_fraction, axis=-1) * mask[:,:,0], axis=-1) / tf.cast(placeholder_num_features, tf.float32)
        classes_fraction = tf.reduce_mean(classes_fraction)

        accuracy = tf.cast(tf.equal(predicted_adj_matrix, truths), tf.float32) * mask
        accuracy = self.reduce_mean_variable_vertices(accuracy)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(truths, depth=2), logits=net) * mask
        loss = tf.reduce_mean(loss, axis=-1)
        loss = tf.reduce_sum(loss, axis=-1) / tf.cast(placeholder_num_features, tf.float32)
        loss = tf.reduce_mean(loss)

        return_dict = {
            "accuracy" : accuracy,
            "loss" : loss,
            "classes_fraction" : classes_fraction,
            "predicted_adj_matrix" : predicted_adj_matrix,
        }

        return return_dict

    def build_classification_segments(self, graph_features, placeholders):
        keys = ['placeholder_cell_adj_matrix', 'placeholder_row_adj_matrix', 'placeholder_col_adj_matrix']

        gt_sampled_adj_matrices = []
        predicted_sampled_adj_matrices = []
        losses = []
        accuracies = []
        classes_fractions = []
        samples = []
        for type in range(3):
            gt_sampled_adj_matrix = placeholders[keys[type]]

            sampled_indices, computation_graph, gt_matrix = self.do_monte_carlo_sampling(graph_features, gt_sampled_adj_matrix)
            return_dict = self.build_classification_model((computation_graph,gt_matrix))
            predicted_sampled_adj_matrices.append(return_dict['predicted_adj_matrix'])
            losses.append(return_dict['loss'])
            accuracies.append(return_dict['accuracy'])
            classes_fractions.append(return_dict['classes_fraction'])
            gt_sampled_adj_matrices.append(gt_matrix)
            samples.append(sampled_indices)

        summary_training_loss_cells =  tf.summary.scalar('training_loss_cells', losses[0])
        summary_training_loss_rows =  tf.summary.scalar('training_loss_rows', losses[1])
        summary_training_loss_cols =  tf.summary.scalar('training_loss_cols', losses[2])

        summary_validation_loss_cells =  tf.summary.scalar('validation_loss_cells', losses[0])
        summary_validation_loss_rows =  tf.summary.scalar('validation_loss_rows', losses[1])
        summary_validation_loss_cols =  tf.summary.scalar('validation_loss_cols', losses[2])

        summary_training_accuracy_cells =  tf.summary.scalar('training_loss_cells', accuracies[0])
        summary_training_accuracy_rows =  tf.summary.scalar('training_loss_rows', accuracies[1])
        summary_training_accuracy_cols =  tf.summary.scalar('training_loss_cols', accuracies[2])

        summary_validation_accuracy_cells =  tf.summary.scalar('validation_loss_cells', accuracies[0])
        summary_validation_accuracy_rows =  tf.summary.scalar('validation_loss_rows', accuracies[1])
        summary_validation_accuracy_cols =  tf.summary.scalar('validation_loss_cols', accuracies[2])

        self.test_x = accuracies[2]

        self.graph_summaries_training = tf.summary.merge([summary_training_loss_cells, summary_training_loss_rows, summary_training_loss_cols,
                                      summary_training_accuracy_cells, summary_training_accuracy_rows, summary_training_accuracy_cols])

        self.graph_summaries_validation = tf.summary.merge([summary_validation_loss_cells, summary_validation_loss_rows, summary_validation_loss_cols,
                                      summary_validation_accuracy_cells, summary_validation_accuracy_rows, summary_validation_accuracy_cols])

        _alpha = gconfig.get_config_param("loss_alpha", "float")
        _beta = gconfig.get_config_param("loss_beta", "float")
        _gamma = gconfig.get_config_param("loss_gamma", "float")

        alpha = _alpha / (_alpha+_beta+_gamma)
        beta = _beta / (_alpha+_beta+_gamma)
        gamma = _gamma / (_alpha+_beta+_gamma)

        total_loss = alpha * losses[0] + beta * losses[1] + gamma * losses[2]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.graph_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        self.graph_prints = tf.print("Accuracy - cells:", accuracies[0], "rows:", accuracies[1], "cols:", accuracies[2], "\n",
                          "Loss     - cells:", losses[0], "rows:", losses[1], "cols:", losses[2], "\n",
                          "Fraction - cells:", classes_fractions[0], "rows:", classes_fractions[1], "cols:", classes_fractions[2], "\n",
                          "Total loss:", total_loss)

        self.graph_gt_sampled_adj_matrices = gt_sampled_adj_matrices
        self.graph_predicted_sampled_adj_matrices = predicted_sampled_adj_matrices
        self.graph_sampled_indices = samples


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

            _graph_vertex_features = placeholders['placeholder_vertex_features']
            vertices_combined_features = tf.concat(
                (_graph_vertex_features, gathered_image_features), axis=-1)

            self.graph_segment.training=self.training
            graph_features = self.graph_segment.build_network_segment(vertices_combined_features)

            self.build_classification_segments(graph_features, placeholders)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.get_variable_scope()))
        print("The model has", get_num_parameters(self.get_variable_scope()), "parameters.")

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
        print("Training Iteration %d:" % iteration_number)
        ops_to_run = self.graph_predicted_sampled_adj_matrices + self.graph_gt_sampled_adj_matrices + \
            self.graph_sampled_indices+ [self.graph_optimizer, self.graph_prints, self.graph_summaries_training]
        ops_result = sess.run(ops_to_run, feed_dict = feed_dict)

        summary_writer.add_summary(ops_result[-1], iteration_number)


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


        print("---------------------------------------------------")
        print("Validation Iteration %d:" % iteration_number)
        ops_to_run = self.graph_predicted_sampled_adj_matrices + self.graph_gt_sampled_adj_matrices + \
            self.graph_sampled_indices + [self.graph_prints, self.graph_summaries_training]
        ops_result = sess.run(ops_to_run, feed_dict = feed_dict)
        print("---------------------------------------------------")

        summary_writer.add_summary(ops_result[-1], iteration_number)

        data = {
            'image' : feeds[1][0],
            'sampled_ground_truths' : [ops_result[3][0], ops_result[4][0], ops_result[5][0]],
            'sampled_predictions' : [ops_result[0][0], ops_result[1][0], ops_result[2][0]],
            'sampled_indices' : [ops_result[6][0], ops_result[7][0], ops_result[8][0]],
            'global_features' : feeds[2][0],
            'vertex_features' : feeds[0][0],
        }

        if iteration_number % self.visualize_validation_results_after == 0:
            print("Visualizing")
            self.visual_feedback_generator.add(iteration_number,  data)


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


        print("Testing Iteration %d:" % iteration_number)
        start=time.time()
        ops_to_run = self.graph_predicted_sampled_adj_matrices + self.graph_gt_sampled_adj_matrices + \
            self.graph_sampled_indices + [self.graph_prints, self.test_x, self.graph_summaries_training]
        ops_result = sess.run(ops_to_run, feed_dict = feed_dict)
        print('\n\nTime taken:',time.time()-start)
        vv =  ops_result[-2]

        summary_writer.add_summary(ops_result[-1], iteration_number)

        result = {
            'image': feeds[1][0],
            'sampled_ground_truths': [ops_result[3][0], ops_result[4][0], ops_result[5][0]],
            'sampled_predictions': [ops_result[0][0], ops_result[1][0], ops_result[2][0]],
            'sampled_indices': [ops_result[6][0], ops_result[7][0], ops_result[8][0]],
            'global_features': feeds[2][0],
            'vertex_features': feeds[0][0],
        }

        # if vv==1:

        self.inference_output_streamer.add(result)

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

    def wrap_up(self):
        if self.training:
            pass
        else:
            self.inference_output_streamer.close()


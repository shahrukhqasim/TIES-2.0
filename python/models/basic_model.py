from models.model_interface import ModelInterface
from overrides import overrides
from libs.configuration_manager import ConfigurationManager as gconfig
import tensorflow as tf


class BasicModel(ModelInterface):
    def initialize(self):
        self._make_model()
        self.max_vertices = gconfig.get_config_param("max_vertices", "int")
        self.num_data_dims = gconfig.get_config_param("num_data_dims", "int")
        self.image_height = gconfig.get_config_param("image_height", "int")
        self.image_width = gconfig.get_config_param("image_width", "int")
        self.max_words_len = gconfig.get_config_param("max_words_len", "int")
        self.num_batch = gconfig.get_config_param("num_batch", "int")


    def _make_placeholders(self):
        self._placeholder_image = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.max_entries, self.n_space])


    def _make_model(self):
        self._make_placeholders()

    @overrides
    def get_saver(self):
        pass

    @overrides
    def run_training_iteration(self, sess, summary_writer, iteration_number):
        pass

    @overrides
    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        pass


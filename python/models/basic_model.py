from models.model_interface import ModelInterface
from overrides import overrides


class BasicModel(ModelInterface):
    def initialize(self):
        pass

    @overrides
    def get_saver(self):
        pass

    @overrides
    def run_training_iteration(self, sess, summary_writer, iteration_number):
        pass

    @overrides
    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        pass


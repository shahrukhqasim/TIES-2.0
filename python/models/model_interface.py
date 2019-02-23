




class ModelInterface:
    def initialize(self, training):
        raise Exception("Not implemented error")

    def get_saver(self):
        raise Exception("Not implemented error")

    def run_training_iteration(self, sess, summary_writer, iteration_number):
        raise Exception("Not implemented error")

    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        raise Exception("Not implemented error")

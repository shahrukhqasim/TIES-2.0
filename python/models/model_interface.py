




class ModelInterface:
    def _get_feed_dict_validation(self, iteration_number):
        pass

    def _get_fetches_validation(self, iteration_number):
        pass

    def _get_feed_dict(self, iteration_number):
        pass

    def _get_fetches(self, iteration_number):
        pass

    def _post_fetch(self, fetches_result, iteration_number):
        pass

    def _post_fetch_validation(self, fetches_result, iteration_number):
        pass

    def get_saver(self):
        pass

    def run_training_iteration(self, sess, summary_writer, iteration_number):
        pass

    def run_validation_iteration(self, sess, summary_writer, iteration_number):
        pass






class ModelInterface:
    def get_feed_dict_validation(self, iteration_number):
        pass

    def get_fetches_validation(self, iteration_number):
        pass

    def get_feed_dict(self, iteration_number):
        pass

    def get_fetches(self, iteration_number):
        pass

    def post_fetch(self, fetches_result, iteration_number):
        pass

    def post_fetch_validation(self, fetches_result, iteration_number):
        pass

    def get_saver(self):
        pass

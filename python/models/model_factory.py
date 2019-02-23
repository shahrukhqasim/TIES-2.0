from models.model_interface import ModelInterface



class ModelFactory:
    def get_model(self):
        return ModelInterface() # TODO: Fix this
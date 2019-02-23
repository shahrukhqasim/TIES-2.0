from models.model_interface import ModelInterface
from libs.configuration_manager import ConfigurationManager as gconfig
from models.basic_model import BasicModel


class ModelFactory:
    def get_model(self):
        model = gconfig.get_config_param("model", "str")
        if model == "basic_conv_graph":
            return BasicModel()
        else:
            return ModelInterface() # TODO: Fix this

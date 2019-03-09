from models.model_interface import ModelInterface
from libs.configuration_manager import ConfigurationManager as gconfig
from models.basic_model import BasicModel
from models.fast_conv_segment import FastConvSegment
from models.dgcnn_segment import DgcnnSegment
from models.garnet_segment import GarNetSegment
from models.fcnn_segment import FcnnSegment
from models.gravnet_segment import GravnetSegment

class ModelFactory:
    def get_model(self):
        model = gconfig.get_config_param("model", "str")
        if model == "basic_conv_graph":
            model = BasicModel()
        elif model == "conv_graph_dgcnn_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(DgcnnSegment())
        elif model == "conv_graph_garnet_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(GarNetSegment())
        elif model == "conv_fcnn_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(FcnnSegment())
        elif model == "conv_grav_net_fast_conv":
            model = BasicModel()
            model.set_conv_segment(FastConvSegment())
            model.set_graph_segment(GravnetSegment())
        else:
            return ModelInterface() # TODO: Fix this

        return model

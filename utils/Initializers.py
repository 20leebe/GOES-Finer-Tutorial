import torch
import numpy
from helpers import get_logger


LOGGER = get_logger(name = "Initializers", log_file="worldclim-dataset.log")

class Initializer:
    @staticmethod
    def siren_init(linear_layer, in_features, is_first=False, omega_0=30.0):

        LOGGER.debug("Initializing siren weights")

        with torch.no_grad():
            if is_first :
                LOGGER.debug("Initializing first weights")
                linear_layer.weight.uniform_(-1 / in_features, 1 / in_features)

            else:
                LOGGER.debug("Initializing hidden weights")
                bound = numpy.sqrt(6 / in_features) / omega_0
                linear_layer.weight.uniform_(-bound, bound)

    @staticmethod
    def finer_init(linear_layer, in_features, is_first=False, omega_0=30.0, c = 6.0, first_k=10, hidden_k=10):
        LOGGER.debug("Initializing finer weights")

        with torch.no_grad():
            if is_first :
                LOGGER.debug("Initializing first weights")
                linear_layer.weight.uniform_(-1 / in_features, 1 / in_features)
                linear_layer.bias.uniform_(-first_k, first_k)
            else:
                LOGGER.debug("Initializing hidden weights")
                bound = numpy.sqrt(c / in_features) / omega_0
                linear_layer.weight.uniform_(-bound, bound)
                linear_layer.bias.uniform_(-hidden_k, hidden_k)


    @staticmethod
    def gaussian_init(linear_layer, weight_init=0.1, bias_init=1.0, bias=True):
        LOGGER.debug("Initializing gaussian weights")

        with torch.no_grad():
            linear_layer.weight.uniform_(-weight_init, weight_init)
            if bias and bias_init is not None:
                linear_layer.bias.uniform_(-bias_init, bias_init)

    @staticmethod
    def gaussian_finer_init(linear_layer, weight_init=2.0, k=10):
        with torch.no_grad():
            LOGGER.debug("Initializing hidden weights")
            linear_layer.weight.uniform_(-weight_init, weight_init)
            linear_layer.bias.uniform_(-k, k)

    @staticmethod
    def mlp_init(linear_layer, weight_init, bias_init, bias):
        with torch.no_grad():
            linear_layer.weight.uniform_(-weight_init, weight_init)
            if bias and bias_init:
                linear_layer.bias.uniform_(-bias_init, bias_init)


    @staticmethod
    def wire_init(linear_layer, weight_init, bias_init, bias):
        LOGGER.debug("Initializing wire weights")

        with torch.no_grad():
            linear_layer.weight.uniform_(-weight_init, weight_init)
            if bias and bias_init is not None:
                linear_layer.bias.uniform_(-bias_init, bias_init)


    @staticmethod
    def wire_finer_init():
        pass

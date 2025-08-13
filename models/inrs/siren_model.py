# define siren layer
# define siren model
# define logic for residual layer

import torch
import torch.nn as nn
import numpy
from helpers import get_logger
from utils.activations import siren_activation
from utils.Initializers import Initializer


LOGGER = get_logger(name = "SirenModel", log_file="worldclim-dataset.log")

class SirenLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        """
        Initializes the SirenLayer with the given parameters.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            is_first (bool, optional): Indicates if this is the first layer. Defaults to False.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.
        """
        super().__init__()
        LOGGER.info("Siren Layer")
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.out_features = out_features
        self.activation = siren_activation(self.omega_0)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        Initializer.siren_init(self.linear, in_features, is_first, omega_0)




    def forward(self, x):
        """
        Applies the SirenLayer to the input tensor.

        This applies the linear layer, then applies the siren activation function to the output.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SirenLayer.
        """
        LOGGER.info("SIREN LAYER")
        # Apply the linear layer
        out = self.linear(x)
        # Apply the siren activation function
        return self.activation(out)


class SirenResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30., is_first=False):
        """
        Initializes the SirenResidualLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.
        """
        super().__init__()
        LOGGER.info("SIREN RESIDUAL LAYER")
        self.omega_0 = omega_0
        self.in_features = in_features
        self.out_features = out_features
        self.activation = siren_activation(self.omega_0)

        # Define a linear layer
        self.first_linear = nn.Linear(in_features, out_features, bias=bias)
        self.last_linear = nn.Linear(out_features, out_features, bias=bias)
        # Initialize weights
        Initializer.siren_init(self.first_linear, in_features, is_first, omega_0)
        Initializer.siren_init(self.last_linear, in_features, is_first, omega_0)



    def forward(self, x):
        """
        Perform the forward pass for the SirenResidualLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the SirenResidualLayer with residual connection.
        """
        LOGGER.info("SIREN RESIDUAL LAYER")
        # Apply the linear layer and siren activation, then add the input tensor for residual connection
        out = self.first_linear(x)
        first_activation = self.activation(out)

        out = self.last_linear(first_activation)
        out = out + x

        return self.activation(out)


class SirenModel(nn.Module):
    def __init__(self, in_features=2, out_features=2, hidden_layers=5, hidden_features=128, bias=True, final_bias=False, first_omega_0=30., hidden_omega_0=30., residual_net=False):
        """
        Initializes a SirenModel with the given parameters.

        Args:
            in_features (int, optional): The number of input features. Defaults to 2.
            out_features (int, optional): The number of output features. Defaults to 2.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 5.
            hidden_features (int, optional): The number of features in the hidden layers. Defaults to 128.
            first_omega_0 (float, optional): The frequency scaling factor for the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): The frequency scaling factor for the hidden layers. Defaults to 30.
            residual_net (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super().__init__()
        LOGGER.info("SIREN MODEL")
        self.net = []
        # build the first layer
        self.net.append(SirenLayer(in_features, hidden_features, bias=bias,is_first=True, omega_0=first_omega_0))

        # build the hidden layers
        for i in range(hidden_layers):
            if residual_net:
                # build a residual layer
                LOGGER.info(f"SIREN RESIDUAL LAYER {i}")
                self.net.append(SirenResidualLayer(hidden_features, hidden_features, bias=bias, omega_0=hidden_omega_0))
            else:
                # build a normal layer
                LOGGER.info(f"SIREN LAYER {i}")
                self.net.append(SirenLayer(hidden_features, hidden_features, bias=bias, is_first=False, omega_0=hidden_omega_0))

        # build the final layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            final_layer.weight.uniform_(-numpy.sqrt(6/hidden_features)/hidden_omega_0, numpy.sqrt(6/hidden_features)/hidden_omega_0)
        self.net.append(final_layer)
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """
        Execute the forward pass through the SirenModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the model's network.
        """
        LOGGER.info("FORWARDING SIREN MODEL")
        # Forward the input tensor through the network
        return self.net(x)


# ##### FINER LAYERS


class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.0, is_first=False, is_last=False, first_k=10, hidden_k=10):
        """
        Initializes a layer with configurable features and biases.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layer. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.0.
            is_first (bool, optional): Indicates if this is the first layer. Defaults to False.
            is_last (bool, optional): Indicates if this is the last layer. Defaults to False.
            first_bias (float, optional): The bias for the first layer. Defaults to None.
            hidden_bias (float, optional): The bias for hidden layers. Defaults to None.
        """
        super().__init__()

        # Assign input and output features
        self.in_features = in_features
        self.out_features = out_features
        # Frequency scaling factor
        self.omega_0 = omega_0
        # Flags for first and last layers
        self.is_first = is_first
        self.is_last = is_last
        # Bias values
        self.first_k = first_k
        self.hidden_k = hidden_k
        self.activation = siren_activation(self.omega_0, with_finer=True)

        # Define a linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Initialize weights
        Initializer.finer_init(self.linear, in_features, is_first=self.is_first, omega_0=self.omega_0, first_k=self.first_k, hidden_k=self.hidden_k)

    def forward(self, x):
        """
        Applies the FinerLayer to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the FinerLayer.
        """
        LOGGER.debug("FINER LAYER")
        # Apply the linear layer
        out = self.linear(x)
        # Apply the siren activation function
        return self.activation(out)

class FinerResidualLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega_0=30.0, hidden_k=10):
        """
        Initializes a FinerResidualLayer.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            omega_0 (float, optional): The frequency scaling factor. Defaults to 30.0.
            hidden_bias (float, optional): The bias for the hidden layers. Defaults to None.
        """
        super().__init__()

        LOGGER.info("BUILDING FINER RESIDUAL LAYER")
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.hidden_k = hidden_k
        self.activation = siren_activation(self.omega_0, with_finer=True)

        self.first_linear = nn.Linear(in_features, out_features, bias=bias)
        self.last_linear = nn.Linear(out_features, out_features, bias=bias)
        Initializer.finer_init(self.first_linear, in_features, is_first=False, omega_0=self.omega_0, hidden_k=self.hidden_k)
        Initializer.finer_init(self.last_linear, out_features, is_first=False, omega_0=self.omega_0, hidden_k=self.hidden_k)

    def forward(self, x):
        """
        Forward passes through the FinerResidualLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the FinerResidualLayer.
        """
        LOGGER.debug("FINER RESIDUAL LAYER")
        out = self.first_linear(x)
        first_activation = self.activation(out)
        out = self.last_linear(first_activation)
        out = out + x
        return self.activation(out)

class FinerModel(nn.Module):
    def __init__(self, in_features=2, out_features=2, bias=True, final_bias=False, hidden_layers=5, hidden_features=128, first_omega_0=30., hidden_omega_0=30.,
                 first_k=None, hidden_k=None, residual_net=False):
        """
        Initializes the FinerModel.

        Args:
            in_features (int, optional): The number of input features. Defaults to 2.
            out_features (int, optional): The number of output features. Defaults to 2.
            bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
            hidden_layers (int, optional): The number of hidden layers. Defaults to 5.
            hidden_features (int, optional): The number of features in the hidden layers. Defaults to 128.
            first_omega_0 (float, optional): The frequency scaling factor for the first layer. Defaults to 30.
            hidden_omega_0 (float, optional): The frequency scaling factor for the hidden layers. Defaults to 30.
            first_bias (float, optional): The maximum absolute value of the bias for the first layer. Defaults to None.
            hidden_bias (float, optional): The maximum absolute value of the bias for the hidden layers. Defaults to None.
            residual_net (bool, optional): Whether to use residual connections. Defaults to False.
        """
        super().__init__()

        LOGGER.info("INITIALIZING FINER MODEL")
        assert bias, "Bias must be True for FinerModel"
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, bias=bias, is_first=True, omega_0=first_omega_0, first_k=first_k))

        for i in range(hidden_layers):
            # build the hidden layers
            LOGGER.info("BUILDING FINER MODEL")
            if residual_net:
                LOGGER.info(f"RESIDUAL LAYER {i}")
                self.net.append(FinerResidualLayer(hidden_features, hidden_features, bias=True, omega_0=hidden_omega_0, hidden_k=hidden_k))
            else:
                LOGGER.info(f"NORMAL LAYER {i}")
                self.net.append(FinerLayer(hidden_features, hidden_features, bias=True, is_first=False, omega_0=hidden_omega_0, hidden_k=hidden_k))

        # build the final layer
        final_layer = nn.Linear(hidden_features, out_features, bias=final_bias)
        with torch.no_grad():
            final_layer.weight.uniform_(-numpy.sqrt(6/hidden_features)/hidden_omega_0, numpy.sqrt(6/hidden_features)/hidden_omega_0)
        self.net.append(final_layer)
        self.net = nn.Sequential(*self.net)


    def forward(self, x):
        """
        Forward pass through the FinerModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        LOGGER.debug("FORWARDING FINER MODEL")
        return self.net(x)

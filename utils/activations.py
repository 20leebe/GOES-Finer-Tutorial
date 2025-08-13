# define activation functions for implicit neural representations

import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import get_logger

LOGGER = get_logger(name = "Activations", log_file="worldclim-dataset.log")


class siren_activation(nn.Module):
    def __init__(self, omega_f=1, with_finer=False):
        """
        Initializes the activation function with the given parameters.

        Args:
            omega (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the frequency scaling factor
        self.omega_f = omega_f
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the siren activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the siren activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        LOGGER.info("APPLYING SIREN ACTIVATION FUNCTION")
        if self.with_finer:
            LOGGER.debug("Applying finer adjustments")
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            mod_x = self.omega_f * alpha * x
        else:
            LOGGER.debug("Applying SIREN adjustments")
            # Modulate x with omega
            mod_x = self.omega_f * x

        LOGGER.debug("Applying sine function to modulated x")
        # Apply sine function to the modulated x
        return torch.sin(mod_x)


class gaussian_activation(nn.Module):
    def __init__(self, scale, with_finer=False, omega_f=2.5):
        """
        Initializes the activation function with the given parameters.

        Args:
            scale (float): The scaling factor for the activation function.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
            omega_f (float, optional): The frequency scaling factor. Defaults to 1.
        """
        super().__init__()
        # Set the scaling factor for the activation function
        self.scale = scale
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer
        # Set the frequency scaling factor
        self.omega_f = omega_f

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the Gaussian activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the Gaussian activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the Gaussian activation.
        """

        LOGGER.info("APPLYING Gaussian ACTIVATION FUNCTION")
        if self.with_finer:
            LOGGER.debug("Applying finer adjustments")
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            x = torch.sin(self.omega_f * alpha * x)

            LOGGER.debug("Applying Gaussian function to finer modulated x")
            # Apply the Gaussian activation function
            return torch.exp(-(self.scale/self.omega_f * x)**2)
        else:
            LOGGER.debug("Applying Gaussian function to modulated x")
            # Apply the Gaussian activation function
            return torch.exp(-(self.scale * x)**2)


class wire_activation(nn.Module):
    def __init__(self, scale, omega_w, with_finer=False, omega_f=2.5):
        """
        Initializes the wire activation function with the given parameters.

        Args:
            scale (float): The scaling factor for the wire activation function.
            omega_w (float): The frequency scaling factor for the wire activation function.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
            omega_f (float, optional): The frequency scaling factor for the finer adjustments. Defaults to 1.
        """
        super().__init__()
        self.scale = scale
        self.omega_w= omega_w
        self.with_finer = with_finer
        self.omega_f = omega_f

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            return torch.abs(x) + 1

    @staticmethod
    def generate_complex_alpha(x):
        """
        Generates the complex alpha value for the wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The complex alpha value.
        """
        with torch.no_grad():
            # Calculate the absolute value of the real and imaginary parts of x
            alpha_real = torch.abs(x.real) + 1
            alpha_imag = torch.abs(x.imag) + 1

        # Scale the real and imaginary parts of x by the corresponding alpha values
        real_scaled = x.real * alpha_real
        imag_scaled = x.imag * alpha_imag

        # Combine the scaled real and imaginary parts to form the complex alpha value
        return torch.complex(real_scaled, imag_scaled)

    def forward(self, x):
        """
        Applies the wire activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard wire activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the wire activation.
        """

        LOGGER.info("APPLYING WIRE ACTIVATION FUNCTION")
        if self.with_finer:
            LOGGER.debug("Applying finer adjustments")
            if x.is_complex():
                LOGGER.debug("Applying complex finer adjustments")
                # Generate the complex alpha value for finer adjustments
                alpha = self.generate_complex_alpha(x)
                # Modulate x with omega and complex alpha

                LOGGER.debug("Applying complex sine function")
                modulated_x = torch.sin(self.omega_f * alpha)
            else:
                LOGGER.debug("Applying real finer adjustments")
                # Generate the alpha value for finer adjustments
                alpha = self.generate_alpha(x)
                # Modulate x with omega and alpha
                modulated_x = torch.sin(self.omega_f * alpha * x)

            LOGGER.debug("Applying finer modulated wire activation function")
            # Apply the wire activation function with modulation
            return torch.exp((1j * self.omega_w / self.omega_f * modulated_x) -
                             torch.abs(self.scale / self.omega_f * modulated_x)**2)
        else:
            # Apply the standard wire activation function
            LOGGER.debug("Standard wire modulation")
            modulated_x = x

        LOGGER.debug("Applying standard wire activation function")
        # Apply the exponential function to the modulated output
        return torch.exp((1j * self.omega_w * modulated_x) -
                         torch.abs(self.scale * modulated_x)**2)





# ######################### UNSURE IF IMPLEMENTED CORRECTLY ##########################

class hosc_activation(nn.Module):
    def __init__(self, beta, omega_f=1, with_finer=False):
        """
        Initializes the hosc_activation function with the given parameters.

        Args:
            beta (float): The scaling factor for the tanh function.
            omega_f (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the scaling factor for the tanh function
        self.beta = beta
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer
        # Set the frequency scaling factor
        self.omega_f = omega_f

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the hosc_activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the hosc_activation function to the input tensor.

        If `with_finer` is True, then the function applies finer adjustments to the input tensor.
        Otherwise, it applies the standard sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        if self.with_finer:
            # Generate the alpha value for the finer adjustments
            alpha = self.generate_alpha(x)
            # Apply the finer adjustments
            mod_x = torch.sin(self.omega * alpha * x)
            return torch.tanh(self.beta / self.omega_f * mod_x)
        else:
            # Apply the standard sinc activation function
            mod_x = torch.sin(self.omega * x)
            # Apply the tanh function with the given scaling factor
            return torch.tanh(self.beta * mod_x)


class sinc_activation(nn.Module):
    def __init__(self, omega_f=1, with_finer=False):
        """
        Initializes the activation function with the given parameters.

        Args:
            omega (float, optional): The frequency scaling factor. Defaults to 1.
            with_finer (bool, optional): Whether to apply finer adjustments. Defaults to False.
        """
        super().__init__()
        # Set the frequency scaling factor
        self.omega_f = omega_f
        # Determine if finer adjustments should be applied
        self.with_finer = with_finer

    @staticmethod
    def generate_alpha(x):
        """
        Generates the alpha value for the sinc activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The alpha value.
        """
        with torch.no_grad():
            # The formula for alpha is |x| + 1
            return torch.abs(x) + 1

    def forward(self, x):
        """
        Applies the sinc activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the sinc activation.
        """
        if self.with_finer:
            # Generate alpha for finer adjustments
            alpha = self.generate_alpha(x)
            # Modulate x with omega and alpha
            mod_x = self.omega_f * alpha * x
        else:
            # Modulate x with omega
            mod_x = self.omega_f * x
        # Apply sinc function to the modulated x
        return torch.sinc(mod_x)

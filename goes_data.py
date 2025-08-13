import torch
from torch.utils.data import Dataset
from utils.utils import xarray_to_tensor, normalize_tensor, denormalize_tensor

class GoesData(Dataset):
    
    def __init__(self, config_file):
        
        coords, tensor = xarray_to_tensor(config_file)
        self.coords = coords
        self.tensor = tensor
        self.tensor_min = None
        self.tensor_max = None
        self.coords_min = None
        self.coords_max = None
        
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.tensor[idx]
    
    def normalize(self):
        """
        Normalizes self.tensor to [-1, 1] range per feature.
        Stores min and max values for later denormalization.
        """
        self.tensor, self.tensor_min, self.tensor_max = normalize_tensor(self.tensor)
        self.coords, self.coords_min, self.coords_max = normalize_tensor(self.coords)
    
    def denormalize(self, normalized_tensor):
        """
        De-normalizes a tensor from [-1, 1] back to original scale.
        
        Args:
            normalized_tensor (torch.Tensor): Tensor to be denormalized.

        Returns:
            torch.Tensor: Original-scale tensor.
        """
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Data must be normalized before it can be denormalized.")
        return denormalize_tensor(normalized_tensor, self.min_vals, self.max_vals)

        

    


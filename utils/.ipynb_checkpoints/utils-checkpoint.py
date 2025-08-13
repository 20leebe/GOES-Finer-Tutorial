import yaml
import torch
from train import train
import xarray as xr
import torch.nn.functional as F
import matplotlib.pyplot as plt


def normalize_tensor(tensor):
    """
    Normalizes the tensor to be between -1 and 1 for each column value independently
    Args:
        tensor: The tensor to be normalized
       
    Returns:
     torch.Tensor The tensor where each column is scaled to have values between 0 and 1
    
    """
    min_vals = tensor.min(dim=0, keepdim=True).values
    max_vals = tensor.max(dim=0, keepdim=True).values
    normalized = 2 * (tensor - min_vals) / (max_vals - min_vals + 1e-8) - 1
    return normalized, min_vals, max_vals 

def denormalize_tensor(normalized_tensor, min_vals, max_vals):
    """
    De-normalizes a tensor from the [-1, 1] range back to the original scale.
    
    Args:
        normalized_tensor (torch.Tensor): The normalized tensor in the [-1, 1] range.
        min_vals (torch.Tensor): The original minimum values per column (shape: [1, num_features]).
        max_vals (torch.Tensor): The original maximum values per column (shape: [1, num_features]).

    Returns:
        torch.Tensor: The tensor rescaled back to the original data range.
    """
    return 0.5 * (normalized_tensor + 1) * (max_vals - min_vals + 1e-8) + min_vals

def create_coords(lat, lon):
    """
    Converts the latitude and and longitude into pytorch tensors then creates a mesh grid that pairs every latitude value with every longitude value
    Args:
        lat: A list of latitude values
        lon: A list of longitude values
        
    Returns:
    torch.Tensor: A 2D tensor of shape (num_lat * num_lon, 2), where each row is a [lat, lon] pair.
    """
    coords = torch.stack(torch.meshgrid(
        torch.tensor(lat), torch.tensor(lon), indexing="ij"), dim=-1)
    return coords.view(-1, 2)


def create_tensor_from_dataset(dataset, bands):
    """
    Converts selected bands from an xarray dataset into a 2D PyTorch tensor.

    For each specified band, the function extracts the data values, stacks them into a single tensor 
    with shape (lat, lon, B), and then flattens it into shape (lat * lon, B), where B is the number of bands.

    Args:
        dataset (xarray.Dataset): The dataset containing the spatial band data.
        bands (list of str): List of band names to extract from the dataset.

    Returns:
        torch.Tensor: A 2D tensor of shape (num_pixels, num_bands), where each row corresponds to a spatial pixel 
                      and each column to a band value.
    """
    values = [torch.tensor(dataset[band].values) for band in bands]
    tensor = torch.stack(values, -1)  # shape: (lat, lon, B)
    return tensor.view(-1, len(bands))


def filter_valid(coords, tensor):
    """
    Filters the data to remove any tensor value that has nan and removes the coordinates associated with the tensor values
    
    Args:
    
    """
    valid_mask = ~torch.isnan(tensor).any(dim=-1)
    return coords[valid_mask], tensor[valid_mask]


def load_data(config_file):
    """
    Loads the data based on the information from the config file (should be .yaml file).
    It then filters the data based on the yaml file
    config_file: the file which stores the configuration settings for the data
    """
    file_path = config_file['data']['file_addr']
    data = xr.open_zarr(file_path)
    data = data.assign_coords(lon=((data.lon + 180) % 360 - 180))
    
    bands = config_file['data']['bands']
    lat_min = config_file['data']['lat_min']
    lat_max = config_file['data']['lat_max']
    lon_min = config_file['data']['lon_min']
    lon_max = config_file['data']['lon_max']
    time_slice = config_file['data']['time_slice']
    
    data_subset = data[bands]
    data_subset = data_subset.isel(t=time_slice).sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    
    return data_subset  


def xarray_to_tensor(config_file):
    """
    Converts the xarray from the config file to a tensor mapping latitude and longitude to the band values.
    Removes the NaN values
    Args:
        config_file: The yaml file that has the information 
    """
    dataset = load_data(config_file)

    bands = config_file['data']['bands']
    lat = dataset.lat.values
    lon = dataset.lon.values

    tensor = create_tensor_from_dataset(dataset, bands)
    coords = create_coords(lat, lon)

    coords, tensor = filter_valid(coords, tensor)

    return coords, tensor


def plot_band_comparison(tensor, predictions, coords, band_index, band_name=None, cmap='viridis', point_size=10):
    """
    Plots actual vs predicted values for a given band using latitude and longitude.

    Parameters:
        tensor (ndarray): Actual values, shape (N, num_bands)
        predictions (ndarray): Predicted values, shape (N, num_bands)
        coords (ndarray): Coordinates, shape (N, 2), format [lat, lon]
        band_index (int): Index of the band to plot (0-based)
        band_name (str): Optional name for the band (e.g., 'Red')
        cmap (str): Colormap to use
        point_size (int): Size of each scatter point
    """
    lat = coords[:, 0]
    lon = coords[:, 1]
    
    if band_name is None:
        band_name = f"Band {band_index + 1}"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{band_name} - Actual vs Predicted", fontsize=16)

    # Actual
    sc1 = axs[0].scatter(lon, lat, c=tensor[:, band_index], cmap=cmap, s=point_size)
    axs[0].set_title("Actual")
    axs[0].set_xlabel("Longitude")
    axs[0].set_ylabel("Latitude")
    plt.colorbar(sc1, ax=axs[0])

    # Predicted
    sc2 = axs[1].scatter(lon, lat, c=predictions[:, band_index], cmap=cmap, s=point_size)
    axs[1].set_title("Predicted")
    axs[1].set_xlabel("Longitude")
    axs[1].set_ylabel("Latitude")
    plt.colorbar(sc2, ax=axs[1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

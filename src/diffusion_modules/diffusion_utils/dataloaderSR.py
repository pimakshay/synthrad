import torchvision, torch
from torch.utils.data import DataLoader
from torchvision import transforms 
from torchvision.transforms import Compose
import os
PATH = '/home/akshay/Documents/cse_sem_5/thesis/PI_diffusionModels/code/test-annotated-diffusion/run_scripts/data'

# Assume you have two dataloaders: low_res_loader and high_res_loader

# Iterate over the dataloaders and join corresponding items
joined_data = []
# for low_res_item, high_res_item in zip(low_res_loader, high_res_loader):
#     low_res_image = low_res_item['image']
#     high_res_image = high_res_item['image']
#     joined_item = {
#         'low_resolution': low_res_image,
#         'high_resolution': high_res_image
#     }
#     joined_data.append(joined_item)

# # Create a new dataloader for the joined data
# joined_loader = DataLoader(joined_data, batch_size=batch_size, shuffle=True)

# # Now you can iterate over the joined dataloader and access both low-resolution and high-resolution images in each batch
# for batch in joined_loader:
#     low_res_images = batch['low_resolution']
#     high_res_images = batch['high_resolution']
    # Perform further operations or processing on the low-resolution and high-resolution images


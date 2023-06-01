import os
import torch
import torchvision
from torchvision import transforms 
import numpy as np
import matplotlib.pyplot as plt

def show_images(datset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15)) 
    for i, img in enumerate(datset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(img[0])


def load_car_data(PATH="."):
    '''
    :PATH: path to the folder where data will be downloaded
    '''
    data = torchvision.datasets.StanfordCars(root=PATH, download=True)
    return data


def load_transformed_dataset(IMG_SIZE = 64, PATH='./data'):
    '''Converts a pillow image to a tensor'''
    data_transforms = [
        transforms.Resize(IMG_SIZE), #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: t)#(t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform, split='test')
    return torch.utils.data.ConcatDataset([train, test]) #test and train set merged

def show_tensor_image(image):
    '''Does reverse transformations from a tensor image to a pillow image'''
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))
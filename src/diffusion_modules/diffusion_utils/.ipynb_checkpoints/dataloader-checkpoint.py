import torchvision, torch
from torchvision import transforms 
from torchvision.transforms import Compose
import os

def load_MNIST(image_size, channels):
    transform = Compose([transforms.Resize(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])  

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )        
    val_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )          
    return train_dataset, val_dataset

def load_EMNIST(image_size, channels):
    transform = Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])  

    train_dataset = torchvision.datasets.EMNIST(
        root="./data", train=True, transform=transform, download=True, split="byclass"
    ) 
    val_dataset = torchvision.datasets.EMNIST(
        root="./data", train=False, transform=transform, download=True, split="byclass"
    )  
    return train_dataset, val_dataset        

def load_FashionMNIST(image_size, channels):
    transform = Compose([transforms.Resize(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda t: (t * 2) - 1)
    ])  

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )        
    val_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )          
    return train_dataset, val_dataset

def load_CIFAR10(image_size, channels):
    transforms_cifar10 = transforms.Compose([transforms.Resize(image_size), torchvision.transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])      
                                            
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', download=True, transform=transforms_cifar10)  
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms_cifar10)      
    return train_dataset, test_dataset

def load_StanfordCars(image_size, channels):
    PATH = './data'
    '''Converts a pillow image to a tensor'''
    data_transforms = [
        transforms.Resize(image_size), #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform, split='test')
    return train, test #torch.utils.data.ConcatDataset([train, test]) #test and train set merged


def load_DTD(image_size, channels):
    PATH = './data'
    '''Converts a pillow image to a tensor'''
    data_transforms = [
        transforms.Resize(image_size), #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.DTD(root=PATH, download=True, 
                                         transform=data_transform)

    val = torchvision.datasets.DTD(root=PATH, download=True, 
                                         transform=data_transform, split='val')
    train = torch.utils.data.ConcatDataset([train, val])

    test = torchvision.datasets.DTD(root=PATH, download=True, 
                                         transform=data_transform, split='test')
    return train, test #torch.utils.data.ConcatDataset([train, test]) #test and train set merged


def load_Celeba(image_size, channels):
    PATH = './data'
    '''Converts a pillow image to a tensor'''
    data_transform = transforms.Compose([
                                        transforms.Resize(image_size),
                                        # transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    train = torchvision.datasets.CelebA(root=PATH, download=True, 
                                         transform=data_transform)

#     val = torchvision.datasets.CelebA(root=PATH, download=True, 
#                                          transform=data_transform, split='val')
#     train = torch.utils.data.ConcatDataset([train, val])

    test = torchvision.datasets.CelebA(root=PATH, download=True, 
                                         transform=data_transform, split='test')
    return train, test #torch.utils.data.ConcatDataset([train, test]) #test and train set merged


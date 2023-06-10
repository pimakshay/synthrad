import torchvision, torch
from torchvision import transforms 
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, Dataset
# from diffusion_modules.diffusion_utils import dataloaderDenoise, dataloaderSR

PATH = '/home/akshay/Documents/cse_sem_5/thesis/PI_diffusionModels/code/test-annotated-diffusion/run_scripts/data'
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


def load_CIFAR10(image_size, channels):
    transforms_cifar10 = transforms.Compose([transforms.Resize(image_size), torchvision.transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])      

    noisy_transforms_cifar10 = transforms.Compose([transforms.Resize(image_size), torchvision.transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                             dataloaderDenoise.AddNoise("gaussian")])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', download=True, transform=transforms_cifar10)  
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms_cifar10)   

    noisy_train_dataset = torchvision.datasets.CIFAR10(
        root='./data', download=True, transform=noisy_transforms_cifar10)  
    noisy_val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=noisy_transforms_cifar10)         
    return train_dataset, val_dataset, noisy_train_dataset, noisy_val_dataset

def load_StanfordCars(image_size, channels):
    # PATH = './data'
    '''Converts a pillow image to a tensor'''
    data_transform = transforms.Compose([
        transforms.Resize(image_size), #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        # transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ])
    noisy_data_transform = transforms.Compose([
        transforms.Resize(image_size), #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(image_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        dataloaderDenoise.AddNoise("gaussian")
    ])

    train = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform)

    val = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=data_transform, split='test')
    
    noisy_train = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=noisy_data_transform)

    noisy_val = torchvision.datasets.StanfordCars(root=PATH, download=True, 
                                         transform=noisy_data_transform, split='test')    
    return train, val, noisy_train, noisy_val #torch.utils.data.ConcatDataset([train, test]) #test and train set merged

def load_task1_brain(data_dir, anatomy, image_size, num_of_samples):
    from src.diffusion_modules.diffusion_utils.dataloaderSITK import PrepareData, CreateDataset
    # data_dir = "/home/akshay/Documents/cse_sem_6/synthrad2023/algorithm-template/data/Task1"  # Specify the directory containing the patient folders
    anatomy = "brain"
    max_size = 284 if anatomy=="brain" else 586
    brain_data = PrepareData(data_dir,anatomy=anatomy,max_size=max_size,trim_x_slices=0.15,num=num_of_samples)
    brain_dataset = CreateDataset( brain_data, phase="train", image_size=image_size, lr_flip=0.5)
    return brain_dataset

def load_task2_brain(data_dir, anatomy, image_size, num_of_samples, rescale):
    from src.diffusion_modules.diffusion_utils.dataloaderSITK_T2 import PrepareData, CreateDataset
    # data_dir = "/home/akshay/Documents/cse_sem_6/synthrad2023/algorithm-template/data/Task1"  # Specify the directory containing the patient folders
    anatomy = "brain"
    max_size = 298 if anatomy=="brain" else 586
    brain_data = PrepareData(data_dir,anatomy=anatomy,max_size=max_size,trim_x_slices=0.15,num=num_of_samples)
    brain_dataset = CreateDataset( brain_data, phase="train", image_size=image_size, rescale=rescale, lr_flip=0.5)
    return brain_dataset

def load_t1_brain(data_dir):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize the images to a specific size
        transforms.ToTensor()  # Convert images to tensors
    ])    
    
    prior_dataset = torch.Tensor(np.load(data_dir+"/brain/prior_dataset.npz")["x_prior"])
    # cond_dataset = np.load(data_dir+"/brain/cond_dataset.npz")
    # return cond_dataset["x_cond"], prior_dataset["x_prior"]
    return MedicalDataset(prior_dataset, transform), MedicalDataset(prior_dataset, transform)

def get_conditioned_dataloader(dataloader, cond):
    """
    gets conditioned dataloader

    Args:
        image (torch.Tensor): Input image tensor of shape (C, H, W).
        noise_type (str): Type of noise to be applied. Options: 'gaussian', 'salt', 'pepper', 's&p', 'speckle'.

    Returns:
        torch.Tensor: Image tensor with added noise.
    """    
    if cond=="SR":
        conditioned_dataloader = None #callSR()
    elif cond=="denoise":
        conditioned_dataloader = dataloaderDenoise.get_denoise_dataset(dataloader)
    
    return conditioned_dataloader

def show_images(dataset, num_samples=24, cols=4, random=True, save_all=False, filename="abc.png"):
    """ Plot400s some samples from the dataset """
    random_img_idx = np.random.randint(0, high=len(dataset), size=len(dataset), dtype=int)
#     plt.figure(figsize=(15,15)) 
    rows = int(num_samples//cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14,14))    
    img_count=0
    for i in range(rows):
        for j in range(cols):
            random_index = random_img_idx[i*cols + j] if random else img_count#int(np.random.random()*len(dataset))
#             if i == num_samples:
#                 break
    #         plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
    #         img = dataset[random_index] # use for cifar10, mnist
            img = dataset[random_index] # use for stanford_Cars
            out = img[0].numpy().transpose((1,2,0)) #reshape(ddpmModel.image_size, ddpmModel.image_size)
            axes[i, j].imshow(out)    
            img_count = img_count + 1    
    if save_all:
        assert filename is not None, "Filename missing.."
        plt.savefig(filename)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, cond_dataset=None, prior_dataset=None):
        self.cond_dataset = cond_dataset
        self.prior_dataset = prior_dataset

    def __getitem__(self, index):
        cond_image = self.cond_dataset[index]
        prior_image = self.prior_dataset[index]
        image = {'x_cond': cond_image,
                'x_prior': prior_image}
        return image
    
    def __len__(self):
        return min(len(self.cond_dataset), len(self.prior_dataset))

class MedicalDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        return x
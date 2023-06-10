import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import SimpleITK as sitk


class PrepareData():
    def __init__(self, data_dir, anatomy='brain', max_size=298, trim_x_slices=0.15, num=5):
        self.data_dir = data_dir   
        self.anatomy = anatomy
        self.file_paths = self.get_file_paths() 
        self.load_n_samples = num
        self.max_size = (max_size,max_size) #self.calculate_max_size() #(max_size,max_size) 
        self.output_size = np.maximum(self.max_size[0], self.max_size[1])
        self.trim_x_slices = trim_x_slices

    def get_file_paths(self):
        pattern = os.path.join(self.data_dir, self.anatomy, '*') #, "patient_*")
        file_paths = []
        for patient_dir in glob(pattern):
            ct_path = os.path.join(patient_dir, "ct.nii.gz")
            cbct_path = os.path.join(patient_dir, "cbct.nii.gz")
            mask_path = os.path.join(patient_dir, "mask.nii.gz")
            if os.path.exists(ct_path) and os.path.exists(cbct_path) and os.path.exists(mask_path):
                file_paths.append((ct_path, cbct_path, mask_path))
        return file_paths
    
    def load_images(self, rescale=False):
        ct_stacked = []
        cbct_stacked = []
        for i,file_path in enumerate(self.file_paths):
            if i < self.load_n_samples:
                ct_path, cbct_path, mask_path = file_path

                ct_arr = sitk.GetArrayFromImage(self.rescale_image(sitk.ReadImage(ct_path)) if rescale else sitk.ReadImage(ct_path))
                # ct_arr = self.pad_images(ct_arr.transpose((0, 2, 1)))
                ct_arr = self.pad_images(ct_arr)
                ct_stacked.append(np.stack(ct_arr))
                
                cbct_arr = sitk.GetArrayFromImage(self.rescale_image(sitk.ReadImage(cbct_path)) if rescale else sitk.ReadImage(cbct_path))
                # cbct_arr = self.pad_images(cbct_arr.transpose((0, 2, 1)))
                cbct_arr = self.pad_images(cbct_arr)
                cbct_stacked.append(np.stack(cbct_arr))
            else: 
                break
        ct_concat = np.concatenate(ct_stacked, axis=0)
        cbct_concat = np.concatenate(cbct_stacked, axis=0)
        return ct_concat, cbct_concat

    def rescale_image(self, image):
        filter = sitk.RescaleIntensityImageFilter()
        filter.SetOutputMaximum(255)
        filter.SetOutputMinimum(0)
        rescaled_img = filter.Execute(image)
        return rescaled_img

    def pad_images(self, images):

        padded_images = []

        for i,image in enumerate(images):
            if i>(self.trim_x_slices)*len(images) and i<(1-self.trim_x_slices)*len(images):
                size = image.shape
                pad_x = max(0, (self.output_size - size[0]) // 2)
                pad_y = max(0, (self.output_size - size[1]) // 2)
                pad_width = ((pad_x, self.output_size - size[0] - pad_x), (pad_y, self.output_size - size[1] - pad_y))
                padded_image = np.pad(image, pad_width, mode='edge')
                padded_images.append(padded_image)
        return padded_images

    def calculate_max_size(self):
        max_width = max_height = 0
        for file_paths in self.file_paths:
            file_path = file_paths[0]  # file path for ct images
            image = sitk.ReadImage(file_path)
            w = image.GetWidth()
            h = image.GetHeight()
            max_width = np.maximum(max_width, w)
            max_height = np.maximum(max_height, h)
        return max_width, max_height


class CreateDataset(Dataset):
    def __init__(self, prepare_data:PrepareData, phase='train', image_size=64, rescale=False, lr_flip=0.5):
        self.prepare_data = prepare_data
        self.ct_data, self.cbct_data = prepare_data.load_images(rescale)
        if phase == 'train':
            self.transforms = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Lambda(self.rescale_image),
                transforms.Resize(image_size),
#                 transforms.RandomHorizontalFlip(lr_flip),
                transforms.ToTensor(),
#                 transforms.Normalize((0.5), (0.5))
#                 transforms.Lambda(lambda t: (t * 2) - 1)
            ])

        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(image_size),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
#                 transforms.Lambda(lambda t: (t * 2) - 1)
            ])

    def __len__(self):
        assert len(self.ct_data) == len(self.cbct_data)
        return len(self.ct_data)

    def __getitem__(self, idx):
        '''
        x_prior: ct images --> to be generated as output
        x_cond: cbct images --> conditioned on cbct images
        '''
        transformed_ct = self.transforms(self.ct_data[idx])
        transformed_cbct = self.transforms(self.cbct_data[idx])      
        image = dict(x_prior=transformed_ct, 
                     x_cond=transformed_cbct)
        return image 
    
    def __getitem__(self, idx:list):
        '''
        x_prior: ct images --> to be generated as output
        x_cond: cbct images --> conditioned on cbct images
        '''
        transformed_ct = self.transforms(self.ct_data[idx])
        transformed_cbct = self.transforms(self.cbct_data[idx])      
        image = dict(x_prior=transformed_ct, 
                     x_cond=transformed_cbct)
        return image     

    def rescale_image(self, image):
        filter = sitk.RescaleIntensityImageFilter()
        filter.SetOutputMaximum(255)
        filter.SetOutputMinimum(0)
        rescaled_img = filter.Execute(image)
        return rescaled_img


class SimpleITKDataset(Dataset):
    def __init__(self, data_dir, anatomy):
        self.data_dir = data_dir
        self.anatomy = anatomy
        self.file_paths = self.get_file_paths()
        self.max_size = (284,284) if anatomy=="brain" else self.calculate_max_size()
        self.output_size = np.maximum(self.max_size[0], self.max_size[1])
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_paths = self.file_paths[idx]
        image = self.load_image(file_paths)
        # Add any preprocessing steps if required
        return image

    def get_file_paths(self):
        pattern = os.path.join(self.data_dir, self.anatomy, '*') #, "patient_*")
        file_paths = []
        for patient_dir in glob(pattern):
            ct_path = os.path.join(patient_dir, "ct.nii.gz")
            cbct_path = os.path.join(patient_dir, "cbct.nii.gz")
            mask_path = os.path.join(patient_dir, "mask.nii.gz")
            if os.path.exists(ct_path) and os.path.exists(cbct_path) and os.path.exists(mask_path):
                file_paths.append((ct_path, cbct_path, mask_path))
        return file_paths

    def load_image(self, file_paths):
        ct_path, cbct_path, mask_path = file_paths
        ct_image = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_image)
        ct_array = ct_array.astype(float)
        ct_array = self.pad_images(ct_array)

        cbct_image = sitk.ReadImage(cbct_path)
        cbct_array = sitk.GetArrayFromImage(cbct_image)
        cbct_array = cbct_array.astype(float)
        cbct_array = self.pad_images(cbct_array)

        mask_image = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask_array = mask_array.astype(float)
        mask_array = self.pad_images(mask_array)

        return {
            'x_prior': ct_array,
            'x_cond': cbct_array,
            'mask': mask_array
        }

    def calculate_max_size(self):
        max_width = max_height = 0
        for file_paths in self.file_paths:
            file_path = file_paths[0] # file path for ct images
            image = sitk.ReadImage(file_path)
            w = image.GetWidth()
            h = image.GetHeight()
            max_width = np.maximum(max_width, w)
            max_height = np.maximum(max_height, h)
        return max_width, max_height
    
    def pad_images(self, images):

        padded_images = []

        for image in images:
            size = image.shape
            pad_x = max(0, (self.output_size - size[0]) // 2)
            pad_y = max(0, (self.output_size - size[1]) // 2)
            pad_width = ((pad_x, self.output_size - size[0] - pad_x), (pad_y, self.output_size - size[1] - pad_y))
            padded_image = np.pad(image, pad_width, mode='edge')
            padded_images.append(padded_image)
        return padded_images

def split_dataset(dataset, batch_size=4, tvt_ratio=[0.8,0.1,0.1]):
    dataset_size = len(dataset)
    # Define the sizes for train, validation, and test sets
    train_size = int(tvt_ratio[0] * dataset_size)  # 80% for training
    val_size = int(tvt_ratio[1] * dataset_size)   # 10% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 10% for testing

    # Perform the random split
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for each set
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, test_dataset 

def show_images(dataset, vis_sample_num=0, scan_type='ct', num_slices=10, freq=10, cols=3, random=False):
    '''
    :vis_sample_num: sample num to be visualized
    :num_slices: num of slices to visualize from the given vis_sample_num
    :scan_type: scan type like cbct, ct, mask
    :freq: freq of slices to visualize: 0, freq, 2*freq, 3*freq, 4*freq
    :cols: three dimensions of voxel image [0,1,2]
    '''
    rows=num_slices
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(14,14))
    for i in range(len(dataset) // freq):
            if i < num_slices:
                img_dim0 = dataset[vis_sample_num][scan_type][i*freq,:,:] #[i*freq]
                img_dim1 = dataset[vis_sample_num][scan_type][:,i*freq,:]
                img_dim2 = dataset[vis_sample_num][scan_type][:,:,i*freq]
                axes[i, 0].imshow(img_dim0)      
                axes[i, 1].imshow(img_dim1) 
                axes[i, 2].imshow(img_dim2) 
    # plt.savefig('stanford_cars_trainset.png')


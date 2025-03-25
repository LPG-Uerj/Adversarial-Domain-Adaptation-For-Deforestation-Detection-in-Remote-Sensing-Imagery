import os.path
import torch
import scipy.io as sio
import tensorflow as tf
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import random
import sys 


class RemoteSensingUnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two remote sensing images data_T1 and data_T2 respectively.
    That can be loaded using the scripts Amazonia_Legal_RO, Amazonia_Legal_PA, and Cerrado_Biome_MA. They are loaded in
    __init__.py script.
    
    """

    def __init__(self, dataset_s, dataset_t, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        
        half_dim = self.opt.crop_size // 2
        npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))

        self.data_T1 = np.concatenate((dataset_s.images_norm[0], dataset_s.images_norm[1]), axis = 2)
        self.data_T1 = np.pad(self.data_T1, pad_width = npad, mode = 'reflect')
        self.data_T2 = np.concatenate((dataset_t.images_norm[0], dataset_t.images_norm[1]), axis = 2)
        self.data_T2 = np.pad(self.data_T2, pad_width = npad, mode = 'reflect')
        
        self.source_diff_reference = dataset_s.images_diff[0]
        self.source_diff_reference = np.pad(self.source_diff_reference, pad_width = npad, mode = 'reflect')
        self.target_diff_reference = dataset_t.images_diff[0]
        self.target_diff_reference = np.pad(self.target_diff_reference, pad_width = npad, mode = 'reflect')
        
        self.central_pixels_coordinates_T1 = dataset_s.central_pixels_coordinates # get the patch central pixel coordinates
        self.central_pixels_coordinates_T2 = dataset_t.central_pixels_coordinates # get the patch central pixel coordinates
        self.T1_size = self.central_pixels_coordinates_T1.shape[0]  # get the size of dataset A
        self.T2_size = self.central_pixels_coordinates_T2.shape[0]  # get the size of dataset B
        print(self.T1_size)
        print(self.T2_size)

        if self.opt.use_task_loss:
            self.source_class_reference = dataset_s.new_reference
            # self.source_class_reference = np.pad(self.source_class_reference, pad_width = npad, mode = 'reflect')
            self.target_class_reference = dataset_t.new_reference
            # self.target_class_reference = np.pad(self.target_class_reference, pad_width = npad, mode = 'reflect')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B
            A (tensor)       -- an image in the source domain
            A_ref(tensor)    -- its corresponding difference reference
            B (tensor)       -- an image in the target domain
            B_ref(tensor)    -- its corresponding difference reference

        """
        
        T1_coor = self.central_pixels_coordinates_T1[index, :]
        # print(T1_coor[0], T1_coor[1], T1_coor[2])
        T2_coor = self.central_pixels_coordinates_T2[index, :]
        
        T1_coor = np.reshape(T1_coor, (-1, np.size(self.central_pixels_coordinates_T1, 1)))
        # print(T1_coor)
        T2_coor = np.reshape(T2_coor, (-1, np.size(self.central_pixels_coordinates_T2, 1)))
        
        if self.opt.phase == 'train':
            T1_img = self.Patch_Extraction_Train(self.data_T1, T1_coor, self.opt.crop_size, False, 'reflect')
            T2_img = self.Patch_Extraction_Train(self.data_T2, T2_coor, self.opt.crop_size, False, 'reflect')
            
            T1_ref = self.Patch_Extraction_Train(self.source_diff_reference, T1_coor, self.opt.crop_size, False, 'reflect')
            T2_ref = self.Patch_Extraction_Train(self.target_diff_reference, T2_coor, self.opt.crop_size, False, 'reflect')

            if self.opt.use_task_loss:
                T1_class_ref = self.Patch_Extraction_Train(np.expand_dims(self.source_class_reference, axis = 2),
                    T1_coor, self.opt.crop_size, True, 'reflect')
            
                T1_data = np.concatenate((T1_img , T1_ref, T1_class_ref), axis = 1)
                T2_data = np.concatenate((T2_img , T2_ref), axis = 1)

            else:
                T1_data = np.concatenate((T1_img , T1_ref), axis = 1)
                T2_data = np.concatenate((T2_img , T2_ref), axis = 1)
            
        if self.opt.phase == 'test':
            T1_data = self.Patch_Extraction_Test(self.data_T1, T1_coor, self.opt, True)
            T2_data = self.Patch_Extraction_Test(self.data_T2, T2_coor, self.opt, False)
        
        T1_data = T1_data[0, :, :, :]
        T2_data = T2_data[0, :, :, :]

        if self.opt.use_tile_configuration:
            T1_data = self.Task_transform(T1_data, T1_coor[0,2])
        else:
            T1_data = self.RemoteSensing_Transforms(T1_data)
        T2_data = self.RemoteSensing_Transforms(T2_data)        
           
        if self.opt.phase == 'train':
            # mog: original pedro
            # T1_ref = np.zeros((7, T1_data.shape[1], T1_data.shape[2]))
            # T2_ref = np.zeros((7, T2_data.shape[1], T2_data.shape[2]))
            # T1_ref[:7, :, :] = T1_data[-7:, :, :]
            # T2_ref[:7, :, :] = T2_data[-7:, :, :]
            # T1_img = T1_data[:-7,:, :]
            # T2_img = T2_data[:-7,:, :]

            T1_ref = T1_data[14:21, :, :]
            T2_ref = T2_data[14:21, :, :]

            T1_img = T1_data[:14, :, :]
            T2_img = T2_data[:14, :, :]

            if self.opt.use_task_loss:
                T1_class_ref = T1_data[-1, :, :]
                # print(T1_class_ref.shape)
                # print(T1_class_ref[0:10,0:10])
                # T2_class_ref = T2_data[:, -1,:, :]

        else:
            T1_img = T1_data
            T2_img = T2_data
            
            
        A = torch.Tensor(T1_img)
        B = torch.Tensor(T2_img)
        if self.opt.phase == 'train':
            A_ref = torch.Tensor(T1_ref)
            B_ref = torch.Tensor(T2_ref)

            if self.opt.use_task_loss:
                A_class_ref =  torch.Tensor(T1_class_ref).long()
        
        if self.opt.phase == 'test':    
            return {'A': A, 'B': B}
        else:
            if not self.opt.use_task_loss:
                return {'A': A, 'A_ref': A_ref, 'B': B, 'B_ref': B_ref}
            else:
                return {'A': A, 'A_ref': A_ref, 'B': B, 'B_ref': B_ref,
                'A_class_ref': A_class_ref} # , 'B_class_ref': B_class_ref}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.T1_size, self.T2_size)
    
    def Patch_Extraction_Train(self, data, central_pixels_indexs, patch_size, padding, mode):
        
        half_dim = patch_size // 2
        data_rows = np.size(data, 0)
        data_cols = np.size(data, 1)
        data_depth = np.size(data, 2)
        num_samp = np.size(central_pixels_indexs , 0)
        
        patches_cointainer = np.zeros((num_samp, data_depth, patch_size, patch_size)) 
        
        if padding:
            if mode == 'zeros':
                upper_padding = np.zeros((half_dim, data_cols, data_depth))
                left_padding = np.zeros((data_rows + half_dim, half_dim, data_depth))
                bottom_padding = np.zeros((half_dim, half_dim + data_cols, data_depth))
                right_padding = np.zeros((2 * half_dim + data_rows, half_dim, data_depth))
                
                #Add padding to the data     
                data_padded = np.concatenate((upper_padding, data), axis=0)
                data_padded = np.concatenate((left_padding, data_padded), axis=1)
                data_padded = np.concatenate((data_padded, bottom_padding), axis=0)
                data_padded = np.concatenate((data_padded, right_padding), axis=1)
            if mode == 'reflect':
                npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))
                data_padded = np.pad(data, pad_width = npad, mode = 'reflect')
        else:
            data_padded = data
        
        for i in range(num_samp):
            patches_cointainer[i, :, :, :] = np.transpose(data_padded[int(central_pixels_indexs[i , 0]) - half_dim  : int(central_pixels_indexs[i , 0]) + half_dim , int(central_pixels_indexs[i , 1]) - half_dim : int(central_pixels_indexs[i , 1]) + half_dim , :],(2, 0, 1))
                    
        return patches_cointainer

    def Patch_Extraction_Test(self, data, patches_coordinates, opt, from_source):
        
        num_samples = np.size(patches_coordinates, 0)
        rows_size = data.shape[0]
        cols_size = data.shape[1]
        data_depth= np.size(data, 2)
        data_patch = np.zeros((num_samples, data_depth, opt.crop_size, opt.crop_size))
        
        if from_source:
            overlap = round(opt.crop_size * opt.overlap_porcent_s)
        else:
            overlap = round(opt.crop_size * opt.overlap_porcent_t)
        
        overlap -= overlap % 2
        stride = opt.crop_size - overlap
        
        step_row = (stride - rows_size % stride) % stride
        step_col = (stride - cols_size % stride) % stride
        
        pad_tuple = ((overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col) , (0 , 0))
        data_pad = np.pad(data, pad_tuple, mode='symmetric')
        for i in range(num_samples):
            
            data_patch[i, :, :, :] = np.transpose(data_pad[int(patches_coordinates[i , 0]) : int(patches_coordinates[i , 2]), int(patches_coordinates[i , 1]) : int(patches_coordinates[i , 3]),:],(2, 0, 1))
        
        return data_patch
        
    def RemoteSensing_Transforms(self, data):
        # The transformations here were accomplished using tensorflow-cpu framework
        input_nc = np.size(data, 0)
        data = np.transpose(data, (1, 2, 0))
        
        if 'resize' in self.opt.preprocess:
            data = tf.image.resize(data, [self.opt.load_size, self.opt.load_size],
                                    method=tf.image.ResizeMethod.BICUBIC) 
                    
        if 'crop' in self.opt.preprocess:
            data = tf.image.random_crop(data, size=[self.opt.crop_size, self.opt.crop_size, input_nc])
            
        if not self.opt.no_flip:
            
            data = tf.image.random_flip_left_right(data)
            
        if self.opt.convert:
            pass
        
        return np.transpose(data, (2, 0, 1))

    def Task_transform(self, data, aug):
        # 0 - no transform
        # 1 - 90 degrees counterclockwise
        # 2 - 180 degrees turn
        # 3 - 270 degrees counterclowise (90 clockwise)
        # 4 - mirror image vertically
        # 5 - mirror + 90 counter
        # 6 - mirror + 180 counter
        # 7 - mirror + 270 counter
        aug = 2
        if aug == 0:
            return data
        
        out_data = np.zeros((data.shape))
        for i in range(data.shape[0]):
            if aug == 1:
                out_data[i] = np.rot90(data[i])
            elif aug == 2:
                out_data[i] = np.rot90(data[i], 2)
            elif aug == 3:
                out_data[i] = np.rot90(data[i], 3)
            elif aug == 4:
                out_data[i] = np.flip(data[i], axis = 1)
            elif aug == 5:
                out_data[i] = np.rot90(np.flip(data[i], axis = 1))
            elif aug == 6:
                out_data[i] = np.rot90(np.flip(data[i], axis = 1), 2)
            elif aug == 7:
                out_data[i] = np.rot90(np.flip(data[i], axis = 1), 3)

            else:
                raise Exception("Transform out of index (n<0 or n>7)")
        return out_data

        
        
import os
import numpy as np
import tifffile

import torch
import torch.nn as nn

from . import networks, utils

import sys
sys.path.append('..')
from model_architectures import GAN
from model_architectures import GAN_parameters


class opt_param():
    crop_size = 256 # equals patch size
    overlap_porcent_s = 0.7
    overlap_porcent_t = 0.7
    save_real = False

    batch_size = 50
    save_path = './recon_test/'
    # rows_size_s = 0
    # cols_size_s = 0

class reconstruction_parameters():
    def __init__(self, args, domain):
        # rec_param = reconstruction_parameters
        self.on_cpu = args.run_rec_on_cpu
        self.save_real = False
        if args.full_image:
            self.crop_size = np.max(domain.conc_image.shape)
            self.batch_size = 1
            self.overlap_porcent_s = 0
        else:
            self.crop_size = args.rec_size
            self.batch_size = args.rec_batch
            self.overlap_porcent_s = args.overlap_porcent

        self.de_normalize = args.de_normalize
        self.save_tiff = args.save_tiff
        self.save_mat = args.save_mat

        self.save_only_present = False
        self.save_only_present_rgb = False

        self.rows_size_s = domain.conc_image.shape[0]
        # print("rows_size", self.rows_size_s)
        self.cols_size_s = domain.conc_image.shape[1]
        # print("cols_size", self.cols_size_s)
        self.output_nc = domain.conc_image.shape[-1]
        # print("output_nc", self.output_nc)
        
        # crop_size = args.rec
        # overlap_porcent_s = 0.7
        # overlap_porcent_t = 0.7
        # save_real = False

        # batch_size = 50
        # save_path = './recon_test/'
        # # rows_size_s = 0
        # # cols_size_s = 0


def Coordinates_Definition_Test(rows_size, cols_size, opt, from_source):
        
        if from_source:
            overlap = round(opt.crop_size * opt.overlap_porcent_s)
        else:
            overlap = round(opt.crop_size * opt.overlap_porcent_t)
        
        overlap -= overlap % 2
        stride = opt.crop_size - overlap
        
        step_row = (stride - rows_size % stride) % stride
        step_col = (stride - cols_size % stride) % stride
        
        k1, k2 = (rows_size + step_row)//stride, (cols_size + step_col)//stride
        coordinates = np.zeros((k1 * k2 , 4))
        counter = 0
        for i in range(k1):
            for j in range(k2):
                coordinates[counter, 0] = i * stride
                coordinates[counter, 1] = j * stride
                coordinates[counter, 2] = i * stride + opt.crop_size
                coordinates[counter, 3] = j * stride + opt.crop_size
                counter += 1
        
        return coordinates

def Patch_Extraction_Test(data, patches_coordinates, opt, from_source):
        
        num_samples = np.size(patches_coordinates, 0)
        # rows_size = data.shape[0]
        # cols_size = data.shape[1]
        # data_depth= np.size(data, 2)
        rows_size = data.shape[1]
        cols_size = data.shape[2]
        data_depth= data.shape[0]
        data_patch = np.zeros((num_samples, data_depth, opt.crop_size, opt.crop_size))
        # print(data_patch.shape)
        
        if from_source:
            overlap = round(opt.crop_size * opt.overlap_porcent_s)
        else:
            overlap = round(opt.crop_size * opt.overlap_porcent_t)
        
        overlap -= overlap % 2
        stride = opt.crop_size - overlap
        
        step_row = (stride - rows_size % stride) % stride
        step_col = (stride - cols_size % stride) % stride
        
        pad_tuple = ((0 , 0), (overlap//2, overlap//2 + step_row) , (overlap//2, overlap//2 + step_col))
        data_pad = np.pad(data, pad_tuple, mode='symmetric')
        # print(data_pad.shape)
        # data_pad = utils.channels_last2first(data_pad)
        # print(data_pad.shape)
        # tifffile.imsave('tmpimg.tiff', data_pad, photometric='rgb')
        for i in range(num_samples):
            #print(patches_coordinates[i, :])
            data_patch[i, :, :, :] = data_pad[:,int(patches_coordinates[i , 0]) : int(patches_coordinates[i , 2]), int(patches_coordinates[i , 1]) : int(patches_coordinates[i , 3])]
        
        return data_patch

        
class RemoteSensingPatchesContainer():
    
    def __init__(self, scalers, opt):
        
        self.opt = opt
        self.scalers = scalers
        # self.save_path = self.opt.results_dir + self.opt.name + '/images/'
        self.save_path = opt.save_path
        # Creating the directories
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 
        #Computing the coordinates and prepare the container for the patches
        self.overlap_s = round(self.opt.crop_size * self.opt.overlap_porcent_s)
        self.overlap_s -= self.overlap_s % 2
        self.stride_s = self.opt.crop_size - self.overlap_s
        
        # self.overlap_t = round(self.opt.crop_size * self.opt.overlap_porcent_t)
        # self.overlap_t -= self.overlap_t % 2
        # self.stride_t = self.opt.crop_size - self.overlap_t
        
        self.step_row_s = (self.stride_s - self.opt.rows_size_s % self.stride_s) % self.stride_s
        self.step_col_s = (self.stride_s - self.opt.cols_size_s % self.stride_s) % self.stride_s
        
        # self.step_row_t = (self.stride_t - self.opt.rows_size_t % self.stride_t) % self.stride_t
        # self.step_col_t = (self.stride_t - self.opt.cols_size_t % self.stride_t) % self.stride_t
        
        
        self.k1_s, self.k2_s = (self.opt.rows_size_s + self.step_row_s)//self.stride_s, (self.opt.cols_size_s + self.step_col_s)//self.stride_s
        self.coordinates_s = np.zeros((self.k1_s * self.k2_s , 4))
        
        # self.k1_t, self.k2_t = (self.opt.rows_size_t + self.step_row_t)//self.stride_t, (self.opt.cols_size_t + self.step_col_t)//self.stride_t
        # self.coordinates_t = np.zeros((self.k1_t * self.k2_t , 4))
        
        if self.opt.save_real:
            self.patchcontainer_real_A = np.zeros((self.k1_s * self.stride_s, self.k2_s * self.stride_s, self.opt.output_nc))
            # self.patchcontainer_real_B = np.zeros((self.k1_t * self.stride_t, self.k2_t * self.stride_t, self.opt.output_nc))
        
        # self.patchcontainer_fake_A = np.zeros((self.k1_t * self.stride_t, self.k2_t * self.stride_t, self.opt.output_nc))
        self.patchcontainer_fake_B = np.zeros((self.k1_s * self.stride_s, self.k2_s * self.stride_s, self.opt.output_nc))
        
        counter = 0
        for i in range(self.k1_s):
            for j in range(self.k2_s):
                self.coordinates_s[counter, 0] = i * self.stride_s
                self.coordinates_s[counter, 1] = j * self.stride_s
                self.coordinates_s[counter, 2] = i * self.stride_s + self.opt.crop_size
                self.coordinates_s[counter, 3] = j * self.stride_s + self.opt.crop_size
                counter += 1
                
        # counter = 0
        # for i in range(self.k1_t):
        #     for j in range(self.k2_t):
        #         self.coordinates_t[counter, 0] = i * self.stride_t
        #         self.coordinates_t[counter, 1] = j * self.stride_t
        #         self.coordinates_t[counter, 2] = i * self.stride_t + self.opt.crop_size
        #         self.coordinates_t[counter, 3] = j * self.stride_t + self.opt.crop_size
        #         counter += 1
    
    def store_current_visuals(self, visuals, index):
        start_index = index
        for label in visuals:
            index = start_index
            for image in visuals[label]:
                # image_tensor = image.data
                # image_numpy = image_tensor[0].cpu().float().numpy()
                image_numpy = image.cpu().float().numpy()
                image_numpy = np.transpose(image_numpy,(1, 2, 0))
                
                # if label == 'fake_A':
                #     if index < self.opt.size_t:
                #         self.patchcontainer_fake_A[int(self.coordinates_t[index, 0]) : int(self.coordinates_t[index, 0]) + int(self.stride_t), 
                #                                 int(self.coordinates_t[index, 1]) : int(self.coordinates_t[index, 1]) + int(self.stride_t), :] = image_numpy[int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),
                #                                                                                                                         int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),:]
                if label == 'fake_B':
                    if index < self.opt.size_s:
                        self.patchcontainer_fake_B[int(self.coordinates_s[index, 0]) : int(self.coordinates_s[index, 0]) + int(self.stride_s), 
                                                int(self.coordinates_s[index, 1]) : int(self.coordinates_s[index, 1]) + int(self.stride_s), :] = image_numpy[int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),
                                                                                                                                        int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),:]
                if self.opt.save_real:
                    if label == 'real_A':
                        if index < self.opt.size_s:
                            self.patchcontainer_real_A[int(self.coordinates_s[index, 0]) : int(self.coordinates_s[index, 0]) + int(self.stride_s), 
                                                int(self.coordinates_s[index, 1]) : int(self.coordinates_s[index, 1]) + int(self.stride_s), :] = image_numpy[int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),
                                                                                                                                        int(self.overlap_s//2) : int(self.overlap_s//2) + int(self.stride_s),:]
                    # if label == 'real_B':
                    #     if index < self.opt.size_t:
                    #         self.patchcontainer_real_B[int(self.coordinates_t[index, 0]) : int(self.coordinates_t[index, 0]) + int(self.stride_t), 
                    #                             int(self.coordinates_t[index, 1]) : int(self.coordinates_t[index, 1]) + int(self.stride_t), :] = image_numpy[int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),
                    #                                                                                                                     int(self.overlap_t//2) : int(self.overlap_t//2) + int(self.stride_t),:]
                    

                index += 1                                                                                                                    
                
    def save_images(self):
        
        # fake_img_A = self.patchcontainer_fake_A[:self.k1_t*self.stride_t - self.step_row_t, :self.k2_t*self.stride_t - self.step_col_t, :]
        fake_img_B = self.patchcontainer_fake_B[:self.k1_s*self.stride_s - self.step_row_s, :self.k2_s*self.stride_s - self.step_col_s, :]
        # Applaying the normalizers back
        # scaler_1 = self.scalers[0]
        # scaler_2 = self.scalers[1]
        scaler_1 = self.scalers
        scaler_2 = self.scalers
        
        if self.opt.de_normalize:
            # fake_img_A_reshaped = fake_img_A.reshape((fake_img_A.shape[0] * fake_img_A.shape[1], fake_img_A.shape[2]))
            fake_img_B_reshaped = fake_img_B.reshape((fake_img_B.shape[0] * fake_img_B.shape[1], fake_img_B.shape[2]))
            
            # fake_img_inv_A = scaler_1.inverse_transform(fake_img_A_reshaped)
            fake_img_inv_B = scaler_2.inverse_transform(fake_img_B_reshaped)
            
            # fake_img_norm_A = fake_img_inv_A.reshape((fake_img_A.shape[0], fake_img_A.shape[1], fake_img_A.shape[2]))
            fake_img_norm_B = fake_img_inv_B.reshape((fake_img_B.shape[0], fake_img_B.shape[1], fake_img_B.shape[2]))
            fake_img_B = fake_img_norm_B
        # Saving the fake images
        #saving generated images in .npy to use their in the classifier evaluation
        # np.save(self.save_path + 'Adapted_Target', fake_img_norm_A)

        # mog: adicionar dps npy
        np.save(self.save_path, fake_img_B)
        #saving generated images in .mat to use in visualization purposes
        if self.opt.save_tiff:
            # print(fake_img_B.shape)
            # print("min:", np.min(fake_img_B))
            # print("max:", np.max(fake_img_B))
            # print("********shape is: ", fake_img_B.shape)
            if self.opt.save_only_present_rgb:
                tifffile.imsave(self.save_path + '.tiff', fake_img_B[:,:, 7:7+3], photometric='rgb')
                print("present image saved on:", self.save_path + '.tiff')
            elif self.opt.save_only_present:
                tifffile.imsave(self.save_path + '.tiff', fake_img_B[:,:, 7:], photometric='rgb')
                print("present image saved on:", self.save_path + '.tiff')
            else:
                tifffile.imsave(self.save_path + '.tiff', fake_img_B, photometric='rgb')
                print("saved on:", self.save_path + '.tiff')
                
        # sio.savemat(self.save_path + 'Adapted_Target.mat', {'fake_A': fake_img_norm_A})
        if self.opt.save_mat:
            sio.savemat(self.save_path + '.mat', {'fake_B': fake_img_B})

        # if self.opt.save_real:
        #     real_img_A = self.patchcontainer_real_A[:self.k1_s*self.stride_s - self.step_row_s, :self.k2_s*self.stride_s - self.step_col_s, :]
        #     # real_img_B = self.patchcontainer_real_B[:self.k1_t*self.stride_t - self.step_row_t, :self.k2_t*self.stride_t - self.step_col_t, :]
        #     real_img_A_reshaped = real_img_A.reshape((real_img_A.shape[0] * real_img_A.shape[1], real_img_A.shape[2]))
        #     # real_img_B_reshaped = real_img_B.reshape((real_img_B.shape[0] * real_img_B.shape[1], real_img_B.shape[2]))
        #     real_img_inv_A = scaler_1.inverse_transform(real_img_A_reshaped)
        #     # real_img_inv_B = scaler_2.inverse_transform(real_img_B_reshaped)
            
        #     real_img_norm_A = real_img_inv_A.reshape((real_img_A.shape[0], real_img_A.shape[1], real_img_A.shape[2]))
        #     # real_img_norm_B = real_img_inv_B.reshape((real_img_B.shape[0], real_img_B.shape[1], real_img_B.shape[2]))
        #     #saving generated images in .mat to use in visualization purposes

        #     tifffile.imsave(self.save_path + 'Real_Source.tiff', real_img_norm_A, photometric='rgb')
        #     # sio.savemat(self.save_path + 'Real_Source.mat', {'real_A': real_img_norm_A})
        #     # sio.savemat(self.save_path + 'Real_Target.mat', {'real_B': real_img_norm_B})

        
def overlap_reconstruction(domain_image, domain_scaler, args, rec_options, path_to_weights, save_path, eval, alt_way = False, net_ = False):

    if eval == True:
        # save_path = save_path + "adapted_conc_" + args.dataset + "_eval"
        file_name = "adapted_conc_" + args.dataset + "_eval"
        save_path = os.path.join(save_path, file_name)
    else:
        file_name = "adapted_conc_" + args.dataset
        save_path = os.path.join(save_path, file_name)
    
    # do = domain
    # img = do.conc_image
    img = domain_image
    opt = rec_options
    # opt.rows_size_s = img.shape[0]
    # opt.cols_size_s = img.shape[1]
    # opt.output_nc = img.shape[-1]
    opt.save_path = save_path
    scaler = domain_scaler
    rspc = RemoteSensingPatchesContainer(scaler, opt)
    
    # if not single_image:
    coord = Coordinates_Definition_Test (img.shape[0], img.shape[1], opt, from_source = True)
    opt.size_s = coord.shape[0]
    iterator = torch.utils.data.DataLoader(coord, batch_size = opt.batch_size)

    img = utils.channels_last2first(img)

    if opt.on_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    if alt_way:
        optm = GAN_parameters.cyclegan_model_options
        G_A2B = networks.define_G(optm.input_nc, optm.output_nc, optm.ngf, optm.netG, 
            optm.norm, not optm.no_dropout, optm.linear_output, optm.init_type, 
            optm.init_gain, optm.gpu_ids)
        G_A2B.name = "G_A2B_model"
        if net_:
            G_A2B = G_A2B.module
            # G_A2B = nn.DataParallel(G_A2B)
        # print(torch.load(path_to_weights))
        G_A2B.load_state_dict(torch.load(path_to_weights))
        G_A2B.eval()
        G_A2B.to(device)
    
    else:
        G_A2B = GAN.modelGenerator(channels = opt.output_nc)
        G_A2B.load_state_dict(torch.load(path_to_weights))
        G_A2B.eval()
        G_A2B.to(device)
    
    counter = 0
    with torch.no_grad():
        for this_coord in iterator:
            output = {}
            output['fake_B'] = {}

            batch = Patch_Extraction_Test(img, this_coord, opt, from_source = True)
            batch = torch.from_numpy(batch).float().to(device)
            adapted_batch = G_A2B(batch)
            # adapted_batch = G_A2B(batch)[0]
            output['fake_B'] = adapted_batch
            
            rspc.store_current_visuals(output, counter)
            counter += len(this_coord)

    rspc.save_images()
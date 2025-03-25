#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8
import sys
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
import json
import numpy as np
import torch
import torch.nn as nn
import utils
import random
import tifffile
import glob

from Tools import *
import DeepLabV3plus
import GAN
from torchinfo import summary

from pedro.models import networks
# import GAN_parameters
from pedro.models.image_pool import ImagePool
from pedro.models import cycle_gan_model
import tensorflow as tf



def train(source, target, args, global_args, opt):
    print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())
    device = torch.device("cuda:0")
    output_path = args.output_path

    path_pre = []
    path_pre.append(output_path)
    print("make dir")
    for i in range (len(path_pre)):
        p = path_pre[i]
        if (os.path.isdir(p) == False):
            os.makedirs(p)
            print('\''+ p + '\'' + " path created")

    patch_size = args.patch_size
    num_classes = args.num_classes
    channels = args.channels

    mini_batch_size = args.batch_size
    # num_mini_batches = args.num_batches
    num_epochs = args.num_epochs
    save_model_each_n_epoch = args.save_model_each_n_epoch

    # losses weights (lambdas) 
    A2B_cyclic_loss_lambda = args.A2B_cyclic_loss_lambda
    B2A_cyclic_loss_lambda = args.B2A_cyclic_loss_lambda
    indentity_loss_lambda = args.indentity_loss_lambda
    descriminator_loss_lambda = args.descriminator_loss_lambda
    semantic_loss_lambda = args.semantic_loss_lambda
    diff_loss_lambda = args.diff_loss_lambda

    use_diff_loss = args.use_diff_loss

    # Learning rates
    descriminator_learning_rate = args.descriminator_learning_rate
    generator_learning_rate = args.generator_learning_rate
    semantic_learning_rate = args.semantic_learning_rate

    # semantic train parameters
    weights = args.weights
    gamma = args.gamma
    dilation_rates = args.dilation_rates

    source_domain = global_args.source_domain
    target_domain = global_args.target_domain

    use_se = args.use_semantic_loss
    print("[*] Semantic_loss:", use_se)

    save_previews = args.save_adaptation_previews
    continue_train = args.continue_train

    opt.continue_train = continue_train
    
    # if continue_train:
    #     checkpoint_path = glob.glob(os.path.join(global_args.cyclegan_models_path, "checkpoint*"))
    #     checkpoint_path = checkpoint_path[0]
    #     checkpoint = torch.load(checkpoint_path)


    print("loading datasets...")
    total_patches_per_epoch = args.total_patches_per_epoch
    train_source = source.cycle_coor
    train_target = target.cycle_coor
    total_patches_per_epoch = min((len(train_source), len(train_target), total_patches_per_epoch))
    num_mini_batches = total_patches_per_epoch//mini_batch_size
    print("[*] Patches per epoch:", num_mini_batches * mini_batch_size)

    opt.niter = total_patches_per_epoch

    # aug_idx = augmentations_index(np.zeros(total_patches_per_epoch))


    # if use_se:
    #     # print("[*] Semantic_loss2:", use_se)
    #     deep_lab_v3p = DeepLabV3plus.create(args)
    #     deep_lab_v3p = deep_lab_v3p.to(device)
        
    class ModelFitter(utils.ModelFitter):
        def __init__(self):
            super().__init__(num_epochs, num_mini_batches, output_path=output_path)
            
        def initialize(self):
            # Used as storage folder name
            self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '_test'

            # labels
            # if args.patch_size == 64:
            #     label_shape = (mini_batch_size,) + (1, 6, 6) #  change shape when patch size is not 256
            # else:
            #     label_shape = (mini_batch_size,) + (1, 30, 30) #  change shape when patch size is not 256
            # ones = np.ones(shape=label_shape) * self.REAL_LABEL
            # zeros = ones * 0
            # self.ones = torch.from_numpy(ones).float().to(device)
            # self.zeros = torch.from_numpy(zeros).float().to(device)
            # print("ones", ones.shape)
            # print("ones", ones)
            
            # Image pools used to update the discriminators
            # self.synthetic_pool_A = GAN.ImagePool(self.synthetic_pool_size)
            # self.synthetic_pool_B = GAN.ImagePool(self.synthetic_pool_size)

            # self.synthetic_pool_A = ImagePool(self.synthetic_pool_size)
            # self.synthetic_pool_B = ImagePool(self.synthetic_pool_size)

            self.img = {}
            self.img[source_domain] = utils.channels_last2first(source.conc_image)
            self.img[target_domain] = utils.channels_last2first(target.conc_image)

            self.gt = {}
            self.gt[source_domain] = source.new_reference
            self.gt[target_domain] = target.new_reference

            # if self.use_diff_loss:
            self.diff_ref = {}
            self.diff_ref[source_domain] = utils.channels_last2first(source.diff_reference)
            self.diff_ref[target_domain] = utils.channels_last2first(target.diff_reference)
            # print(self.source_fixed_preview_patch.shape)
            # self.last_checkpoint = ''

            # if continue_train:
            #     self.last_checkpoint = checkpoint_path

            #     self.start_epoch = checkpoint["epoch"] + 1
            #     self. metrics = checkpoint["metrics"]
            #     self.G_A2B.load_state_dict(checkpoint["G_A2B_model"])
            #     self.G_B2A.load_state_dict(checkpoint["G_B2A_model"])
            #     self.D_A.load_state_dict(checkpoint["D_A_model"])
            #     self.D_B.load_state_dict(checkpoint["D_B_model"])
            #     self.opt_D_A.load_state_dict(checkpoint["D_A_optimizer"])
            #     self.opt_D_B.load_state_dict(checkpoint["D_B_optimizer"])
            #     self.opt_G_A2B.load_state_dict(checkpoint["G_A2B_optimizer"])
            #     self.opt_G_B2A.load_state_dict(checkpoint["G_B2A_optimizer"])

            np.random.shuffle(train_source)
            np.random.shuffle(train_target)

            self.model = cycle_gan_model.CycleGANModel(opt)
            self.model.setup(opt)
            self.model.save_dir = global_args.cyclegan_models_path
                
        def pre_epoch(self, epoch):
            # np.random.shuffle(aug_idx)
            # np.random.shuffle(train_source)
            # np.random.shuffle(train_target)

            self.source_set = np.zeros((total_patches_per_epoch,2))
            self.source_set[:,:2] = train_source[:total_patches_per_epoch]
            # self.source_set[:,2] = aug_idx
            self.source_set = torch.utils.data.DataLoader(self.source_set, 
                batch_size = mini_batch_size, shuffle = True)


            self.target_set = np.zeros((total_patches_per_epoch,2))
            self.target_set[:,:2] = train_target[:total_patches_per_epoch]
            # self.target_set[:,2] = aug_idx            
            self.target_set = torch.utils.data.DataLoader(self.target_set, 
                batch_size = mini_batch_size, shuffle = True)

            self.src_iter = iter(self.source_set)
            self.tgt_iter = iter(self.target_set)

            # self.G_A2B.train()
            # self.G_B2A.train()
            # self.D_A.train()
            # self.D_B.train()
            
        def get_batch(self, epoch, batch, batch_data):
            src_batch = next(self.src_iter)
            src_coor = src_batch
            # src_aug = src_batch[:,2]
            # if self.use_diff_loss:
            A , y, A_ref = patch_extraction(self.img[source_domain], 
                self.gt[source_domain], src_coor, patch_size, 
                diff_reference_extract = True,
                diff_reference = self.diff_ref[source_domain])
            # else:
            #     x , y = patch_extraction(self.img[source_domain], self.gt[source_domain], 
            #         src_coor, patch_size, aug_batch = src_aug)
            # batch_data.append(x)
            # batch_data.append(y)

            tgt_batch = next(self.tgt_iter)
            tgt_coor = tgt_batch
            # tgt_aug = tgt_batch[:,2]
            # if self.use_diff_loss:
            B , y, B_ref = patch_extraction(self.img[target_domain], 
                self.gt[target_domain], tgt_coor, patch_size,
                diff_reference_extract = True,
                diff_reference = self.diff_ref[target_domain])
            # else:
            #     x , y = patch_extraction(self.img[target_domain], self.gt[target_domain], 
            #     tgt_coor, patch_size, aug_batch = tgt_aug)
            # batch_data.append(x)
            # batch_data.append(y)
            # if self.use_diff_loss:
            #     batch_data.append(z1)
            #     batch_data.append(z2)

            del y

            A = self.RemoteSensing_Transforms(opt, A)
            B = self.RemoteSensing_Transforms(opt, B)

            self.b_data = {}
            self.b_data['A'] = torch.from_numpy(A).float()
            self.b_data['A_ref'] = torch.from_numpy(A_ref).float()
            self.b_data['B'] = torch.from_numpy(B).float()
            self.b_data['B_ref'] = torch.from_numpy(B_ref).float()
            
        def train(self, epoch, b_data, batch_data, metrics, iteration):
            # self.model
            self.model.set_input(self.b_data)
            self.model.optimize_parameters()
            # print(self.model.loss_D_A.item())

            metrics["DA_loss"] = self.model.loss_D_A.item()
            metrics["DB_loss"] = self.model.loss_D_B.item()
            metrics["D_loss"] = self.model.loss_D_A.item() + self.model.loss_D_B.item()
            
            metrics["gA_d_loss_synthetic"] = self.model.loss_G_A.item()
            metrics["gB_d_loss_synthetic"] = self.model.loss_G_B.item()
            
            metrics["reconstruction_loss_A"] = self.model.loss_cycle_A.item()
            metrics["reconstruction_loss_B"] = self.model.loss_cycle_B.item()
            metrics["reconstruction_loss"] = self.model.loss_cycle_A.item() + self.model.loss_cycle_B.item()
            
            metrics["GA_identity_loss"] = self.model.loss_idt_A.item()
            metrics["GB_identity_loss"] = self.model.loss_idt_B.item()

            metrics["GA_diff_loss"] = self.model.loss_diff_A.item()
            metrics["GB_diff_loss"] = self.model.loss_diff_B.item()

            # metrics["GA_loss"] = self.modelGA_loss.item()
            # metrics["GB_loss"] = self.modelGB_loss.item()
            metrics["G_loss"] = self.model.loss_idt_A.item()
                                                                                                                    
            
        def post_epoch(self, epoch, metrics):
            print('\n')

            if epoch==num_epochs-1 or epoch % (save_model_each_n_epoch//2) == 0:
                directory = self.model.save_dir
                if not os.path.exists(directory):
                    os.makedirs(directory)
                self.model.save_networks(epoch)
                print("Checkpoint Saved:", self.model.save_dir)


            # if save_previews:
            #     print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
            #     self.saveImages(epoch, self.source_fixed_preview_patch, 
            #         self.target_fixed_preview_patch)
            # if epoch==num_epochs-1 or epoch % (save_model_each_n_epoch//2) == 0:
            #     self.saveModel(self.G_A2B, epoch)
            #     self.saveModel(self.G_B2A, epoch)

            #     checkpoint_path = '{}/saved_models/checkpoint_epoch_{}'.format(output_path, epoch)
            #     checkpoint_data = {
            #         # general train
            #         'epoch': epoch,
            #         'metrics': metrics,
            #         # models
            #         'G_A2B_model': self.G_A2B.state_dict(),
            #         'G_B2A_model': self.G_B2A.state_dict(),
            #         'D_A_model': self.D_A.state_dict(),
            #         'D_B_model': self.D_B.state_dict(),
                    
            #         # optimizers
            #         'D_A_optimizer': self.opt_D_A.state_dict(),
            #         'D_B_optimizer': self.opt_D_B.state_dict(),
            #         'G_A2B_optimizer': self.opt_G_A2B.state_dict(),
            #         'G_B2A_optimizer': self.opt_G_B2A.state_dict()
            #         }
            #     if use_se:
            #         checkpoint_data['Semantic_model'] = deep_lab_v3p.state_dict()
            #         checkpoint_data['Semantic_optimizer'] = self.optim_seg.state_dict()
                
            #     torch.save(checkpoint_data, checkpoint_path)

            #     if epoch > 0:
            #         os.remove(self.last_checkpoint)
            #     self.last_checkpoint = checkpoint_path
            #     print("Checkpoint Saved:", checkpoint_path)



    #===============================================================================
    # Help functions

        def saveImages(self, epoch, coor_A, coor_B, num_saved_images=1):
            self.G_A2B.eval()
            self.G_B2A.eval()
            device = next(self.G_A2B.parameters()).device
            directory = os.path.join(output_path, 'images')
            if not os.path.exists(os.path.join(directory, 'A')):
                os.makedirs(os.path.join(directory, 'A'))
                os.makedirs(os.path.join(directory, 'B'))
                os.makedirs(os.path.join(directory, 'Atest'))
                os.makedirs(os.path.join(directory, 'Btest'))

            testString = ''
            # print(coor_A.shape)
            # print(coor_B.shape)
            with torch.no_grad():
                for i in range(2):
                    if i == 1:
                        coor_A = np.zeros((1,2))
                        coor_A[0] = train_source[random.randint(0,(len(train_source)-1))]
                        coor_B = np.zeros((1,2))
                        coor_B[0] = train_target[random.randint(0,(len(train_target)-1))]
                        testString = 'test'                        

                    a , c = patch_extraction(self.img[source_domain], 
                        self.gt[source_domain], coor_A, patch_size)
                    a = torch.from_numpy(a).float().to(device)
                    # print(a.shape)
                    a2b = self.G_A2B(a)
                    back2a = self.G_B2A(a2b)

                    b , c = patch_extraction(self.img[target_domain], 
                        self.gt[target_domain], coor_B, patch_size)
                    b = torch.from_numpy(b).float().to(device)
                    b2a = self.G_A2B(b)
                    back2b = self.G_B2A(b2a)

                    output_a = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]*3))
                    output_a[:,:,:patch_size,:patch_size] = a.cpu()
                    output_a[:,:,:patch_size,patch_size:patch_size*2] = a2b.cpu()
                    output_a[:,:,:patch_size,patch_size*2:] = back2a.cpu()
                    # print(output_a.shape)
                    output_a = output_a.reshape((output_a.shape[0] * output_a.shape[1], output_a.shape[2], output_a.shape[3]))
                    output_a = utils.channels_first2last(output_a)
                    original = output_a
                    image_reshaped = original.reshape((original.shape[0] * original.shape[1], original.shape[2]))
                    output_a = source.scaler.inverse_transform(image_reshaped)
                    output_a = output_a.reshape((original.shape))
                    output_a = np.rint(output_a)
                    # print(output_a.shape)
                    save_path = '{}/images/{}/epoch{}_sample{}.tiff'.format(output_path,'A' + testString, epoch, i)
                    tifffile.imsave(save_path, output_a, photometric='rgb')
                    # save_path = '{}/images/{}/epoch{}_sample{}.mat'.format(output_path,'A' + testString, epoch, i)
                    # sio.savemat(save_path, {'preview_a': output_a})
                    print(save_path)

                    output_b = np.zeros((b.shape[0], b.shape[1], b.shape[2], b.shape[3]*3))
                    output_b[:,:,:patch_size,:patch_size] = b.cpu()
                    output_b[:,:,:patch_size,patch_size:patch_size*2] = b2a.cpu()
                    output_b[:,:,:patch_size,patch_size*2:] = back2b.cpu()
                    # print(output_b.shape)
                    output_b = output_b.reshape((output_b.shape[0] * output_b.shape[1], output_b.shape[2], output_b.shape[3]))
                    output_b = utils.channels_first2last(output_b)
                    original = output_b
                    image_reshaped = original.reshape((original.shape[0] * original.shape[1], original.shape[2]))
                    output_b = target.scaler.inverse_transform(image_reshaped)
                    output_b = output_b.reshape((original.shape))
                    output_b = np.rint(output_b)
                    # print(output_b.shape)
                    save_path = '{}/images/{}/epoch{}_sample{}.tiff'.format(output_path,'B' + testString, epoch, i)
                    tifffile.imsave(save_path, output_b, photometric='rgb')
                    # sio.savemat(save_path, {'preview_b': output_b})
                    print(save_path)

    #===============================================================================
    # Save and load

        def saveModel(self, model, epoch):
            # Create folder to save model architecture and weights
            directory = os.path.join(output_path, 'saved_models')
            if not os.path.exists(directory):
                os.makedirs(directory)

            model_path_w = '{}/saved_models/{}_weights_epoch_{}.pt'.format(output_path, model.name, epoch)
            torch.save(model.state_dict(), model_path_w)
            #model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
            #model.save_weights(model_path_m)
            #json_string = model.to_json()
            #with open(model_path_m, 'w') as outfile:
            #    json.dump(json_string, outfile)
            print('{} has been saved in saved_models/'.format(model.name))

        def RemoteSensing_Transforms(self, opt, data):
            # The transformations here were accomplished using tensorflow-cpu framework
            # print("shape b4", data.shape)
            input_nc = np.size(data, 1)
            # print(input_nc)
            out_data = []
            for i in range(len(data)):
                # print("shape b42", data[i].shape)
                this_data = np.transpose(data[i], (1, 2, 0))
                # print("shape A4", this_data.shape)
                if 'resize' in opt.preprocess:
                    this_data = tf.image.resize(this_data, [opt.load_size, opt.load_size], 
                        method=tf.image.ResizeMethod.BICUBIC) 
                
                if 'crop' in opt.preprocess:
                    this_data = tf.image.random_crop(this_data, size=[opt.crop_size, opt.crop_size, input_nc])
                
                if not opt.no_flip:
                    this_data = tf.image.random_flip_left_right(this_data)
                
                # if opt.convert:
                #     pass
                
                this_data = np.transpose(this_data, (2, 0, 1))
                # print("shape A42", this_data.shape)
                out_data.append(this_data)
                # print(data[i][0][0])
                # print(this_data[0][0])
            out_data = np.array(out_data)
            return out_data

    ModelFitter().fit()        
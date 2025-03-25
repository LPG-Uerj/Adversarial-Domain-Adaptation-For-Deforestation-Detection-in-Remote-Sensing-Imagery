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

# from pedro.models import networks
# import GAN_parameters
# from pedro.models.image_pool import ImagePool



def train(source, target, args, global_args):
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
    
    if continue_train:
        checkpoint_path = glob.glob(os.path.join(global_args.cyclegan_models_path, "checkpoint*"))
        checkpoint_path = checkpoint_path[0]
        checkpoint = torch.load(checkpoint_path)


    print("loading datasets...")
    total_patches_per_epoch = args.total_patches_per_epoch
    train_source = source.cycle_coor
    train_target = target.cycle_coor
    total_patches_per_epoch = min((len(train_source), len(train_target), total_patches_per_epoch))
    num_mini_batches = total_patches_per_epoch//mini_batch_size
    print("[*] Patches per epoch:", num_mini_batches * mini_batch_size)

    aug_idx = augmentations_index(np.zeros(total_patches_per_epoch))


    if use_se:
        # print("[*] Semantic_loss2:", use_se)
        deep_lab_v3p = DeepLabV3plus.create(args)
        deep_lab_v3p = deep_lab_v3p.to(device)
        
    class ModelFitter(utils.ModelFitter):
        def __init__(self):
            super().__init__(num_epochs, num_mini_batches, output_path=output_path)
            
        def initialize(self):
           #self.img_shape = image_shape
            # self.channels = channels
            # Hyper parameters
            # self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
            # self.lambda_2 = 10.0  # Cyclic loss weight B_2_A
            # self.lambda_D = 1  # Weight for loss from discriminator guess on synthetic images
            # self.learning_rate_D = 2e-4
            # self.learning_rate_G = 2e-4
            self.lambda_1 = A2B_cyclic_loss_lambda  # Cyclic loss weight A_2_B
            self.lambda_2 = B2A_cyclic_loss_lambda  # Cyclic loss weight B_2_A
            self.lambda_D = descriminator_loss_lambda  # Weight for loss from discriminator guess on synthetic images
            self.lambda_I = indentity_loss_lambda
            self.seloss = semantic_loss_lambda # weight for Semantic loss
            self.lambda_Diff = diff_loss_lambda # weight for diff_loss

            # self.learning_rate_D = 2e-4
            # self.learning_rate_G = 2e-4
            self.learning_rate_D = descriminator_learning_rate
            self.learning_rate_G = generator_learning_rate
            self.generator_iterations = 1  # Number of generator training iterations in each training loop
            self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
            self.beta_1 = 0.5
            self.beta_2 = 0.999
            self.lambda_S = 0.2
            #self.batch_size = 5
            self.epochs = 200  # choose multiples of 10 since the models are save each 10th epoch
            self.save_interval = 1
            # opt = GAN_parameters.cyclegan_model_options
            self.synthetic_pool_size = 50

            self.use_se = use_se # True for use Semantic loss
            # self.seloss = 0.1 # weight for Semantic loss
            if self.use_se:
                # print("[*] Semantic_loss3:", self.use_se)
                self.gamma = gamma
                self.semantic_learning_rate = semantic_learning_rate
                self.class_weights = torch.FloatTensor(weights).cuda()
                self.seg_loss_fn = FocalLoss(weight = self.class_weights, gamma = self.gamma)       
                self.optim_seg = torch.optim.Adam(deep_lab_v3p.parameters(), lr = self.semantic_learning_rate)
                # self.seg_loss_fn = nn.CrossEntropyLoss()
                # self.optim_seg = torch.optim.Adadelta(deep_lab_v3p.parameters())

            self.use_diff_loss = use_diff_loss
            if self.use_diff_loss:
                self.diff_loss = nn.L1Loss()
            
            # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
            self.use_identity_learning = True
            
            # mog: originally not used
            # PatchGAN 
            self.use_patchgan = True

            # Tweaks
            self.REAL_LABEL = 1.0  # Use e.g. 0.9 to avoid training the discriminators to zero loss
            # Used as storage folder name
            self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + '_test'

            
            # Discriminators
            self.D_A = GAN.modelDiscriminator(name='D_A_model', channels = channels)#Model(inputs=image_A, outputs=guess_A, name='D_A_model')
            self.D_A = self.D_A.to(device)
            self.D_B = GAN.modelDiscriminator(name='D_B_model', channels = channels)#Model(inputs=image_B, outputs=guess_B, name='D_B_model')
            self.D_B = self.D_B.to(device)


            # self.D_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
            # self.D_A.name = "D_A_model"
            # self.D_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
            #                                 opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, opt.gpu_ids)
            # self.D_B.name = "D_B_model"
            
            # print (type(self.D_A))

            # optimizer
            self.opt_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.learning_rate_D, betas=(self.beta_1, self.beta_2))
            self.opt_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.learning_rate_D, betas=(self.beta_1, self.beta_2))

            self.loss_D = nn.MSELoss()#nn.BCELoss() MSE or BCEloss
            print(next(self.D_A.parameters()).device)
            print(device)

            # ======= Generator model ==========
            # Do note update discriminator weights during generator training
            #self.D_A_static.trainable = False
            #self.D_B_static.trainable = False

            # Generators
            self.G_A2B = GAN.modelGenerator(name='G_A2B_model', channels = channels)
            self.G_A2B = self.G_A2B.to(device)
            self.G_B2A = GAN.modelGenerator(name='G_B2A_model', channels = channels)
            self.G_B2A = self.G_B2A.to(device)


            # self.G_A2B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            #                             not opt.no_dropout, opt.linear_output, opt.init_type, opt.init_gain, opt.gpu_ids)
            # self.G_A2B.name = "G_A2B_model"
            # self.G_B2A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
            #                             not opt.no_dropout, opt.linear_output, opt.init_type, opt.init_gain, opt.gpu_ids)
            # self.G_B2A.name = "G_B2A_model"

            # summary(self.G_A2B, (10, 14, 64, 64))
            # summary(self.G_B2A, (10, 14, 64, 64))

            # self.G_A2B.summary()
            self.opt_G_A2B = torch.optim.Adam(self.G_A2B.parameters(), lr=self.learning_rate_G, betas=(self.beta_1, self.beta_2))
            self.opt_G_B2A = torch.optim.Adam(self.G_B2A.parameters(), lr=self.learning_rate_G, betas=(self.beta_1, self.beta_2))
            print(next(self.G_A2B.parameters()).device)
            self.cycle_loss = nn.L1Loss()
            if self.use_identity_learning:
                #self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
                #self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')
                self.loss_G_iden = nn.L1Loss()

            # labels
            if args.patch_size == 64:
                label_shape = (mini_batch_size,) + (1, 6, 6) #  change shape when patch size is not 256
            else:
                label_shape = (mini_batch_size,) + (1, 30, 30) #  change shape when patch size is not 256
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0
            self.ones = torch.from_numpy(ones).float().to(device)
            self.zeros = torch.from_numpy(zeros).float().to(device)
            # print("ones", ones.shape)
            # print("ones", ones)
            
            # Image pools used to update the discriminators
            self.synthetic_pool_A = GAN.ImagePool(self.synthetic_pool_size)
            self.synthetic_pool_B = GAN.ImagePool(self.synthetic_pool_size)

            # self.synthetic_pool_A = ImagePool(self.synthetic_pool_size)
            # self.synthetic_pool_B = ImagePool(self.synthetic_pool_size)

            self.img = {}
            self.img[source_domain] = utils.channels_last2first(source.conc_image)
            self.img[target_domain] = utils.channels_last2first(target.conc_image)

            self.gt = {}
            self.gt[source_domain] = source.new_reference
            self.gt[target_domain] = target.new_reference

            if self.use_diff_loss:
                self.diff_ref = {}
                self.diff_ref[source_domain] = utils.channels_last2first(source.diff_reference)
                self.diff_ref[target_domain] = utils.channels_last2first(target.diff_reference)

            self.source_fixed_preview_patch = np.zeros((1,2))
            self.source_fixed_preview_patch[0] = train_source[random.randint(0,(len(train_source)-1))]
            self.target_fixed_preview_patch = np.zeros((1,2))
            self.target_fixed_preview_patch[0] = train_target[random.randint(0,(len(train_target)-1))]
            # print(self.source_fixed_preview_patch.shape)
            self.last_checkpoint = ''

            if continue_train:
                self.last_checkpoint = checkpoint_path

                self.start_epoch = checkpoint["epoch"] + 1
                self. metrics = checkpoint["metrics"]
                self.G_A2B.load_state_dict(checkpoint["G_A2B_model"])
                self.G_B2A.load_state_dict(checkpoint["G_B2A_model"])
                self.D_A.load_state_dict(checkpoint["D_A_model"])
                self.D_B.load_state_dict(checkpoint["D_B_model"])
                self.opt_D_A.load_state_dict(checkpoint["D_A_optimizer"])
                self.opt_D_B.load_state_dict(checkpoint["D_B_optimizer"])
                self.opt_G_A2B.load_state_dict(checkpoint["G_A2B_optimizer"])
                self.opt_G_B2A.load_state_dict(checkpoint["G_B2A_optimizer"])

                
        def pre_epoch(self, epoch):
            np.random.shuffle(aug_idx)
            np.random.shuffle(train_source)
            np.random.shuffle(train_target)

            self.source_set = np.zeros((len(aug_idx),3))
            self.source_set[:,:2] = train_source[:total_patches_per_epoch]
            self.source_set[:,2] = aug_idx
            self.source_set = torch.utils.data.DataLoader(self.source_set, 
                batch_size = mini_batch_size)


            self.target_set = np.zeros((len(aug_idx),3))
            self.target_set[:,:2] = train_target[:total_patches_per_epoch]
            self.target_set[:,2] = aug_idx            
            self.target_set = torch.utils.data.DataLoader(self.target_set, 
                batch_size = mini_batch_size)

            self.src_iter = iter(self.source_set)
            self.tgt_iter = iter(self.target_set)

            self.G_A2B.train()
            self.G_B2A.train()
            self.D_A.train()
            self.D_B.train()
            
        def get_batch(self, epoch, batch, batch_data):
            src_batch = next(self.src_iter)
            src_coor = src_batch[:,:2]
            src_aug = src_batch[:,2]
            if self.use_diff_loss:
                x , y, z1 = patch_extraction(self.img[source_domain], self.gt[source_domain], 
                    src_coor, patch_size, aug_batch = src_aug, diff_reference_extract = True,
                    diff_reference = self.diff_ref[source_domain])
            else:
                x , y = patch_extraction(self.img[source_domain], self.gt[source_domain], 
                    src_coor, patch_size, aug_batch = src_aug)
            batch_data.append(x)
            batch_data.append(y)

            tgt_batch = next(self.tgt_iter)
            tgt_coor = tgt_batch[:,:2]
            tgt_aug = tgt_batch[:,2]
            if self.use_diff_loss:
                x , y, z2 = patch_extraction(self.img[target_domain], self.gt[target_domain], 
                tgt_coor, patch_size, aug_batch = tgt_aug, diff_reference_extract = True,
                diff_reference = self.diff_ref[target_domain])
            else:
                x , y = patch_extraction(self.img[target_domain], self.gt[target_domain], 
                tgt_coor, patch_size, aug_batch = tgt_aug)
            batch_data.append(x)
            batch_data.append(y)
            
            if self.use_diff_loss:
                batch_data.append(z1)
                batch_data.append(z2)
            
        def train(self, epoch, batch, batch_data, metrics, iteration):
            # ======= Discriminator training ==========
            self.opt_D_A.zero_grad()
            self.opt_D_B.zero_grad()
            real_images_A = torch.from_numpy(batch_data[0]).float().requires_grad_().to(device)
            real_images_B = torch.from_numpy(batch_data[2]).float().requires_grad_().to(device)
            synthetic_images_B = self.G_A2B(real_images_A)
            # synthetic_images_B = self.G_A2B(real_images_A)[0]
            synthetic_images_A = self.G_B2A(real_images_B)
            # synthetic_images_A = self.G_B2A(real_images_B)[0]
            synthetic_images_A = self.synthetic_pool_A.query(synthetic_images_A).to(device)
            synthetic_images_B = self.synthetic_pool_B.query(synthetic_images_B).to(device)

            for _ in range(self.discriminator_iterations):
                x = self.D_A(real_images_A)
                y = self.ones
                DA_loss_real = self.loss_D(x, y)
                x = self.D_B(real_images_B)
                DB_loss_real = self.loss_D(x, y)
                print('[*****]', synthetic_images_A.shape)
                # exit()
                x = self.D_A(synthetic_images_A)
                y= self.zeros
                DA_loss_synthetic = self.loss_D(x, y)
                x = self.D_B(synthetic_images_B)
                DB_loss_synthetic = self.loss_D(x, y)
                DA_loss = 0.5*DA_loss_real + 0.5*DA_loss_synthetic
                DB_loss = 0.5*DB_loss_real + 0.5*DB_loss_synthetic
                D_loss = DA_loss + DB_loss
                metrics["DA_loss"] = DA_loss.item()
                metrics["DB_loss"] = DB_loss.item()
                metrics["D_loss"] = D_loss.item()
                D_loss.backward()
                self.opt_D_A.step()
                self.opt_D_B.step()
                


                # ======= Generator training ==========
            self.opt_G_A2B.zero_grad()
            self.opt_G_B2A.zero_grad()
            y = self.ones


            for _ in range(self.generator_iterations):
                #synthetic_images_B = self.G_A2B(real_images_A)
                #synthetic_images_A = self.G_B2A(real_images_B)
                x = self.D_A(synthetic_images_A)
                gA_d_loss_synthetic = self.lambda_D*self.loss_D(x, y)
                x = self.D_B(synthetic_images_B)
                gB_d_loss_synthetic = self.lambda_D*self.loss_D(x, y)
                # reconstructed_images_B = self.G_A2B(synthetic_images_A)
                reconstructed_images_B = self.G_A2B(synthetic_images_A)[0]
                # reconstructed_images_A = self.G_B2A(synthetic_images_B)
                reconstructed_images_A = self.G_B2A(synthetic_images_B)[0]
                reconstruction_loss_A = self.lambda_1*self.cycle_loss(real_images_A, reconstructed_images_A)
                reconstruction_loss_B = self.lambda_2*self.cycle_loss(real_images_B, reconstructed_images_B)
                # Store training data
            metrics["gA_d_loss_synthetic"] = gA_d_loss_synthetic.item()
            metrics["gB_d_loss_synthetic"] = gB_d_loss_synthetic.item()
            metrics["reconstruction_loss_A"] = reconstruction_loss_A.item()
            metrics["reconstruction_loss_B"] = reconstruction_loss_B.item()

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B            
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            metrics["reconstruction_loss"] = reconstruction_loss.item()

                # Identity training
            if self.use_identity_learning:
                # identity_B = self.G_A2B(real_images_B)
                identity_B = self.G_A2B(real_images_B)[0]
                G_A2B_identity_loss = self.lambda_I*self.loss_G_iden(identity_B, real_images_B)
                # identity_A = self.G_B2A(real_images_A)
                identity_A = self.G_B2A(real_images_A)[0]
                G_B2A_identity_loss = self.lambda_I*self.loss_G_iden(identity_A, real_images_A)                
                GA_loss += G_B2A_identity_loss
                GB_loss += G_A2B_identity_loss
                metrics["GA_identity_loss"] = G_B2A_identity_loss.item()
                metrics["GB_identity_loss"] = G_A2B_identity_loss.item()

            if self.use_diff_loss:
                half_channels = channels//2
                real_diff_A = torch.from_numpy(batch_data[4]).float().requires_grad_().to(device)
                diff_A = synthetic_images_B[:, half_channels:,:,:] - synthetic_images_B[:, 0:half_channels,:,:]
                G_A2B_diff_loss = self.lambda_Diff * self.diff_loss(diff_A, real_diff_A)
                GA_loss += G_A2B_diff_loss
                real_diff_B = torch.from_numpy(batch_data[5]).float().requires_grad_().to(device)
                diff_B = synthetic_images_A[:, half_channels:,:,:] - synthetic_images_A[:, 0:half_channels,:,:]
                G_B2A_diff_loss = self.lambda_Diff * self.diff_loss(diff_B, real_diff_B)
                GB_loss += G_B2A_diff_loss
                metrics["GA_diff_loss"] = G_A2B_diff_loss.item()
                metrics["GB_diff_loss"] = G_B2A_diff_loss.item()

            metrics["GA_loss"] = GA_loss.item()
            metrics["GB_loss"] = GB_loss.item()
            G_loss = GA_loss + GB_loss
            metrics["G_loss"] = G_loss.item()
            G_loss.backward()
            self.opt_G_B2A.step()
            self.opt_G_A2B.step()

                # ======= segmentation training ==========
            if self.use_se:
                # print("[*] Semantic_loss:4", use_se)
                self.optim_seg.zero_grad()
                self.opt_G_A2B.zero_grad()
                x = torch.from_numpy(batch_data[0]).float().requires_grad_().to(device)
                # x = deep_lab_v3p(self.G_A2B(x))
                x = deep_lab_v3p(self.G_A2B(x)[0])
                y = torch.from_numpy(batch_data[1]).long().to(device)
                loss = self.seloss*self.seg_loss_fn(x, y)
                loss.backward()
                self.optim_seg.step()
                self.opt_G_A2B.step()
                metrics["seg_loss"] = loss.item()
                metrics["seg_acc"] = (x.argmax(1) == y).sum().item() / (x.shape[0] * patch_size * patch_size)
            
            # mog: removed 
            # if iteration % 20 == 0:
            #         # Save temporary images continously
            #     self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)                                                                                                          
            
        def post_epoch(self, epoch, metrics):
            print('\n')
            if save_previews:
                print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
                self.saveImages(epoch, self.source_fixed_preview_patch, 
                    self.target_fixed_preview_patch)
            if epoch==num_epochs-1 or epoch % (save_model_each_n_epoch//2) == 0:
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

                checkpoint_path = '{}/saved_models/checkpoint_epoch_{}'.format(output_path, epoch)
                checkpoint_data = {
                    # general train
                    'epoch': epoch,
                    'metrics': metrics,
                    # models
                    'G_A2B_model': self.G_A2B.state_dict(),
                    'G_B2A_model': self.G_B2A.state_dict(),
                    'D_A_model': self.D_A.state_dict(),
                    'D_B_model': self.D_B.state_dict(),
                    
                    # optimizers
                    'D_A_optimizer': self.opt_D_A.state_dict(),
                    'D_B_optimizer': self.opt_D_B.state_dict(),
                    'G_A2B_optimizer': self.opt_G_A2B.state_dict(),
                    'G_B2A_optimizer': self.opt_G_B2A.state_dict()
                    }
                if use_se:
                    checkpoint_data['Semantic_model'] = deep_lab_v3p.state_dict()
                    checkpoint_data['Semantic_optimizer'] = self.optim_seg.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)

                if epoch > 0:
                    os.remove(self.last_checkpoint)
                self.last_checkpoint = checkpoint_path
                print("Checkpoint Saved:", checkpoint_path)



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

    ModelFitter().fit()
    


from parameters.training_parameters import Train_cyclegan, Global
from dataset_preprocessing.dataset_select import select_domain
if __name__=='__main__':
    global_parameters = Global()
    train_parameters = Train_cyclegan()
    
    base_path = train_parameters.output_path    
    
    source, source_params = select_domain(global_parameters.source_domain)
    target, target_params = select_domain(global_parameters.target_domain)
    
    source_params.patches_dimension = train_parameters.patch_size
    source_params.stride = source_params.train_source_stride
    ready_domain(source, source_params, train_set = False, cyclegan_set = True)
    
    target_params.patches_dimension = train_parameters.patch_size
    target_params.stride = target_params.train_source_stride
    ready_domain(target, target_params, train_set = False, cyclegan_set = True)

    train(source, target, train_parameters, global_parameters)
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import numpy as np
import torch
import torch.nn as nn
import Desmatamento.code.utilities.utils as utils
import sys

from Desmatamento.code.utilities.training_utils import *
import DeepLabV3plus
import GAN


def train(source, source_args, args, global_args, gan_models_path):
    print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())
    device = torch.device("cuda:0")
    source_domain = global_args.source_domain
    IMAGE_SAVE_PATH = args.savedata_folder + source_domain + '/'

    path_pre = []
    path_pre.append(IMAGE_SAVE_PATH)
    print("make dir")
    for i in range (len(path_pre)):
        p = path_pre[i]
        if (os.path.isdir(p) == False):
            os.makedirs(p)
            print('\''+ p + '\'' + " path created")

    patch_size = args.patch_size
    num_classes = args.num_classes
    channels = args.channels

    weights = args.weights
    gamma = args.gamma
    learning_rate = args.learning_rate

    starting_model_index = args.starting_model_index

    early_skip = args.early_skip
    early_skip_limit = args.early_skip_limit


    # dilation_rates = args.dilation_rates

    MODEL_PATH = gan_models_path
    output_path = MODEL_PATH
    model_names = [name for name in os.listdir(MODEL_PATH) if
                   os.path.splitext(name)[1] == '.pt' and name.split('_')[1] == 'A2B']
    model_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(model_names)

    epochs_per_model = args.epochs_per_model
    mini_batch_size = args.batch_size
    # num_mini_batches = args.num_batches
    iteration_modifier = args.iteration_modifier
    if iteration_modifier < 1:
        iteration_modifier = 1
    num_epochs = epochs_per_model*(len(model_names)-1 - starting_model_index)
    print("[*] Total number of epochs:", num_epochs)

    global best_gan_epoch
    best_gan_epoch = ''


    print("loading datasets...")
    train_set = source.augmented_train_set
    num_mini_batches = int(train_set.shape[0]//mini_batch_size//iteration_modifier)


    def changedataset(path_to_weights):
        print("loading adapted datasets...")
        source.Prepare_GAN_Set(source_args, path_to_weights, eval = True, 
            adapt_back_to_original = False)
        # train_set = source.adapted_train_set
        

    class ModelFitter(Desmatamento.code.utilities.utils.ModelFitter):
        def __init__(self):
            super().__init__(num_epochs, num_mini_batches, output_path=output_path)
            
        def initialize(self):
        	# deeplab parameters
            self.gamma = gamma
            self.learning_rate = learning_rate
            self.class_weights = torch.FloatTensor(weights).cuda()
            self.seg_loss_fn = FocalLoss(weight = self.class_weights, gamma = self.gamma)
            self.deep_lab_v3p = DeepLabV3plus.create(args)
            self.deep_lab_v3p = self.deep_lab_v3p.to(device)

            self.model_num = 1 + starting_model_index

            #early stop parameters
            self.early_skip = early_skip
            self.early_skip_limit = early_skip_limit
            self.early_skip_count = 0
            self.epoch_for_next_model = 0
            self.this_model_best_f1 = 0

            self.validation_set = {}
            self.img = {}
            self.gt = {}
            self.running_model = ''
            self.running_model_epoch = model_names[0].split('_')[-1].split('.')[0]
            # print(self.running_model_epoch)
            global best_gan_epoch
            best_gan_epoch = model_names[0].split('_')[-1]

        def pre_epoch(self, epoch):
            if epoch % epochs_per_model == 0:
                del self.deep_lab_v3p
                self.deep_lab_v3p = DeepLabV3plus.create(args)
                self.deep_lab_v3p = self.deep_lab_v3p.to(device)
                self.optim_seg = torch.optim.Adam(self.deep_lab_v3p.parameters(), lr = self.learning_rate)
                
                self.epoch_for_next_model = epoch + epochs_per_model
                print(self.epoch_for_next_model)
                self.this_model_best_f1 = 0
                self.early_skip_count = 0
               
                # self.G_A2B = GAN.modelGenerator(name='G_A2B_model')
                print(model_names[self.model_num])
                path_to_weights = MODEL_PATH + model_names[self.model_num]
                self.running_model = model_names[self.model_num]
                self.running_model_epoch = model_names[self.model_num].split('_')[-1].split('.')[0]
                # metrics["model_" + self.running_model_epoch] = {}
                # print(metrics)
                changedataset(path_to_weights)
                self.model_num += 1

                self.validation_set[source_domain] = torch.utils.data.DataLoader(source.central_pixels_coor_vl, 
                batch_size = mini_batch_size)
                self.img[source_domain] = Desmatamento.code.utilities.utils.channels_last2first(source.adapted_image)
                self.gt[source_domain] = source.new_reference
            
            self.epoch_train_set = torch.utils.data.DataLoader(train_set, 
            	batch_size = mini_batch_size, shuffle = True)
            self.tr_coor = iter(self.epoch_train_set)

            self.deep_lab_v3p.train()

        def get_batch(self, epoch, batch, batch_data):
            batch = next(self.tr_coor)
            coor_batch = batch[:,:2]
            aug_batch = batch[:, 2]

            x , y = patch_extraction(self.img[source_domain], 
            	self.gt[source_domain], coor_batch, patch_size, aug_batch = aug_batch)
            batch_data.append(x)
            batch_data.append(y)

            
        def train(self, epoch, batch, batch_data, metrics, iteration):
            self.optim_seg.zero_grad()
            x = torch.from_numpy(batch_data[0]).float().requires_grad_().to(device)
            x = self.deep_lab_v3p(x)
            y = torch.from_numpy(batch_data[1]).long().to(device)
            loss = self.seg_loss_fn(x, y)
            loss.backward()
            self.optim_seg.step()
            metrics["seg_loss"] = loss.item()

            x = x.argmax(1)
            x = x.cpu()
            y = y.cpu()

            acc, f1, rec, prec, alert = return_metrics(y,x)
            metrics["seg_acc"] = acc
            metrics["seg_f1"] = f1
            metrics["seg_rec"] = rec
            metrics["seg_prec"] = prec
            metrics["seg_alert"] = alert

        def load_model_and_weights(self, model):
            #path_to_model = os.path.join('generate_images', 'models', '{}.json'.format(model.name))
            path_to_weights = os.path.join('generate_images', 'models', '{}.pt'.format(model.name))
            #model = model_from_json(path_to_model)
            model.load_state_dict(torch.load(path_to_weights))
            
        def post_epoch(self, epoch, metrics):
            self.deep_lab_v3p.eval()
            with torch.no_grad():
                for domain in (self.validation_set):
                    print("validation domain", domain)
                    count = 0
                    loss = 0
                    acc = 0
                    f1 = 0
                    rec = 0
                    prec = 0
                    alert = 0
                    for x in (self.validation_set[domain]):
                        count += 1
                        n = x.shape[0]
                        coor_batch = x
                        x , y = patch_extraction(self.img[domain], self.gt[domain], coor_batch, patch_size)
                        x = torch.from_numpy(x).float().to(device)
                        x = self.deep_lab_v3p(x)
                        y = torch.from_numpy(y).long().to(device)
                        loss += self.seg_loss_fn(x, y).item()

                        x = x.argmax(1)
                        x = x.cpu()
                        y = y.cpu()
                        a, b, c, d, e = return_metrics(y,x)
                        acc += a
                        f1 += b
                        rec += c
                        prec += d
                        alert += e

                    num_slices = count
                    metrics[f"val_loss_{domain}"] = loss / num_slices
                    metrics[f"val_acc_{domain}"] = acc / num_slices
                    metrics[f"val_f1_{domain}"] = f1 / num_slices
                    metrics[f"val_rec_{domain}"] = rec / num_slices
                    metrics[f"val_prec_{domain}"] = prec / num_slices
                    metrics[f"val_alert_{domain}"] = alert / num_slices

            f1 = metrics["val_f1_" + source_domain]
            global best_gan_epoch
            if epoch > 0 and f1 > np.max(self.history["val_f1_" + source_domain]):
                path = f"{output_path}/model_{epoch:02d}_{f1:.4f}.pt"
                best_gan_epoch = self.running_model
                best_gan_epoch = best_gan_epoch.split('_')[-1]
                print("[*] Best GAN epoch updated:", best_gan_epoch)
                print(f"saving model weights to '{path}'")
                torch.save(self.deep_lab_v3p.state_dict(), path)

            elif epoch == 0 or epoch == (num_epochs-1) or epoch%10 == 0:
                path = f"{output_path}/model_{epoch:02d}_{acc:.4f}.pt"
                print(f"saving model weights to '{path}'")
                torch.save(self.deep_lab_v3p.state_dict(), path)

            if f1 > self.this_model_best_f1 or epoch % epochs_per_model == 0:
                print("[*] This model best f1score =", f1)
                self.this_model_best_f1 = f1
                if self.early_skip:
                    self.early_skip_count = 0
            else:
                if self.early_skip:
                    self.early_skip_count += 1
            if self.early_skip:                    
                print("Countdown to early skip: " + str(self.early_skip_count) + '/' + str(self.early_skip_limit))
            print("[*] Best GAN so far:", best_gan_epoch)

    print("Starting train...")
    ModelFitter().fit()
    # print(np.max(self.history["val_f1_" + source_domain]))
    return best_gan_epoch

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
import json
import numpy
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import torch
import torch.nn as nn
import utils

from Tools import *
import DeepLabV3plus
# import GAN
from scipy.special import softmax


def train(source, target, args, train_source_args, global_args, best_source_model):

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
    iteration_modifier = args.iteration_modifier
    if iteration_modifier < 1:
        iteration_modifier = 1
    num_epochs = args.num_epochs

    # losses weights (lambdas)
    semantic_loss_lambda = args.semantic_loss_lambda
    cyclic_loss_lambda = args.cyclic_loss_lambda
    feature_discriminator_loss_lambda = args.feature_discriminator_loss_lambda

    # Learning rates
    learning_rate = args.learning_rate
    # descriminator_learning_rate = args.descriminator_learning_rate
    # encoder_learning_rate = args.encoder_learning_rate

    weights = args.weights
    gamma = args.gamma

    dilation_rates = args.dilation_rates

    use_early_stop = args.early_stop
    early_stop_limit = args.early_stop_limit

    source_domain = global_args.source_domain
    target_domain = global_args.target_domain

    task_loss = args.task_loss
    pseudo_label = args.pseudo_label
    feature_adapt = args.feature_adaptation

    assert (task_loss) or (pseudo_label)

    large_latent_space = args.large_latent_space
    label_type = args.label_type #  1 - use a matrix as label for each domain
    # 0 - use a scaler as label for each domain, 1 - use a matrix as label for each domain, 2 - create merged/mixed matrices as labels
    train_encoder_only_on_target_domain = args.train_encoder_only_on_target_domain 
    discriminator_type = args.discriminator_type # 0 for best discriminator_type

    
    best_source_model_path = best_source_model
    # target_to_source_model_path = global_args.A2B_best_gan_model


    # 'best_target_model_path' is the best on source (through validation), which
    # in a normal use case is the only ground truth available
    # 'theorical_best_target_model' is the best on target (trough validation), but
    # in a normal use case its ground truth wouldn't be available
    global best_target_model_path, theorical_best_target_model
    best_target_model_path = ''
    theorical_best_target_model = ''


    print("loading datasets...")
    # mog: remeber that these are just coordinates, not patches
    # the used domains are S->T, T and T->S
    source_adapted_train_set = source.augmented_train_set
    target_default_train_set = target.augmented_train_set
    # target_adapted_train_set = target.augmented_train_set
    print(len(source_adapted_train_set), len(target_default_train_set))
    shorter_set = min((len(source_adapted_train_set), len(target_default_train_set)))
    print(shorter_set)
    num_mini_batches = (int(shorter_set//mini_batch_size))//iteration_modifier
    print("[*] Patches per epoch:", num_mini_batches * mini_batch_size)



        
    print("Defining and loading Networks...")   
    deep_lab_v3p = DeepLabV3plus.create(args)
    deep_lab_v3p = deep_lab_v3p.to(device)

    source_deep_lab_v3p = DeepLabV3plus.create(train_source_args)
    source_deep_lab_v3p = source_deep_lab_v3p.to(device) # source semantic model
    checkpoint1 = torch.load(best_source_model_path)
    source_deep_lab_v3p.load_state_dict(checkpoint1)
    source_deep_lab_v3p.eval()

    class ck(nn.Module):
        def __init__(self, i, k, use_normalization):
            super(ck, self).__init__()
            self.conv_block = self.build_conv_block(i, k, use_normalization)

        def build_conv_block(self, i, k, use_normalization):
            conv_block = []                       
            conv_block += [nn.Conv2d(i, k, 1)]
            if use_normalization:
                conv_block += [nn.BatchNorm2d(k)]
            conv_block += [nn.ReLU()]
            return nn.Sequential(*conv_block)

        def forward(self, x):
            out = self.conv_block(x)
            return out

    if feature_adapt:

        encoder = nn.Sequential(
            deep_lab_v3p.part1,
            deep_lab_v3p.part2,
            deep_lab_v3p.aspp
        ).to(device)
            
        discriminator_num_output_classes = 1
        discriminator = []
        if discriminator_type == 3:
            discriminator.extend((
                InvBottleneck(256, 6, 64, 3, 1),
                InvBottleneck(64, 6, 16, 3, 1),
                InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
            ))
        elif discriminator_type == 2:
            discriminator.extend((
                InvBottleneck(256, 6, 64, 1, 1),
                InvBottleneck(64, 6, 16, 1, 1),
                InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
            ))
        elif discriminator_type == 1:
            discriminator.extend((
                InvBottleneck(256, 6, discriminator_num_output_classes, 1, 1),
            ))
        else:
            assert discriminator_type == 0
            discriminator.extend((
                ck(256, 256, False),
                ck(256, 256, False),
                nn.Conv2d(256, discriminator_num_output_classes, 1),
            ))
        del discriminator_num_output_classes
        discriminator = nn.Sequential(*discriminator).to(device)

    # print("are the tensors equal?")

    def compare_models(model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('Models match perfectly! :)')

    # this_enc = nn.Sequential(
    #     deep_lab_v3p.part1,
    #     deep_lab_v3p.part2,
    #     deep_lab_v3p.aspp).to(device)
    # # print(list(encoder.parameters()) == list(this_enc.parameters()))
    # # torch.all(torch.eq(tens_a, tens_b))
    # # print(torch.all(torch.eq(deep_lab_v3p, source_deep_lab_v3p)))
    # # compare_models(deep_lab_v3p, source_deep_lab_v3p)
    # compare_models(encoder, this_enc)
    # del this_enc
    # assert 0 == 1

    class ModelFitter(utils.ModelFitter):
        def __init__(self):
            super().__init__(num_epochs, num_mini_batches, output_path=output_path)
            
        def initialize(self):
            self.lambda_S = semantic_loss_lambda
            self.lambda_C = cyclic_loss_lambda
            self.lambda_D = feature_discriminator_loss_lambda

            self.use_early_stop = use_early_stop
            self.early_stop_limit = 0
            if self.use_early_stop:
                self.early_stop_limit = early_stop_limit

            self.gamma = gamma
            self.learning_rate = learning_rate

            print("weights:", weights)
            print("gamma:", self.gamma)
            print("learning rate: ", self.learning_rate)

            self.class_weights = torch.FloatTensor(weights).cuda()
            self.seg_loss_fn = FocalLoss(weight = self.class_weights, gamma = self.gamma)       
            self.optim_seg = torch.optim.Adam(deep_lab_v3p.parameters(), lr = self.learning_rate)

            self.feature_patch_size = patch_size // (8 if large_latent_space else 16)
            self.feature_shape = (mini_batch_size//2, 1, self.feature_patch_size, self.feature_patch_size)

            # self.cycseg_loss_fn = nn.KLDivLoss()
            self.disc_loss_fn = nn.MSELoss() # nn.CrossEntropyLoss() # MSE or CE
            
            # mog: precisa tirar comentario para usar feature
            if feature_adapt:
	            self.optim_enc = torch.optim.Adadelta(encoder.parameters())
	            self.optim_disc = torch.optim.Adadelta(discriminator.parameters())
            self.train_adaptation = self.train_adaptation_regular

            self.validation_set = {}
            self.validation_set_size = {}
            # mog: definir se usa o original ou o adaptado pra validação
            self.validation_set[source_domain] = torch.utils.data.DataLoader(source.central_pixels_coor_vl, 
                batch_size = mini_batch_size)
            self.validation_set_size[source_domain] = len(source.central_pixels_coor_vl)
            self.validation_set[target_domain] = torch.utils.data.DataLoader(target.central_pixels_coor_vl, 
                batch_size = mini_batch_size)
            self.validation_set_size[target_domain] = len(target.central_pixels_coor_vl)

            self.img = {}
            # mog: source.conc_image just used on validation
            # self.img[source_domain] = utils.channels_last2first(source.conc_image)
            # source_domain in here is adapated
            self.img[source_domain] = utils.channels_last2first(source.adapted_image)
            self.img[target_domain] = utils.channels_last2first(target.conc_image)
            self.img[target_domain + "_adapted"] = utils.channels_last2first(target.adapted_image)

            self.gt = {}
            self.gt[source_domain] = source.new_reference
            self.gt[target_domain] = target.new_reference    

            
        def pre_epoch(self, epoch):
            self.epoch_source_adapted_set = torch.utils.data.DataLoader(source_adapted_train_set, 
                batch_size = mini_batch_size, shuffle = True)
            self.source_adapted_tr_coor = iter(self.epoch_source_adapted_set)
            self.epoch_target_set = torch.utils.data.DataLoader(target_default_train_set, 
                batch_size = mini_batch_size, shuffle = True)
            self.target_tr_coor = iter(self.epoch_target_set)
            # self.epoch_target_adapted_set = torch.utils.data.DataLoader(target_adapted_train_set, 
            #     batch_size = mini_batch_size, shuffle = True)
            # self.target_adapted_tr_coor = iter(self.epoch_target_adapted_set)

            deep_lab_v3p.train()
            
        def get_batch(self, epoch, batch, batch_data):
            batch = next(self.source_adapted_tr_coor)
            coor_batch = batch[:,:2]
            aug_batch = batch[:, 2]

            x , y = patch_extraction(self.img[source_domain], 
                self.gt[source_domain], coor_batch, patch_size, aug_batch = aug_batch)
            batch_data.append(x) # 0
            batch_data.append(y) # 1

            batch = next(self.target_tr_coor)
            coor_batch = batch[:,:2]
            aug_batch = batch[:, 2]

            x , y = patch_extraction(self.img[target_domain], 
                self.gt[target_domain], coor_batch, patch_size, aug_batch = aug_batch)
            batch_data.append(x) # 2
            batch_data.append(y) # 3

            # batch = next(self.target_tr_coor)
            # coor_batch = batch[:,:2]
            # aug_batch = batch[:, 2]
            # mog: batch same as actual target
            
            x , y = patch_extraction(self.img[target_domain + "_adapted"], 
                self.gt[target_domain], coor_batch, patch_size, aug_batch = aug_batch)
            batch_data.append(x) # 4
            # batch_data.append(y)

        def train(self, epoch, batch, batch_data, metrics, iteration):
            self.optim_seg.zero_grad()
            loss = 0
            if pseudo_label:
                x = torch.from_numpy(batch_data[4][:self.feature_shape[0]]).float().to(device)
                ## mog: pseudo label here \/
                # gera pseudo e compara com a classificacao de Ft
                x = source_deep_lab_v3p(x)
                # T->S = Fs(T->S)
                xpred = x.argmax(1)
                y = torch.from_numpy(batch_data[2][:self.feature_shape[0]]).float().requires_grad_().to(device)
                y = deep_lab_v3p(y)
                # T = Ft(T)
                ypred = y.argmax(1)
                cycloss = self.seg_loss_fn(y, xpred) + self.seg_loss_fn(x, ypred)
                metrics["kl_loss"] = cycloss.item()
                loss += self.lambda_C*cycloss
                # just used if used solo:
                x = ypred
                y = xpred

            # /\
            if task_loss:
                x = torch.from_numpy(batch_data[0][:self.feature_shape[0]]).float().requires_grad_().to(device)
                x = deep_lab_v3p(x)
                y = torch.from_numpy(batch_data[1][:self.feature_shape[0]]).long().to(device)
                loss = self.seg_loss_fn(x, y)
                metrics["seg_loss"] = loss.item()
                loss += self.lambda_S*loss # change 0.35 for different cross-domain semantic consistent loss weight
                x = x.argmax(1)

            loss.backward()
            self.optim_seg.step()        
            # metrics["seg_acc"] = (x.argmax(1) == y).sum().item() / (x.shape[0] * patch_size * patch_size)
            x = x.cpu()
            y = y.cpu()
            # print(return_metrics(y,x))
            # print(calcula_metricas(y,x))

            acc, f1, rec, prec, alert = return_metrics(y,x)
            metrics["seg_acc"] = acc
            metrics["seg_f1"] = f1
            metrics["seg_rec"] = rec
            metrics["seg_prec"] = prec
            metrics["seg_alert"] = alert

            # domain adaptation:
            # mog: feature adaptation*
            if feature_adapt:
                self.train_adaptation(epoch, batch, batch_data, metrics)

        def train_adaptation_regular(self, epoch, batch, batch_data, metrics):
            # discriminator training:
            self.optim_disc.zero_grad()
            if label_type < 2:
                x = numpy.concatenate((batch_data[0][:self.feature_shape[0]], batch_data[2][:self.feature_shape[0]]))
                x = torch.from_numpy(x).float().requires_grad_().to(device)
                i = numpy.concatenate((numpy.ones(self.feature_shape), numpy.zeros(self.feature_shape)))
                #j = numpy.ones(i.shape) #- i
                y = torch.from_numpy(i).float().to(device)
                # inv_y = torch.from_numpy(j).long().squeeze().to(device)
                inv_y = y
                x = discriminator(encoder(x))
                if label_type == 0:
                    x = x.squeeze()
            else:
                assert label_type == 2
                x = [torch.from_numpy(batch_data[0][:self.feature_shape[0]]).float().requires_grad_().to(device)]
                x.append(torch.from_numpy(batch_data[2][:self.feature_shape[0]]).float().requires_grad_().to(device))
                i = numpy.where(numpy.random.uniform(size=self.feature_shape) < .5, 1, 0)
                j = numpy.ones(self.feature_shape) #- i
                y = torch.from_numpy(i).long().squeeze().to(device)
                inv_y = torch.from_numpy(j).long().squeeze().to(device)
                i = torch.from_numpy(i).float().requires_grad_().to(device)
                j = torch.from_numpy(j).float().requires_grad_().to(device)
                x = discriminator(i*encoder(x[0]) + j*encoder(x[1]))
            loss = self.disc_loss_fn(x, y)
            loss.backward()
            metrics["disc_loss"] = loss.item()
            # if use_wasserstein_loss:
            #     metrics["disc_acc"] = 0.5 # there is no easy/meaningful way to compute accuracy when using MSE loss
            # else:
            metrics["disc_acc"] = 0.5#(x.argmax(1) == y).sum().item() / (x.shape[0] * self.feature_patch_size * self.feature_patch_size) # decomment for CE
            if metrics["disc_acc"] < 1: 
                self.optim_disc.step()
            # encoder training:        
            self.optim_enc.zero_grad()
            if train_encoder_only_on_target_domain:
                x = batch_data[2][:self.feature_shape[0]]
                x = torch.from_numpy(x).float().requires_grad_().to(device)
                inv_y = numpy.ones(self.feature_shape)
                inv_y = torch.from_numpy(inv_y).float().to(device)
                x = discriminator(encoder(x))
            elif label_type < 2:
                x = numpy.concatenate((batch_data[0][:self.feature_shape[0]], batch_data[2][:self.feature_shape[0]]))
                x = torch.from_numpy(x).float().requires_grad_().to(device)
                x = discriminator(encoder(x))
            else:
                x = [torch.from_numpy(batch_data[0][:self.feature_shape[0]]).float().requires_grad_().to(device)]
                x.append(torch.from_numpy(batch_data[2][:self.feature_shape[0]]).float().requires_grad_().to(device))
                x = discriminator(i*encoder(x[0]) + j*encoder(x[1]))
            if label_type == 0:
                x = x.squeeze()
            loss = self.lambda_D*self.disc_loss_fn(x, inv_y) #change 0.01 for different feature domain adversarial loss weight
            loss.backward()
            if metrics["disc_acc"] > 0: # training the encoder only if the discriminator is not too bad when use CE loss but when MSE keep 0
                self.optim_enc.step()
            metrics["enc_loss"] = loss.item()
            
        def post_epoch(self, epoch, metrics):
            deep_lab_v3p.eval()
            with torch.no_grad():
                for domain in (self.validation_set):
                    print("validation domain", domain)
                    count = 0
                    acc, loss, f1, rec, prec, alert  = 0, 0, 0, 0, 0, 0
                    val_x = np.zeros((self.validation_set_size[domain], num_classes-1, patch_size, patch_size))
                    val_y = np.zeros((self.validation_set_size[domain], patch_size, patch_size))
                    this_batch = 0
                    for x in (self.validation_set[domain]):
                        count += 1
                        this_batch_size = x.shape[0]
                        coor_batch = x
                        x , y = utils.patch_extraction(self.img[domain], self.gt[domain], coor_batch, patch_size)
                        x = torch.from_numpy(x).float().to(device)
                        x = deep_lab_v3p(x)
                        y = torch.from_numpy(y).long().to(device)
                        loss += self.seg_loss_fn(x, y).item()

                        x = x.cpu().numpy()
                        y = y.cpu().numpy()
                        val_x[this_batch:this_batch + this_batch_size] = x[:,:2]
                        val_y[this_batch:this_batch + this_batch_size] = y
                        this_batch += this_batch_size

                    val_x = softmax(val_x, axis = 1)
                    val_x = val_x.argmax(1)
                    acc, f1, rec, prec, alert = return_metrics(val_y, val_x)

                    metrics[f"val_acc_{domain}"] = acc
                    metrics[f"val_f1_{domain}"] = f1
                    metrics[f"val_rec_{domain}"] = rec
                    metrics[f"val_prec_{domain}"] = prec
                    metrics[f"val_alert_{domain}"] = alert 
                    metrics[f"val_loss_{domain}"] = loss / count

            # this_enc = nn.Sequential(
            #     deep_lab_v3p.part1,
            #     deep_lab_v3p.part2,
            #     deep_lab_v3p.aspp).to(device)
            # # print(list(encoder.parameters()) == list(this_enc.parameters()))
            # # torch.all(torch.eq(tens_a, tens_b))
            # # print(torch.all(torch.eq(deep_lab_v3p, source_deep_lab_v3p)))
            # # compare_models(deep_lab_v3p, source_deep_lab_v3p)
            # compare_models(encoder, this_enc)
            # del this_enc
            # assert 0 == 1

            f1 = metrics["val_f1_" + target_domain]
            f1_visible = metrics["val_f1_" + source_domain]
            global best_target_model_path
            global theorical_best_target_model
            if self.use_early_stop and epoch > 0:
                if f1 < np.max(self.history["val_f1_" + target_domain]):
                    self.early_stop_count += 1
                else:
                    self.early_stop_count = 0
                
                print("Countdown to early stop: " + str(self.early_stop_count) + '/' + str(self.early_stop_limit))
                
                if self.early_stop_count == self.early_stop_limit:
                    self.early_stop = True
                    path = os.path.join(output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
                    print(f"saving model weights to '{path}'")
                    torch.save(deep_lab_v3p.state_dict(), path)

            if epoch > 0 and f1 > np.max(self.history["val_f1_" + target_domain]):
                path = os.path.join(output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
                theorical_best_target_model = path
                print("[*] Theorical best model updated:", path)
                print(f"saving model weights to '{path}'")
                torch.save(deep_lab_v3p.state_dict(), path)

            # if epoch > 0 and f1_visible > np.max(self.history["val_f1_" + source_domain]):
            #     path = f"{output_path}/model_{epoch:02d}_{f1:.4f}.pt"
            #     best_target_model_path = path
            #     print("[*] Best model updated:", path)
            #     print(f"saving model weights to '{path}'")
            #     torch.save(deep_lab_v3p.state_dict(), path)


            elif epoch == 0 or epoch == (num_epochs-1) or epoch%10 == 0:
                path = os.path.join(output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
                print(f"saving model weights to '{path}'")
                torch.save(deep_lab_v3p.state_dict(), path)
                if epoch == 0:
                    best_target_model_path = path
                    theorical_best_target_model = path

            print("[*] Best visible model so far:", best_target_model_path)
            print("[*] Best model so far:", theorical_best_target_model)

    ModelFitter().fit()
    # return best_target_model_path, theorical_best_target_model
    return theorical_best_target_model





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import glob
import itertools
import time
import json
import numpy
import PIL.Image
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import torch
import torch.nn as nn
import Desmatamento.code.utilities.utils as utils


# In[ ]:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())
# user parameters
patch_size = 256
num_classes = 6

feature_adapt = True

path_pre = []

# original:
# source_domain = "hannover"
# target_domain = "vaihingen"
# rectarget_domain = "recvaihingen"

# experimento_1:
# AJUSTAR TBM BOOTSTRAP
source_domain = "potsdam" # USAR ADAPTADO/RECONSTRUIDO
target_domain = "vaihingen"
rectarget_domain = "recvaihingen"

output_path = f"./tmp/models/target_seg/experimento_1"
path_pre.append(output_path)

sourcemodel_file_path = "./tmp/models/source_seg/potsdamirgwithimg300/bak_model_299_0.4297.pt"

print("loading datasets...")
data = {}
data["potsdam"] = {}
images, ground_truth = utils.load_potsdam_irg2(resize_factor=(5/9)) # ADAPTADO
data["potsdam"]["images"] = images
data["potsdam"]["gt"] = ground_truth

# data["vaihingen"] = {}
# images, ground_truth = utils.load_vaihingen_irg2()
# data["vaihingen"]["images"] = images
# data["vaihingen"]["gt"] = ground_truth

# data["recvaihingen"] = {}
# images, ground_truth = utils.load_recvaihingen_irg()
# data["recvaihingen"]["images"] = images
# data["recvaihingen"]["gt"] = ground_truth

# acclist = [0.0]
######
#####
####
###
##
#
# FIM EXPERIMENTO 1



# experimento_2:
# AJUSTAR TBM BOOTSTRAP
# source_domain = "vaihingen" # USAR ADAPTADO/RECONSTRUIDO
# target_domain = "potsdam"
# rectarget_domain = "recpotsdam"

# output_path = f"./tmp/models/target_seg/nada"
# path_pre.append(output_path)
# sourcemodel_file_path = "./tmp/models/source_seg/vaihingenirgwithimg300/bak_model_299_0.4743.pt"
# if os.path.isfile(sourcemodel_file_path) == False:
#     raise ValueError("There is no such file or folder: " + sourcemodel_file_path)

# print("loading datasets...")
# data = {}

# data["vaihingen"] = {}
# images, ground_truth = utils.load_recvaihingen_irg() # ADAPTADO
# data["vaihingen"]["images"] = images
# data["vaihingen"]["gt"] = ground_truth

# data["potsdam"] = {}
# images, ground_truth = utils.load_potsdam_irg2(resize_factor=(5/9))
# data["potsdam"]["images"] = images
# data["potsdam"]["gt"] = ground_truth

# data["recpotsdam"] = {}
# images, ground_truth = utils.load_potsdam_adap()
# data["recpotsdam"]["images"] = images
# data["recpotsdam"]["gt"] = ground_truth

# acclist = [0.0]
######
#####
####
###
##
#
# FIM EXPERIMENTO 2


# draw_new_bootstrap_set = False
# # bootstrap_paths = {"hannover":"/home/hanletia/work/Semantic/hannover/bootstrap.json", "vaihingen":"/home/hanletia/work/Semantic/vaihingen/bootstrap.json"}
# #bootstrap_paths = {"potsdam":"/home/josematheus/Desktop/ProjetoHan/Pots/bootstrap.json", "vaihingen":"/home/josematheus/Desktop/ProjetoHan/Vaih/bootstrap.json"}

# bootstrap_paths = {"potsdam":"./tmp/bootstrap/pots/bootstrap.json", "vaihingen":"./tmp/bootstrap/vaih/bootstrap.json"}
# path_pre.append("./tmp/bootstrap/pots/")
# path_pre.append("./tmp/bootstrap/vaih/")

# new_augmentations = False
# horizontal_flip = False
# vertical_flip = False
# x_shear_range = 16
# y_shear_range = 16
# rotation_range = 45 # different from cyclegan 
# #augmentations_path = "/home/josematheus/Desktop/ProjetoHan/data/augmentations.npz"
# augmentations_path = "./tmp/augmentations/seg/augmentations.npz"
# path_pre.append("./tmp/augmentations/seg/")

large_latent_space = True
dilation_rates = ()

#device = torch.device("cpu")
device = torch.device("cuda:0")

mini_batch_size = 18
num_mini_batches = 250
num_epochs = 300

use_wasserstein_loss = False # future work
label_type = 1 #  1 - use a matrix as label for each domain
# 0 - use a scaler as label for each domain, 1 - use a matrix as label for each domain, 2 - create merged/mixed matrices as labels
train_encoder_only_on_target_domain = True 
discriminator_type = 0 # 0 for best discriminator_type

# print("make dir")
# for i in range (len(path_pre)):
#     p = path_pre[i]
#     if (os.path.isdir(p) == False):
#         os.makedirs(p)
#         print(os.path.isdir(p))

# # In[ ]:

# print("drawing bootstrap sets...")
# bootstrap = {}
# if draw_new_bootstrap_set:
#     for key in ("potsdam", "vaihingen"):
#         bootstrap[key] = utils.draw_bootstrap_sets(len(data[key]["images"]))
#         with open(bootstrap_paths[key], "w") as f:
#             json.dump(bootstrap[key], f)
# else:
#     for key, path in bootstrap_paths.items():
#         bootstrap[key] = utils.load_bootstrap_sets(path)

# # experimento_1:      
# bootstrap["recvaihingen"] = bootstrap["vaihingen"]

# # experimento_2:
# # bootstrap["recpotsdam"] = bootstrap["potsdam"]

# print("creating augmentation transforms...")
# if new_augmentations:
#     transforms = numpy.empty((1000000, 3, 3), dtype=numpy.float64)
#     for i in range(transforms.shape[0]):
#         transforms[i,:,:] = utils.get_random_transform(horizontal_flip, vertical_flip, x_shear_range, y_shear_range, rotation_range, 1.0, 1.0)
#     numpy.savez_compressed(augmentations_path, transforms=transforms)
# else:
#     transforms = numpy.load(augmentations_path)["transforms"]
# if not transforms.flags["C_CONTIGUOUS"]:
#     transforms = numpy.ascontiguousarray(transforms)

# print("generating validation data...")
# validation_data = {}
# for key, dataset in data.items():
#     validation_data[key] = {}
#     images, ground_truth = utils.get_validation_data(dataset["images"], dataset["gt"], bootstrap[key][0][1], patch_size, num_classes)
#     validation_data[key]["images"] = images
#     validation_data[key]["gt"] = ground_truth


# # In[ ]:


# print(f"source domain: {source_domain}")
# print(f"rectarget domain: {rectarget_domain}")
# print(f"target domain: {target_domain}")
# source_image_set = bootstrap[source_domain][0][0]
# rectarget_image_set = bootstrap[target_domain][0][0]
# target_image_set = bootstrap[target_domain][0][0]
# In[ ]:

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


class InvBottleneck(nn.ModuleList):
    def __init__(self, prev_filters, t, c, n, s, initial_dilation=1, dilation=1):
        super().__init__()
        for sub_index in range(n):
            _c0 = prev_filters if sub_index == 0 else c
            _c1 = t * _c0
            _s = s if sub_index == 0 else 1
            _d = initial_dilation if sub_index == 0 else dilation
            self.append(nn.Sequential(
                nn.Conv2d(_c0, _c1, 1),
                nn.BatchNorm2d(_c1),
                nn.ReLU6(),
                nn.ReplicationPad2d(_d),
                nn.Conv2d(_c1, _c1, 3, stride=_s, dilation=_d, groups=_c1),
                nn.BatchNorm2d(_c1),
                nn.ReLU6(),
                nn.Conv2d(_c1, c, 1),
                nn.BatchNorm2d(c)
            ).to(device))
    
    def forward(self, x):
        for sub_index, layer in enumerate(self):
            x = layer(x) if sub_index == 0 else layer(x) + x
        return x
    
MobileNetv2_part1 = nn.Sequential(
    nn.ReplicationPad2d(1),
    nn.Conv2d(3, 32, 3, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU6(),
    InvBottleneck(32, 1, 16, 1, 1),
    InvBottleneck(16, 6, 24, 2, 2)
).to(device)

if large_latent_space:
    MobileNetv2_part2 = nn.Sequential(
        InvBottleneck(24, 6, 32, 3, 2),
        InvBottleneck(32, 6, 64, 4, 1, dilation=2),
        InvBottleneck(64, 6, 96, 3, 1, initial_dilation=2, dilation=2),
        InvBottleneck(96, 6, 160, 3, 1, initial_dilation=2, dilation=4),
        InvBottleneck(160, 6, 320, 1, 1, initial_dilation=4)
    ).to(device)
else:
    MobileNetv2_part2 = nn.Sequential(
        InvBottleneck(24, 6, 32, 3, 2),
        InvBottleneck(32, 6, 64, 4, 2),
        InvBottleneck(64, 6, 96, 3, 1),
        InvBottleneck(96, 6, 160, 3, 1, dilation=2),
        InvBottleneck(160, 6, 320, 1, 1, initial_dilation=2)
    ).to(device)
    
MobileNetv2p_part1 = nn.Sequential(
    nn.ReplicationPad2d(1),
    nn.Conv2d(3, 32, 3, stride=2),
    nn.BatchNorm2d(32),
    nn.ReLU6(),
    InvBottleneck(32, 1, 16, 1, 1),
    InvBottleneck(16, 6, 24, 2, 2)
).to(device)

if large_latent_space:
    MobileNetv2p_part2 = nn.Sequential(
        InvBottleneck(24, 6, 32, 3, 2),
        InvBottleneck(32, 6, 64, 4, 1, dilation=2),
        InvBottleneck(64, 6, 96, 3, 1, initial_dilation=2, dilation=2),
        InvBottleneck(96, 6, 160, 3, 1, initial_dilation=2, dilation=4),
        InvBottleneck(160, 6, 320, 1, 1, initial_dilation=4)
    ).to(device)
else:
    MobileNetv2p_part2 = nn.Sequential(
        InvBottleneck(24, 6, 32, 3, 2),
        InvBottleneck(32, 6, 64, 4, 2),
        InvBottleneck(64, 6, 96, 3, 1),
        InvBottleneck(96, 6, 160, 3, 1, dilation=2),
        InvBottleneck(160, 6, 320, 1, 1, initial_dilation=2)
    ).to(device)

class AtrousSpatialPyramidPooling(nn.ModuleList):
    def __init__(self):
        super().__init__()
        latent_space_size = patch_size // (8 if large_latent_space else 16)
        # global average pooling
        self.append(nn.Sequential(
            nn.AvgPool2d(latent_space_size),
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.Upsample(latent_space_size)
        ).to(device))
        # 1x1 conv
        self.append(nn.Sequential(
            nn.Conv2d(320, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU6()
        ).to(device))
        # atrous conv
        for d in dilation_rates:
            self.append(nn.Sequential(
                nn.ReplicationPad2d(d),
                nn.Conv2d(320, 320, 3, dilation=d, groups=320),
                nn.Conv2d(320, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU6()
            ).to(device))
        
    def forward(self, x):
        results = []
        for layer in self:
            results.append(layer(x))
        return torch.cat(results, 1)
    
class DeepLabv3p(nn.Module):
    def __init__(self, src=False):
        super().__init__()
        self.part1 = MobileNetv2_part1 if src==False else MobileNetv2p_part1
        self.part2 = MobileNetv2_part2 if src==False else MobileNetv2p_part2
        self.aspp = nn.Sequential(
            AtrousSpatialPyramidPooling(),
            nn.Conv2d(256*(2+len(dilation_rates)), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU6()
        ).to(device)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2 if large_latent_space else 4, mode="bilinear")
        ).to(device)
        self.skip_connection = nn.Sequential(
            nn.Conv2d(24, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU6()
        ).to(device)
        self.final = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(304, 304, 3, groups=304),
            nn.Conv2d(304, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(256, 256, 3, groups=256),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU6(),
            nn.Conv2d(256, num_classes, 1),
            nn.Upsample(scale_factor=4, mode="bilinear")
        ).to(device)
        
    def forward(self, x):
        x = self.part1(x)
        x = torch.cat((self.upsample(self.aspp(self.part2(x))), self.skip_connection(x)), 1)
        return self.final(x)
    
deep_lab_v3p = DeepLabv3p(src=False)
utils.load_deep_lab_v3p_weights(deep_lab_v3p, {0:1, 1:2, 2:0})

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


def patch_data(img, isgt = False):
	cut = patch_size
	h = img.shape[2]//cut
	w = img.shape[1]//cut
	if (isgt):
		channels = 1 
		output_img = numpy.zeros((h*w, cut, cut))
	else:
		channels = 3
		output_img = numpy.zeros((h*w, channels, cut, cut))
	# print(output_img.shape)
	n = 0
	for i in range (h):
		for j in range (w):
			left = cut * i
			right = cut * (i+1)
			top = cut * j
			bottom = cut * (j+1)
			newImg = img[:channels, top:bottom,left:right]
			output_img[n] = newImg
			n = n+1
	if(isgt):
		return output_img
	return output_img, h, w


def rec_gt(patches, h, w):
	# print(patches[0].shape)
	cut = patch_size
	rec = numpy.zeros((w*cut, h*cut))
	n = 0
	# print(rec.shape)
	for i in range (h):
		for j in range(w):
			left = cut * i
			right = cut * (i+1)
			top = cut * j
			bottom = cut * (j+1)
			rec[top:bottom,left:right] = patches[n]
			n = n+1
	return rec

def raster_gt(gt):
	new_gt = numpy.zeros((gt.shape[0],gt.shape[1],3))

	masks = []
	masks.append((gt == 0))
	masks.append((gt == 1))
	masks.append((gt == 2))
	masks.append((gt == 3))
	masks.append((gt == 4))
	masks.append((gt == 5))

	new_gt[:][masks[0]] = [255,255,255]
	new_gt[:][masks[1]] = [0,0,255]
	new_gt[:][masks[2]] = [0,255,255]
	new_gt[:][masks[3]] = [0,255,0]
	new_gt[:][masks[4]] = [255,255,0]
	new_gt[:][masks[5]] = [255,0,0] 

	return new_gt


hannoverdeep_lab_v3p = DeepLabv3p(src=True) # source semantic model 
# load the pre-trained source semantic model
# mog: melhor selecionar modelo estaticamente
# models = {model: int(model.split("_")[2]) for model in glob.iglob(f"{sourcemodel_path}/model_*.pt")} # change folder
# best = list(models.keys())[0]
# for model, epo in models.items():
#     if epo > models[best]:
#         best = model

best = sourcemodel_file_path

print(best)

checkpoint1=torch.load(best)
hannoverdeep_lab_v3p.load_state_dict(checkpoint1)
hannoverdeep_lab_v3p.eval()

del best

# source_domain = "potsdam"
# augmentations_path = "./tmp/augmentations/seg/augmentations.npz"
# transforms = numpy.load(augmentations_path)["transforms"]
# image_id = 1
# image = data[source_domain]["images"][image_id]
# gt = data[source_domain]["gt"][image_id]
# x, y = utils.get_mini_batch(mini_batch_size, patch_size, image, gt, num_classes, utils.get_slice(transforms, 0, mini_batch_size))
# batch_data = []
# batch_data.append(x)
# batch_data.append(y)


# img_patches = batch_data[0]
# gt_patches = batch_data[1]

img = data["potsdam"]["images"][1]
gt = data["potsdam"]["gt"][1]
img_patches, h, w = patch_data(img)
gt_patches = patch_data(gt, isgt = True)

with torch.no_grad():
    n = 10
    x = torch.from_numpy(img_patches[:n]*(1.0/255.0)).float().to(device)
    x = hannoverdeep_lab_v3p(x)
    y = torch.from_numpy(gt_patches[:n]).long().to(device)
    # loss += seg_loss_fn(x, y).item()
    x = x.argmax(1)
    acc = (x == y).sum().item() / (n * patch_size * patch_size)
    # x = x.cpu()
    # print (loss)
    print (acc)


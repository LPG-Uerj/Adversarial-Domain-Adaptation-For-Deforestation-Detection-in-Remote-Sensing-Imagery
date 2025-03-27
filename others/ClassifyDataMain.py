# In[ ]:


# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
import torch.nn.functional as F # mog: for focal loss
import skimage # para o prodes

import Amazonia_Legal as ama
import Cerrado_Biome as cer

import Networks
from Desmatamento.code.utilities.training_utils import *
import modeling.deeplab as d3plus



print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())

patch_size = 64
num_classes = 3
channels = 14

mini_batch_size = 50

large_latent_space = True
dilation_rates = ()

run_test = True
prodes = True
test_only_deforestation = False


output_path = f"./class_output"
if (os.path.isdir(output_path) == False):
	os.makedirs(output_path)

validation_check = False #Change Path

# defo_test_path = f"./tmp/models/source_seg/teste/model_99_0.8210.pt"
# defo_source_only_path = f"./tmp/models/source_seg/teste2/model_99_0.6796.pt"
# sou_tar_teste_path = f"./tmp/models/source_seg/teste3/model_49_0.6705.pt"
# teste_renan_1_path = f"./tmp/models/source_seg/teste_renan1/model_49_95.2413.pt"
teste_renan_2_path = f"./tmp/models/source_seg/teste_renan_2/model_41_98.6185.pt"
xception_test_1_path = f"./tmp/models/source_seg/torch_xception_1/model_25_98.2443.pt"
xception_test_2_path = f"./tmp/models/source_seg/torch_xception_2/model_21_98.3198.pt"


models = {}
# models["defo_test"] = defo_test_path
# models["defo_source_only"] = defo_source_only_path
# models["sou_tar_teste"] = sou_tar_teste_path
# models["teste_renan_1"] = teste_renan_1_path
# models["teste_renan_2"] = teste_renan_2_path
models["xception_test_1"] = xception_test_1_path
models["xception_test_2"] = xception_test_2_path



print("loading data...")
data = {}
# bootstrap = {}

class Data():
	def __init__(self, data, t1_name, t2_name, buffer=False, buffer_dimension_out = 0,
		buffer_dimension_in = 0, ndvi=False):
		self.dataset_main_path = "./main_data/"
		self.dataset = data + '/'
		self.images_section = ""
		self.reference_section = ""
		self.data_t1_name = "img_" + t1_name + '_' + data
		self.data_t2_name = "img_" + t2_name + '_' + data 
		self.reference_t1_name = "gt_" + t1_name + '_' + data
		self.reference_t2_name = "gt_" + t2_name + '_' + data
		self.buffer = buffer
		self.buffer_dimension_out = buffer_dimension_out
		self.buffer_dimension_in = buffer_dimension_in
		self.compute_ndvi = ndvi

# ro = Data("RO/", "img_past_RO", "img_present_RO")

class Bootstrap():
	def __init__(self, data, patches_dimension, stride = 1, horizontal_blocks = 2,
		vertical_blocks = 2, porcent_of_last_reference_in_actual_reference = 100,
		phase = 'train', fixed_tiles = True, defined_before = False, path = "./tmp/bootstrap/"):
		self.phase = phase
		self.fixed_tiles = fixed_tiles
		self.defined_before = defined_before
		self.checkpoint_dir_posterior = path + data
		self.save_checkpoint_path = path + data + '/'
		
		self.horizontal_blocks = horizontal_blocks
		self.vertical_blocks = vertical_blocks
		self.patches_dimension = patches_dimension
		self.stride = stride
		self.porcent_of_last_reference_in_actual_reference = porcent_of_last_reference_in_actual_reference

		p = self.save_checkpoint_path
		if os.path.isdir(p) == False:
			os.makedirs(p)
			print('\'' + p + '\'' + " path created")

# RO
ro = ama.AMAZON_RO(Data("RO", "past", "present", buffer = True,
	buffer_dimension_out = 4, buffer_dimension_in = 2))
ro_bootstrap = Bootstrap("RO", patches_dimension = patch_size, stride = 16,
	horizontal_blocks = 5, vertical_blocks = 5)

shapes = ro.images_norm[0].shape
new_image = numpy.zeros((shapes[0], shapes[1], shapes[2]*2))
new_image[:,:,:shapes[2]] = ro.images_norm[0]
new_image[:,:,shapes[2]:] = ro.images_norm[1]
data["RO"] = {}
data["RO"]["image"] = Desmatamento.code.utilities.utils.channels_last2first(new_image)
print("input image shape:", data["RO"]["image"].shape)
del shapes
del new_image

new_ref = (ro.references[0]*2) + ro.references[1]
new_ref[:][(new_ref == 3)] = 2
new_ref[:][(new_ref == 4)] = 2
data["RO"]["gt"] = new_ref
print("class values", numpy.unique(data["RO"]["gt"]))
del new_ref

# PA
pa = cer.CERRADO(Data("PA", "past", "present", buffer = True,
	buffer_dimension_out = 0, buffer_dimension_in = 2))
pa_bootstrap = Bootstrap("PA", patches_dimension = patch_size, stride = 16,
	horizontal_blocks = 3, vertical_blocks = 5)

shapes = pa.images_norm[0].shape
new_image = numpy.zeros((shapes[0], shapes[1], shapes[2]*2))
new_image[:,:,:shapes[2]] = pa.images_norm[0]
new_image[:,:,shapes[2]:] = pa.images_norm[1]
data["PA"] = {}
data["PA"]["image"] = Desmatamento.code.utilities.utils.channels_last2first(new_image)
print("input image shape:", data["PA"]["image"].shape)
del shapes
del new_image

new_ref = (pa.references[0]*2) + pa.references[1]
new_ref[:][(new_ref == 3)] = 2
new_ref[:][(new_ref == 4)] = 2
data["PA"]["gt"] = new_ref
print("class values", numpy.unique(data["PA"]["gt"]))
del new_ref

if run_test:
	pa.Tiles_Configuration(pa_bootstrap, 0)
	pa_bootstrap.phase = 'test'
	pa_bootstrap.stride = patch_size
	pa.Coordinates_Creator(pa_bootstrap, 0)
	print(pa.central_pixels_coor_ts.shape)
	if test_only_deforestation:
		pa.central_pixels_coor_ts = get_defo_coor(data["PA"]["gt"], pa.central_pixels_coor_ts, patch_size)
	print(pa.central_pixels_coor_ts.shape)

device = torch.device("cuda:0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
    

print("defining segmentation network...")


def patch_data(img, isgt = False):
	cut = patch_size
	if (isgt):
		h = img.shape[0]//cut
		w = img.shape[1]//cut
		channel = 1 
		output_img = numpy.zeros((h*w, cut, cut))
		
	else:
		h = img.shape[1]//cut
		w = img.shape[2]//cut
		channel = channels
		output_img = numpy.zeros((h*w, channel, cut, cut))
		
	# print(output_img.shape)
	n = 0
	for i in range (h):
		for j in range (w):
			top = cut * i
			bottom = cut * (i+1)
			left = cut * j
			right = cut * (j+1)
			if isgt:
				newImg = img[top:bottom,left:right]
			else:
				newImg = img[:channel, top:bottom,left:right]
			output_img[n] = newImg
			n = n+1
	if(isgt):
		return output_img
	return output_img, h, w


def rec_gt(patches, h, w):
	# print(patches[0].shape)
	cut = patch_size
	rec = numpy.zeros((h*cut, w*cut))
	n = 0
	# print(rec.shape)
	for i in range (h):
		for j in range(w):
			top = cut * i
			bottom = cut * (i+1)
			left = cut * j
			right = cut * (j+1)
			rec[top:bottom,left:right] = patches[n]
			n = n+1
	return rec

def raster_gt(gt):
	new_gt = numpy.zeros((gt.shape[0],gt.shape[1],3))
	masks = []
	masks.append((gt == 0))
	masks.append((gt == 1))
	# masks.append((gt == 2))
	# masks.append((gt == 3))
	# masks.append((gt == 4))
	# masks.append((gt == 5))

	new_gt[:][masks[0]] = [0,0,0]
	new_gt[:][masks[1]] = [255,255,255]
	# new_gt[:][masks[2]] = [0,255,255]
	# new_gt[:][masks[3]] = [0,255,0]
	# new_gt[:][masks[4]] = [255,255,0]
	# new_gt[:][masks[5]] = [255,0,0] 

	return new_gt

def calcula_metricas(ref, pred, patch_size):
  total = 0
  fp, tp = 0, 0
  fn, tn = 0, 0
  c2 = 2
  c1 = 1
  c0 = 0

#  result_image = np.zeros((9, 368,520, 3), dtype = np.float32)

  for k in range(len(ref)):
    for i in range(patch_size): #patch_size ou tile_size
      for j in range(patch_size):  #patch_size ou tile_size
        if(ref[k,i,j]!=c2):
          if(ref[k,i,j]==c1):
            if(pred[k,i,j]==1):
              tp = tp+1
      #        result_image[k,i,j] = [0, 255, 255 ] 
            else:
              fn = fn+1
       #       result_image[k,i,j] = [255 ,0 , 0 ] 
          elif(ref[k,i,j]==c0):
            if(pred[k,i,j]==0):
              tn = tn+1
         #     result_image[k,i,j] = [255, 255, 255 ] 
            else:
              fp = fp+1
         #     result_image[k,i,j] = [0, 0 , 255 ] 
        else:
        #  result_image[k,i,j] = [255, 255, 255 ] 
          total = total+1
  #  cv2.imwrite("resultados/dissertacao/DLCD-4/4tiles/result"+str(k)+".png", result_image[k])
          
  prescision=0.0
  f1score = 0.0       
  recall = tp/(tp+fn)
  if(tp+fp!=0):
    prescision = tp/(tp+fp)
  if(prescision!=0):
    f1score = 100*(2*prescision*recall)/(prescision+recall)
  P = tp+fn
  N = tn+fp
  overall = 100*(tp+tn)/(P+N)
  alert_rate = 100*(tp+fp)/(P+N)
  recall = recall * 100
  prescision = prescision * 100  

  print("Total: ", total)
  print("tp: ", tp)
  print("fp: ", fp)
  print("tn: ", tn)
  print("fn: ", fn)
  print("Overall: %.2f" % overall)
  print("F1-Score: %.2f" % f1score)
  print("Recall: %.2f" % recall)
  print("Prescision: %.2f" % prescision)
  print("Alert Rate: %.2f" % alert_rate)
  print("Confusion Matrix: \n["+str(tp)+" "+str(fp)+"]\n["+str(fn)+" "+str(tn)+"]")


# def test():
# 	torch.utils.data.DataLoader(pa.central_pixels_coor_ts, batch_size = mini_batch_size)

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

print ("starting classification...")
metrics = {}
for model in (models):
	print(f"model: {model}")
	# seg_loss_fn = nn.CrossEntropyLoss()
	seg_loss_fn = FocalLoss(weight = torch.FloatTensor([0.3, 0.7, 0]).cuda(), gamma = 1)
	# classifier = Networks.DeepLabv3p()
	classifier = d3plus.DeepLab(num_classes=num_classes,
	                    backbone='xception',
	                    output_stride=8
	                    # sync_bn=args.sync_bn,
	                    # freeze_bn=args.freeze_bn
	                    )
	classifier.cuda()
	classifier.load_state_dict(torch.load(models[model]))
	classifier.eval()

	metrics[model] = {}

	with torch.no_grad():
		if run_test:
			print("Running test... ")
			if prodes:
				print("Prodes ON")
			else:
				print("Prodes OFF")

			img = data['PA']["image"]
			gt = data['PA']["gt"]
			domain = 'PA_test'
			metrics[model][domain] = {}
			metrics[model][domain]["domain_loss"] = 0
			metrics[model][domain]["domain_acc"] = 0
			metrics[model][domain]["loss"] = []
			metrics[model][domain]["acc"] = []
			metrics[model][domain]["f1"] = []
			metrics[model][domain]["rec"] = []
			metrics[model][domain]["prec"] = []
			# test_loader = torch.utils.data.DataLoader(pa.central_pixels_coor_ts, batch_size = mini_batch_size)
			test_loader = pa.central_pixels_coor_ts
			acc = 0
			loss = 0
			f1 = 0
			rec = 0
			prec = 0
			count = 0
			
			# for x in (test_loader):
			x = test_loader
			count += 1
			n = x.shape[0]
			coor_batch = x
			x , y = Desmatamento.code.utilities.utils.patch_extraction(img, gt, coor_batch, patch_size)
			x = torch.from_numpy(x).float().to(device)
			x = classifier(x)
			y = torch.from_numpy(y).long().to(device)
			loss += seg_loss_fn(x, y).item()

			x = x.argmax(1)
			x = x.cpu().numpy()
			y = y.cpu().numpy()
			x[:][(x == 2)] = 0
			# y[:][(y == 2)] = 0


			if prodes:
				# for i,j in (x,y):
				# print(1)

				# print('x bf', np.count_nonzero(x == 0))
				# print('x bf', np.count_nonzero(x == 1))
				# print('x bf', np.count_nonzero(x == 2))
				x_prodes = skimage.morphology.area_opening(x.astype('int'), area_threshold = 69, connectivity = 1)
				# print('x_prodes', np.count_nonzero(x_prodes == 0))
				# print('x_prodes', np.count_nonzero(x_prodes == 1))
				# print('x_prodes', np.count_nonzero(x_prodes == 2))
				eliminated_samples = x - x_prodes
				# print('eliminated', np.count_nonzero(eliminated_samples == 0))
				# print('eliminated', np.count_nonzero(eliminated_samples == 1))
				# print('eliminated', np.count_nonzero(eliminated_samples == 2))

				# print('y bf', np.count_nonzero(y == 0))
				# print('y bf', np.count_nonzero(y == 1))
				# print('y bf', np.count_nonzero(y == 2))

				y_prodes = y + eliminated_samples
				# print('y_prodes bf', np.count_nonzero(y_prodes == 0))
				# print('y_prodes bf', np.count_nonzero(y_prodes == 1))
				# print('y_prodes bf', np.count_nonzero(y_prodes == 2))
				# print('y_prodes bf', np.count_nonzero(y_prodes == 3))
				# print('y_prodes bf', np.count_nonzero(y_prodes == 4))
				# y_prodes[:][y_prodes == 2] = 0
				# y_prodes[:][y_prodes == 3] = 0

				# y_prodes[:][y_prodes == 3] = 2

				# print('y_prodes af', np.count_nonzero(y_prodes == 0))
				# print('y_prodes af', np.count_nonzero(y_prodes == 1))
				# print('y_prodes af', np.count_nonzero(y_prodes == 2))
				# print('y_prodes af', np.count_nonzero(y_prodes == 3))
				print("y unique",numpy.unique(y_prodes))
				y = y_prodes

			else:
				y[:][(y == 2)] = 0

			new_x = x + 2*y
			new_x[new_x == 2] = 0
			new_x[new_x == 3] = 1
			new_x[new_x == 4] = 0
			new_x[new_x == 5] = 0
			new_x[new_x == 6] = 0
			new_x[new_x == 7] = 0
			y[:][(y == 2)] = 0
			y[:][(y == 3)] = 0
			print("y unique",numpy.unique(y))
			print("new_x unique",numpy.unique(new_x))
			x = new_x
			calcula_metricas(y,x,patch_size)


			x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2])
			y = y.reshape(y.shape[0] * y.shape[1] * y.shape[2])
			
			


			a, b, c, d, e = compute_metrics(y,x)
			acc += a
			f1 += b
			rec += c
			prec += d

				# print("break")
				# break
			num_slices = count
			metrics[model][domain]['domain_loss'] = loss / num_slices
			metrics[model][domain]['domain_acc'] = acc / num_slices
			metrics[model][domain]['f1'] = f1 / num_slices
			metrics[model][domain]['rec'] = rec / num_slices
			metrics[model][domain]['prec'] = prec / num_slices

			print(f"{domain} loss: {metrics[model][domain]['domain_loss']}")
			print(f"{domain} acc: {metrics[model][domain]['domain_acc']}")
			print(f"{domain} f1: {metrics[model][domain]['f1']}")
			print(f"{domain} rec: {metrics[model][domain]['rec']}")
			print(f"{domain} prec: {metrics[model][domain]['prec']}")
			# save_folder = f"{output_path}/{model}"
			# if (os.path.isdir(save_folder) == False):
			# 	os.makedirs(save_folder)
			# save_history = f"{save_folder}/{domain}_history.json"
			# with open(save_history, "w") as f:
			# 	json.dump(metrics[model][domain], f)

		else:
			for domain in (data):
				print(f"domain: {domain}")

				img = data[domain]["image"]
				gt = data[domain]["gt"]
				# coor = data[domain]["coor"]

				
				# self.tr_it = 0
				# self.tr_coor = iter(self.train_set)

				# self.img = utils.channels_last2first(data["past"]["image"])
				# self.gt = data["past"]["gt"]

				metrics[model][domain] = {}
				metrics[model][domain]["domain_loss"] = 0
				metrics[model][domain]["domain_acc"] = 0
				metrics[model][domain]["loss"] = []
				metrics[model][domain]["acc"] = []
				metrics[model][domain]["f1"] = []
				metrics[model][domain]["rec"] = []
				metrics[model][domain]["prec"] = []

				# if (validation_check):
				# 	iterations = bootstrap[domain]
				# 	print("Validation data classification")
				# else:
				# iterations = numpy.arange(len(img))
				print("All data classification")

				# for i in range(len(img)):
				# print(f"img: {i}")

				loss = 0
				acc = 0

				img_patches, h, w = patch_data(img)
				gt_patches = patch_data(gt, isgt = True)
				print(gt_patches.shape)
				classification = numpy.zeros((gt_patches.shape))
				n = gt_patches.shape[0]

				x = torch.from_numpy(img_patches).float().to(device)
				x = classifier(x)
				y = torch.from_numpy(gt_patches).long().to(device)
				
				loss += seg_loss_fn(x, y).item()
				# x = x.argmax(1)
				# acc += (x == y).sum().item() / (n * patch_size * patch_size)

				x = x.argmax(1)
				x = x.cpu()
				y = y.cpu()
				x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2])
				y = y.reshape(y.shape[0] * y.shape[1] * y.shape[2])
				# print(x.shape)
				# print(y.shape)
				x[:][(x == 2)] = 0
				y[:][(y == 2)] = 0
				# print(numpy.unique(x))
				# print(numpy.unique(y))
				acc, f1, rec, prec, conf = compute_metrics(y,x)
				# metrics["seg_acc"] = acc
				# metrics["seg_f1"] = f1
				# metrics["seg_rec"] = rec
				# metrics["seg_prec"] = prec
				# metrics["seg_conf"] = conf

				# x = x.cpu()
				# y = y.cpu()
				# print(type(x[0,0,0]))
				classification = x
				# print(classification)
				print(f"img loss: {loss}")
				print(f"img acc: {acc}")
				metrics[model][domain]["loss"].append(loss)
				metrics[model][domain]["acc"].append(acc)
				metrics[model][domain]["f1"].append(f1)
				metrics[model][domain]["rec"].append(rec)
				metrics[model][domain]["prec"].append(prec)
				# acc, f1, rec, prec, conf = compute_metrics(y.astype(int),x.astype(int))
				
				# metrics[model][domain]["seg_acc"] = acc
				# metrics[model][domain]["seg_f1"] = f1
				# metrics[model][domain]["seg_rec"] = rec
				# metrics[model][domain]["seg_prec"] = prec
				# metrics[model][domain]["seg_conf"] = conf
				
				classification = rec_gt(classification, h, w)
				classification = raster_gt(classification)
				classification = PIL.Image.fromarray(classification.astype('uint8'))
				save_folder = f"{output_path}/{model}"
				if (os.path.isdir(save_folder) == False):
					os.makedirs(save_folder)
				save_path = f"{save_folder}/{domain}_.png"
				classification.save(save_path)
				metrics[model][domain]["domain_loss"] = sum(metrics[model][domain]["loss"])/len(metrics[model][domain]["loss"])
				metrics[model][domain]["domain_acc"] = sum(metrics[model][domain]["acc"])/len(metrics[model][domain]["acc"])
				print(f"{domain} loss: {metrics[model][domain]['domain_loss']}")
				print(f"{domain} acc: {metrics[model][domain]['domain_acc']}")
				print(f"{domain} f1: {metrics[model][domain]['f1']}")
				print(f"{domain} rec: {metrics[model][domain]['rec']}")
				print(f"{domain} prec: {metrics[model][domain]['prec']}")
				save_history = f"{save_folder}/{domain}_history.json"
				with open(save_history, "w") as f:
					json.dump(metrics[model][domain], f)

	del classifier
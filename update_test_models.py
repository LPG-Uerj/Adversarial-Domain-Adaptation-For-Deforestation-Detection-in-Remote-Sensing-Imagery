# In[ ]:


# coding: utf-8
import os
os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import glob
import json
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils

import skimage # para o prodes

from Tools import *
import DeepLabV3plus
# import modeling.deeplab as d3plus



print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())

def test(domain, args, model):
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Using device:', device)

	patch_size = args.patch_size
	num_classes = args.num_classes
	channels = args.channels

	mini_batch_size = args.batch_size

	weights = args.weights
	gamma = args.gamma

	# large_latent_space = True
	# dilation_rates = ()

	run_test = True
	prodes = args.run_prodes
	# test_only_deforestation = False


	# output_path = f"./class_output"
	# if (os.path.isdir(output_path) == False):
	# 	os.makedirs(output_path)

	# validation_check = False #Change Path

	# source_domain = global_args.source_domain
	# target_domain = global_args.target_domain

	# models = {}
	# models["source"] = global_args.best_source_model
	# models["target"] = global_args.best_target_model
	# models["ideal_target"] = global_args.theorical_best_target_model



	print("loading data...")

	# domains = {}
	# domains[source_domain] = source
	# domains[target_domain] = target
	    

	print("defining segmentation network...")


	def patch_data(img, isgt = False):
		cut = patch_size
		if (isgt):
			h = img.shape[0]//cut
			w = img.shape[1]//cut
			channel = 1 
			output_img = np.zeros((h*w, cut, cut))
			
		else:
			h = img.shape[1]//cut
			w = img.shape[2]//cut
			channel = channels
			output_img = np.zeros((h*w, channel, cut, cut))
			
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
		rec = np.zeros((h*cut, w*cut))
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
		new_gt = np.zeros((gt.shape[0],gt.shape[1],3))
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

	def calcula_metricass(ref, pred, patch_size):
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

	  # print("Total: ", total)
	  # print("tp: ", tp)
	  # print("fp: ", fp)
	  # print("tn: ", tn)
	  # print("fn: ", fn)
	  # print("Overall: %.2f" % overall)
	  # print("F1-Score: %.2f" % f1score)
	  # print("Recall: %.2f" % recall)
	  # print("Prescision: %.2f" % prescision)
	  # print("Alert Rate: %.2f" % alert_rate)
	  # print("Confusion Matrix: \n["+str(tp)+" "+str(fp)+"]\n["+str(fn)+" "+str(tn)+"]")

	  return overall, f1score, recall, prescision, alert_rate 


	# def test():
	# 	torch.utils.data.DataLoader(pa.central_pixels_coor_ts, batch_size = mini_batch_size)


	print ("starting classification...")
	metrics = {}
	# for model in (models):
	print(f"model: {model}")
	# seg_loss_fn = nn.CrossEntropyLoss()
	seg_loss_fn = FocalLoss(weight = torch.FloatTensor(weights).to(device), gamma = gamma)
	classifier = DeepLabV3plus.create(args)
	# classifier = d3plus.DeepLab(num_classes=num_classes,
	#                     backbone='xception',
	#                     output_stride=8
	#                     # sync_bn=args.sync_bn,
	#                     # freeze_bn=args.freeze_bn
	#                     )
	classifier.to(device)
	classifier.load_state_dict(torch.load(model))
	classifier.eval()

	metrics = {}

	with torch.no_grad():
		if run_test:
			print("Running test... ")
			if prodes:
				print("Prodes ON")
			else:
				print("Prodes OFF")

			# for domain in (domains):
			print("domain: ", domain)
			img = utils.channels_last2first(domain.conc_image)
			gt = domain.new_reference
			# domain = 'PA_test'
			metrics[domain] = {}

			test_loader = domain.central_pixels_coor_ts
			test_loader = torch.utils.data.DataLoader(test_loader, batch_size = mini_batch_size)
			acc = 0
			loss = 0
			f1 = 0
			rec = 0
			prec = 0
			count = 0
			alert = 0
			
			for x in (test_loader):
				# x = test_loader
				count += 1
				# n = x.shape[0]
				coor_batch = x
				x , y = patch_extraction(img, gt, coor_batch, patch_size)
				x = torch.from_numpy(x).float().to(device)
				x = classifier(x)
				y = torch.from_numpy(y).long().to(device)
				loss += seg_loss_fn(x, y).item()

				x = x.argmax(1)
				x = x.cpu()
				y = y.cpu()
				# print("unique x", np.unique(x))
				x[:][(x == 2)] = 0
				# y[:][(y == 2)] = 0


				# if prodes:
				# 	# for i,j in (x,y):
				# 	# print(1)

				# 	# print('x bf', np.count_nonzero(x == 0))
				# 	# print('x bf', np.count_nonzero(x == 1))
				# 	# print('x bf', np.count_nonzero(x == 2))
				# 	x_prodes = skimage.morphology.area_opening(x.astype('int'), area_threshold = 69, connectivity = 1)
				# 	# print('x_prodes', np.count_nonzero(x_prodes == 0))
				# 	# print('x_prodes', np.count_nonzero(x_prodes == 1))
				# 	# print('x_prodes', np.count_nonzero(x_prodes == 2))
				# 	eliminated_samples = x - x_prodes
				# 	# print('eliminated', np.count_nonzero(eliminated_samples == 0))
				# 	# print('eliminated', np.count_nonzero(eliminated_samples == 1))
				# 	# print('eliminated', np.count_nonzero(eliminated_samples == 2))

				# 	# print('y bf', np.count_nonzero(y == 0))
				# 	# print('y bf', np.count_nonzero(y == 1))
				# 	# print('y bf', np.count_nonzero(y == 2))

				# 	y_prodes = y + eliminated_samples
				# 	# print('y_prodes bf', np.count_nonzero(y_prodes == 0))
				# 	# print('y_prodes bf', np.count_nonzero(y_prodes == 1))
				# 	# print('y_prodes bf', np.count_nonzero(y_prodes == 2))
				# 	# print('y_prodes bf', np.count_nonzero(y_prodes == 3))
				# 	# print('y_prodes bf', np.count_nonzero(y_prodes == 4))
				# 	# y_prodes[:][y_prodes == 2] = 0
				# 	# y_prodes[:][y_prodes == 3] = 0

				# 	# y_prodes[:][y_prodes == 3] = 2

				# 	# print('y_prodes af', np.count_nonzero(y_prodes == 0))
				# 	# print('y_prodes af', np.count_nonzero(y_prodes == 1))
				# 	# print('y_prodes af', np.count_nonzero(y_prodes == 2))
				# 	# print('y_prodes af', np.count_nonzero(y_prodes == 3))
				# 	print("y unique",np.unique(y_prodes))
				# 	y = y_prodes

				# else:
				# 	y[:][(y == 2)] = 0

				# new_x = x + 2*y
				# new_x[new_x == 2] = 0
				# new_x[new_x == 3] = 1
				# new_x[new_x == 4] = 0
				# new_x[new_x == 5] = 0
				# new_x[new_x == 6] = 0
				# new_x[new_x == 7] = 0
				# y[:][(y == 2)] = 0
				# y[:][(y == 3)] = 0
				# print("y unique",np.unique(y))
				# print("new_x unique",np.unique(new_x))
				# x = new_x
				
				

				# print(calcula_metricass(y,x,patch_size))
				# print(return_metrics(y,x))
				# a, b, c, d, e = compute_metrics(y,x)
				a, b, c, d, e = return_metrics(y,x)
				acc += a
				f1 += b
				rec += c
				prec += d
				alert += e

				# break
			num_slices = count
			metrics[domain]['loss'] = loss / num_slices
			metrics[domain]['acc'] = acc / num_slices
			metrics[domain]['f1'] = f1 / num_slices
			metrics[domain]['rec'] = rec / num_slices
			metrics[domain]['prec'] = prec / num_slices
			metrics[domain]["alert"] = alert / num_slices

			# print(f"{domain} loss: {metrics[model][domain]['loss']}")
			# print(f"{domain} acc: {metrics[model][domain]['acc']}")
			# print(f"{domain} f1: {metrics[model][domain]['f1']}")
			# print(f"{domain} rec: {metrics[model][domain]['rec']}")
			# print(f"{domain} prec: {metrics[model][domain]['prec']}")
			# print(f"{domain} alert: {metrics[model][domain]['alert']}")

			return metrics
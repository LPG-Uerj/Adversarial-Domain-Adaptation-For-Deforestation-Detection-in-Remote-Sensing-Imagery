# import parameters
from Tools import *
# from Tools import ready_domain
import utils
import time
import json
import os

from Amazon_PA import *
from Amazon_RO import *
from Cerrado_MA import *
# import GAN

import train_source
import train_cyclegan
import evaluate_cyclegan
import evaluate_cyclegan_inference
import train_adapted_target
import test_models




def train_source_model(source, source_args, target, target_args, train_args, global_args):
	rounds = train_args.number_of_rounds
	this_models = []
	base_path = train_args.output_path
	
	# source_args.patches_dimension = train_args.patch_size
	source_args.stride = source_args.train_source_stride
	ready_domain(source, source_args, train_set = True, augmented = True)

	# target_args.patches_dimension = train_args.patch_size
	target_args.stride = target_args.train_source_stride
	ready_domain(target, target_args, train_set = True, augmented = False)
	print(len(target.central_pixels_coor_vl))
	print(len(source.central_pixels_coor_vl))
	
	for i in range(global_args.train_source_starting_round, rounds):
		train_args.output_path = os.path.join(base_path, f"round_{i}")
		print(train_args.output_path)
		best_source_model = train_source.train(source, target, train_args, global_args)
		this_models.append(best_source_model)
	return this_models

def train_cyclegan_model(source, source_args, target, target_args, train_args, global_args):
	source_args.patches_dimension = train_args.patch_size
	source_args.stride = source_args.train_source_stride
	ready_domain(source, source_args, train_set = False, cyclegan_set = True)
	
	target_args.patches_dimension = train_args.patch_size
	target_args.stride = target_args.train_source_stride
	ready_domain(target, target_args, train_set = False, cyclegan_set = True)

	train_cyclegan.train(source, target, train_args, global_args)
	# trained_models_path = train_args.output_path + '/saved_models/'
	# return trained_models_path


def evalutate_cyclegan_model(source, source_args, train_args, global_args, gan_models_path):
	assert gan_models_path != ''
	
	# source_args.patches_dimension = train_args.patch_size
	source_args.stride = source_args.train_source_stride
	ready_domain(source, source_args, train_set = True, augmented = True)

	# best_gan_epoch = evaluate_cyclegan.train(source, source_args, 
	# 	train_args, global_args, gan_models_path)
	best_gan_epoch = evaluate_cyclegan_inference.evaluate(source, source_args, 
		train_args, global_args, gan_models_path)
	return best_gan_epoch

def train_target_model(source, source_args, target, target_args, train_args, 
	train_source_args, global_args, source_model, A2B_gan_model, B2A_gan_model):

	assert os.path.isfile(source_model)
	assert os.path.isfile(A2B_gan_model)
	assert os.path.isfile(B2A_gan_model)
	assert A2B_gan_model != B2A_gan_model
	
	rounds = train_args.number_of_rounds
	this_models = []
	base_path = train_args.output_path

	# source_args.patches_dimension = train_args.patch_size
	source_args.stride = source_args.train_source_stride
	ready_domain(source, source_args, train_set = True, augmented = True)
	source.Prepare_GAN_Set(source_args, A2B_gan_model, eval = False, load = False)

	# target_args.patches_dimension = train_args.patch_size
	target_args.stride = target_args.train_source_stride
	ready_domain(target, target_args, train_set = True, augmented = True)
	target.Prepare_GAN_Set(target_args, B2A_gan_model, 	eval = False, load = False)

	for i in range(global_args.train_adapted_target_starting_round, rounds):
		train_args.output_path = os.path.join(base_path, f"round_{i}")
		print(train_args.output_path)
		theorical_best_target_model = train_adapted_target.train(source, 
		target, train_args, train_source_args, global_args, source_model)
		this_models.append(theorical_best_target_model)
	return this_models

def run_test(source, source_args, target, target_args, test_args, global_args, 
	models, file_name, last_gans):

	ready_domain(source, source_args, train_set = False, test_set = True)
	ready_domain(target, target_args, train_set = False, test_set = True)

	if test_args.test_transforms:
		if last_gans == []:
			last_gans = jason(os.path.join(global_args.base_path, "last_gans_paths.json"), 'r')

		if global_args.inverse_cyclegan:
			source_args.best_generator = last_gans[1]
			target_args.best_generator = last_gans[0]
		else:
			source_args.best_generator = last_gans[0]
			target_args.best_generator = last_gans[1]

		A2B_best_gan_model = source_args.best_generator
		B2A_best_gan_model = target_args.best_generator

		source.Prepare_GAN_Set(source_args, A2B_best_gan_model, eval = False, load = False)
		target.Prepare_GAN_Set(target_args, B2A_best_gan_model,	eval = False, load = False)

	test_models.test(source, target, test_args, global_args, models, file_name)

	return 0
	
	
def main(parameters, file_name = 'parameters.py'):
# if __name__=='__main__':

	total_time = time.time()
	Global = parameters.Global
	models = {}
	last_gans = []

	# Prepare root path
	if (os.path.isdir(Global.base_path) == False):
		os.makedirs(Global.base_path)
		print('\''+ Global.base_path + '\'' + " path created")
	# Save used parameters
	# with open(file_name, "r") as f:
	# 	lines = f.readlines()
	# 	with open(os.path.join(Global.base_path, file_name.split('.')[0] + ".txt"), 'w') as f1:
	# 		for line in lines:
	# 			f1.write(line)


	# Load data and initial preparation
	if Global.source_domain == "PA":
		source = AM_PA(parameters.PA)
		source_args = parameters.PA
	elif Global.source_domain == "RO":
		source = AM_RO(parameters.RO)
		source_args = parameters.RO
	elif Global.source_domain == "MA":
		source = CE_MA(parameters.MA)
		source_args = parameters.MA

	if Global.target_domain == "PA":
		target = AM_PA(parameters.PA)
		target_args = parameters.PA
	elif Global.target_domain == "RO":
		target = AM_RO(parameters.RO)
		target_args = parameters.RO
	elif Global.target_domain == "MA":
		target = CE_MA(parameters.MA)
		target_args = parameters.MA
		

	print('[*] Source domain' , type(source))
	print('[*] Target domain' , type(target))

	# Train steps \/

	# ------ deforestation train on source domain ------
	if not Global.skip_train_source:
		print("[*] Train Source...")
		best_source_models = train_source_model(source, source_args, target, target_args, 
			parameters.Train_source, Global)
		
		print(best_source_models[0])
		models["source_group"] = best_source_models
		jason(os.path.join(Global.base_path, "best_source_models.json"), 'w', 
			models["source_group"])
		print('[*] Best source models' , models["source_group"])

		if Global.train_source_both_ways:
			print("[*] Train GT Target...")
			parameters.Train_source.output_path = os.path.join(Global.default_segmentation_path,
				'gt_target')
			best_gt_target_models = train_source_model(target, target_args, source, source_args, 
				parameters.Train_source, Global)

			print(best_source_models[0])
			models["gt_target_group"] = best_gt_target_models
			jason(os.path.join(Global.base_path, "best_gt_target_models.json"), 'w', 
				models["gt_target_group"])
			print('[*] Best source models' , models["gt_target_group"])

	else:
		if Global.train_source_json != '':
			models["source_group"] = jason(Global.train_source_json, 'r')
			print("[*] Loaded Source Json...")
		if Global.train_gt_target_json != '':
			models["gt_target_group"] = jason(Global.train_gt_target_json, 'r')
			print("[*] Loaded GT Target Json...")

		# jason(os.path.join(Global.base_path, "best_models.json"), 'w', models)
	# ------ end ------


	# ------ cyclegan training ------
	
	if not Global.skip_train_cyclegan:
		print("[*] Train CycleGAN...")
		train_cyclegan_model(source, source_args, target, target_args, 
			parameters.Train_cyclegan, Global)

		print('[*] Cyclegan trained models in', Global.cyclegan_models_path)
		epochs = str(parameters.Train_cyclegan.num_epochs - 1)
		last_AB = os.path.join(Global.cyclegan_models_path, Global.base_A2B_name + epochs + ".pt")
		last_BA = os.path.join(Global.cyclegan_models_path, Global.base_B2A_name + epochs + ".pt")
		last_gans = [last_AB, last_BA]
		# last_gans["G_AB"] = last_AB
		# last_gans["G_BA"] = last_BA
		del epochs

		jason(os.path.join(Global.base_path, "last_gans_paths.json"), 'w', 
			last_gans)

	elif Global.train_cyclegan_json != '':
		last_gans = jason(Global.train_cyclegan_json, 'r')
		print("[*] Loaded CycleGAN Json...")
		# with open(os.path.join(Global.base_path, 'last_gans_paths.json'), 'w') as f:
		# 	json.dump(last_gans, f)
	# ------ end ------
	

	# # ------ cyclegan evaluation ------
	# best_gan_epoch = evalutate_cyclegan_model(source, source_args, parameters.Evaluate_cyclegan, 
	# 	Global, gan_models_path)

	# A2B_best_gan_model = gan_models_path + Global.base_A2B_name + best_gan_epoch + '.pt'
	# B2A_best_gan_model = gan_models_path + Global.base_B2A_name + best_gan_epoch + '.pt'
	# source_args.best_generator = A2B_best_gan_model
	# target_args.best_generator = B2A_best_gan_model

	# print('[*] Best Source to Target GAN', A2B_best_gan_model)
	# print('[*] Best Target to Source GAN', B2A_best_gan_model)
	# gans = [A2B_best_gan_model, B2A_best_gan_model]
	# models["G_A2B"] = A2B_best_gan_model
	# models["G_B2A"] = B2A_best_gan_model

	# with open(os.path.join(Global.base_path, "best_gans" + "_" + 
	# 	file_name + ".json"), 'w') as f:
	# 	json.dump(gans, f)
	# with open(os.path.join(Global.base_path, 'best_models.json'), 'w') as f:
	# 	json.dump(models, f)
	# # ------ end ------


	# ------ deforestation train on target domain -----
	if not Global.skip_train_adapted_target:
		print("[*] Train Adpated Target...")
		if models == {}:
			models["source_group"] = jason(os.path.join(Global.base_path, 
				"best_source_models.json"), 'r')
		if last_gans == []:
			last_gans = jason(os.path.join(Global.base_path, "last_gans_paths.json"), 'r')

		if Global.inverse_cyclegan:
			source_args.best_generator = last_gans[1]
			target_args.best_generator = last_gans[0]
		else:
			source_args.best_generator = last_gans[0]
			target_args.best_generator = last_gans[1]

		source_args.best_source_model = models["source_group"][0]

		best_source_model = source_args.best_source_model
		A2B_best_gan_model = source_args.best_generator
		B2A_best_gan_model = target_args.best_generator
		print(best_source_model)
		print(A2B_best_gan_model)
		print(B2A_best_gan_model)

		best_target_models = train_target_model(source, source_args, target, target_args, 
			parameters.Train_adapted_target, parameters.Train_source, Global, best_source_model, 
			A2B_best_gan_model, B2A_best_gan_model)

		# print('[*] Best target model on source (through validation)', best_model)
		print('[*] Theorical best model (on target, through validation)', best_target_models)
		# models["adapted_target"] = best_model
		models["adapted_target_group"] = best_target_models
		# models["source"] = best_source_model

		# with open(Global.base_path + "/best_models.json", "w") as f:
		# 	json.dump(models, f)
		jason(os.path.join(Global.base_path, "best_adapted_target_models" 
			+ ".json"), 'w', models["adapted_target_group"])

	elif Global.train_adapted_target_json != '':
		models["adapted_target_group"] = Global.train_adapted_target_json
	else:
		p = os.path.join(Global.base_path, "best_adapted_target_models.json")
		if os.path.isfile(p):
			models["adapted_target_group"] = jason(p, 'r')
	# ------ end ------


	# ------ deforestation test with source, best target and theorical best target models ------
	# ------ on source and target images ------
	if not Global.skip_test:
		print("[*] Testing Models...")
		# if models == {}:
		# 	models["source_group"] = jason(os.path.join(Global.base_path, 
		# 			"best_source_models.json"), 'r')
		# 	models["adapted_target_group"] = jason(os.path.join(Global.base_path, 
		# 			"best_adapted_target_models.json"), 'r')
		# 	path = os.path.join(Global.base_path, "best_gt_target_models.json")
		# 	if os.path.isfile(path):
		# 		# print ("gt_target_group")
		# 		models["gt_target_group"] = jason(path, 'r')
		# 	del path
		
		#--- use it to test single model VVVV ---
		# del models
		# models = {}
		# path = "./experimentos/PA-RO_A1/best_source_models.json"
		# models["source_group"] = jason(path, 'r')
		# --- end here ---
		
		run_test(source, source_args, target, target_args, parameters.Test_models, Global, models, file_name, last_gans)
	# ------ end ------




	## ------ Total time ------
	total_time = time.time() - total_time
	if (total_time > 3600):
		print("{:.2f} hours".format(round(total_time/3600, 2)))
	elif (total_time > 60):
		print("{:.2f}minutes".format(round(total_time/60, 2)))
	else:
		print("{:.2f} seconds".format(round(total_time, 2)))



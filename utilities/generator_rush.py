import os

import sys
sys.append('..')
from dataset_preprocessing.Amazon_PA import *
from dataset_preprocessing.Amazon_RO import *
from dataset_preprocessing.Cerrado_MA import *

from Desmatamento.code.utilities.reconstruction_tool import *

def save_gans_adpted_domains(domain, domain_args, gan_models_path, save_path, 
	rec_options, A2B_or_B2A = "A2B", only_last_gan = False, pedro = False, net_ = False):

	path = save_path + '/'

	MODEL_PATH = gan_models_path

	if not net_:
		model_names = [name for name in os.listdir(MODEL_PATH) if
					os.path.splitext(name)[1] == '.pt' and name.split('_')[1] == A2B_or_B2A]
		model_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	else:
		if A2B_or_B2A == "A2B":
			A2B_or_B2A = "_net_G_A"
		elif A2B_or_B2A == "B2A":
			A2B_or_B2A = "_net_G_B"

		model_names = [name for name in os.listdir(MODEL_PATH) if 
			name.split(name.split('_')[0])[1] == A2B_or_B2A + '.pth']
		
		if A2B_or_B2A == "_net_G_A":
			if "latest_net_G_A.pth" in model_names:
				model_names.remove("latest_net_G_A.pth")
		elif A2B_or_B2A == "_net_G_B":
			if "latest_net_G_B.pth" in model_names:
				model_names.remove("latest_net_G_B.pth")
		
		model_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	if not net_:
		begin = 1
	else:
		begin = 0
	
	
	print(model_names)
	size = len(model_names)
	if only_last_gan:
		if not net_:
			model_epoch = model_names[-1].split('_')[-1].split('.')[0]
		else:
			model_epoch = model_names[-1].split('_')[0]
			print(model_epoch)
		save_path = path + model_epoch + '_'
		path_to_weights = MODEL_PATH + model_names[-1]
		print(model_names[-1])
		overlap_reconstruction(domain.conc_image, domain.scaler, domain_args, 
			rec_options, path_to_weights, save_path, eval = False, pedro = pedro,
			net_ = net_)
	else:
		for i in range (begin, size):
			if not net_:
				model_epoch = model_names[i].split('_')[-1].split('.')[0]
			else:
				model_epoch = model_names[i].split('_')[0]
			# print (model_epoch)
			# if int(model_epoch) % 10 != 0:
			# 	continue
			# if int(model_epoch) > 20:
			# 	continue
			save_path = path + model_epoch + '_'
			# print (save_path)
			path_to_weights = MODEL_PATH + model_names[i]
			print (model_names[i])
			
			overlap_reconstruction(domain.conc_image, domain.scaler, domain_args, 
				rec_options, path_to_weights, save_path, eval = False, pedro = pedro,
				net_ = net_)


def check_labels(path):
    path = path.split('/')[-1]
    source = path.split('-')[0]
    target = path.split('-')[1].split('_')[0]
    if source[0] == 's':
        source = source[1:]
    if target[0] == 't':
        target = target[1:]
#     print("source", source)
#     print("target",target)
    ftype = path.split('_')[1]
    letter = ftype[0]
    number = ftype.split(letter)[1]
#     print (letter)
#     print(number)
    
    if letter == 'A':
        model = "CN"
    elif letter == 'B':
        model = "CT"
    elif letter == 'D':
        model = "CD"
    model += f'_s{source}_t{target}'
        
    # if number == '1':
    #     model += "-T"
    # elif number == '2':
    #     model += "-P"
    # elif number == '3':
    #     model += "-TP"
    # elif number == '4':
    #     model += "-TPF"
    # model += f"({target})"
    
    return source, target, model, path

def check_folders(path):
    if not os.path.isdir(path):
        return 0
    else: 
    	return 1

def run_all(list_of_folders, only_last_gan, pedro = False):
	for i in range(len(list_of_folders)):
		path = os.path.join(experiments_path, list_of_folders[i], default_gan_folder)
		print(path)
		check_result = check_folders(path)

		if check_result != 0:
			source_name, target_name, model, file_name = check_labels(list_of_folders[i])

			if source_name == 'PA':
				source = pa
				source_args = pa_args
			elif source_name == 'RO':
				source = ro
				source_args = ro_args
			elif source_name == 'MA':
				source = ma
				source_args = ma_args

			if target_name == 'PA':
				target = pa
				target_args = pa_args
			elif target_name == 'RO':
				target = ro
				target_args = ro_args
			elif target_name == 'MA':
				target = ma
				target_args = ma_args

			print(model)

			# source
			save_folder = f"{model}({source_name})"
			save_path = os.path.join(output_path, save_folder)
			print(save_path)
			rec_param = reconstruction_parameters(source_args, source)
			rec_param.save_tiff = True
			save_gans_adpted_domains(source, source_args, path, save_path, rec_param,
				A2B_or_B2A = "A2B", only_last_gan = only_last_gan, pedro = pedro)

			# target
			save_folder = f"{model}({target_name})"
			save_path = os.path.join(output_path, save_folder)
			print(save_path)
			rec_param = reconstruction_parameters(target_args, target)
			rec_param.save_tiff = True
			save_gans_adpted_domains(target, target_args, path, save_path, rec_param,
				A2B_or_B2A = "B2A", only_last_gan = only_last_gan, pedro = pedro)
		else:
			# print(check_result)
			print(f"No cyclegan in '{list_of_folders[i]}'")
			continue


import parameters_1 as parameters1
from exp_param import parameters_sMA_tPA_A1 as parameters2

pa = AM_PA(parameters1.PA)
pa_args = parameters1.PA

# ro = AM_RO(parameters1.RO)
# ro_args = parameters1.RO

ma = CE_MA(parameters2.MA)
ma_args = parameters2.MA


experiments_path = "./experimentos/"
default_gan_folder = "models/cyclegan/saved_models/"

output_path = "./domain_adapts/"
# only_last_gan = False

list_of_folders = os.listdir(experiments_path)
# print(list_of_folders)

# run_all(list_of_folders, only_last_gan)

# output_path = './GAN_test/og_sRO_tMA_CDNorm/'
# path = "./CycleGAN_D/checkpoints/og_sRO_tMA_CDNorm/"

# "./tmp/CG_CN(T_comentado)/models/cyclegan/saved_models/"
# "./tmp/CG_Adversarial/models/cyclegan/saved_models/"
# "./tmp/CG_CD/models/cyclegan/saved_models/"
# "./tmp/CG_CN/models/cyclegan/saved_models/"
# "./tmp/CG_CT/models/cyclegan/saved_models/"
# "./tmp/CG_CNsemIdt/models/cyclegan/saved_models/"

# "./newGAN/newGAN_sPA_tRO_CN/models/cyclegan/saved_models/"
# "./newGAN/newGAN_sPA_tRO_CT/models/cyclegan/saved_models/"
# "./newGAN/newGAN_sRO_tMA_CN/models/cyclegan/saved_models/"
# "./newGAN/newGAN_sRO_tMA_CT/models/cyclegan/saved_models/"
# "./newGAN/newGANnewTrain_sRO_tMA_CN/models/cyclegan/saved_models/"


# "./CycleGAN_D/checkpoints/prove/"
# "./CycleGAN_D/checkpoints/sRO_tMA_CN/prove/"
# "./CycleGAN_D/checkpoints/nada/"
# "./CycleGAN_D/checkpoints/mogsRO_tMA_CN/"
# "./CycleGAN_D/checkpoints/mog_sRO_tMA_CT/"
# "./CycleGAN_D/checkpoints/mog_sRO_tMA_CT2/"
# "./CycleGAN_D/checkpoints/mog_sRO_tMA_CD/"

# og
# ./CycleGAN_D/checkpoints/og_sRO_tMA_CN/
# ./CycleGAN_D/checkpoints/og_sRO_tMA_CD/
# ./CycleGAN_D/checkpoints/og_sRO_tMA_CT/
# ./CycleGAN_D/checkpoints/og_sRO_tMA_CDNorm/

# mog
# ./CycleGAN_D/checkpoints/mog_sRO_tMA_CN/
# ./CycleGAN_D/checkpoints/mog_sRO_tMA_CDNorm/
# ./CycleGAN_D/checkpoints/mog_sRO_tMA_CT/



### GAN og_exp
## MA-RO
# ./cyclegan_models/MA-RO/og_sRO_tMA_CN/
# ./cyclegan_models/MA-RO/og_sRO_tMA_CDNorm/

# ./cyclegan_models/MA-RO/og_sMA_tRO_CT/
# ./cyclegan_models/MA-RO/og_sMA_tRO_CTDNorm/

# ./cyclegan_models/MA-RO/og_sRO_tMA_CT/
# ./cyclegan_models/MA-RO/og_sRO_tMA_CTDNorm/


## PA-MA
# ./cyclegan_models/PA-MA/og_sPA_tMA_CN/
# ./cyclegan_models/PA-MA/og_sPA_tMA_CDNorm/

# ./cyclegan_models/PA-MA/og_sMA_tPA_CT/
# ./cyclegan_models/PA-MA/og_sMA_tPA_CTDNorm/

# ./cyclegan_models/PA-MA/og_sPA_tMA_CT/
# ./cyclegan_models/PA-MA/og_sPA_tMA_CTDNorm/


## RO-PA
# /cyclegan_models/RO-PA/og_sPA_tRO_CN/
# /cyclegan_models/RO-PA/og_sPA_tRO_CDNorm/

# /cyclegan_models/RO-PA/og_sPA_tRO_CT/
# /cyclegan_models/RO-PA/og_sPA_tRO_CTDNorm/

# /cyclegan_models/RO-PA/og_sRO_tPA_CT/
# /cyclegan_models/RO-PA/og_sRO_tPA_CTDNorm/




output_path = './GAN_exp/PA-MA/og_sPA_tMA_CTDNorm/'
path = "./cyclegan_models/PA-MA/og_sPA_tMA_CTDNorm/"

rec_param = reconstruction_parameters(pa_args, pa)
rec_param.save_tiff = True
rec_param.save_only_present = True # [7:,:,:]
rec_param.save_only_present_rgb = True # [7:7+3,:,:]
# A2B
# B2A
save_gans_adpted_domains(pa, pa_args, path, output_path, rec_param, 
	A2B_or_B2A = "A2B", only_last_gan = True, pedro = True, net_ = True)

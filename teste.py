# from Amazonia_Legal import *
# import RO_args as ro
# from Cerrado_Biome import *
# import PA_args as pa
# import CyGAN

# model_path = "./tmp/models/cyclegan/pos/saved_models/20200903-192153_test/G_B2A_model_weights_epoch_49.pt"
# y = ready_domain(AMAZON_RO, ro, True)
# y.Prepare_GAN_Set(ro.Args(), CyGAN, model_path)
# print("dapted shape", y.adapted_image.shape)
# print(y.adapted_train_set[0:20])
# print(y.adapted_validation_set[0:20])

# import parameters
# from Tools import *
# import utils

# from Amazonia_Legal import *
# from Cerrado_Biome import *
# # import RO_args
# # import PA_args

# def this_domain(args):
# 	args.patches_dimension = parameters.Train_source.patch_size

# if __name__=='__main__':
# 	print(parameters.Train_source.patch_size)
# 	print(parameters.PA.patches_dimension)
# 	this_domain(parameters.PA)
# 	print(parameters.PA.patches_dimension)



# def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
#             self.G_A2B.eval()
#             self.G_B2A.eval()
#             device = next(self.G_A2B.parameters()).device
#             directory = os.path.join(output_path, 'images', self.date_time)
#             if not os.path.exists(os.path.join(directory, 'A')):
#                 os.makedirs(os.path.join(directory, 'A'))
#                 os.makedirs(os.path.join(directory, 'B'))
#                 os.makedirs(os.path.join(directory, 'Atest'))
#                 os.makedirs(os.path.join(directory, 'Btest'))

#             testString = ''

#             real_image_Ab = None
#             real_image_Ba = None
#             for i in range(num_saved_images + 1):
#                 if i == num_saved_images:
#                     real_image_At = real_image_A[177]
#                     real_image_Bt = real_image_B[3]
#                     real_image_At = np.expand_dims(real_image_At, axis=0)
#                     real_image_Bt = np.expand_dims(real_image_Bt, axis=0)
#                     testString = 'test'
                    
#                 else:
#                     index_A = random.randint(1,(len(real_image_A)-2))
#                     index_B = random.randint(1,(len(real_image_B)-2))
#                     real_image_At = real_image_A[index_A:index_A+1]
#                     real_image_Bt = real_image_B[index_B:index_B+1]
#                     if len(real_image_A.shape) < 4:
#                         real_image_A = np.expand_dims(real_image_A, axis=0)
#                         real_image_B = np.expand_dims(real_image_B, axis=0)
                        
#                 with torch.no_grad():
#                     real_image_At = torch.from_numpy(real_image_At).float().to(device)
#                     real_image_Bt = torch.from_numpy(real_image_Bt).float().to(device)
#                     synthetic_image_B = self.G_A2B(real_image_At)
#                     synthetic_image_A = self.G_B2A(real_image_Bt)
#                     reconstructed_image_A = self.G_B2A(synthetic_image_B)
#                     reconstructed_image_B = self.G_A2B(synthetic_image_A)n()








# import GAN
# import torch
# from Cerrado_Biome import *
# from Tools import *
# import parameters
# import utils
# import numpy as np
# import matplotlib.pyplot as plt

# a2b_path = f'./tmp/models/cyclegan/pos/saved_models/20200903-192153_test/G_A2B_model_weights_epoch_49.pt'
# b2a_path = f'/home/josematheus/Desktop/Mestrado/Desmatamento/code/tmp/models/cyclegan/pos/saved_models/20200903-192153_test/G_B2A_model_weights_epoch_49.pt'

# G_A2B = GAN.modelGenerator()
# G_A2B.load_state_dict(torch.load(a2b_path))
# G_A2B.eval()
# G_A2B.cuda()

# G_B2A = GAN.modelGenerator()
# G_B2A.load_state_dict(torch.load(b2a_path))
# G_B2A.eval()
# G_B2A.cuda()

# pa = CERRADO(parameters.PA)
# # ready_domain(pa, parameters.PA, train_set = True, augmented = False)
# # x = pa.central_pixels_coor_tr[0]
# x = np.zeros((1,2))
# print(x)
# print(x.shape)
# original, nada = patch_extraction(utils.channels_last2first(pa.images), 
# 		pa.new_reference, x, parameters.PA.patches_dimension)
# with torch.no_grad():
# 	a , b = patch_extraction(utils.channels_last2first(pa.conc_image), 
# 		pa.new_reference, x, parameters.PA.patches_dimension)
# 	print(a.shape)
# 	s = torch.from_numpy(a).float().cuda()
# 	print(s.shape)
# 	# print(s[0, 0, :10,:10])
# 	s2t = G_A2B(s)
# 	# print(s2t[0, 0, :10,:10])
# 	back2s = G_B2A(s2t)
# 	# print(back2s[0, 0, :10,:10])
# 	output_a = np.zeros((a.shape[0], a.shape[1], a.shape[2], a.shape[3]*3))
# 	output_a[:,:,:64,0:64] = s.cpu()
# 	output_a[:,:,:64,64:128] = s2t.cpu()
# 	output_a[:,:,:64,128:] = back2s.cpu()
# 	# print(output_a.shape)
# 	# print(output_a[0,0])
# # print(pa.scaler)
# output_a = output_a.reshape((output_a.shape[0] * output_a.shape[1], output_a.shape[2], output_a.shape[3]))
# print(output_a.shape)
# output_a = utils.channels_first2last(output_a)
# print(output_a.shape)
# image_reshaped = output_a.reshape((output_a.shape[0] * output_a.shape[1], output_a.shape[2]))
# print(image_reshaped.shape)
# # scaler = pa.scaler.fit(image_reshaped)
# output = pa.scaler.inverse_transform(image_reshaped)
# output = output.reshape((output_a.shape))
# output = np.rint(output)
# print(output)
# plt.imshow(output[:,:,0:3])
# sio.savemat('image.mat', {'label': output})

# a = a.reshape((a.shape[0] * a.shape[1], a.shape[2], a.shape[3]))
# print(a.shape)
# a = utils.channels_first2last(a)
# print(a.shape)
# reshaped_a = a.reshape((a.shape[0]*a.shape[1], a.shape[2]))
# print(reshaped_a[0:10,0:3])
# scaler = pa.scaler
# new_a = scaler.inverse_transform(reshaped_a)
# print("here",new_a[0:10,0:3])
# new_a = new_a.reshape((a.shape))
# print(new_a.shape)
# print("now",np.rint(new_a[0:10,0,0:3]))
# original = original.reshape((original.shape[0] * original.shape[1], original.shape[2], original.shape[3]))
# original = utils.channels_first2last(original)
# print(original.shape)
# print("original", original[0:10,0,0:3])





# import torch
# from Cerrado_Biome import *
# from Tools import *
# import parameters
# import utils
# import numpy as np


# pa = CERRADO(parameters.PA)
# # ready_domain(pa, parameters.PA, train_set = True, augmented = False)
# # x = pa.central_pixels_coor_tr[0]
# x = np.ones((1,2))
# print(x)
# print(x.shape)
# a , y = patch_extraction(utils.channels_last2first(pa.conc_image), 
# 	pa.new_reference, x, parameters.PA.patches_dimension)
# print(y.shape)
# y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
# print(np.unique(y))
# y[(y == 2)] = 0
# print(y.shape)
# xargmax = np.zeros((y.shape))
# print(xargmax.shape)
# print(compute_metrics(y,xargmax))

# print(np.zeros((1,10,10)))
# x = np.array([[
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#   [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
#   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#   [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   ]])
# y = np.array([[
#   [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#   [0, 0, 0, 2, 0, 0, 2, 0, 0, 0],
#   [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#   [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
#   [0, 0, 0, 0, 1, 2, 0, 0, 0, 1],
#   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#   ]])
# # x = x.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
# # y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
# y_mask = (y == 1)
# print("y tp", np.count_nonzero(y_mask))
# # y_n_mask = (y == 0)
# dc_mask = (y == 2)
# # print(dc_mask)
# print("y tn", np.count_nonzero(y_mask))
# # x_p_mask = (x == 1)
# # x_n_mask = (x == 0)
# # print("fn",fn)
# return_metrics(y, x)
# final_mask = (((x * 2) - (y_mask * 1) - (dc_mask*4)))

# values for x:
# 2 e 0
# values for y:
# 1 e 0
# values for dc - dont care:
# 4 e 0

#### x - y - dc:
# not possible -> (2 - 1 - 4 = -3)
# not possible -> (0 - 1 - 4 = -5)

# 2 - 0 - 4 = -2 -> dc
# 0 - 0 - 4 = -4 -> dc

# 0 - 1 - 0 = -1 -> fn
# 2 - 1 - 0 = 1 -> tp
# 2 - 0 - 0 = 2 -> fp
# 0 - 0 - 0 = 0 -> tn 
# print(np.count_nonzero((x == y_mask) != dc_mask))

# dc = np.count_nonzero((final_mask < -1))
# print("dc",dc)
# fn = np.count_nonzero((final_mask == -1))
# tp = np.count_nonzero((final_mask == 1))
# fp = np.count_nonzero((final_mask == 2))
# tn = np.count_nonzero((final_mask == 0))

# print("tp",tp)
# print("fp", fp)
# print("tn", tn)
# print("fn", fn)

# prescision=0.0
# f1score = 0.0       
# recall = tp/(tp+fn)
# if(tp+fp!=0):
# 	prescision = tp/(tp+fp)
# if(prescision!=0):
# 	f1score = 100*(2*prescision*recall)/(prescision+recall)
# P = tp+fn
# N = tn+fp
# overall = 100*(tp+tn)/(P+N)
# alert_rate = 100*(tp+fp)/(P+N)
# recall = recall * 100
# prescision = prescision * 100
# print("Overall: %.2f" % overall)
# print("F1-Score: %.2f" % f1score)
# print("Recall: %.2f" % recall)
# print("Prescision: %.2f" % prescision)
# print("Alert Rate: %.2f" % alert_rate)

# print((np.where(y == 0)))
# print(y)

# print(compute_metrics(y,x))
# x = x.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
# y = y.reshape((y.shape[0] * y.shape[1] * y.shape[2]))
# calcula_metricas(y,x)


# x = x.argmax(1)

# 		    x = x.cpu()
# 		    y = y.cpu()
# 		    x = x.reshape(x.shape[0] * x.shape[1] * x.shape[2])
# 		    y = y.reshape(y.shape[0] * y.shape[1] * y.shape[2])
# 		    # print(x.shape)
# 		    # print(y.shape)
# 		    x[:][(x == 2)] = 0
# 		    y[:][(y == 2)] = 0
# 		    # print(np.unique(x))
# 		    # print(np.unique(y))
# 		    acc, f1, rec, prec, conf = compute_metrics(y,x)

# 		    # acc, f1, rec, prec, alert = calcula_metricas(y,x)
# 		    metrics["seg_acc"] = acc
# 		    metrics["seg_f1"] = f1
# 		    metrics["seg_rec"] = rec
# 		    metrics["seg_prec"] = prec
# 		    # metrics["seg_conf"] = conf

# best_source_model = './tmp/models/source_seg/main_test_PA_source/model_05_54.3222.pt'
# A2B_best_gan_model = './tmp/models/cyclegan/main_test_PA_source/saved_models/G_A2B_model_weights_epoch_20.pt'
# B2A_best_gan_model = './tmp/models/cyclegan/main_test_PA_source/saved_models/G_B2A_model_weights_epoch_20.pt'
# print('[*] Best Source model',best_source_model)
# print('[*] Best Source to Target GAN', A2B_best_gan_model)
# print('[*] Best Target to Source GAN', B2A_best_gan_model)
# # models = {}
# # models["souce"] = best_source_model
# models = [best_source_model, A2B_best_gan_model, B2A_best_gan_model]
# # models.append("yolo")
# print (models)
# import json
# with open("testee.json", "w") as f:
# 		json.dump(models, f)

# # with open("teste.txt", "w") as f:
# #     for s in models:
# #         f.write(str(s) +"\n")


# import parameters
# from Amazonia_Legal import *
# from Cerrado_Biome import *
# from Tools import *





from reconstruction_tool import *
import tifffile
import GAN
import utils
import numpy as np
import torch

# opt = opt_parameters()
# pa = CERRADO(parameters.PA)
# print(pa.conc_image.shape)
# path_to_weights = './tmp/new_test_cycle_130epoc_10_batch5_weight0/models/cyclegan/saved_models/G_A2B_model_weights_epoch_90.pt'


def overlap_reconstruction(domain, args, rec_options, path_to_weights, save_path, eval,
	save_mat = False, save_tiff = False, de_normalize = False, single_image = False):

	if eval == True:
		save_path = save_path + "adapted_conc_" + args.dataset + "_eval"
	else:
		save_path = save_path + "adapted_conc_" + args.dataset
	
	do = domain
	img = do.conc_image
	opt = rec_options
	opt.rows_size_s = img.shape[0]
	opt.cols_size_s = img.shape[1]
	opt.output_nc = img.shape[-1]
	opt.save_path = save_path
	scaler = do.scaler
	rspc = RemoteSensingPatchesContainer(scaler, opt)
	
	if not single_image:
		coord = Coordinates_Definition_Test (img.shape[0], img.shape[1], opt, from_source = True)
		opt.size_s = coord.shape[0]
		iterator = torch.utils.data.DataLoader(coord, batch_size = opt.batch_size)

	img = utils.channels_last2first(img)
	
	G_A2B = GAN.modelGenerator(channels = opt.output_nc)
	G_A2B.load_state_dict(torch.load(path_to_weights))
	G_A2B.eval()
	G_A2B.cuda()
	
	counter = 0
	with torch.no_grad():
		for this_coord in iterator:
			output = {}
			output['fake_B'] = {}

			batch = Patch_Extraction_Test(img, this_coord, opt, from_source = True)
			batch = torch.from_numpy(batch).float().cuda()
			adapted_batch = G_A2B(batch)
			output['fake_B'] = adapted_batch
			
			rspc.store_current_visuals(output, counter)
			counter += len(this_coord)

	rspc.save_images(save_mat = save_mat, save_tiff = save_tiff, de_normalize = de_normalize)

# overlap_reconstruction(pa, parameters.PA, opt, path_to_weights, './recaste_in_pacheval/', eval = True, save_tiff = True)


def save_gans_adpted_domains(gan_models_path, save_path, rec_options,
	save_mat = False, save_tiff = False, de_normalize = False, single_image = False):

	path = save_path + '/'

	pa = CERRADO(parameters.PA)
	ro = AMAZON_RO(parameters.RO)
	source = pa
	source_args = parameters.PA

	# if parameters.Global.source_domain == "PA":
	# 	source = pa
	# 	source_args = parameters.PA
	# 	target = ro
	# 	target_args = parameters.RO
	# else:
	# 	source = ro
	# 	source_args = parameters.RO
	# 	target = pa
	# 	target_args = parameters.PA

	MODEL_PATH = gan_models_path
	model_namesA2B = [name for name in os.listdir(MODEL_PATH) if
				os.path.splitext(name)[1] == '.pt' and name.split('_')[1] == 'A2B']
	model_namesA2B.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

	model_namesB2A = [name for name in os.listdir(MODEL_PATH) if
				os.path.splitext(name)[1] == '.pt' and name.split('_')[1] == 'B2A']
	model_namesB2A.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	print(model_namesA2B)
	
	for i in range (1, len(model_namesA2B)):
		model_epoch = model_namesA2B[i].split('_')[-1].split('.')[0]
		print (model_epoch)
		if int(model_epoch) % 10 != 0:
			continue
		# if int(model_epoch) > 20:
		# 	continue
		save_path = path + model_epoch + '_'
		print (save_path)
		path_to_weights = MODEL_PATH + model_namesA2B[i]

		overlap_reconstruction(source, source_args, rec_options, path_to_weights, 
			save_path, eval = False, save_mat = save_mat, save_tiff = save_tiff, 
			de_normalize = de_normalize, single_image = single_image)
		# adapt_domain(source, source_args, path_to_weights, save_path, eval = False,
		# 	save_tiff = save_tiff, de_normalize = de_normalize)

		# path_to_weights = MODEL_PATH + model_namesB2A[i]
		# overlap_reconstruction(target, target_args, rec_options, path_to_weights, 
		# 	save_path, eval = False, save_mat = save_mat, save_tiff = save_tiff, 
		# 	de_normalize = de_normalize)

		# adapt_domain(target, target_args, path_to_weights, save_path, eval = False, 
		# 	save_tiff = de_normalize, de_normalize = de_normalize)

# gan_models_path = './tmp/main_test_PA_source/models/cyclegan/saved_models/'
# gan_models_path = "./tmp/new_test_cycle_120epoc_10_batch5_weight03/models/cyclegan/saved_models/"
# gan_models_path = "./tmp/new_test_cycle_130epoc_10_batch5_weight0/models/cyclegan/saved_models/"
gan_models_path = []
# gan_models_path.append("./tmp/new_test_cycle_130epoc_5_batch5_weight03_identity/models/cyclegan/saved_models/")
# gan_models_path.append("./tmp/new_test_cycle_130epoc_5_batch5_sameweight1_identity/models/cyclegan/saved_models/")
# gan_models_path.append("./tmp/new_test_cycle_130epoc_5_batch5_NoSem_identity/models/cyclegan/saved_models/")
# gan_models_path.append("./tmp/new_test_cycle_130epoc_5_batch5_weight03_identity_notAlltgt/models/cyclegan/saved_models/")

# gan_models_path.append("./tmp/new_test_cycle_130epoc_5_batch5_sameweight1_identity/models/cyclegan/saved_models/")

# gan_models_path.append("./tmp/new_test_cycle64_400epoc_10_batch20_weight03_identity/models/cyclegan/saved_models/")
# gan_models_path.append("./tmp/new_test_cycle64_400epoc_10_batch20_weight03_identity_crossentropy/models/cyclegan/saved_models/")
gan_models_path.append("./tmp/new_test_cycle_40epoc_5_batch5_weight03_identity_crossentropy/models/cyclegan/saved_models/")

# save_path = "./all_gan_models_adaptations_weight_0"
# save_path = "./all_gan_models_adaptations_weight_03"
# save_path = "./all_gan_models_adaptations"
save_path = []
# save_path.append("./adapt_03_identity")
# save_path.append("./adapt_sameWeight_identity")
# save_path.append("./adapt_noClassification_identity")
# save_path.append("./adapt_03_identity_notAllTarget")

# save_path.append("./adapt_256On64_Focal")
# save_path.append("./adapt_256On64_Cross")
save_path.append("./adapt_40-256On256_Cross")
single_image = False

# opt = opt_parameters()
# mog: to run, uncomment
# for i in range(len(save_path)):	
# 	print("image rec iteration:", i)
# 	save_gans_adpted_domains(gan_models_path[i], save_path[i], opt, 
# 		save_tiff = True, de_normalize = True, single_image = single_image)






# import json
# # json_path = "./tmp/new_test_cycle64_400epoc_10_batch20_weight03_identity/best_gans_paths_parameters_3.py.json"
# # json_path = "./tmp/new_test_cycle64_400epoc_10_batch20_weight03_identity/best_models_paths.json"
# with open('this_file.txt', 'a') as f:
# 	f.write("oh shit, here we go again\n")
# 	f.write("die motherfucker")
# 	print(f)

# for model in models:
# 	model_name = model.split('/')[-1]
# 	print(model_name)
# 	for 
# 	metrics = updated_test_models.test(source, test_args, model)
# 	with open(Global.base_path + '/' + file_name + "_source" + 
# 		source_args.dataset + '_' + model_name + 'txt', 'a') as f:
# 		f.write('--------' + model_name + ':--------\n')
# 		f.write(metrics[model].keys()[0])
# 		for metric in metrics:
# 			f.write(

# data = {
# 	# general train
# 	'epoch': 0,
# 	'metrics': 1,
# 	# models
# 	'G_A2B_model': 2,
# 	'G_B2A_model': 3,
# 	'D_A_model': 4,
# 	'D_B_model': 5,
# 	'Semantic_model': 6,
# 	# optimizers
# 	'D_A_optimizer': 7,
# 	'D_B_optimizer': 8,
# 	'G_A2B_optimizer': 9,
# 	'G_B2A_optimizer': 0
	

# }
# data['Semantic_optimizer'] = 11

# print(type(data))
# print(data)


# output = {}
# output['fake_B'] = {}
# output['fake_B'] = np.arange(50)

# for label in output:
# 	for image in output[label]:
# 		print (label)
# 		print (image)
# 	# print(image[0].data)


# import DeepLabV3plus
# import parameters


# pa = CERRADO(parameters.PA)
# # ro = AMAZON_RO(parameters.RO)
# source = pa
# source_args = parameters.PA

# best_source_model_path = './tmp/pre_train_source_seg/model_05_54.3222.pt'
# train_source_args = parameters.Train_source
# patch_size = 128

# source_args.patches_dimension = patch_size
# source.Coordinates_Domain_Adaptation(source_args)
# # ready_domain(source, source_args, train_set = False, cyclegan_set = True)

# source_deep_lab_v3p = DeepLabV3plus.create(train_source_args)
# source_deep_lab_v3p = source_deep_lab_v3p.cuda() # source semantic model
# checkpoint1 = torch.load(best_source_model_path)
# source_deep_lab_v3p.load_state_dict(checkpoint1)
# source_deep_lab_v3p.eval()

# loss = 0
# x, y = patch_extraction(utils.channels_last2first(source.conc_image), 
# 	source.new_reference, source.no_stride_coor, patch_size)
# with torch.no_grad():
# 	x = torch.from_numpy(x).float().cuda()
# 	x = source_deep_lab_v3p(x)
# 	y = torch.from_numpy(y).long().cuda()
# 	# loss += self.seg_loss_fn(x, y).item()
# print(loss)



# yolo = np.arange(1.0, 0, -1.0/(19))
# print(yolo)
# yolo2 = np.transpose(np.array(np.where(yolo >= 0.5)))
# print(yolo2)
# yolo2 = np.argwhere(yolo >= 0.5)
# print(yolo2)
# print(yolo.max(0))

# min_array = np.zeros((1 , ))
# print(min_array.shape)
# positive_map_init = np.zeros_like(yolo)
# z2 = np.transpose(np.array(np.where(yolo >= 0.5)))
# positive_map_init[z2[:].astype('int')] = 1
# print(positive_map_init)



# group_models = {}
# group_models["source_group"] = []
# group_models["source_group"].append(0)
# print(group_models)
# models = []
# models.append(0)
# print(models)
# print(models[0])

# print(type(0))
# print(type(str(0)))

# models = {}
# print(models)
# print(models == 1)


# import DeepLabV3plus
# import parameters

# args = parameters.Train_source
# device = torch.device("cuda:0")

# deep_lab_v3p = DeepLabV3plus.create(args)
# deep_lab_v3p = deep_lab_v3p.to(device)

# source_deep_lab_v3p = DeepLabV3plus.create(args)

# discriminator_type = 0

# class ck(nn.Module):
#     def __init__(self, i, k, use_normalization):
#         super(ck, self).__init__()
#         self.conv_block = self.build_conv_block(i, k, use_normalization)

#     def build_conv_block(self, i, k, use_normalization):
#         conv_block = []                       
#         conv_block += [nn.Conv2d(i, k, 1)]
#         if use_normalization:
#             conv_block += [nn.BatchNorm2d(k)]
#         conv_block += [nn.ReLU()]
#         return nn.Sequential(*conv_block)

#     def forward(self, x):
#         out = self.conv_block(x)
#         return out


# class InvBottleneck(nn.ModuleList):
#     def __init__(self, prev_filters, t, c, n, s, initial_dilation=1, dilation=1):
#         super().__init__()
#         for sub_index in range(n):
#             _c0 = prev_filters if sub_index == 0 else c
#             _c1 = t * _c0
#             _s = s if sub_index == 0 else 1
#             _d = initial_dilation if sub_index == 0 else dilation
#             self.append(nn.Sequential(
#                 nn.Conv2d(_c0, _c1, 1),
#                 nn.BatchNorm2d(_c1),
#                 nn.ReLU6(),
#                 nn.ReplicationPad2d(_d),
#                 nn.Conv2d(_c1, _c1, 3, stride=_s, dilation=_d, groups=_c1),
#                 nn.BatchNorm2d(_c1),
#                 nn.ReLU6(),
#                 nn.Conv2d(_c1, c, 1),
#                 nn.BatchNorm2d(c)
#             ).to(device))
    
#     def forward(self, x):
#         for sub_index, layer in enumerate(self):
#             x = layer(x) if sub_index == 0 else layer(x) + x
#         return x

# encoder = nn.Sequential(
#     deep_lab_v3p.part1,
#     deep_lab_v3p.part2,
#     deep_lab_v3p.aspp
# ).to(device)
    
# discriminator_num_output_classes = 1
# discriminator = []
# if discriminator_type == 3:
#     discriminator.extend((
#         InvBottleneck(256, 6, 64, 3, 1),
#         InvBottleneck(64, 6, 16, 3, 1),
#         InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
#     ))
# elif discriminator_type == 2:
#     discriminator.extend((
#         InvBottleneck(256, 6, 64, 1, 1),
#         InvBottleneck(64, 6, 16, 1, 1),
#         InvBottleneck(16, 6, discriminator_num_output_classes, 1, 1),
#     ))
# elif discriminator_type == 1:
#     discriminator.extend((
#         InvBottleneck(256, 6, discriminator_num_output_classes, 1, 1),
#     ))
# else:
#     assert discriminator_type == 0
#     discriminator.extend((
#         ck(256, 256, False),
#         ck(256, 256, False),
#         nn.Conv2d(256, discriminator_num_output_classes, 1),
#     ))
# del discriminator_num_output_classes
# discriminator = nn.Sequential(*discriminator).to(device)

import torch
import glob
# print(glob.glob(os.path.join("./tmp/nada/models/cyclegan/saved_models", "checkpoint*")))
# checkpoint = torch.load(glob.glob(os.path.join("./experimentos/PA-RO_A1/models/cyclegan/saved_models", "checkpoint*")))
# print(checkpoint['epoch'])

# checkpoint_data = {
#     # general train
#     'epoch': epoch,
#     'metrics': metrics,
#     # models
#     'G_A2B_model': self.G_A2B.state_dict(),
#     'G_B2A_model': self.G_B2A.state_dict(),
#     'D_A_model': self.D_A.state_dict(),
#     'D_B_model': self.D_B.state_dict(),
    
#     # optimizers
#     'D_A_optimizer': self.opt_D_A.state_dict(),
#     'D_B_optimizer': self.opt_D_B.state_dict(),
#     'G_A2B_optimizer': self.opt_G_A2B.state_dict(),
#     'G_B2A_optimizer': self.opt_G_B2A.state_dict()
#     }
# if use_se:
#     checkpoint_data['Semantic_model'] = deep_lab_v3p.state_dict()
#     checkpoint_data['Semantic_optimizer'] = self.optim_seg.state_dict()

# print("hello world")
# import skimage
# import numpy as np

# # ones = np.ones((100,100))
# ones = np.random.choice([0, 1], size=(10,10), p=[5./10, 5./10])
# print(ones)
# print(ones.shape)
# print(np.unique(ones))
# print(int(np.sum(ones)))
# positive_map_init_ = skimage.morphology.area_opening(ones.astype('int'), area_threshold = 5, connectivity=1)
# print()
# print(positive_map_init_)
# print(positive_map_init_.shape)
# print(np.unique(positive_map_init_))
# print(np.sum(positive_map_init_))
# diff = ones - positive_map_init_
# print(diff)
# print(diff.shape)
# print(np.unique(diff))
# print(np.sum(diff))
# what_i_want = ones - diff
# print(what_i_want)
# print(what_i_want.shape)
# print(np.unique(what_i_want))
# print(np.sum(what_i_want))
# [[0 0 0 1 1 1 1 0 1 0]
#  [0 1 1 1 0 0 1 0 0 1]
#  [0 1 1 0 1 1 1 0 1 1]
#  [0 1 1 0 1 1 0 0 1 1]
#  [1 0 0 1 1 1 0 0 0 0]
#  [1 0 0 1 1 0 1 1 1 1]
#  [0 1 0 0 0 1 0 1 0 0]
#  [0 1 1 0 0 1 1 1 0 0]
#  [1 0 1 1 0 1 0 0 1 1]
#  [0 0 0 0 1 0 1 0 1 1]]
# (10, 10)
# [0 1]
# 52
# [[0 0 0 1 1 1 1 0 0 0]
#  [0 1 1 1 0 0 1 0 0 1]
#  [0 1 1 0 1 1 1 0 1 1]
#  [0 1 1 0 1 1 0 0 1 1]
#  [0 0 0 1 1 1 0 0 0 0]
#  [0 0 0 1 1 0 1 1 1 1]
#  [0 1 0 0 0 1 0 1 0 0]
#  [0 1 1 0 0 1 1 1 0 0]
#  [0 0 1 1 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]]
# (10, 10)
# [0 1]
# 42
# [[0 0 0 0 0 0 0 0 1 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]
#  [1 0 0 0 0 0 0 0 1 1]
#  [0 0 0 0 1 0 1 0 1 1]]
# (10, 10)
# [0 1]
# 10
# [[0 0 0 1 1 1 1 0 0 0]
#  [0 1 1 1 0 0 1 0 0 1]
#  [0 1 1 0 1 1 1 0 1 1]
#  [0 1 1 0 1 1 0 0 1 1]
#  [0 0 0 1 1 1 0 0 0 0]
#  [0 0 0 1 1 0 1 1 1 1]
#  [0 1 0 0 0 1 0 1 0 0]
#  [0 1 1 0 0 1 1 1 0 0]
#  [0 0 1 1 0 1 0 0 0 0]
#  [0 0 0 0 0 0 0 0 0 0]]
# (10, 10)
# [0 1]
# 42



# from Amazon_PA import *
# from Amazon_RO import *
# from Tools import *
# import exp_param.parameters_PA_RO_A1

# source_args = exp_param.parameters_PA_RO_A1.PA
# source = AM_PA(source_args)
# ready_domain(source, source_args, train_set = True, augmented = True)



# import numpy as np

# file_path = './main_data/PA/img_present_PA.npy'
# img = np.load(file_path)
# # img = img[:,0:1700,:]
# # img = img[:,:,0:1440]
# print(img.shape)



from Amazon_PA import *
from Amazon_RO import *
from Tools import *
import parameters
import torch

source_args = parameters.PA
source = AM_PA(source_args)
target_args = parameters.RO
target = AM_RO(target_args)
print(type(source.diff_reference))
print(source.diff_reference.shape)

real_diff_A = utils.channels_last2first(source.diff_reference)
real_diff_B = utils.channels_last2first(target.diff_reference)
# print(type(real_diff_A))

real_diff_A = torch.from_numpy(real_diff_A).float()
real_diff_B = torch.from_numpy(real_diff_B).float()
print(real_diff_A.size())


real_diff_A_ = (torch.norm(real_diff_A, dim = 1)).mean()   #Norm2(A_2 - A_1)
# print(real_diff_A_)
real_diff_B_ = (torch.norm(real_diff_B, dim = 1)).mean()   #Norm2(B_2 - B_1)
print(real_diff_A_)
print(real_diff_B_)
print()
shape_A = real_diff_A.size()
shape_B = real_diff_B.size()
diff_A_ = real_diff_B_     #Norm2(G_A(A_2) - G_A(A_1))
diff_B_ = real_diff_A_     #Norm2(G_B(B_2) - G_B(B_1))

# Normalizing the differences of real and generated images
real_diff_A_norm = real_diff_A/real_diff_A_  # (A_2 - A_1)/Norm2(A_2 - A_1)
real_diff_B_norm = real_diff_B/real_diff_B_  # (B_2 - B_1)/Norm2(B_2 - B_1)
print(real_diff_A_norm.shape)
print(real_diff_B_norm.shape)
diff_A_norm = real_diff_A/diff_A_     # (G_A(A_2) - G_A(A_1))/Norm2(G_A(A_2) - G_A(A_1))
diff_B_norm = real_diff_B/diff_B_  

loss_diff_A = (torch.norm(diff_A_norm - real_diff_A_norm, dim=1)).mean()
loss_diff_B = (torch.norm(diff_B_norm - real_diff_B_norm, dim=1)).mean()
print(loss_diff_A)
print(loss_diff_B)
print()

seize = 3
zeros = np.zeros((seize, shape_A[0], shape_A[1], shape_A[2]))
for i in range(seize):
	zeros[i] = real_diff_A
zeros = torch.from_numpy(zeros).float()
print(zeros.shape)
ones = np.repeat(real_diff_A_, seize)
print(ones.shape)
print(zeros[i].shape)
print(ones[i])
yolo = np.zeros(seize)
yolo = torch.from_numpy(yolo).float()
for i in range(seize):
	yolo[i] = zeros[i]/ones[i]
print(yolo)

# from Amazon_PA import *
# from Amazon_RO import *
# from Cerrado_MA import *
# from exp_param import parameters_PA_RO_A1 as parameters1
# from exp_param import parameters_sMA_tPA_A1 as parameters2
# import tifffile
# import os

# # os.mkdir('./original_conc')
# pa = AM_PA(parameters1.PA)
# # pa_args = parameters1.PA
# tifffile.imsave("./original_conc/pa_.tiff", pa.conc_image, photometric='rgb')

# # if domain == "RO":
# ro = AM_RO(parameters1.RO)
# # ro_args = parameters1.RO
# tifffile.imsave("./original_conc/ro_.tiff", ro.conc_image, photometric='rgb')

# # if domain == "MA":
# ma = CE_MA(parameters2.MA)
# # ma_args = parameters2.MA
# tifffile.imsave("./original_conc/ma_.tiff", ma.conc_image, photometric='rgb')


# import GAN_parameters
# # from image_pool import ImagePool
# from pedro.models import cycle_gan_model
# # from Amazon_PA import *
# from Cerrado_MA import *
# from Amazon_RO import *
# # from exp_param import parameters_PA_RO_A1 as parameters1
# # from exp_param import parameters_sMA_tPA_A1 as parameters2
# from newGAN_param import NewGANnewTrain_sRO_tMA_CN as parameters
# # /newGAN_param/NewGANnewTrain_sRO_tMA_CN.py
# import numpy as np

# # import tensorflow as tf
# import pedro_train_cyclegan

# ro = AM_RO(parameters.RO)
# ro_args = parameters.RO
# ready_domain(ro, ro_args, train_set = False, cyclegan_set = True)

# ma = CE_MA(parameters.MA)
# ma_args = parameters.MA
# ready_domain(ma, ma_args, train_set = False, cyclegan_set = True)

# opt = GAN_parameters.cyclegan_model_options
# pedro_train_cyclegan.train(ro, ma, parameters.Train_cyclegan,
# 	parameters.Global, opt)


# def RemoteSensing_Transforms(opt, data):
#         # The transformations here were accomplished using tensorflow-cpu framework
#         # print("shape b4", data.shape)
#         input_nc = np.size(data, 1)
#         # print(input_nc)
#         out_data = []
#         for i in range(len(data)):
#         	# print("shape b42", data[i].shape)
#         	this_data = np.transpose(data[i], (1, 2, 0))
#         	# print("shape A4", this_data.shape)
#         	if 'resize' in opt.preprocess:
#         		this_data = tf.image.resize(this_data, [opt.load_size, opt.load_size], 
#         			method=tf.image.ResizeMethod.BICUBIC) 
        	
#         	if 'crop' in opt.preprocess:
#         		this_data = tf.image.random_crop(this_data, size=[opt.crop_size, opt.crop_size, input_nc])
        	
#         	if not opt.no_flip:
#         		this_data = tf.image.random_flip_left_right(this_data)
        	
#         	# if opt.convert:
#         	#     pass
        	
#         	this_data = np.transpose(this_data, (2, 0, 1))
#         	# print("shape A42", this_data.shape)
#         	out_data.append(this_data)
#         	# print(data[i][0][0])
#         	# print(this_data[0][0])
#         out_data = np.array(out_data)
#         return out_data




# A , y, Aref = patch_extraction(utils.channels_last2first(pa.conc_image),
#         pa.new_reference, pa.central_pixels_coor_tr[0:10], 64, 
#         diff_reference_extract = True,
#         diff_reference = utils.channels_last2first(pa.diff_reference))
# B , y, Bref = patch_extraction(utils.channels_last2first(ma.conc_image),
#         ma.new_reference, ma.central_pixels_coor_tr[0:10], 64, 
#         diff_reference_extract = True,
#         diff_reference = utils.channels_last2first(ma.diff_reference))
# del y

# opt = GAN_parameters.cyclegan_model_options
# A = RemoteSensing_Transforms(opt, A)
# B = RemoteSensing_Transforms(opt, B)
# # print(A.shape)
# # break
# data = {}
# data['A'] = torch.from_numpy(A).float()
# data['A_ref'] = torch.from_numpy(Aref).float()
# data['B'] = torch.from_numpy(B).float()
# data['B_ref'] = torch.from_numpy(Bref).float()




# model = cycle_gan_model.CycleGANModel(opt)
# model.setup(opt)

# model.set_input(data)
# model.optimize_parameters()




# class Building(object):
#      def __init__(self, floors):
#          self._floors = [None]*floors
#      def __setitem__(self, floor_number, data):
#           self._floors[floor_number] = data
#      def __getitem__(self, floor_number):
#           return self._floors[floor_number]

# building1 = Building(4) # Construct a building with 4 floors
# building1[0] = 'Reception'
# building1[1] = 'ABC Corp'
# building1[2] = 'DEF Inc'
# print( building1[2] )





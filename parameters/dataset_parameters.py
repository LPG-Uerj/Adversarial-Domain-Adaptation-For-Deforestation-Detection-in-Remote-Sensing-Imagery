import os
from .training_parameters import Evaluate_cyclegan

'''
	This is an example of parameters to be used for the dataset handling
	of each domain/image
'''
class PA():
	dataset_main_path = "./main_data/"
	dataset = "PA"
	images_section = ""
	reference_section = ""
	data_t1_name = "/img_" + "past" + '_' + dataset
	data_t2_name = "/img_" + "present" + '_' + dataset 
	reference_t1_name = "/new_gt_" + "past" + '_' + dataset
	reference_t2_name = "/gt_" + "present" + '_' + dataset
	buffer = True
	buffer_dimension_out = 0
	buffer_dimension_in = 2
	compute_ndvi = False

	phase = "train"
	fixed_tiles = True
	defined_before = False
	checkpoint_dir_posterior = "./tmp/bootstrap/" + dataset
	save_checkpoint_path = "./tmp/bootstrap/" + dataset + '/'
	
	horizontal_blocks = 3
	vertical_blocks = 5
	patches_dimension = 64
	# stride = 16
	porcent_of_last_reference_in_actual_reference = 100

	p = save_checkpoint_path

	adapted_save_path = f'./tmp/adapted_data/{dataset}/'
	eval_save_path = Evaluate_cyclegan.savedata_folder + dataset + '/'
	adapted_file_name = f'adapted_conc_{dataset}'

	# stride per train
	stride = 19 # global if you want
	train_source_stride = stride
	train_cyclegan_stride = stride
	evaluate_cyclegan_stride = stride
	train_target_stride = stride

	# reconstruction parameters
	run_rec_on_cpu = False

	full_image = True
	save_tiff = False
	save_mat = False
	de_normalize = False

	rec_size = 1024
	overlap_porcent = 0
	rec_batch = 1

	# best models - if already trained; else use = ''
	best_source_model = ''
	generator_type = 'A2B'  # for evaluation, if cyclegan alreay trained
	best_generator = ''



class RO():
	dataset_main_path = "./main_data/"
	dataset = "RO"
	images_section = ""
	reference_section = ""
	data_t1_name = "/img_" + "past" + '_' + dataset
	data_t2_name = "/img_" + "present" + '_' + dataset 
	reference_t1_name = "/new_gt_" + "past" + '_' + dataset
	reference_t2_name = "/gt_" + "present" + '_' + dataset
	buffer = True
	buffer_dimension_out = 4
	buffer_dimension_in = 2
	compute_ndvi = False

	phase = "train"
	fixed_tiles = True
	defined_before = False
	checkpoint_dir_posterior = "./tmp/bootstrap/" + dataset
	save_checkpoint_path = "./tmp/bootstrap/" + dataset + '/'
	
	horizontal_blocks = 10
	vertical_blocks = 10
	patches_dimension = 64
	# stride = 16
	porcent_of_last_reference_in_actual_reference = 100

	p = save_checkpoint_path

	adapted_save_path = f'./tmp/adapted_data/{dataset}/'
	eval_save_path = Evaluate_cyclegan.savedata_folder + dataset + '/'
	adapted_file_name = f'adapted_conc_{dataset}'

	# stride per train
	stride = 23 # global if you want
	train_source_stride = stride
	train_cyclegan_stride = stride
	evaluate_cyclegan_stride = stride
	train_target_stride = stride

	# reconstruction parameters
	run_rec_on_cpu = True

	full_image = True
	save_tiff = False
	save_mat = False
	de_normalize = False

	rec_size = 1024
	overlap_porcent = 0
	rec_batch = 1

	# best models - if already trained; else use = ''
	best_source_model = ''
	generator_type = 'B2A'  # for evaluation, if cyclegan alreay trained
	best_generator = ''


class MA():
	dataset_main_path = "./main_data/"
	dataset = "MA"
	images_section = ""
	reference_section = ""
	data_t1_name = "/img_" + "past" + '_' + dataset
	data_t2_name = "/img_" + "present" + '_' + dataset 
	reference_t1_name = "/gt_" + "past" + '_' + dataset
	reference_t2_name = "/gt_" + "present" + '_' + dataset
	buffer = True
	buffer_dimension_out = 2
	buffer_dimension_in = 0
	compute_ndvi = False

	phase = "train"
	fixed_tiles = True
	defined_before = False
	checkpoint_dir_posterior = "./tmp/bootstrap/" + dataset
	save_checkpoint_path = "./tmp/bootstrap/" + dataset + '/'
	
	horizontal_blocks = 5
	vertical_blocks = 3
	patches_dimension = 64
	# stride = 16
	porcent_of_last_reference_in_actual_reference = 100

	p = save_checkpoint_path

	adapted_save_path = f'./tmp/adapted_data/{dataset}/'
	eval_save_path = Evaluate_cyclegan.savedata_folder + dataset + '/'
	adapted_file_name = f'adapted_conc_{dataset}'

	# stride per train
	stride = 19 # global if you want
	train_source_stride = stride
	train_cyclegan_stride = stride
	evaluate_cyclegan_stride = stride
	train_target_stride = stride

	# reconstruction parameters
	run_rec_on_cpu = False

	full_image = True
	save_tiff = False
	save_mat = False
	de_normalize = False

	rec_size = 1024 # if not full_image
	overlap_porcent = 0
	rec_batch = 1

	# best models - if already trained; else use = ''
	best_source_model = ''
	generator_type = 'B2A'  # for evaluation, if cyclegan alreay trained
	best_generator = ''
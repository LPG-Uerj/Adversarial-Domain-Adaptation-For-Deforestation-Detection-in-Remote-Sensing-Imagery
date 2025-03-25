import os

class Global():

	source_domain = "PA"
	target_domain = "RO"
	save_folder = 'CG_CD'

	patch_size = 64
	num_classes = 3
	channels = 14

	weights = [0.3, 0.7, 0] # classes weights
	gamma = 1
	learning_rate = 0.001

	dilation_rates = (3,6)
	large_latent_space = True

	# which trains skip - False to train; True to skip 
	skip_train_source = True
	train_source_starting_round = 0
	train_source_json = ''
	train_gt_target_json = ''
	train_source_both_ways = True

	skip_train_cyclegan = False
	continue_train_cyclegan = False
	train_cyclegan_json = ''
	inverse_cyclegan = False

	skip_train_adapted_target = True
	train_adapted_target_starting_round = 0
	train_adapted_target_json = ''

	skip_test = True
	######

	tmp_path = './tmp'
	base_path = os.path.join(tmp_path, save_folder)

	models_path = 'models'
	default_segmentation_path = os.path.join(base_path, models_path, 'source_seg')
	# default_cyclegan_path = base_path + models_path + '/cyclegan/'
	default_cyclegan_path = os.path.join(base_path, models_path, 'cyclegan')
	cyclegan_models_path = os.path.join(default_cyclegan_path, 'saved_models')
	# default_evaldata_path = base_path + '/eval/'
	default_evaldata_path = os.path.join(base_path, 'eval')
	# default_adapted_target_path = base_path + models_path + '/target_adapted_seg/'
	default_adapted_target_path = os.path.join(base_path, models_path, 'target_adapted_seg')

	test_result_save_folder = os.path.join(base_path, 'test_results')
	
	base_A2B_name = 'G_A2B_model_weights_epoch_'
	base_B2A_name = 'G_B2A_model_weights_epoch_'


class Train_source():
	number_of_rounds = 10
	output_path = Global.default_segmentation_path
	
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	
	# stride = 28

	batch_size = 16
	iteration_modifier = 1
	num_epochs = 100

	weights = Global.weights
	gamma = Global.gamma
	learning_rate = Global.learning_rate

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	early_stop = True
	early_stop_limit = 10



class Train_cyclegan():
	output_path = Global.default_cyclegan_path
	save_adaptation_previews = False
	continue_train = Global.continue_train_cyclegan

	patch_size = 64
	num_classes = Global.num_classes
	channels = Global.channels

	batch_size = 10 # set 5 when use Semantic loss
	total_patches_per_epoch = 6000 # 6000 por padrão
	num_epochs = 300
	save_model_each_n_epoch = 10

	# losses weights (lambad)
	A2B_cyclic_loss_lambda = 10.0
	B2A_cyclic_loss_lambda = 10.0
	descriminator_loss_lambda = 1.0
	indentity_loss_lambda = 5.0
	semantic_loss_lambda = 0.0 # only used if us use_semantic_loss = True
	diff_loss_lambda = 10.0 # only used if use_diff_loss = True

	use_diff_loss = True

	# Learning rates
	descriminator_learning_rate = 2e-4
	generator_learning_rate = 2e-4
	semantic_learning_rate = Train_source.learning_rate # only used if us use_semantic_loss = True

	use_semantic_loss = False # True for use Semantic loss
	# values for semantic train:
	# weights = [1.0,1.0,1.0]
	# weights = Train_source.weights
	weights = [0.3, 0.7, 0.3]
	gamma = Train_source.gamma
	# learning_rate = Train_source.learning_rate
	dilation_rates = Train_source.dilation_rates
	large_latent_space = Train_source.large_latent_space


class Evaluate_cyclegan():
	savedata_folder = Global.default_evaldata_path
	# model_path = '' # pegar da cyclegan
	# output_path = Global.default_segmentation_path + save_folder
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	# stride = 16

	batch_size = 25
	iteration_modifier = 1
	epochs_per_model = 20

	weights = Global.weights
	gamma = Global.gamma
	learning_rate = Global.learning_rate

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	starting_model_index = 0	# (this + 1) * 10  
								# example: 5 -> (5 + 1) * 10 = model 60
								
	early_skip = True		# skip to next model evaluation
	early_skip_limit = 10


class Train_adapted_target():
	number_of_rounds = 10
	output_path = Global.default_adapted_target_path

	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels

	batch_size = 16
	iteration_modifier = 1
	num_epochs = 100

	# losses weights (lambdas)
	semantic_loss_lambda = 1.0
	cyclic_loss_lambda = 0.35
	feature_discriminator_loss_lambda = 0.01

	# Learning rates
	learning_rate = Global.learning_rate
	# descriminator_learning_rate =
	# encoder_learning_rate = 

	weights = Global.weights
	gamma = Global.gamma

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	early_stop = True
	early_stop_limit = 10

	task_loss = True
	pseudo_label = False
	feature_adaptation = False # use or not feature adpatation
	# feature adaptation specific
	label_type = 1
	train_encoder_only_on_target_domain = False
	discriminator_type = 0

	assert (task_loss) or (pseudo_label)

class Test_models():
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	output_path = Global.test_result_save_folder

	test_only_on_proper_domain = True
	test_FS_models = True
	test_FS_on_both_domains = True # only use if test_FS_models = True

	test_transforms = False

	prodes_polygon_removal = True  # use prodes 69 polygon removal

	# for differents thresholds
	change_threshold = True
	number_of_points = 100
		
	batch_size = 4610

	weights = Global.weights
	gamma = Global.gamma

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space


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
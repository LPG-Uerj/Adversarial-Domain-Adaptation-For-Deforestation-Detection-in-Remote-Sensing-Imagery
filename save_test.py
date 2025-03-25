class Global():

	source_domain = "PA"
	target_domain = "RO"
	save_folder = 'nada'
	# save_folder = 'new_test_cycle_130epoc_5_batch5_weight03_identity_notAlltgt'

	patch_size = 64
	num_classes = 3
	channels = 14

	weights = [0.3, 0.7, 0]
	gamma = 1
	learning_rate = 0.001

	dilation_rates = (3,6)
	large_latent_space = True


	tmp_path = './tmp'
	base_path = tmp_path + '/' + save_folder

	models_path = '/models'
	default_segmentation_path = base_path + models_path + '/source_seg/'
	default_cyclegan_path = base_path + models_path + '/cyclegan/'
	default_evaldata_path = base_path + '/eval/'
	default_adapted_target_path = base_path + models_path + '/target_adapted_seg/'

	
	base_A2B_name = 'G_A2B_model_weights_epoch_'
	base_B2A_name = 'G_B2A_model_weights_epoch_'


class Train_source():
	# save_folder = Global.save_folder
	# output_path = Global.default_segmentation_path + save_folder
	output_path = Global.default_segmentation_path
	
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	
	stride = 16

	batch_size = 16
	iteration_modifier = 5
	num_epochs = 5

	weights = Global.weights
	gamma = Global.gamma
	learning_rate = Global.learning_rate

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	early_stop = True
	early_stop_limit = 10



class Train_cyclegan():
	# save_folder = Global.save_folder
	# output_path = Global.default_cyclegan_path + save_folder
	output_path = Global.default_cyclegan_path
	save_adaptation_previews = False
	continue_train = False

	# patch_size = Global.patch_size
	patch_size = 256
	num_classes = Global.num_classes
	channels = Global.channels
	source_stride = 24
	target_stride = 24

	batch_size = 5 # set 5 when use Semantic loss
	# num_batches = 250 # iterations per epoch
	total_patches_per_epoch = 6000
	num_epochs = 132
	save_model_each_n_epoch = 5

	use_semantic_loss = True # True for use Semantic loss
	# values for semantic train:
	# weights = [1.0,1.0,1.0]
	# weights = Train_source.weights
	weights = [0.3, 0.7, 0.3]
	gamma = Train_source.gamma
	learning_rate = Train_source.learning_rate
	dilation_rates = Train_source.dilation_rates
	large_latent_space = Train_source.large_latent_space


class Evaluate_cyclegan():
	savedata_folder = Global.default_evaldata_path
	# model_path = '' # pegar da cyclegan
	# output_path = Global.default_segmentation_path + save_folder
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	stride = 16

	batch_size = 24
	iteration_modifier = 5
	epochs_per_model = 5

	weights = Global.weights
	gamma = Global.gamma
	learning_rate = Global.learning_rate

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	# early_stop = True
	# early_stop_limit = 10


class Train_adapted_target():
	# save_folder = Global.save_folder
	# output_path = Global.default_adapted_target_path + save_folder
	output_path = Global.default_adapted_target_path

	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	stride = 16

	batch_size = 16
	iteration_modifier = 5
	num_epochs = 5

	weights = Global.weights
	gamma = Global.gamma
	learning_rate = Global.learning_rate

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	early_stop = True
	early_stop_limit = 10

	feature_adaptation = False # use or not feature adpatation
	# feature adaptation specific
	label_type = 1
	train_encoder_only_on_target_domain = True
	discriminator_type = 0


class Test_models():
	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels

	batch_size = 2000

	weights = Global.weights
	gamma = Global.gamma

	dilation_rates = Global.dilation_rates
	large_latent_space = Global.large_latent_space

	run_prodes = False  # use prodes 69 polygon removal


class PA():
	dataset_main_path = "./main_data/"
	dataset = "PA"
	images_section = ""
	reference_section = ""
	data_t1_name = "/img_" + "past" + '_' + dataset
	data_t2_name = "/img_" + "present" + '_' + dataset 
	reference_t1_name = "/gt_" + "past" + '_' + dataset
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
	stride = 16
	porcent_of_last_reference_in_actual_reference = 100

	p = save_checkpoint_path

	adapted_save_path = f'./tmp/adapted_data/{dataset}/'
	eval_save_path = Evaluate_cyclegan.savedata_folder + dataset + '/'
	adapted_file_name = f'adapted_conc_{dataset}'


class RO():
	dataset_main_path = "./main_data/"
	dataset = "RO"
	images_section = ""
	reference_section = ""
	data_t1_name = "/img_" + "past" + '_' + dataset
	data_t2_name = "/img_" + "present" + '_' + dataset 
	reference_t1_name = "/gt_" + "past" + '_' + dataset
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
	
	horizontal_blocks = 3
	vertical_blocks = 5
	patches_dimension = 64
	stride = 16
	porcent_of_last_reference_in_actual_reference = 100

	p = save_checkpoint_path

	adapted_save_path = f'./tmp/adapted_data/{dataset}/'
	eval_save_path = Evaluate_cyclegan.savedata_folder + dataset + '/'
	adapted_file_name = f'adapted_conc_{dataset}'
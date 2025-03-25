class Global():

	source_domain = "PA"
	target_domain = "RO"
	save_folder = 'nada'
	# save_folder = 'results2'

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
	
	# stride = 28

	batch_size = 16
	iteration_modifier = 1
	num_epochs = 100

	# # losses weights
	# focal_loss_weight = 

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
	# source_stride = 24
	# target_stride = 24

	batch_size = 5 # set 5 when use Semantic loss
	# num_batches = 250 # iterations per epoch
	total_patches_per_epoch = 6000
	num_epochs = 42
	save_model_each_n_epoch = 5

	# losses weights (lambad)
	A2B_cyclic_loss_lambda = 10.0
	B2A_cyclic_loss_lambda = 10.0
	descriminator_loss_lambda = 1.0
	indentity_loss_lambda = 1.0
	semantic_loss_lambda = 0.1 # only used if us use_semantic_loss = True

	# Learning rates
	descriminator_learning_rate = 2e-4
	generator_learning_rate = 2e-4
	semantic_learning_rate = Train_source.learning_rate # only used if us use_semantic_loss = True

	use_semantic_loss = True # True for use Semantic loss
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
	# save_folder = Global.save_folder
	# output_path = Global.default_adapted_target_path + save_folder
	output_path = Global.default_adapted_target_path

	patch_size = Global.patch_size
	num_classes = Global.num_classes
	channels = Global.channels
	# stride = 16
	# source_stride = 16
	# target_stride = 28

	batch_size = 16
	iteration_modifier = 5
	num_epochs = 5

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
	stride = 16 # global if you want
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
	best_source_model = './tmp/new_ref_source_PA/models/source_seg/model_31_65.1563.pt'
	generator_type = 'A2B'  # for evaluation, if cyclegan alreay trained
	best_generator = './tmp/new_test_cycle_130epoc_5_batch5_weight03_identity/models/cyclegan/saved_models/G_A2B_model_weights_epoch_130.pt'



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
	stride = 28 # global if you want
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
	best_source_model = './tmp/new_ref_source_RO/models/source_seg/model_03_20.7114.pt'
	generator_type = 'B2A'  # for evaluation, if cyclegan alreay trained
	best_generator = './tmp/new_test_cycle_130epoc_5_batch5_weight03_identity/models/cyclegan/saved_models/G_B2A_model_weights_epoch_130.pt'
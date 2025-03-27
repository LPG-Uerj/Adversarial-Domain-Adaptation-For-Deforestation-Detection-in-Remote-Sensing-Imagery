import os
'''
    This is an example of parameters to be used during training
'''
class Global():

    # Target and source domain definition
    source_domain = "RO"		# Source domain is the one in which you have labels available
    target_domain = "MA"		# Target domain is the one with no labels available
    save_folder = 'experiments'		# Folder for saving data

    # Definition of model training parameters
    patch_size = 64				# Patch size that will be used for all images
    num_classes = 3				# Number of classes - other than forested an deforested, there is also an ignore label
    channels = 14				# Number of channels on image - originally we use two 7 channels (past and present) images concatenated, resulting in a 14 channel image/array

    weights = [0.3, 0.7, 0] 	# Classes weights, the third one is 0 because it is ignored
    gamma = 1					# Gamma value for focal loss
    learning_rate = 0.001		# Learning rate for training

    dilation_rates = (3,6)		# Dilation is used to make a "ignore label" in the intersection of 'forested' and 'deforested' labels
    large_latent_space = True	# Used for building the model architeture, in doubt keep it True

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
    '''
        Parameters related to the supervised training of the source model.
        These parameters were thought out in the context of multispectral images of large resolution
        which are wuite common in the context of remote sensing (satellite imagery).
        The model is also supposed to be a binary classifier, with a third class of pixels to be ignored.
        The model architeture is DeepLabV3+.
    '''

    number_of_rounds = 1		# Number of times that will run the training, used for experiments
    output_path = Global.default_segmentation_path
    
    patch_size = Global.patch_size
    num_classes = Global.num_classes
    channels = Global.channels
    
    # stride = 28

    batch_size = 16				# Batch size for the training, based on the patch size
    iteration_modifier = 1
    num_epochs = 5				# Number of epochs for the training

    weights = Global.weights
    gamma = Global.gamma
    learning_rate = Global.learning_rate

    dilation_rates = Global.dilation_rates
    large_latent_space = Global.large_latent_space

    # Early Stop Parameters
    early_stop = True			# True if you want to use early stop, False otherwise
    early_stop_limit = 10		# Limit for the early stop, how many epochs with no good results it should stop



class Train_cyclegan():
    '''
        Parameters related to the unsipervised training of the CycleGAN model.
        This model is going to train the domain adaptation from source to target AND target to source.
        Both at the same time.
        In addition to the original CycleGAN model, there is also the possibility to run alongside
        a classifier in other to get more features from the source.
        Both CycleGAN and CycleGAN+Classifier architetures are not original to this code.
    '''
    
    output_path = Global.default_cyclegan_path
    save_adaptation_previews = False
    continue_train = Global.continue_train_cyclegan

    patch_size = Global.patch_size
    num_classes = Global.num_classes
    channels = Global.channels

    batch_size = 10 				# Set 5 when use Semantic loss
    total_patches_per_epoch = 6000 	# Number of patches to iterate on each epoch, keep 6000 by default
    num_epochs = 5					# Number of epochs for the training
    save_model_each_n_epoch = 10	# Create checkpoint each n steps

    # CycleGAN model losses weights (lambad)
    A2B_cyclic_loss_lambda = 10.0	# GAN A2B loss lambda
    B2A_cyclic_loss_lambda = 10.0	# GAN B2A loss lambda
    descriminator_loss_lambda = 1.0	# Descriminator loss lambda
    indentity_loss_lambda = 1.0		# Identity loss lambda
    semantic_loss_lambda = 0.1 		# only used if us use_semantic_loss = True
    diff_loss_lambda = 1.0 			# only used if use_diff_loss = True

    use_diff_loss = False			# True to use L1Loss during training

    # GAN models Learning rates
    descriminator_learning_rate = 2e-4						# Lerning rate for the descriminator
    generator_learning_rate = 2e-4							# Lerning rate for the generator
    semantic_learning_rate = Train_source.learning_rate 	# Only used if us use_semantic_loss = True

    # Parameters for semantic segmentation model
    use_semantic_loss = False 		# True for use Semantic loss
    # values for semantic train:
    # weights = [1.0,1.0,1.0]
    # weights = Train_source.weights
    weights = [0.3, 0.7, 0.3]		# Classes weights
    gamma = Train_source.gamma		# Focal loss gamma
    # learning_rate = Train_source.learning_rate
    dilation_rates = Train_source.dilation_rates	
    large_latent_space = Train_source.large_latent_space


class Evaluate_cyclegan():
    '''
        Parameters related to the evaluation of the generative models resulted from the 
        CycleGAN training step.
    '''

    savedata_folder = Global.default_evaldata_path		# Folder to save results
    # model_path = '' # pegar da cyclegan
    # output_path = Global.default_segmentation_path + save_folder
    patch_size = Global.patch_size						# Patch size
    num_classes = Global.num_classes					# Number of classes
    channels = Global.channels							# Number of channels
    # stride = 16

    batch_size = 25				# Batch size
    iteration_modifier = 1		
    # epochs_per_model = 20		

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
    '''
        Parameters related to the training of the target model.
    '''

    number_of_rounds = 1
    output_path = Global.default_adapted_target_path

    patch_size = Global.patch_size
    num_classes = Global.num_classes
    channels = Global.channels

    batch_size = 16
    iteration_modifier = 1
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

    task_loss = True
    pseudo_label = False
    feature_adaptation = False # use or not feature adpatation
    # feature adaptation specific
    label_type = 1
    train_encoder_only_on_target_domain = False
    discriminator_type = 0

    assert (task_loss) or (pseudo_label)

class Test_models():
    '''
        Parameters related to the testing of the models. Trained in the train adapted step.
    '''
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
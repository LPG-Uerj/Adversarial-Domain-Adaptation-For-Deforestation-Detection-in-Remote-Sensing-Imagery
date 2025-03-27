class cyclegan_model_options():
	# general 
	input_nc = 14
	output_nc = input_nc
	pool_size = 50

	init_type = 'normal'
	init_gain = 0.02
	# norm = 'batch'
	norm = 'instance'
	gpu_ids = [0]

	# generators
	ngf = 64
	netG = 'resnet_9blocks'
	no_dropout = True	
	linear_output = False


	# opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout,
	# opt.linear_output, opt.init_type, opt.init_gain, self.gpu_ids

	# discriminators
	ndf = 64
	netD = 'basic'
	n_layers_D = 3

	# (opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
	# opt.init_type, opt.init_gain, self.gpu_ids)


	# name = ''

	gan_mode = 'lsgan'
	dataset_type = "remote_sensing_images"

	isTrain = True
	direction = "AtoB"
	phase = 'train'	
	
	# preprocess = ['resize', 'crop']
	preprocess = []
	load_size = 64
	crop_size = 64
	no_flip = False
	convert = False

	lr = 2e-4
	beta1 = 0.5

	lambda_A = 10.0
	lambda_B = 10.0

	# \/this * lambda_A
	lambda_identity_t = 0.5 
	lambda_identity_s = 0
	lambda_diff_s = 0
	lambda_diff_t = 0

	lr_policy = 'linear'
	epoch_count = 0
	# niter = 600
	niter_decay = 1
	# continue_train = False
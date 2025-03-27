import os
import sys
import numpy as np
import skimage.morphology
from skimage.morphology import  disk 
from sklearn.preprocessing import StandardScaler
import scipy.io as sio

import sys
sys.path.append('..')
from utilities import training_utils, reconstruction_tool


class AM_RO():
    def __init__(self, args):
        
        self.images_norm = []
        self.references = []
        self.mask = []
        self.coordinates = []
         
        Image_t1_path = args.dataset_main_path + args.dataset + args.images_section + args.data_t1_name + '.npy'
        Image_t2_path = args.dataset_main_path + args.dataset + args.images_section + args.data_t2_name + '.npy'
        Reference_t1_path = args.dataset_main_path + args.dataset + args.reference_section + args.reference_t1_name + '.npy'
        Reference_t2_path = args.dataset_main_path + args.dataset + args.reference_section + args.reference_t2_name + '.npy'
        
        # Reading images and references
        print('[*]Reading images...')
        image_t1 = np.load(Image_t1_path)
        image_t2 = np.load(Image_t2_path)
        reference_t1 = np.load(Reference_t1_path)
        image_t1 = image_t1[:,1:2551,1:5121]
        image_t2 = image_t2[:,1:2551,1:5121]
        reference_t1 = reference_t1[1:2551,1:5121]

        if os.path.exists(Reference_t2_path):
            reference_t2 = np.load(Reference_t2_path)
            reference_t2 = reference_t2[1:2551,1:5121]

        # mog: não usar 'None'
        # mog: não usar ndvi por enquanto
        # elif args.reference_t2_name == 'None':
        #     reference_t2 = np.ones((2550, 5120))
        elif args.reference_t2_name == 'NDVI':
            ndvi_t1 = training_utils.Compute_NDVI_Band(image_t1) 
            ndvi_t2 = training_utils.Compute_NDVI_Band(image_t2)
            reference_t2_1 = np.zeros_like(ndvi_t1)
            reference_t2_2 = np.zeros_like(ndvi_t2)
            reference_t2_1[ndvi_t1 > 0.27] = 0
            reference_t2_1[ndvi_t1 <= 0.27] = 1
            reference_t2_2[ndvi_t2 > 0.27] = 0
            reference_t2_2[ndvi_t2 <= 0.27] = 1
            reference_t2 = reference_t2_2 - reference_t2_1
            reference_t2[reference_t2 == -1] = 0
            reference_t2 = reference_t2[:,:,0]
            
        # Pre-processing references
        if args.buffer:
            print('[*]Computing buffer regions...')
            #Dilating the reference_t1
            reference_t1 = skimage.morphology.dilation(reference_t1, disk(args.buffer_dimension_out))
            # if os.path.exists(Reference_t2_path) or args.reference_t2_name == 'NDVI':
            if os.path.exists(Reference_t2_path):
                #Dilating the reference_t2
                reference_t2_dilated = skimage.morphology.dilation(reference_t2, disk(args.buffer_dimension_out))
                buffer_t2_from_dilation = reference_t2_dilated - reference_t2
                reference_t2_eroded  = skimage.morphology.erosion(reference_t2 , disk(args.buffer_dimension_in))
                buffer_t2_from_erosion  = reference_t2 - reference_t2_eroded
                buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
                reference_t2 = reference_t2 - buffer_t2_from_erosion
                buffer_t2[buffer_t2 == 1] = 2
                reference_t2 = reference_t2 + buffer_t2
                
        # Pre-processing images
        # mog: não usar ndvi por enquanto
        if args.compute_ndvi:
            print('[*]Computing and stacking the ndvi band...')
            ndvi_t1 = training_utils.Compute_NDVI_Band(image_t1)
            ndvi_t2 = training_utils.Compute_NDVI_Band(image_t2)
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
            image_t1 = np.concatenate((image_t1, ndvi_t1), axis=2)
            image_t2 = np.concatenate((image_t2, ndvi_t2), axis=2)
        else:
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
        
        
        # Pre-Processing the images
       
        print('[*]Normalizing the images...')
        scaler = StandardScaler()
        #scaler = MinMaxScaler()
        images = np.concatenate((image_t1, image_t2), axis=2)
        images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))
        
        scaler = scaler.fit(images_reshaped)
        self.scaler = scaler
        images_normalized = scaler.fit_transform(images_reshaped)
        images_norm = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
        image_t1_norm = images_norm[:, :, : image_t1.shape[2]]
        image_t2_norm = images_norm[:, :, image_t2.shape[2]: ]
        
        print(np.min(image_t1_norm))
        print(np.max(image_t1_norm))
        print(np.min(image_t2_norm))
        print(np.max(image_t2_norm))
        
        # Storing the images in a list
        self.images_norm.append(image_t1_norm)
        self.images_norm.append(image_t2_norm)
        # Storing the references in a list
        self.references.append(reference_t1)
        self.references.append(reference_t2)

        
        # concatenate images and ajust ground truths
        shapes = self.images_norm[0].shape
        self.conc_image = np.zeros((shapes[0], shapes[1], shapes[2]*2))
        self.conc_image[:,:,:shapes[2]] = self.images_norm[0]
        self.conc_image[:,:,shapes[2]:] = self.images_norm[1]
        print("concatenated image shape:", self.conc_image.shape)
        self.new_reference = (self.references[0]*2) + self.references[1]
        self.new_reference[:][(self.new_reference == 3)] = 2
        self.new_reference[:][(self.new_reference == 4)] = 2
        print("new reference class values:", np.unique(self.new_reference))


        # create References (ground truths) for Diff_loss
        # diff_referen = image_t2_norm - image_t1_norm
        self.diff_reference = self.images_norm[1] - self.images_norm[0]


    def Tiles_Configuration(self, args):
        #Generating random training and validation tiles
        if args.phase == 'train' or args.phase == 'compute_metrics':
            if args.fixed_tiles:
                if args.defined_before:
                    if args.phase == 'train':
                        files = os.listdir(args.checkpoint_dir_posterior)
                        print(files[i])
                        self.Train_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Train_tiles.npy')
                        self.Valid_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Valid_tiles.npy')
                        np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                        np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                    if args.phase == 'compute_metrics':
                        self.Train_tiles = np.load(args.save_checkpoint_path +  'Train_tiles.npy')
                        self.Valid_tiles = np.load(args.save_checkpoint_path +  'Valid_tiles.npy')
                else:
                    # mog: vou usar o mesmo formato usado no else
                    self.Train_tiles = np.array([2, 6, 13, 24, 28, 35, 37, 46, 47, 53, 58, 60, 64, 71, 75, 82, 86, 88, 93])
                    self.Valid_tiles = np.array([8, 11, 26, 49, 78])
                    self.Undesired_tiles = []
                    # tiles = np.array([17,  8,  7, 20,  2, 14])
                    # tiles = np.array([1, 7, 9, 13, 5, 12])
                    # self.Train_tiles = tiles[:4]
                    # self.Valid_tiles = tiles[4:]
                    # self.Undesired_tiles = []
                    print("Train" + str(self.Train_tiles))
                    print("Valid" + str(self.Valid_tiles))

            else:
                # mog: (horizontal*vertical)blocks | size = 1/4 disto | valid = 1/5
                # mog: no caso escolhi 5x5 = 25 | size = 6 | valid = 1/3
                # tiles = np.random.randint(100, size = 25) + 1
                # self.Train_tiles = tiles[:20]
                # self.Valid_tiles = tiles[20:]
                n_blocks = args.vertical_blocks * args.horizontal_blocks
                tiles = np.random.default_rng().choice(n_blocks, size=6, replace=False) + 1
                self.Train_tiles = tiles[:4]
                self.Valid_tiles = tiles[4:]
                self.Undesired_tiles = []
                np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
        if args.phase == 'test':
            self.Train_tiles = []
            self.Valid_tiles = []
            self.Undesired_tiles = []

    def Coordinates_Creator(self, args, i):
        print('[*]Defining the central patches coordinates...')
        if args.phase == 'train':
            if args.fixed_tiles:
                if i == 0:
                    print(self.images_norm[0].shape)
                    self.mask = training_utils.mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
                    # self.central_pixels_coor_tr, self.y_train, self.central_pixels_coor_vl, self.y_valid = Central_Pixel_Definition(self.mask, self.references[0], self.references[1], args.patches_dimension, args.stride, args.porcent_of_last_reference_in_actual_reference)
                    self.central_pixels_coor_tr, self.central_pixels_coor_vl = training_utils.Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.horizontal_blocks, args.vertical_blocks)
                    # print("here")
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
            else:
                self.mask = training_utils.mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
                self.central_pixels_coor_tr, self.y_train, self.central_pixels_coor_vl, self.y_valid = training_utils.Central_Pixel_Definition(self.mask, self.references[0], self.references[1], args.patches_dimension, args.stride, args.porcent_of_last_reference_in_actual_reference)
        if args.phase == 'test':
            self.mask = training_utils.mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
            self.central_pixels_coor_ts, self.y_test = training_utils.Central_Pixel_Definition_For_Test(self.mask, np.zeros_like(self.references[0]), np.zeros_like(self.references[0]), args.patches_dimension, 1, args.phase)
    
    def Coordinates_Train_Validation(self, args):
        self.mask = training_utils.mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
        self.central_pixels_coor_tr, self.central_pixels_coor_vl = training_utils.Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.horizontal_blocks, args.vertical_blocks)
        
    def Coordinates_Test(self, args):
        self.mask = training_utils.mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
        self.central_pixels_coor_ts = training_utils.Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.vertical_blocks, args.horizontal_blocks, test = True)

    def Coordinates_Cyclegan(self, args):
        self.cycle_coor = training_utils.No_Mask_Coor(self.images_norm[0].shape, args.patches_dimension, args.stride)

    # def Prepare_Cycle_Set(self, args):
 #        sefl.augmented_cycle_train_set = prepare_set_with_aug(self.central_pixels_coor_tr, 
 #            self.new_reference, args.patches_dimension, only_defo = False, multiplier = False)

    def Prepare_Train_Set(self, args):
        self.augmented_train_set = training_utils.prepare_set_with_aug(self.central_pixels_coor_tr, 
            self.new_reference, args.patches_dimension, only_defo = True, multiplier = True)

    def Coordinates_Domain_Adaptation(self, args):
        self.no_stride_coor = training_utils.No_Mask_Coor(self.images_norm[0].shape, args.patches_dimension, args.patches_dimension)

    def Prepare_GAN_Set(self, args, path_to_weights, eval = False, load = False, 
        adapt_back_to_original = False):

        rec_param = reconstruction_tool.reconstruction_parameters(args, self)
        # print(rec_param.crop_size)

        if eval:
            data_path = args.eval_save_path
            file_name = args.adapted_file_name + '_eval'
        else:
            data_path = args.adapted_save_path
            file_name = args.adapted_file_name

        if eval == True or load == False:
            reconstruction_tool.overlap_reconstruction(self.conc_image, self.scaler, args, rec_param, 
                path_to_weights, data_path, eval = eval)

        load_path = data_path + file_name + '.npy'
        self.adapted_image = np.load(load_path)

        if adapt_back_to_original:
            print("before", path_to_weights)

            if path_to_weights.find("G_A2B") != -1:
                splited_path = path_to_weights.split("G_A2B")
                path_to_weights = splited_path[0] + "G_B2A" + splited_path[1]
            else:
                splited_path = path_to_weights.split("G_B2A")
                path_to_weights = splited_path[0] + "G_A2B" + splited_path[1]

            print("after", path_to_weights)
            reconstruction_tool.overlap_reconstruction(self.adapted_image, self.scaler, args, rec_param, 
                path_to_weights, data_path, eval = eval)            
            self.adapted_image = np.load(load_path)
        # self.adapted_train_set = Get_GAN_Set(self.adapted_image.shape, args.patches_dimension, self.augmented_train_set)
        # self.adapted_validation_set = Get_GAN_Set(self.adapted_image.shape, args.patches_dimension, self.central_pixels_coor_vl)

    # def Prepare_GAN_Set(self, args, path):
    #     # print('[*]Normalizing the images...')
    #     gan_image = np.load(path)
    #     self.gan_image = gan_image
    #     self.gan_train_set = Get_GAN_Set(self.gan_image.shape, args.patches_dimension, self.augmented_train_set)
    #     self.gan_validation_set = Get_GAN_Set(self.gan_image.shape, args.patches_dimension, self.central_pixels_coor_vl)
    #     # mog: imagem já foi salva normalizada, depois ver como des-normaliza-la
    #     # shapes = self.images_norm[0].shape
    #     # scaler = StandardScaler()
    #     # #scaler = MinMaxScaler()
    #     # gan_image_t1 = gan_image[:shapes[2]]
    #     # gan_image_t2 = gan_image[shapes[2]:]
    #     # gan_images = np.concatenate((gan_image_t1, gan_image_t2), axis=2)
    #     # gan_images_reshaped = gan_images.reshape((gan_images.shape[0] * gan_images.shape[1], gan_images.shape[2]))
        
    #     # scaler = scaler.fit(gan_images_reshaped)
    #     # gan_images_normalized = scaler.fit_transform(images_reshaped)
    #     # gan_images_norm = gan_images_normalized.reshape((gan_images.shape[0], gan_images.shape[1], gan_images.shape[2]))
    #     # gan_image_t1_norm = images_norm[:, :, : image_t1.shape[2]]
    #     # gan_image_t2_norm = images_norm[:, :, image_t2.shape[2]: ]
    #     # self.gan_image[:7] = gan_image_t1_norm
    #     # self.gan_image[7:] = gan_image_t2_normGAN_Set(self.adapted_image.shape, args.patches_dimension, self.central_pixels_coor_vl)

        # It is important to note that in the case of domain adaptation procedure, the model will use the 
        # same coordinates taken on the source domain to define the samples in the target domain. However, due 
        # to the huge unbalance that images contan, the training code must ensure that each training batch
        # includes samples from both domains. 
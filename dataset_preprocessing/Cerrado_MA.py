import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology
from skimage.morphology import square, disk 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.io as sio

from Tools import *
from reconstruction_tool import *

class CE_MA():
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
        reference_t2 = np.load(Reference_t2_path)
    
        #For Amazon
        #Cutting the last 4 rows of the images and reference
        image_t1 = image_t1[:, 0:1700, 0:1440]
        image_t2 = image_t2[:, 0:1700, 0:1440]
        reference_t1 = reference_t1[0:1700, 0:1440]
        reference_t2 = reference_t2[0:1700, 0:1440]
        
        # Pre-processing references
        if args.buffer:
            print('[*]Computing buffer regions...')
            #Dilating the reference_t1
            reference_t1 = skimage.morphology.dilation(reference_t1, disk(args.buffer_dimension_out))
            #Dilating the reference_t2
            reference_t2_dilated = skimage.morphology.dilation(reference_t2, disk(args.buffer_dimension_out))
            buffer_t2_from_dilation = reference_t2_dilated - reference_t2
            reference_t2_eroded  = skimage.morphology.erosion(reference_t2 , disk(args.buffer_dimension_in))
            buffer_t2_from_erosion  = reference_t2 - reference_t2_eroded
            buffer_t2 = buffer_t2_from_dilation + buffer_t2_from_erosion
            reference_t2 = reference_t2 - buffer_t2_from_erosion
            buffer_t2[buffer_t2 == 1] = 2
            #buffer_t2_from_dilation[buffer_t2_from_dilation == 1] = 2
            #reference_t2 = buffer_t2_from_dilation + reference_t2 
            reference_t2 = reference_t2 + buffer_t2
                
        # Pre-processing images
        if args.compute_ndvi:
            print('[*]Computing and stacking the ndvi band...')
            ndvi_t1 = Compute_NDVI_Band(image_t1)
            ndvi_t2 = Compute_NDVI_Band(image_t2)
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
        
        # Storing the images in a list
        self.images_norm.append(image_t1_norm)
        self.images_norm.append(image_t2_norm)
        # Storing the references in a list
        self.references.append(reference_t1)
        self.references.append(reference_t2)


        # concatenate images and ajust ground truths
        shapes = self.images_norm[0].shape
        self.conc_image = np.zeros((shapes[0], shapes[1], shapes[2]*2))
        self.conc_image[:,:, 0:shapes[2]] = self.images_norm[0]
        self.conc_image[:,:, shapes[2]:] = self.images_norm[1]
        print("concatenated image shape:", self.conc_image.shape)
        print(self.conc_image.max())
        print(self.conc_image.min())
        self.new_reference = (self.references[0]*2) + self.references[1]
        self.new_reference[:][(self.new_reference == 3)] = 2
        self.new_reference[:][(self.new_reference == 4)] = 2
        print("new reference class values:", np.unique(self.new_reference))


        # create References (ground truths) for Diff_loss
        # diff_referen = image_t2_norm - image_t1_norm
        self.diff_reference = self.images_norm[1] - self.images_norm[0]
        
        # if args.training_type == 'domain_adaptation' or args.training_type == 'domain_adaptation_balance':
            
        #     if args.data_t3_year != '':
        #         Image_t3_path = args.dataset_main_path + args.dataset + args.images_section + args.data_t3_name + args.data_type
        #         Image_t4_path = args.dataset_main_path + args.dataset + args.images_section + args.data_t4_name + args.data_type
        #         if args.data_type == '.tif':
        #             image_t3 = Read_TIFF_Image(Image_t3_path)
        #             image_t4 = Read_TIFF_Image(Image_t4_path)
        #         if args.data_type == '.npy':
        #             image_t3 = np.load(Image_t3_path)
        #             image_t4 = np.load(Image_t4_path)
                
        #         image_t3 = image_t3[:,:1700,:1440]
        #         image_t4 = image_t4[:,:1700,:1440]
        #         if args.compute_ndvi:
        #             print('[*]Computing and stacking the ndvi band...')
        #             ndvi_t3 = Compute_NDVI_Band(image_t3)
        #             ndvi_t4 = Compute_NDVI_Band(image_t4)
        #             image_t3 = np.transpose(image_t3, (1, 2, 0))
        #             image_t3 = np.concatenate((image_t3, ndvi_t3), axis=2)
        #             image_t4 = np.transpose(image_t4, (1, 2, 0))
        #             image_t4 = np.concatenate((image_t4, ndvi_t4), axis=2)
        #         else:
        #             image_t3 = np.transpose(image_t3, (1, 2, 0))
        #             image_t4 = np.transpose(image_t4, (1, 2, 0))
                    
        #         # Pre-Processing the images
        #         print('[*]Normalizing the images...')
        #         scaler = StandardScaler()
        #         images = np.concatenate((image_t3, image_t4), axis=2)
        #         images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))
                
        #         scaler = scaler.fit(images_reshaped)
        #         images_normalized = scaler.fit_transform(images_reshaped)
        #         images_norm = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
        #         image_t3_norm = images_norm[:, :, : image_t3.shape[2]]
        #         image_t4_norm = images_norm[:, :, image_t4.shape[2]: ]
                
        #         # Storing the images in a list
        #         self.images_norm.append(image_t3_norm)
        #         self.images_norm.append(image_t4_norm)
        #     else:
        #         print('[!] At least a third image must be included in this type of training')
        #         sys.exit()
            
                
        #     if args.training_type == 'domain_adaptation_balance':
        #         Reference_t3_path = args.dataset_main_path + args.dataset + args.reference_section + args.reference_t3_name + args.data_type
        #         Reference_t4_path = args.dataset_main_path + args.dataset + args.reference_section + args.reference_t4_name + args.data_type
        #         reference_t3 = np.load(Reference_t3_path)
        #         reference_t4 = np.load(Reference_t4_path)
                
        #         reference_t3 = reference_t3[:1700,:1440]
        #         reference_t4 = reference_t4[:1700,:1440]
                
        #         # Pre-processing references
        #         if args.buffer:
        #             print('[*]Computing buffer regions...')
        #             #Dilating the reference_t3
        #             reference_t3 = skimage.morphology.dilation(reference_t3, disk(args.buffer_dimension_out))
        #             #Dilating the reference_t4
        #             reference_t4_dilated = skimage.morphology.dilation(reference_t4, disk(args.buffer_dimension_out))
        #             buffer_t4_from_dilation = reference_t4_dilated - reference_t4
        #             reference_t4_eroded  = skimage.morphology.erosion(reference_t4 , disk(args.buffer_dimension_in))
        #             buffer_t4_from_erosion  = reference_t4 - reference_t4_eroded
        #             buffer_t4 = buffer_t4_from_dilation + buffer_t4_from_erosion
        #             reference_t4 = reference_t4 - buffer_t4_from_erosion
        #             buffer_t4[buffer_t4 == 1] = 2
        #             #buffer_t2_from_dilation[buffer_t2_from_dilation == 1] = 2
        #             #reference_t2 = buffer_t2_from_dilation + reference_t2 
        #             reference_t4 = reference_t4 + buffer_t4    
                
        #         self.references_t1.append(reference_t3)
        #         self.references_t2.append(reference_t4)
    
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
                    self.Train_tiles = np.array([1, 5, 12, 13])
                    self.Valid_tiles = np.array([6, 7])
                    self.Undesired_tiles = []
                    print("Train" + str(self.Train_tiles))
                    print("Valid" + str(self.Valid_tiles))
                    # print("Test" + str(self.Test_tiles))
            else:
                # tiles = np.random.randint(100, size = 25) + 1
                # self.Train_tiles = tiles[:20]
                # self.Valid_tiles = tiles[20:]
                # np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                # np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                #This nedd to be redefined
                n_blocks = args.vertical_blocks * args.horizontal_blocks
                tiles = np.random.default_rng().choice(n_blocks, size=4, replace=False) + 1
                print(tiles)
                self.Train_tiles = tiles[:4]
                self.Valid_tiles = tiles[4:]
                self.Undesired_tiles = []
                np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                
        if args.phase == 'test':
            tiles = np.array([1, 7, 9, 13, 5, 12, 2, 3, 4, 6, 8, 10, 11, 14, 15])
            self.Train_tiles = tiles[:4]
            self.Valid_tiles = tiles[4:6]
            self.Test_tiles = tiles[6:]
            self.Undesired_tiles = []
            print("Train" + str(self.Train_tiles))
            print("Valid" + str(self.Valid_tiles))
            print("Test" + str(self.Test_tiles))
            # self.Train_tiles = []
            # self.Valid_tiles = []
            # self.Undesired_tiles = []
            
    def Coordinates_Creator(self, args, i):
        print('[*]Defining the central patches coordinates...')
        if args.phase == 'train':
            if args.fixed_tiles:
                if i == 0:
                    print("Cerrado i = 0")
                    self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
                    # self.central_pixels_coor_tr, self.y_train, self.central_pixels_coor_vl, self.y_valid = Central_Pixel_Definition(self.mask, self.references_t1[0], self.references_t2[0], args.patches_dimension, args.stride, args.porcent_of_last_reference_in_actual_reference)
                    self.central_pixels_coor_tr, self.central_pixels_coor_vl = Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.vertical_blocks, args.horizontal_blocks)
                    # if args.training_type == 'domain_adaptation_balance':
                    #     self.central_pixels_coor_tr_t, self.y_train_t, self.central_pixels_coor_vl_t, self.y_valid_t = Central_Pixel_Definition(self.mask, self.references_t1[1], self.references_t2[1], args.patches_dimension, args.stride, args.porcent_of_last_reference_in_actual_reference)
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
            else:
                self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
                sio.savemat(args.save_checkpoint_path + 'mask.mat', {'mask': self.mask})
                self.central_pixels_coor_tr, self.y_train, self.central_pixels_coor_vl, self.y_valid = Central_Pixel_Definition(self.mask, self.references_t1[0], self.references_t2[0], args.patches_dimension, args.stride, args.porcent_of_last_reference_in_actual_reference)
        if args.phase == 'test':
            print("Test PA")
            self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
            self.central_pixels_coor_ts = Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.vertical_blocks, args.horizontal_blocks, test = True)
            # self.central_pixels_coor_ts, self.y_test = Central_Pixel_Definition_For_Test(self.mask, np.zeros_like(self.references_t1[0]), np.zeros_like(self.references_t2[0]), args.patches_dimension, 1, args.phase)
            
        # It is important to note that in the case of domain adaptation procedure, the model will use the 
        # same coordinates taken on the source domain to define the samples in the target domain. However, due 
        # to the huge unbalance that images contan, the training code must ensure that each training batch
        # includes samples from both domains.

    def Coordinates_Train_Validation(self, args):
        self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
        self.central_pixels_coor_tr, self.central_pixels_coor_vl = Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.horizontal_blocks, args.vertical_blocks)

    def Coordinates_Test(self, args):
        self.mask = mask_creation(self.images_norm[0].shape[0], self.images_norm[0].shape[1], args.horizontal_blocks, args.vertical_blocks, self.Train_tiles, self.Valid_tiles, self.Undesired_tiles)
        self.central_pixels_coor_ts = Coor_Pixel_Definition(self.mask, args.patches_dimension, args.stride, args.vertical_blocks, args.horizontal_blocks, test = True)

    def Coordinates_Cyclegan(self, args):
        self.cycle_coor = No_Mask_Coor(self.images_norm[0].shape, args.patches_dimension, args.stride)

    # def Prepare_Cycle_Set(self, args):
    #     sefl.augmented_cycle_train_set = prepare_set_with_aug(self.central_pixels_coor_tr, 
    #         self.new_reference, args.patches_dimension, only_defo = False, multiplier = False)

    def Prepare_Train_Set(self, args):
        self.augmented_train_set = prepare_set_with_aug(self.central_pixels_coor_tr, 
            self.new_reference, args.patches_dimension, only_defo = True, multiplier = True)

    def Coordinates_Domain_Adaptation(self, args):
        self.no_stride_coor = No_Mask_Coor(self.images_norm[0].shape, args.patches_dimension, args.patches_dimension)

    def Prepare_GAN_Set(self, args, path_to_weights, eval = False, load = False, 
        adapt_back_to_original = False):

        rec_param = reconstruction_parameters(args, self)
        # print(rec_param.crop_size)

        if eval:
            data_path = args.eval_save_path
            file_name = args.adapted_file_name + '_eval'
        else:
            data_path = args.adapted_save_path
            file_name = args.adapted_file_name

        if eval == True or load == False:
        	overlap_reconstruction(self.conc_image, self.scaler, args, rec_param, 
                path_to_weights, data_path, eval = eval, pedro=args.pedro, net_=args.net_)

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
            overlap_reconstruction(self.adapted_image, self.scaler, args, rec_param, 
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
    #     # self.gan_image[7:] = gan_image_t2_norm

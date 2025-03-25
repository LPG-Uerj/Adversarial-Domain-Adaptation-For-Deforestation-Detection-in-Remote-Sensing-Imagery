import os
import sys
import numpy as np
import skimage.morphology
from skimage.morphology import square, disk 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AMAZON_PA():
    def __init__(self, args):
        
        self.images_norm = []
        self.scaler = []
        self.images_diff = []
        
        # Image_t1_path = args.dataroot + args.dataset + args.images_section + args.data_t1_name + '.npy'
        # Image_t2_path = args.dataroot + args.dataset + args.images_section + args.data_t2_name + '.npy'
        Image_t1_path = args.dataroot + args.images_section + args.data_t1_name + '.npy'
        Image_t2_path = args.dataroot + args.images_section + args.data_t2_name + '.npy'
        Reference_t1_path = args.dataroot + args.reference_section + args.reference_t1_name + '.npy'
        Reference_t2_path = args.dataroot + args.reference_section + args.reference_t2_name + '.npy'
        
        # Reading images and references
        print('[*]Reading images...')
        image_t1 = np.load(Image_t1_path)
        image_t2 = np.load(Image_t2_path)
        reference_t1 = np.load(Reference_t1_path)
        reference_t2 = np.load(Reference_t2_path)
        
        #Cutting the last rows of the images
        image_t1 = image_t1[:,1:1099,:]
        image_t2 = image_t2[:,1:1099,:]
        reference_t1 = reference_t1[0:1098,:]
        reference_t2 = reference_t2[0:1098,:]
        
        # Pre-processing images
        if args.compute_ndvi:
            print('[*]Computing and stacking the ndvi band...')            
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
            image_t1 = np.concatenate((image_t1, ndvi_t1), axis=2)
            image_t2 = np.concatenate((image_t2, ndvi_t2), axis=2)
        else:
            image_t1 = np.transpose(image_t1, (1, 2, 0))
            image_t2 = np.transpose(image_t2, (1, 2, 0))
        # Computing the difference image
        image_di = image_t2 - image_t1       
        # Pre-Processing the images
        if args.standard_normalization:
            print('[*]Normalizing the images...')
            scaler = StandardScaler()
           
            images = np.concatenate((image_t1, image_t2), axis=2)
            images_reshaped = images.reshape((images.shape[0] * images.shape[1], images.shape[2]))
            
            scaler = scaler.fit(images_reshaped)
            self.scaler.append(scaler)
            images_normalized = scaler.fit_transform(images_reshaped)
            images_norm = images_normalized.reshape((images.shape[0], images.shape[1], images.shape[2]))
            image_t1_norm = images_norm[:, :, : image_t1.shape[2]]
            image_t2_norm = images_norm[:, :, image_t2.shape[2]: ]
            
            # Storing the images in a list
            self.images_norm.append(image_t1_norm)
            self.images_norm.append(image_t2_norm)
            
            scaler = StandardScaler()
            images_reshaped = image_di.reshape((image_di.shape[0] * image_di.shape[1], image_di.shape[2]))
            scaler = scaler.fit(images_reshaped)
            self.scaler.append(scaler)
            images_normalized = scaler.fit_transform(images_reshaped)
            image_di_norm = images_normalized.reshape((image_di.shape[0], image_di.shape[1], image_di.shape[2]))
            self.images_diff.append(image_di_norm)

        self.conc_img = np.concatenate((self.images_norm[0], self.images_norm[1]), axis = 2)


        if args.ref_buffer:
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

        self.references = []
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

        if args.use_tile_configuration:
            self.vertical_blocks = 5 # fixed for experiments
            self.horizontal_blocks = 3 # fixed for experiments
        # def Tiles_Configuration(self, args):
            #Generating random training and validation tiles
            # if args.phase == 'train' or args.phase == 'compute_metrics':
            if args.phase == 'train':
                if args.fixed_tiles:
                    # if args.defined_before:
                    #     if args.phase == 'train':
                    #         files = os.listdir(args.checkpoint_dir_posterior)
                    #         print(files[i])
                    #         self.Train_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Train_tiles.npy')
                    #         self.Valid_tiles = np.load(args.checkpoint_dir_posterior + '/' + files[i] + '/' + 'Valid_tiles.npy')
                    #         np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                    #         np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                    #     if args.phase == 'compute_metrics':
                    #         self.Train_tiles = np.load(args.save_checkpoint_path +  'Train_tiles.npy')
                    #         self.Valid_tiles = np.load(args.save_checkpoint_path +  'Valid_tiles.npy')
                   
                        tiles = np.array([1, 7, 9, 13, 5, 12, 2, 3, 4, 6, 8, 10, 11, 14, 15])
                        self.Train_tiles = tiles[:4]
                        self.Valid_tiles = tiles[4:6]
                        self.Test_tiles = tiles[6:]
                        self.Undesired_tiles = []
                        print("Train" + str(self.Train_tiles))
                        print("Valid" + str(self.Valid_tiles))
                        print("Test" + str(self.Test_tiles))
                else:
                    # tiles = np.random.randint(100, size = 25) + 1
                    # self.Train_tiles = tiles[:20]
                    # self.Valid_tiles = tiles[20:]
                    # np.save(args.save_checkpoint_path + 'Train_tiles' , self.Train_tiles)
                    # np.save(args.save_checkpoint_path + 'Valid_tiles' , self.Valid_tiles)
                    #This nedd to be redefined
                    n_blocks = self.vertical_blocks * self.horizontal_blocks
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
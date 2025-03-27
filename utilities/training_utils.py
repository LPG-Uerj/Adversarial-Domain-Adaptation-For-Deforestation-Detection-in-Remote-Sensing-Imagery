import json
import numpy as np
import os
import scipy.io as sio
import skimage as sk
import tifffile


import torch
import torch.nn as nn
import torch.nn.functional as F # mog: for focal loss
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

import sys
sys.path.append('..')
from model_architectures import GAN

from . import utils

def Analyse_hyperparameters_test(args, hyperparameters_file_path):
    f = open(hyperparameters_file_path, 'r')
    lines = f.readlines()
    for l in lines:
        fields = l.split(':')            
        if len(fields) > 1:
            field_1 = fields[0].split(' ')
            field_2 = fields[1].split(' ')
            field_head = str(field_1[-1])
            field_content = str((field_2[-1]))
            if field_head == '"epochs"':
                args.epochs = int(field_content[:-2])
            if field_head == '"lr"':
                args.lr = float(field_content[:-2])
            if field_head == '"beta1"':
                args.beta1 = float(field_content[:-2])
            if field_head == '"data_augmentation"':
                if field_content[:-2]=='true':
                    args.data_augmentation = True
                else:
                    args.data_augmentation = False
            if field_head == '"vertical_blocks"':
                args.vertical_blocks = int(field_content[:-2])
            if field_head == '"horizontal_blocks"':
                args.horizontal_blocks = int(field_content[:-2])
            if field_head == '"image_channels"':
                args.image_channels = int(field_content[:-2])
            if field_head == '"patches_dimension"':
                args.patches_dimension = int(field_content[:-2])
            if field_head == '"stride"':
                args.stride = int(field_content[:-2])          
            if field_head == '"compute_ndvi"':
                if field_content[:-2]=='true':
                    args.compute_ndvi = True
                else:
                    args.compute_ndvi = False
            if field_head == '"balanced_tr"':
                if field_content[:-2]=='true':
                    args.balanced_tr = True
                else:
                    args.balanced_tr = False
            if field_head == '"balanced_vl"':
                if field_content[:-2]=='true':
                    args.balanced_vl = True
                else:
                    args.balanced_vl = False
            if field_head == '"buffer"':
                if field_content[:-2]=='true':
                    args.buffer = True
                else:
                    args.buffer = False
            if field_head == '"porcent_of_last_reference_in_actual_reference"':
                args.porcent_of_last_reference_in_actual_reference = int(field_content[:-2])
            if field_head == '"patience"':
                args.patience = int(field_content[:-2])
            if field_head == '"data_t1_year"':
                args.data_t1_year = str(field_content[1:-3])
            if field_head == '"data_t2_year"':
                args.data_t2_year = str(field_content[1:-3])
            if field_head == '"data_t1_name"':
                args.data_t1_name = str(field_content[1:-3])
            if field_head == '"data_t2_name"':
                args.data_t2_name = str(field_content[1:-3])
            if field_head == '"reference_t1_name"':
                args.reference_t1_name = str(field_content[1:-3])
            if field_head == '"reference_t2_name"':
                args.reference_t2_name = str(field_content[1:-3])
            if field_head == '"data_type"':
                args.data_type = str(field_content[1:-3])
            if field_head =='"buffer_dimension_out"':
                args.buffer_dimension_out = int(field_content[:-2])
            if field_head =='"buffer_dimension_in"':
                args.buffer_dimension_in = int(field_content[:-2])
    return args

def save_as_mat(data, name):
    sio.savemat(name, {name: data})

def Read_TIFF_Image(Path):
    img =[]
    #gdal_header = gdal.Open(Path)
    #img = gdal_header.ReadAsArray()
    return img

def Compute_NDVI_Band(Image):
    Image = Image.astype(np.float32)
    nir_band = Image[4, :, :]
    red_band = Image[3, :, :]
    ndvi = np.zeros((Image.shape[1] , Image.shape[2] , 1))
    ndvi[ : , : , 0] = np.divide((nir_band-red_band),(nir_band+red_band))
    return ndvi

def compute_metrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels)
    recall = 100*recall_score(true_labels, predicted_labels)
    prescision = 100*precision_score(true_labels, predicted_labels)
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    return accuracy, f1score, recall, prescision, conf_mat


def return_metrics(ref, pred):
    # print(ref.shape)
    # print(pred.shape)
    assert ref.shape == pred.shape
    y = ref.reshape((ref.shape[0] * ref.shape[1] * ref.shape[2]))
    x = pred.reshape((pred.shape[0] * pred.shape[1] * pred.shape[2]))
    # x[:][(x == 2)] = 0
    x [(x == 2)] = 0
    y_mask = (y == 1)
    dc_mask = (y == 2)
    
    final_mask = ((x * 2) - (y_mask * 1) - (dc_mask*4))

    # values for x:
    # 2 and 0
    # values for y:
    # 1 and 0
    # values for dc - dont care:
    # 4 and 0

    #### x - y - dc:
    # not possible -> (2 - 1 - 4 = -3)
    # not possible -> (0 - 1 - 4 = -5)

    # 2 - 0 - 4 = -2 -> dc
    # 0 - 0 - 4 = -4 -> dc

    # 0 - 1 - 0 = -1 -> fn
    # 2 - 1 - 0 = 1 -> tp
    # 2 - 0 - 0 = 2 -> fp
    # 0 - 0 - 0 = 0 -> tn

    dc = np.count_nonzero(dc_mask)
    # print("dc",dc)
    fn = np.count_nonzero((final_mask == -1))
    tp = np.count_nonzero((final_mask == 1))
    fp = np.count_nonzero((final_mask == 2))
    tn = np.count_nonzero((final_mask == 0))

    prescision=0.0
    f1score = 0.0
    recall = 0.0
    overall = 0.0
    alert_rate = 0.0
    if (tp+fn != 0):      
        recall = tp/(tp+fn)
    if (tp+fp != 0):
        prescision = tp/(tp+fp)
    if(prescision!=0):
        f1score = 100*(2*prescision*recall)/(prescision+recall)
    P = tp+fn
    N = tn+fp
    if (P+N != 0 ):
        overall = 100*(tp+tn)/(P+N)
        alert_rate = 100*(tp+fp)/(P+N)
    recall = recall * 100
    prescision = prescision * 100
    # print("Total: ", dc)
    # print("tp: ", tp)
    # print("fp: ", fp)
    # print("tn: ", tn)
    # print("fn: ", fn)
    # print("Overall: %.2f" % overall)
    # print("F1-Score: %.2f" % f1score)
    # print("Recall: %.2f" % recall)
    # print("Prescision: %.2f" % prescision)
    # print("Alert Rate: %.2f" % alert_rate)

    return overall, f1score, recall, prescision, alert_rate 


def calcula_metricas(ref, pred):
    ref = ref.reshape((ref.shape[0] * ref.shape[1] * ref.shape[2]))
    pred = pred.reshape((pred.shape[0] * pred.shape[1] * pred.shape[2]))
    pred = (pred == 1)
    # codigo renan
    total = 0
    fp, tp = 0, 0
    fn, tn = 0, 0

    shape = ref.shape
    c0 = 0
    c1 = 1
    c2 = 2
    #  result_image = np.zeros((9, 368,520, 3), dtype = np.float32)

    for k in range(len(ref)):
        # for i in range(shape[0]): #patch_size ou tile_size
        #   for j in range(shape[1]):  #patch_size ou tile_size
                if(ref[k]!=c2):
                    if(ref[k]==c1):
                        if(pred[k]==1):
                            tp = tp+1
                        else:
                            fn = fn+1

                    elif(ref[k]==c0):
                        if(pred[k]==0):
                            tn = tn+1
                        else:
                            fp = fp+1
                else:
                    total = total+1
    prescision=0.0
    f1score = 0.0       
    recall = tp/(tp+fn)
    if(tp+fp!=0):
        prescision = tp/(tp+fp)
    if(prescision!=0):
        f1score = 100*(2*prescision*recall)/(prescision+recall)
    P = tp+fn
    N = tn+fp
    overall = 100*(tp+tn)/(P+N)
    alert_rate = 100*(tp+fp)/(P+N)
    recall = recall * 100
    prescision = prescision * 100  

    # print("Total: ", total)
    # print("tp: ", tp)
    # print("fp: ", fp)
    # print("tn: ", tn)
    # print("fn: ", fn)
    # print("Overall: %.2f" % overall)
    # print("F1-Score: %.2f" % f1score)
    # print("Recall: %.2f" % recall)
    # print("Prescision: %.2f" % prescision)
    # print("Alert Rate: %.2f" % alert_rate)
    # print("Confusion Matrix: \n["+str(tp)+" "+str(fp)+"]\n["+str(fn)+" "+str(tn)+"]")


    return overall, f1score, recall, prescision, alert_rate
    

def ready_domain(object, args, train_set, test_set = False, test_only_defo = False,cyclegan_set = False, augmented = True):
    o = object
    o.Tiles_Configuration(args)
    if train_set:
        o.Coordinates_Train_Validation(args)
        if augmented:
            o.Prepare_Train_Set(args)
    if cyclegan_set:
        o.Coordinates_Cyclegan(args)
    if test_set:
        o.Coordinates_Test(args)
        if test_only_defo:
            o.central_pixels_coor_ts = get_defo_coor(o.new_reference, 
                o.central_pixels_coor_ts, o.patch_dimension)

def prepare_set_with_aug(coor, ref, patch_size, only_defo = False, multiplier = False):
    # coor = data.central_pixels_coord_tr
    # ref = data.new_reference
    print("original", coor.shape)
    if only_defo == True:
        coor = get_defo_coor(ref, coor, patch_size)
        print("defo only", coor.shape)

    if multiplier == True:
        augs, coor = augmentations_multiplier_index(coor)
        print("after multiplier", coor.shape)
    else:
        augs = augmentations_index(coor)

    set = train_set_prepare(coor, augs)
    print("set shape", set.shape)
    
    return set

def get_defo_coor(reference, coordinates, patch_size):
    coor_list = []
    # count = 0
    for c in coordinates:
        # print(c)
        a = int(c[0])
        b = int(c[1])
        ref = reference[a:a+patch_size, b:b+patch_size]
        if np.count_nonzero(ref == 1) > 0 :
        # if 1 in ref:
            # print(c)
            coor_list.append(c)
            # count += 1

    out_coor = np.zeros((len(coor_list),2))
    # print (out_coor.shape)
#2867
    for i in range(len(coor_list)):
        out_coor[i] = coor_list[i]
    
    # print (out_coor.shape)
    return out_coor

def replicate_coordinates(coordinates, step):
    size = len(coordinates)
    out_coor = np.zeros((size*step, coordinates.shape[1]))
    # print(type(out_coor))
    # print(out_coor.shape)
    # print(type(coordinates[0,0]))
    # print(coordinates[0].shape)
    for i in range (size):
        # a = coordinates[i]
        a = i*step
        b = a + step
        for j in range (step):
            out_coor[a+j,0] = coordinates[i,0]
            out_coor[a+j,1] = coordinates[i,1]

        # out_coor[a:b,0] = coordinates[i][0]
        # out_coor[a:b,1] = coordinates[i][1]
        # print(out_coor[a:b])
    # print(type(out_coor))
    return out_coor
        

def augmentations_multiplier_index(coordinates):
    size = len(coordinates)
    step = 8 # number of transforms
    aug_index = np.zeros((size*step))
    # print(aug_index.shape)
    for i in range(size):
        a = i*step
        b = a + step
        aug_index[a:b] = np.arange(8)
        # print(aug_index[a:b])
    new_coor = replicate_coordinates(coordinates, step)
    return aug_index, new_coor

def augmentations_index(coordinates, save = False, output_path = None):
    size = len(coordinates)
    aug_index = np.random.randint(8, size = size)
    if save == True:
        if output_path == None:
            raise TypeError("No output path defined")
        elif os.path.isdir(output_path) == False:
            os.makedirs(output_path)

        np.save(output_path + '/aug_index.npy', aug_index)

    return aug_index

def load_augmentions_index(path, domain):
    full_file_path = path + domain + '/aug_index.npy' 
    if os.path.isfile(full_file_path) == False:
        raise TypeError("There is no such file" + full_file_path)
    return np.load(full_file_path)

def apply_augmentations(data, aug_index, tf = False):
    assert len(data) == len(aug_index)
    augmented_data = data
    # print('shape data', data.shape)
    if len(data.shape) > 3:
        if tf:
            for i in range(len(data)):
                for j in range(data.shape[-1]):
                    augmented_data[i][:,:,j] = transform(data[i][:,:,j], aug_index[i])

        else:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    augmented_data[i,j] = transform(data[i,j], aug_index[i])
    else:
        for i in range(len(data)):
            augmented_data[i] = transform(data[i], aug_index[i])

    return augmented_data

def transform(data, aug):
    # 0 - no transform
    # 1 - 90 degrees counterclockwise
    # 2 - 180 degrees turn
    # 3 - 270 degrees counterclowise (90 clockwise)
    # 4 - mirror image vertically
    # 5 - mirror + 90 counter
    # 6 - mirror + 180 counter
    # 7 - mirror + 270 counter
    if aug == 0:
        return data
    elif aug == 1:
        return np.rot90(data)
    elif aug == 2:
        return np.rot90(data, 2)
    elif aug == 3:
        return np.rot90(data, 3)
    elif aug == 4:
        return np.flip(data, axis = 1)
    elif aug == 5:
        return np.rot90(np.flip(data, axis = 1))
    elif aug == 6:
        return np.rot90(np.flip(data, axis = 1), 2)
    elif aug == 7:
        return np.rot90(np.flip(data, axis = 1), 3)

    else:
        raise Exception("Transform out of index (n>7 or n<0)")

def train_set_prepare(coor_index, aug_index):
    aux = coor_index
    aux = np.zeros((aux.shape[0], aux.shape[1]+1))
    aux[:,:2] = coor_index
    # print(aux.shape)
    aux[:, 2] = aug_index
    trains_set = aux

    return trains_set

def Data_Augmentation_Definition(central_pixels_coor, labels):
    num_sample = np.size(central_pixels_coor , 0)
    data_cols = np.size(central_pixels_coor , 1)
    num_classes = np.size(labels, 1)
    
    #central_pixels_coor_augmented = np.zeros((8 * num_sample, data_cols + 1))
    central_pixels_coor_augmented = np.zeros((3 * num_sample, data_cols + 1))
    #labels_augmented = np.zeros((8 * num_sample, num_classes))
    labels_augmented = np.zeros((3 * num_sample, num_classes))
    counter = 0
    for s in range(num_sample):
        central_pixels_coor_x_0 = central_pixels_coor[s, :]
        labels_y_0 = labels[s, :]        
        #Storing
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 0
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 1
        labels_augmented[counter, :] = labels_y_0
        counter += 1
        
        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 2
        labels_augmented[counter, :] = labels_y_0
        counter += 1
        
        central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        central_pixels_coor_augmented[counter, 2] = 3
        labels_augmented[counter, :] = labels_y_0
        counter += 1
        
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 4
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 5
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 6
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
        # central_pixels_coor_augmented[counter, 0 : 2] = central_pixels_coor_x_0
        # central_pixels_coor_augmented[counter, 2] = 7
        # labels_augmented[counter, :] = labels_y_0
        # counter += 1
        
    return central_pixels_coor_augmented, labels_augmented

def Data_Augmentation_Execution(data, transformation_indexs):
    data_rows = np.size(data , 1)
    data_cols = np.size(data , 2)
    data_depth = np.size(data , 3)
    num_sample = np.size(data , 0)
    
    data_transformed = np.zeros((num_sample, data_rows, data_cols, data_depth))
    counter = 0
    for s in range(num_sample):
        data_x_0 = data[s, :, :, :]
        transformation_index = transformation_indexs[s]
        #Rotating
        if transformation_index == 0:
            data_transformed[s, :, :, :] = data_x_0
        # if transformation_index == 1:
        #     data_transformed[s, :, :, :] = sk.transform.rotate(data_x_0, 90) 
        if transformation_index == 1:
            data_transformed[s, :, :, :] = np.rot90(data_x_0)
        if transformation_index == 2:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 0)
        if transformation_index == 3:
            data_transformed[s, :, :, :] = np.flip(data_x_0, 1)
        # if transformation_index == 2:
        #     data_transformed[s, :, :, :] = sk.transform.rotate(data_x_0, 180)
        # if transformation_index == 3:
        #     data_transformed[s, :, :, :] = sk.transform.rotate(data_x_0, 270)
        # #Flipping
        # if transformation_index >= 4: 
        #     data_x_4 = np.flipud(data_x_0)
        #     if transformation_index == 4:
        #         data_transformed[s, :, :, :] = data_x_4
        #     #Rotating again
        #     if transformation_index == 5:
        #         data_transformed[s, :, :, :] = sk.transform.rotate(data_x_4, 90)
        #     if transformation_index == 6:
        #         data_transformed[s, :, :, :] = sk.transform.rotate(data_x_4, 180)
        #     if transformation_index == 7:
        #         data_transformed[s, :, :, :] = sk.transform.rotate(data_x_4, 270)
        
    return data_transformed   
# def Patch_Extraction(data, central_pixels_indexs, patch_size, padding, mode):
    
#     half_dim = patch_size // 2
#     data_rows = np.size(data, 0)
#     data_cols = np.size(data, 1)
#     data_depth = np.size(data, 2)
#     num_samp = np.size(central_pixels_indexs , 0)
    
#     patches_cointainer = np.zeros((num_samp, patch_size, patch_size, data_depth)) 
    
#     if padding:
#         if mode == 'zeros':
#             upper_padding = np.zeros((half_dim, data_cols, data_depth))
#             left_padding = np.zeros((data_rows + half_dim, half_dim, data_depth))
#             bottom_padding = np.zeros((half_dim, half_dim + data_cols, data_depth))
#             right_padding = np.zeros((2 * half_dim + data_rows, half_dim, data_depth))
            
#             #Add padding to the data     
#             data_padded = np.concatenate((upper_padding, data), axis=0)
#             data_padded = np.concatenate((left_padding, data_padded), axis=1)
#             data_padded = np.concatenate((data_padded, bottom_padding), axis=0)
#             data_padded = np.concatenate((data_padded, right_padding), axis=1)
#         if mode == 'reflect':
#             npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))
#             data_padded = np.pad(data, pad_width = npad, mode = 'reflect')
#     else:
#         data_padded = data
    
#     for i in range(num_samp):
#         # print(np.min(data_padded[:, :, :7]))
#         # print(np.max(data_padded[:, :, :7]))
#         # print(np.min(data_padded[:, :, 7:]))
#         # print(np.max(data_padded[:, :, 7:]))
#         patches_cointainer[i, :, :, :] = data_padded[int(central_pixels_indexs[i , 0]) - half_dim  : int(central_pixels_indexs[i , 0]) + half_dim + 1, int(central_pixels_indexs[i , 1]) - half_dim : int(central_pixels_indexs[i , 1]) + half_dim + 1, :]
                
#     return patches_cointainer
def Patch_Extraction(data, central_pixels_indexs, domain_index, patch_size, padding, mode):
    
    half_dim = patch_size // 2
    data_rows = np.size(data[0], 0)
    data_cols = np.size(data[0], 1)
    data_depth = np.size(data[0], 2)
    num_samp = np.size(central_pixels_indexs , 0)
    
    patches_cointainer = np.zeros((num_samp, patch_size, patch_size, data_depth)) 
    
    if padding:
        data_padded = []
        for i in range(len(data)):
            if mode == 'zeros':
                upper_padding = np.zeros((half_dim, data_cols, data_depth))
                # print("upper", upper_padding.shape)
                left_padding = np.zeros((data_rows + half_dim, half_dim, data_depth))
                # print("left", left_padding.shape)
                bottom_padding = np.zeros((half_dim, half_dim + data_cols, data_depth))
                # print("bottom", bottom_padding.shape)
                right_padding = np.zeros((2 * half_dim + data_rows, half_dim, data_depth))
                # print("right", right_padding.shape)
                
                #Add padding to the data     
                data_padded_ = np.concatenate((upper_padding, data[i]), axis=0)
                data_padded_ = np.concatenate((left_padding, data_padded_), axis=1)
                data_padded_ = np.concatenate((data_padded_, bottom_padding), axis=0)
                data_padded_ = np.concatenate((data_padded_, right_padding), axis=1)
                data_padded.append(data_padded_)
            if mode == 'reflect':
                npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))
                data_padded.append(np.pad(data[i], pad_width = npad, mode = 'reflect'))
    else:
        data_padded = data
    # ESto aqui tiene ser revisado para el nuevo contexto.
    for i in range(num_samp):
        data_padded_ = data_padded[int(domain_index[i,0])]
        patches_cointainer[i, :, :, :] = data_padded_[int(central_pixels_indexs[i , 0]) - half_dim  : int(central_pixels_indexs[i , 0]) + half_dim + 1, int(central_pixels_indexs[i , 1]) - half_dim : int(central_pixels_indexs[i , 1]) + half_dim + 1, :]
                
    return patches_cointainer
    
def mask_creation(mask_row, mask_col, num_patch_row, num_patch_col, Train_tiles, Valid_tiles, Undesired_tiles):
    train_index = 1
    teste_index = 2
    valid_index = 3
    undesired_index = 4
    
    patch_dim_row = mask_row//num_patch_row
    patch_dim_col = mask_col//num_patch_col
    print("mask dim row", patch_dim_row)
    print("mask dim col", patch_dim_col)

    
    mask_array = 2 * np.ones((mask_row, mask_col))
    
    train_mask = np.ones((patch_dim_row, patch_dim_col))
    valid_mask = 3 * np.ones((patch_dim_row, patch_dim_col))
    undesired_mask = 4 * np.ones((patch_dim_row, patch_dim_col))
    counter_r = 1
    counter = 1
    for i in range(0, mask_row, patch_dim_row): 
        for j in range(0 , mask_col, patch_dim_col): 
            # print (j)          
            train = np.size(np.where(Train_tiles == counter),1)
            valid = np.size(np.where(Valid_tiles == counter),1)
            undesired = np.size(np.where(Undesired_tiles == counter), 1)
            if train == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = train_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = np.ones((mask_row - i, patch_dim_col))
            if valid == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = valid_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 3 * np.ones((mask_row - i, patch_dim_col))
            if undesired == 1:
                mask_array[i : i + patch_dim_row, j : j + patch_dim_col] = undesired_mask
                if counter_r == num_patch_row:
                    mask_array[i : mask_row, j : j + patch_dim_col] = 4 * np.ones((mask_row - i, patch_dim_col))
            
            counter += 1       
        counter_r += 1
    return mask_array

def Get_GAN_Set(img_shape, patch_size, aug_train_set):
    counter = 0
    print("shape =", aug_train_set.shape)
    print(np.amax(aug_train_set))
    # print(np.amax(aug_train_set, axis = 1))
    print("img_shape", img_shape)
    for i in range(len(aug_train_set)):
        # print(aug_train_set[i][1])
        if (aug_train_set[i][0] + patch_size) < img_shape[0] and (aug_train_set[i][1] + patch_size) < img_shape[1]:
            counter += 1

    print ("counter =" , counter)
    out_coor = np.zeros((counter, aug_train_set.shape[1]))
    print("out_corr shape", out_coor.shape)
    counter = 0
    for i in range (len(aug_train_set)):
        if (aug_train_set[i][0] + patch_size) < img_shape[0] and (aug_train_set[i][1] + patch_size) < img_shape[1]:
            out_coor[counter] = aug_train_set[i]
            counter += 1
    return out_coor


def No_Mask_Coor(img_shape, patch_size, stride):
    rows = img_shape[0]
    cols = img_shape[1]
    # r_limit = rows - patch_size
    # c_limit = cols - patch_size
    print("rows ", rows)
    print("cols ", cols)
    
    if stride <= 0 or stride > patch_size:
        stride = patch_size
    coordinates = np.zeros(((rows//stride)*(cols//stride), 2))
    # print("num of coordinates", coordinates.shape[0])
    position = 0
    print("Getting full image coordinates (without mask), stride", stride)
    print("Number of Coordiantes", coordinates.shape[0])
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            if (i + patch_size) <= rows and (j + patch_size) <= cols:
                coordinates[position, 0] = i
                coordinates[position, 1] = j
                position += 1

    return coordinates


def Coor_Pixel_Definition(mask, patch_size, stride, i_blocks, j_blocks, test = False):
    rows = np.size(mask, 0)
    cols = np.size(mask, 1)
    print("rows ", rows)
    print("cols ", cols)

    cut_rows = rows//i_blocks
    cut_cols = cols//j_blocks
    # i_block_limit = cut_rows - patch_size
    # j_block_limit = cut_cols - patch_size


    # print("cut_rows",cut_rows)
    # print("cut_cols",cut_cols)
    
    # print("i limit", i_block_limit)
    # print("j limit", j_block_limit)

    assert cut_rows >= patch_size
    assert cut_cols >= patch_size

    if stride <= 0 or stride > patch_size:
        stride = patch_size

    next_step = stride + patch_size
    counter_tr = 0
    counter_vl = 0
    counter_ts = 0
    coor_mask = np.zeros((mask.shape))
    for k in range (i_blocks):
        for l in range (j_blocks):
            for i in range(0, cut_rows, stride):
                for j in range(0, cut_cols, stride):
                    a = cut_rows*k + i
                    b = cut_cols*l + j
                    # print (i)
                    if (a + patch_size < cut_rows*(k+1)) and (b + patch_size< cut_cols*(l+1)):
                        if test == False:
                            if mask[a,b] == 1:
                                # if (a + patch_size) < cut_rows and (b + patch_size) < cut_cols:
                                #   if mask[a + patch_size , b + patch_size] == 1:
                                    #Belongs to the training tile
                                counter_tr += 1
                                coor_mask[a,b] = 1

                            if mask[a,b] == 3:
                                # if (a + patch_size) < cut_rows and (b + patch_size) < cut_cols:
                                #   if mask[a + patch_size , b + patch_size] == 3:
                                    #Belongs to the training tile
                                counter_vl += 1
                                coor_mask[a,b] = 3
                        else:
                            if mask[a,b] == 2:
                                counter_ts += 1
                                coor_mask[a,b] = 2

    if test == False:
        coor_tr = np.zeros((counter_tr, 2))
        coor_vl = np.zeros((counter_vl, 2))
        counter_tr = 0
        counter_vl = 0
        # print("coor_tr ", coor_tr.shape)
        # print("coor_vl ", coor_vl.shape)
    else:
        coor_ts = np.zeros((counter_ts, 2))
        counter_ts = 0
        # print("coor_ts ", coor_ts.shape)

    for k in range (i_blocks):
        for l in range (j_blocks):
            for i in range(0, cut_rows, stride):
                for j in range(0, cut_cols, stride):
                    # print (j)
                    a = cut_rows*k + i
                    b = cut_cols*l + j
                    value = coor_mask[a,b]

                    if (a + patch_size < cut_rows*(k+1)) and (b + patch_size< cut_cols*(l+1)):
                        if test == False:
                            if value == 1:
                                coor_tr[counter_tr, 0] = a
                                coor_tr[counter_tr, 1] = b
                                counter_tr += 1
                            if value == 3:
                                coor_vl[counter_vl, 0] = a
                                coor_vl[counter_vl, 1] = b
                                counter_vl += 1
                        else:
                            if value == 2:
                                coor_ts[counter_ts, 0] = a
                                coor_ts[counter_ts, 1] = b
                                counter_ts += 1

    
    if test == False:
        # print("tr", coor_tr[-30:])
        # print("vl", coor_vl[-30:])
        return coor_tr, coor_vl
    else:
        return coor_ts


def Central_Pixel_Definition(mask, last_reference, actual_reference, patch_dimension, stride, porcent_of_last_reference_in_actual_reference):
    
    mask_rows = np.size(mask, 0)
    mask_cols = np.size(mask, 1)
    
    half_dim = patch_dimension//2
    ini_dim = 0
    # half_dim = ini_dim
    upper_padding = np.zeros((half_dim, mask_cols))
    # print("upper", upper_padding.shape)
    left_padding = np.zeros((mask_rows + half_dim, half_dim))
    # print("left", left_padding.shape)
    bottom_padding = np.zeros((half_dim, half_dim + mask_cols))
    # print("bottom", bottom_padding.shape)
    right_padding = np.zeros((2 * half_dim + mask_rows, half_dim))
    # print("right", right_padding.shape)
    
    #Add padding to the mask     
    mask_padded = np.concatenate((upper_padding, mask), axis=0)
    mask_padded = np.concatenate((left_padding, mask_padded), axis=1)
    mask_padded = np.concatenate((mask_padded, bottom_padding), axis=0)
    mask_padded = np.concatenate((mask_padded, right_padding), axis=1)
    
    #Add padding to the last reference
    last_reference_padded = np.concatenate((upper_padding, last_reference), axis=0)
    last_reference_padded = np.concatenate((left_padding, last_reference_padded), axis=1)
    last_reference_padded = np.concatenate((last_reference_padded, bottom_padding), axis=0)
    last_reference_padded = np.concatenate((last_reference_padded, right_padding), axis=1)
    
    #Add padding to the last reference
    actual_reference_padded = np.concatenate((upper_padding, actual_reference), axis=0)
    actual_reference_padded = np.concatenate((left_padding, actual_reference_padded), axis=1)
    actual_reference_padded = np.concatenate((actual_reference_padded, bottom_padding), axis=0)
    actual_reference_padded = np.concatenate((actual_reference_padded, right_padding), axis=1)
    
    #Initializing the central pixels coordinates containers
    central_pixels_coord_tr_init = []
    central_pixels_coord_vl_init = []
    
    if stride == 1:
        central_pixels_coord_tr_init = np.where(mask_padded == 1)
        central_pixels_coord_vl_init = np.where(mask_padded == 3)
        central_pixels_coord_tr_init = np.transpose(np.array(central_pixels_coord_tr_init))
        central_pixels_coord_vl_init = np.transpose(np.array(central_pixels_coord_vl_init))
    else:
        counter_tr = 0
        counter_vl = 0
        for i in range(2 * half_dim, np.size(mask_padded , 0) - 2 * half_dim, stride):
            for j in range(2 * half_dim, np.size(mask_padded , 1) - 2 * half_dim, stride):
                mask_value = mask_padded[i , j]
                #print(mask_value)
                if mask_value == 1:
                    #Belongs to the training tile
                    counter_tr += 1
                    
                if mask_value == 3:
                    #Belongs to the validation tile
                    counter_vl += 1
        
        central_pixels_coord_tr_init = np.zeros((counter_tr, 2))
        central_pixels_coord_vl_init = np.zeros((counter_vl, 2))
        counter_tr = 0
        counter_vl = 0        
        for i in range(2 * half_dim , np.size(mask_padded , 0) - 2 * half_dim, stride):
            for j in range(2 * half_dim , np.size(mask_padded , 1) - 2 * half_dim, stride):
                mask_value = mask_padded[i , j]
                #print(mask_value)
                if mask_value == 1:
                    #Belongs to the training tile
                    central_pixels_coord_tr_init[counter_tr , 0] = int(i)
                    central_pixels_coord_tr_init[counter_tr , 1] = int(j)
                    counter_tr += 1                    
                if mask_value == 3:
                    #Belongs to the validation tile
                    central_pixels_coord_vl_init[counter_vl , 0] = int(i)
                    central_pixels_coord_vl_init[counter_vl , 1] = int(j)
                    counter_vl += 1
                    
    # print(np.shape(central_pixels_coord_tr_init))
    # print(np.shape(central_pixels_coord_vl_init))
    
    #Refine the central pixels coordinates
    counter_tr = 0
    counter_vl = 0
    for i in range(np.size(central_pixels_coord_tr_init , 0)):
        last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        if (last_reference_value != 1) and (actual_reference_value <= 1):
            last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
            number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
            number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
            porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
            if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                counter_tr += 1
    for i in range(np.size(central_pixels_coord_vl_init , 0)):
        last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        if (last_reference_value != 1 ) and (actual_reference_value <= 1):
            last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim  + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
            number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
            number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
            porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
            if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                counter_vl += 1
            
    central_pixels_coord_tr = np.zeros((counter_tr, 2))
    central_pixels_coord_vl = np.zeros((counter_vl, 2))
    y_train_init = np.zeros((counter_tr,1))
    y_valid_init = np.zeros((counter_vl,1))
    counter_tr = 0
    counter_vl = 0
    for i in range(np.size(central_pixels_coord_tr_init , 0)):
        last_reference_value = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_tr_init[i, 0]) , int(central_pixels_coord_tr_init[i, 1])]
        if (last_reference_value != 1 ) and (actual_reference_value <= 1):
            last_reference_patch = last_reference_padded[int(central_pixels_coord_tr_init[i, 0]) - half_dim : int(central_pixels_coord_tr_init[i, 0]) + half_dim + 1, int(central_pixels_coord_tr_init[i, 1]) - half_dim : int(central_pixels_coord_tr_init[i, 1]) + half_dim + 1]
            number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
            number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
            porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
            if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                central_pixels_coord_tr[counter_tr, 0] = central_pixels_coord_tr_init[i , 0]
                central_pixels_coord_tr[counter_tr, 1] = central_pixels_coord_tr_init[i , 1]
                y_train_init[counter_tr, 0] = actual_reference_value
                counter_tr += 1
            
    for i in range(np.size(central_pixels_coord_vl_init , 0)):
        last_reference_value = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        actual_reference_value = actual_reference_padded[int(central_pixels_coord_vl_init[i, 0]) , int(central_pixels_coord_vl_init[i, 1])]
        if (last_reference_value != 1 ) and (actual_reference_value <= 1):
            last_reference_patch = last_reference_padded[int(central_pixels_coord_vl_init[i, 0]) - half_dim : int(central_pixels_coord_vl_init[i, 0]) + half_dim + 1, int(central_pixels_coord_vl_init[i, 1]) - half_dim : int(central_pixels_coord_vl_init[i, 1]) + half_dim + 1]
            number_of_last_reference_pixels_indexs = np.where(last_reference_patch == 1)
            number_of_last_reference_pixels = np.size(number_of_last_reference_pixels_indexs , 1)
            porcent_of_last_reference = (number_of_last_reference_pixels/(patch_dimension*patch_dimension))*100
            if porcent_of_last_reference <= porcent_of_last_reference_in_actual_reference:
                central_pixels_coord_vl[counter_vl, 0] = central_pixels_coord_vl_init[i , 0]
                central_pixels_coord_vl[counter_vl, 1] = central_pixels_coord_vl_init[i , 1]
                y_valid_init[counter_vl, 0] = actual_reference_value
                counter_vl += 1
    # print(np.shape(central_pixels_coord_tr))
    # print(np.shape(central_pixels_coord_vl))
    # print(np.shape(np.where(y_train_init == 0)))
    # print(np.shape(np.where(y_train_init == 1)))
    return central_pixels_coord_tr, y_train_init, central_pixels_coord_vl, y_valid_init

def Central_Pixel_Definition_For_Test(mask, last_reference, actual_reference, patch_dimension, stride, mode):
    
    if mode == 'test':
        mask_rows = np.size(mask, 0)
        mask_cols = np.size(mask, 1)
        
        half_dim = patch_dimension//2
        upper_padding = np.zeros((half_dim, mask_cols))
        left_padding = np.zeros((mask_rows + half_dim, half_dim))
        bottom_padding = np.zeros((half_dim, half_dim + mask_cols))
        right_padding = np.zeros((2 * half_dim + mask_rows, half_dim))
        
        #Add padding to the mask     
        mask_padded = np.concatenate((upper_padding, mask), axis=0)
        mask_padded = np.concatenate((left_padding, mask_padded), axis=1)
        mask_padded = np.concatenate((mask_padded, bottom_padding), axis=0)
        mask_padded = np.concatenate((mask_padded, right_padding), axis=1)
        
        #Add padding to the last reference
        last_reference_padded = np.concatenate((upper_padding, last_reference), axis=0)
        last_reference_padded = np.concatenate((left_padding, last_reference_padded), axis=1)
        last_reference_padded = np.concatenate((last_reference_padded, bottom_padding), axis=0)
        last_reference_padded = np.concatenate((last_reference_padded, right_padding), axis=1)
        
        #Add padding to the last reference
        actual_reference_padded = np.concatenate((upper_padding, actual_reference), axis=0)
        actual_reference_padded = np.concatenate((left_padding, actual_reference_padded), axis=1)
        actual_reference_padded = np.concatenate((actual_reference_padded, bottom_padding), axis=0)
        actual_reference_padded = np.concatenate((actual_reference_padded, right_padding), axis=1)

        mask = mask_padded
        last_reference = last_reference_padded
        actual_reference = actual_reference_padded
    
    #Initializing the central pixels coordinates containers
    central_pixels_coord_ts_init = []
    
    if stride == 1:
        central_pixels_coord_ts_init = np.where(mask == 2)
        central_pixels_coord_ts_init = np.transpose(np.array(central_pixels_coord_ts_init))
    else:
        print('[!] For test stride needs to be 1')     
    # print(np.shape(central_pixels_coord_tr_init))
    # print(np.shape(central_pixels_coord_vl_init))
    
    #Refine the central pixels coordinates
    counter_ts = 0
    for i in range(np.size(central_pixels_coord_ts_init , 0)):
        last_reference_value = last_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        actual_reference_value = actual_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        if (last_reference_value != 1) and (actual_reference_value <= 1):
            counter_ts += 1
            
    central_pixels_coord_ts = np.zeros((counter_ts, 2))
    y_test_init = np.zeros((counter_ts,1))
    counter_ts = 0
    for i in range(np.size(central_pixels_coord_ts_init , 0)):
        last_reference_value = last_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        actual_reference_value = actual_reference[int(central_pixels_coord_ts_init[i, 0]) , int(central_pixels_coord_ts_init[i, 1])]
        if (last_reference_value != 1 ) and (actual_reference_value <= 1):
            central_pixels_coord_ts[counter_ts, 0] = central_pixels_coord_ts_init[i , 0]
            central_pixels_coord_ts[counter_ts, 1] = central_pixels_coord_ts_init[i , 1]
            y_test_init[counter_ts, 0] = actual_reference_value
            counter_ts += 1
                
    # print(np.shape(central_pixels_coord_ts))    
    # print(np.shape(np.where(y_test_init == 0)))
    # print(np.shape(np.where(y_test_init == 1)))
    return central_pixels_coord_ts, y_test_init

def Classification_Maps(Predicted_labels, True_labels, central_pixels_coordinates, hit_map):
        
    Classification_Map = np.zeros((hit_map.shape[0], hit_map.shape[1], 3))
    TP_counter = 0
    FP_counter = 0
    for i in range(central_pixels_coordinates.shape[0]):
        
        T_label = True_labels[i]
        P_label = Predicted_labels[i]
        
        if T_label == 1:
            if P_label == T_label:
                TP_counter += 1
                #True positve
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
            else:
                #False Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0
        if T_label == 0:
            if P_label == T_label:
                #True Negative
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 255
            else:
                #False Positive
                FP_counter += 1
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),0] = 255
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),1] = 0
                Classification_Map[int(central_pixels_coordinates[i , 0]),int(central_pixels_coordinates[i , 1]),2] = 0

    return Classification_Map, TP_counter, FP_counter 
        
def plot_embedding(X, y, d, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)       
    


def adapt_domain(domain, args, path_to_weights, save_path, eval, 
    save_mat = False, save_tiff = False, de_normalize = False):
    
    patch_size = args.patches_dimension
    do = domain        
    channels = do.conc_image.shape[-1]
    if (os.path.isdir(save_path) == False):
        os.makedirs(save_path)
        print(os.path.isdir(save_path))
    if eval == True:
        full_save_path = save_path + "adapted_conc_" + args.dataset + "_eval"
    else:
        full_save_path = save_path + "adapted_conc_" + args.dataset

    G_A2B = GAN.modelGenerator(name='G_A2B_model', channels = channels)
    G_A2B.load_state_dict(torch.load(path_to_weights))
    G_A2B.eval()
    G_A2B.cuda()
    print("loading patches...")
    do.Coordinates_Domain_Adaptation(args)
    x , y = patch_extraction(utils.channels_last2first(do.conc_image), do.new_reference,
     do.no_stride_coor, patch_size)
    del y
    A_testt = x
    A_test = torch.from_numpy(A_testt).float().cuda()
    fakeB = []

    with torch.no_grad():
        for i in range(len(A_test)):
            synthetic_images_B = G_A2B(A_test[i:i+1])
            #synt_B = synthetic_images_B
            synt_B = synthetic_images_B.cpu().data.numpy()
            synt_B = utils.channels_first2last(synt_B.squeeze())
            fakeB.append(synt_B)
    
    # print (len(fakeB))
    k = 0
    shapes = do.conc_image.shape
    adapted_conc_image = np.zeros((shapes[0]//patch_size*patch_size, shapes[1]//patch_size*patch_size, shapes[2]))
    for i in range(shapes[0]//patch_size):
        for j in range(shapes[1]//patch_size):
            a = i*patch_size
            b = j*patch_size
            adapted_conc_image[a:a+patch_size, b:b+patch_size,:] =  fakeB[k]
            k += 1

    # return adapted_conc_image
    if de_normalize:
        print(np.max(adapted_conc_image))
        print(np.min(adapted_conc_image))
        shape = adapted_conc_image.shape
        image_reshaped = adapted_conc_image.reshape((shape[0] * shape[1], shape[2]))
        adapted_conc_image = do.scaler.inverse_transform(image_reshaped)
        adapted_conc_image = adapted_conc_image.reshape((shape))
        adapted_conc_image = np.rint(adapted_conc_image)
        print(np.max(adapted_conc_image))
        print(np.min(adapted_conc_image))

    
    np.save(full_save_path, adapted_conc_image)
    print("Adapted image saved on ", full_save_path)

    if save_mat:
        sio.savemat(full_save_path + '.mat', {'adapted': adapted_conc_image})
        print("Adapted image saved on ", full_save_path + '.mat')
    if save_tiff:
        print (adapted_conc_image.shape)
        tifffile.imsave(full_save_path + '.tiff', utils.channels_last2first(adapted_conc_image), photometric='rgb')
        print("Adapted image saved on ", full_save_path + '.tiff')
    

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
def patch_extraction(img, gt, coor_batch, patch_size, aug_batch = [],
    diff_reference_extract = False, diff_reference = {}):
    batch_size = len(coor_batch)
    out_img = np.zeros((batch_size,img.shape[0],patch_size,patch_size))
    # print(out_img.shape)
    out_gt = np.zeros((batch_size,patch_size,patch_size))
    if diff_reference_extract:
        out_diff = np.zeros((batch_size,diff_reference.shape[0],patch_size,patch_size))
    # print(out_gt.shape)
    if len(coor_batch.shape) > 1:
        for i in range(batch_size):
            a = int(coor_batch[i][0])
            b = int(coor_batch[i][1])
            out_img[i] = img[:,a:a+patch_size, b:b+patch_size]
            out_gt[i] = gt[a:a+patch_size, b:b+patch_size]
            
            if diff_reference_extract:
                out_diff[i] = diff_reference[:, a:a+patch_size, b:b+patch_size]
    else:
        a = int(coor_batch[0])
        b = int(coor_batch[1])
        out_img = img[:,a:a+patch_size, b:b+patch_size]
        out_gt = gt[a:a+patch_size, b:b+patch_size]
        if diff_reference_extract:
            out_diff[i] = diff_reference[:, a:a+patch_size, b:b+patch_size]
    # print(out_img.shape)
    # print(out_gt.shape)
    # print(type(aug_batch))

    if len(aug_batch) > 0:
        out_img = apply_augmentations(out_img, aug_batch)
        out_gt = apply_augmentations(out_gt, aug_batch)
        if diff_reference_extract:
            out_diff = apply_augmentations(out_diff, aug_batch)

    if diff_reference_extract:
        return out_img, out_gt, out_diff
    else:
        return out_img, out_gt


def jason(path, read_or_write, content = ''):
    with open(path, read_or_write) as f:
        if read_or_write == 'w':
            json.dump(content, f)
        if read_or_write == 'r':
            output = json.load(f)
            return output
import tensorflow as tf
import numpy as np
import torch 


def get_batch(source, target, index_src, index_tgt, opt):
    T1_coor = index_src
    T2_coor = index_tgt
    
    T1_coor = np.reshape(T1_coor, (-1, np.size(source.central_pixels_coordinates, 1)))
    T2_coor = np.reshape(T2_coor, (-1, np.size(target.central_pixels_coordinates, 1)))
    
    if opt.phase == 'train':
        T1_img = Patch_Extraction_Train(source.conc_img, T1_coor, opt.crop_size, True, 'reflect')
        T2_img = Patch_Extraction_Train(target.conc_img, T2_coor, opt.crop_size, True, 'reflect')
        
        T1_ref = Patch_Extraction_Train(source.images_diff[0], T1_coor, opt.crop_size, True, 'reflect')
        T2_ref = Patch_Extraction_Train(target.images_diff[0], T2_coor, opt.crop_size, True, 'reflect')

        if opt.use_task_loss:
            T1_class_ref = Patch_Extraction_Train(np.expand_dims(source.new_reference, axis = 2),
                T1_coor, opt.crop_size, True, 'reflect')
            # T2_class_ref = Patch_Extraction_Train(np.expand_dims(target.new_reference, axis = 2),
            #     T2_coor, opt.crop_size, True, 'reflect')
            T1_data = np.concatenate((T1_img , T1_ref, T1_class_ref), axis = 1)
            T2_data = np.concatenate((T2_img , T2_ref), axis = 1)
            # T2_data = np.concatenate((T2_img , T2_ref, T2_class_ref), axis = 1)

        else:             
            T1_data = np.concatenate((T1_img , T1_ref), axis = 1)
            T2_data = np.concatenate((T2_img , T2_ref), axis = 1)
        
    if opt.phase == 'test':
        T1_data = Patch_Extraction_Test(source.conc_img, T1_coor, opt, True)
        T2_data = Patch_Extraction_Test(target.conc_img, T2_coor, opt, False)
        
    # T1_data = T1_data[0, :, :, :]
    # T2_data = T2_data[0, :, :, :]
   
    T1_data = RemoteSensing_Transforms(T1_data, opt)
    T2_data = RemoteSensing_Transforms(T2_data, opt)  

    if opt.phase == 'train':
        T1_ref = T1_data[:, 14:21, :, :]
        T2_ref = T2_data[:, 14:21, :, :]

        T1_img = T1_data[:, :14,:, :]
        T2_img = T2_data[:, :14,:, :]

        if opt.use_task_loss:
            T1_class_ref = T1_data[:, -1,:, :]
            # T2_class_ref = T2_data[:, -1,:, :]
    else:
        T1_img = T1_data
        T2_img = T2_data
        
        
    A = torch.Tensor(T1_img)
    B = torch.Tensor(T2_img)
    if opt.phase == 'train':
        A_ref = torch.Tensor(T1_ref)
        B_ref = torch.Tensor(T2_ref)

        if opt.use_task_loss:
            A_class_ref =  torch.Tensor(T1_class_ref).long()
            # B_class_ref =  torch.Tensor(T2_class_ref).long()
    
    if opt.phase == 'test':    
        return {'A': A, 'B': B}
    else:
        if not opt.use_task_loss:
            return {'A': A, 'A_ref': A_ref, 'B': B, 'B_ref': B_ref}
        else:
            return {'A': A, 'A_ref': A_ref, 'B': B, 'B_ref': B_ref,
            'A_class_ref': A_class_ref} # , 'B_class_ref': B_class_ref}

def Patch_Extraction_Train(data, central_pixels_indexs, patch_size, padding, mode):

    half_dim = patch_size // 2
    data_rows = np.size(data, 0)
    data_cols = np.size(data, 1)
    data_depth = np.size(data, 2)
    num_samp = np.size(central_pixels_indexs , 0)

    # print ("data_rows", data_rows, "data_cols", data_cols, "data_depth", data_depth)

    patches_cointainer = np.zeros((num_samp, data_depth, patch_size, patch_size))
    # print("patches_cointainer:", patches_cointainer.shape) 

    if padding:
        if mode == 'zeros':
            upper_padding = np.zeros((half_dim, data_cols, data_depth))
            left_padding = np.zeros((data_rows + half_dim, half_dim, data_depth))
            bottom_padding = np.zeros((half_dim, half_dim + data_cols, data_depth))
            right_padding = np.zeros((2 * half_dim + data_rows, half_dim, data_depth))
            
            #Add padding to the data     
            data_padded = np.concatenate((upper_padding, data), axis=0)
            data_padded = np.concatenate((left_padding, data_padded), axis=1)
            data_padded = np.concatenate((data_padded, bottom_padding), axis=0)
            data_padded = np.concatenate((data_padded, right_padding), axis=1)
        if mode == 'reflect':
            npad = ((half_dim , half_dim) , (half_dim , half_dim) , (0 , 0))
            # print("npad", np.shape(npad))
            data_padded = np.pad(data, pad_width = npad, mode = 'reflect')
            # print("data_padded", np.shape(data_padded))
    else:
        data_padded = data
    
    for i in range(num_samp):
        patches_cointainer[i, :, :, :] = np.transpose(data_padded[int(central_pixels_indexs[i , 0]) - half_dim  : int(central_pixels_indexs[i , 0]) + half_dim , int(central_pixels_indexs[i , 1]) - half_dim : int(central_pixels_indexs[i , 1]) + half_dim , :],(2, 0, 1))
                
    return patches_cointainer


def RemoteSensing_Transforms(data, opt):
    # The transformations here were accomplished using tensorflow-cpu framework
    datum = data
    out_datum = np.zeros((datum.shape))
    # print (out_datum.shape)
    for i in range(len(datum)):
        data = datum[i]
        input_nc = np.size(data, 0)
        data = np.transpose(data, (1, 2, 0))

        if 'resize' in opt.preprocess:
            data = tf.image.resize(data, [opt.load_size, opt.load_size],
                                    method=tf.image.ResizeMethod.BICUBIC) 
                    
        if 'crop' in opt.preprocess:
            data = tf.image.random_crop(data, size=[opt.crop_size, opt.crop_size, input_nc])
            
        if not opt.no_flip:
            data = tf.image.random_flip_left_right(data)
            
        if opt.convert:
            pass
        
        out_datum[i] = np.transpose(data, (2, 0, 1))

    return out_datum
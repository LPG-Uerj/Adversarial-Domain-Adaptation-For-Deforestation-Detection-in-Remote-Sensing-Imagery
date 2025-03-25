import os
import itertools
import time
import json
import random
import numpy
import PIL.Image
import queue
import concurrent.futures
import traceback
import ctypes
import IPython
from IPython.core.display import clear_output
have_ipython = IPython.get_ipython() != None
import torch
import torch.nn as nn
# import tensorflow

from Tools import *


#hannover_path = "/home/hanletia/Dataset/tmp/data/hannover/"
#hannover = {"recpath":"/home/hanletia/Dataset/tmp/data/hannover/recpt", "evalpath":"/home/hanletia/Dataset/tmp/data/hannover/receval", "gt_path":"/home/hanletia/Dataset/tmp/data/hannover/rec", "images": ("40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55")} ), "path_back": "./tmp/data/vaih", "imagesbak": ("00", "01", "02", "03", "04")}#
vaihingen = {"path":"/home/psoto/matheus/projeto/Projeto Mestrado/Vaih/top", "recpath":"./tmp/data/vaih/rec", "gt_path":"/home/psoto/matheus/projeto/Projeto Mestrado/Vaih/gts_for_participants", "images": ("1", "3", "5", "7", "11", "13", "15", "17", "21", "23", "26", "28", "30", "32", "34", "37"), "path_back": "./tmp/data/vaih", "imagesbak": ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15")}
# vaihingen = {"path":"/content/drive/My Drive/Projeto Mestrado/Vaih/top", "recpath":"./tmp/data/vaih/rec", "gt_path":"/content/drive/My Drive/Projeto Mestrado/Vaih/gts_for_participants", "images": ("1", "3", "5", "7", "11", "13", "15", "17"), "path_back": "./tmp/data/vaih/", "imagesbak": ("00", "01", "02", "03", "04", "05", "06", "07")}# "21", "23", "26", "28", "30", "32", "34", "37"), "path_back": "/content/drive/My Drive/Projeto Mestrado/Vaih/prepross", "imagesbak": ("00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15")}
potsdam = {"path":"/home/psoto/matheus/projeto/Projeto Mestrado/Pots/3_Ortho_IRRG", "gt_path":"/home/psoto/matheus/projeto/Projeto Mestrado/Pots/GroundTruth", "images": ("2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "3_11", "3_12", "3_13", "3_14", "4_10", "4_11", "4_13", "4_14", "4_15") , "gt": ("2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "3_11", "3_12", "3_13", "3_14", "4_10", "4_11", "4_13", "4_14", "4_15") , "recpath":"./tmp/data/pots/rec",  "evalpath":"./tmp/data/pots/eval"}
# mog: 4_12 teve que ser removido porque a label_image estŕ¸Łŕ¸ em formato diferente

# potsdam = {"path":"/home/josematheus/Desktop/ProjetoHan/Pots/3_Ortho_IRRG", "gt_path":"/home/josematheus/Desktop/ProjetoHan/Pots/GroundTruth", "images": ("2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "3_11", "3_12", "3_13", "3_14", "4_10", "4_11", "4_12", "4_13", "4_14", "4_15", "5_10", "5_11", "5_12", "5_13", "5_14", "5_15", "6_7", "6_8", "6_9", "6_10", "6_11", "6_12", "6_13", "6_14", "6_15", "7_7", "7_8", "7_9", "7_10", "7_11", "7_12", "7_13"), "gt": ("2_10", "2_11", "2_12", "2_13", "2_14", "3_10", "3_11", "3_12", "3_13", "3_14", "4_10", "4_11", "4_12", "4_13", "4_14", "4_15", "5_10", "5_11", "5_12", "5_13", "5_14", "5_15", "6_7", "6_8", "6_9", "6_10", "6_11", "6_12", "6_13", "6_14", "6_15", "7_7", "7_8", "7_9", "7_10", "7_11", "7_12", "7_13"), "evalpath":"./tmp/data/pots/rec"}

uni_len = 1

main_path = "./main_data/"

main_pa = "PA/"
main_pre = main_path + main_pa
pa_past_npy = {"len": uni_len, "img": main_pre + "img_past_PA.npy" , "gt": main_pre + "gt_past_PA.npy"}
pa_present_npy = {"len": uni_len, "img": main_pre + "img_present_PA.npy" , "gt": main_pre + "gt_present_PA.npy"}

main_ro = "RO/"
main_pre = main_path + main_ro
ro_past_npy = {"len": uni_len, "img": main_pre + "img_past_RO.npy" , "gt": main_pre + "gt_past_RO.npy"}
ro_present_npy = {"len": uni_len, "img": main_pre + "img_present_RO.npy" , "gt": main_pre + "gt_present_RO.npy"}

#get_batch_lib = ctypes.CDLL("/home/josematheus/Desktop/ProjetoHan/Codigo - MA-Han2019_code/code - main/libget_batch.so")
# get_batch_lib = ctypes.CDLL("./libget_batch.so")
# get_batch_uint8 = get_batch_lib.get_batch_uint8
# get_batch_uint8.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32), 
#                       numpy.ctypeslib.ndpointer(ctypes.c_uint8), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
#                       numpy.ctypeslib.ndpointer(ctypes.c_double),
#                       numpy.ctypeslib.ndpointer(ctypes.c_int64)]
# get_batch_uint8.restype = None
# get_batch_ext_uint8 = get_batch_lib.get_batch_ext_uint8
# get_batch_ext_uint8.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32), 
#                           numpy.ctypeslib.ndpointer(ctypes.c_uint8), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
#                           numpy.ctypeslib.ndpointer(ctypes.c_double), numpy.ctypeslib.ndpointer(ctypes.c_double),
#                           numpy.ctypeslib.ndpointer(ctypes.c_int64)]
# get_batch_ext_uint8.restype = None
# get_batch_float32 = get_batch_lib.get_batch_float32
# get_batch_float32.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32), 
#                       numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
#                       numpy.ctypeslib.ndpointer(ctypes.c_double),
#                       numpy.ctypeslib.ndpointer(ctypes.c_int64)]
# get_batch_float32.restype = None
# get_batch_ext_float32 = get_batch_lib.get_batch_ext_float32
# get_batch_ext_float32.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32), 
#                           numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
#                           numpy.ctypeslib.ndpointer(ctypes.c_double), numpy.ctypeslib.ndpointer(ctypes.c_double),
#                           numpy.ctypeslib.ndpointer(ctypes.c_int64)]
# get_batch_ext_float32.restype = None


def load_image(path, resize_factor=0, use_lanczos=True, as_numpy=True):
    image = PIL.Image.open(path)
    if resize_factor > 0:
        new_size = (int(image.size[0]*resize_factor), int(image.size[1]*resize_factor))
        image = image.resize(new_size, PIL.Image.LANCZOS if use_lanczos else PIL.Image.NEAREST)
    if as_numpy:
        image = numpy.asarray(image, dtype=numpy.float32 if image.mode == "F" else numpy.uint8)
        if len(image.shape) > 2:
            assert len(image.shape) == 3
            image = channels_last2first(image)
        else:
            image = numpy.expand_dims(image, axis=0)
            assert len(image.shape) == 3 and image.shape[0] == 1
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
    return image


def channels_last2first(image):
    temp = numpy.empty((image.shape[2], image.shape[0], image.shape[1]), dtype=image.dtype)
    for c in range(image.shape[2]):
        temp[c,:,:] = image[:,:,c]
    return temp


def channels_first2last(image):
    temp = numpy.empty((image.shape[1], image.shape[2], image.shape[0]), dtype=image.dtype)
    for c in range(image.shape[0]):
        temp[:,:,c] = image[c,:,:]
    return temp


def split_data(data, time, n):
    save_path = ''
    if data+time == 'PApast':
        image, gt = load_pa_past()
        main = main_pa
        save_path = main_path + main
    elif data+time == 'ROpast':
        image, gt = load_ro_past()
        main = main_ro
        save_path = main_path + main
    elif data+time == 'PApresent':
        image, gt = load_pa_present()
        main = main_pa
        save_path = main_path + main
    elif data+time == 'ROpresent':
        image, gt = load_ro_present()
        main = main_ro
        save_path = main_path + main
    else:
        raise ValueError("There is no such data")
    div = n
    divx = gt.shape[0]//3
    divy = gt.shape[1]//3
    for i in range(div):
        if (i < div - 1):
            outimg = image[:,:, i*divy : (i+1)*divy]
            outgt = gt[:, i*divy : (i+1)*divy]
        else:
            outimg = image[:,:, i*divy :]
            outgt = gt[:, i*divy : ]
        numpy.save(save_path + 'img_' + time + '_' + data + '_' + str(i) + '.npy', outimg)
        numpy.save(save_path + 'gt_' + time + '_' + data + '_' + str(i) + '.npy', outgt)

def make_split(n):
    if uni_len < 2 :
        indexA = ['RO', 'PA']
        indexB = ['past', 'present']
        for i,j in itertools.product(indexA, indexB):
            print(i, j)
            utils.split_data(i, j, n)
    else:
        print("Adjust 'uni_len' to < 2 first")


def load_pa_past():
    n = pa_past_npy['len']
    if (n > 1):
        image = []
        ground_truth = []
        path_img = pa_past_npy["img"].split('.npy')
        path_gt = pa_past_npy["gt"].split('.npy')
        for i in range(n):
            item = '_' + str(i) + '.npy'
            image.append(numpy.load(path_img[0] + item))
            ground_truth.append(numpy.load(path_gt[0] + item))
    else:
        image = numpy.load(pa_past_npy["img"])
        ground_truth = numpy.load(pa_past_npy["gt"])
    
    return image, ground_truth

def load_pa_present():
    n = pa_present_npy['len']
    if (n > 1):
        image = []
        ground_truth = []
        path_img = pa_present_npy["img"].split('.npy')
        path_gt = pa_present_npy["gt"].split('.npy')
        for i in range(n):
            item = '_' + str(i) + '.npy'
            image.append(numpy.load(path_img[0] + item))
            ground_truth.append(numpy.load(path_gt[0] + item))
    else:
        image = numpy.load(pa_present_npy["img"])
        ground_truth = numpy.load(pa_present_npy["gt"])
    return image, ground_truth

def load_ro_past():
    n = ro_past_npy['len']
    if (n > 1):
        image = []
        ground_truth = []
        path_img = ro_past_npy["img"].split('.npy')
        path_gt = ro_past_npy["gt"].split('.npy')
        for i in range(n):
            item = '_' + str(i) + '.npy'
            image.append(numpy.load(path_img[0] + item))
            ground_truth.append(numpy.load(path_gt[0] + item))
    else:
        image = numpy.load(ro_past_npy["img"])
        ground_truth = numpy.load(ro_past_npy["gt"])
    return image, ground_truth

def load_ro_present():
    n = ro_present_npy['len']
    if (n > 1):
        image = []
        ground_truth = []
        path_img = ro_present_npy["img"].split('.npy')
        path_gt = ro_present_npy["gt"].split('.npy')
        for i in range(n):
            item = '_' + str(i) + '.npy'
            image.append(numpy.load(path_img[0] + item))
            ground_truth.append(numpy.load(path_gt[0] + item))
    else:
        image = numpy.load(ro_present_npy["img"])
        ground_truth = numpy.load(ro_present_npy["gt"])
    return image, ground_truth


def load_hannover_irg(resize_factor=0):
    images = []
    ground_truth = []
    for i in hannover["images"]:
        path = os.path.join(hannover["recpath"], f"top{i}.png")
        images.append(load_image(path, resize_factor=resize_factor))
        path = os.path.join(hannover["gt_path"], f"gt{i}.png")
        ground_truth.append(load_image(path, resize_factor = resize_factor, use_lanczos=False))
    return images, ground_truth

def load_hannover_eval(resize_factor=0):
    images = []
    ground_truth = []
    for i in hannover["images"]:
        path = os.path.join(hannover["evalpath"], f"top{i}.png")
        images.append(load_image(path, resize_factor=resize_factor))
        path = os.path.join(hannover["gt_path"], f"gt{i}.png")
        ground_truth.append(load_image(path, resize_factor = resize_factor, use_lanczos=False))
    return images, ground_truth

def load_hannover_adap(resize_factor=0):
    images = []
    ground_truth = []
    for i in hannover["images"]:
        path = os.path.join(hannover["evalpath"], f"top{i}.png")
        images.append(load_image(path, resize_factor=resize_factor))
    for i, j in itertools.product(range(4), range(4)):
        path = os.path.join(hannover_path, f"gt{i}{j}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_hannover_irgb(resize_factor=0, skip_eilenriede=False):
    images = []
    ground_truth = []
    for i, j in itertools.product(range(4), range(4)):
        if (i + 4*j) == 3 and skip_eilenriede:
            continue
        path = os.path.join(hannover_path, f"ir{i}{j}.png")
        image = load_image(path, resize_factor=resize_factor)
        path = os.path.join(hannover_path, f"rgb{i}{j}.png")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
        images.append(image)
        path = os.path.join(hannover_path, f"gt{i}{j}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth


def load_hannover_irgbd(resize_factor=0, skip_eilenriede=False):
    images = []
    ground_truth = []
    for i, j in itertools.product(range(4), range(4)):
        if (i + 4*j) == 3 and skip_eilenriede:
            continue
        path = os.path.join(hannover_path, f"ir{i}{j}.png")
        image = load_image(path, resize_factor=resize_factor)
        path = os.path.join(hannover_path, f"rgb{i}{j}.png")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        image = numpy.asarray(image, dtype=numpy.float32) / 255.0
        path = os.path.join(hannover_path, f"dom{i}{j}.tif")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        del temp_image
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
        images.append(image)
        path = os.path.join(hannover_path, f"gt{i}{j}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_potsdam_irg(resize_factor=0):
    images = []
    ground_truth = []
    n = 0
    for i in potsdam["images"]:
        path = os.path.join(potsdam["path"], f"top_potsdam_{i}_IRRG.tif")
        images.append(load_image(path, resize_factor=resize_factor))
    for j in potsdam["gt"]:
        path = os.path.join(potsdam["gt_path"], f"top_potsdam_{j}_label.tif")
        n = n+1
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_potsdam_irg2(resize_factor=0):
    images = []
    ground_truth = []
    n = 0
    m = 0
    for i in potsdam["images"]:
        path = os.path.join(potsdam["path"], f"top_potsdam_{i}_IRRG.tif")
        images.append(load_image(path, resize_factor=resize_factor))
        
    #     if m == 1:
    #         break
    #     m = m+1
    # m = 0
    for j in potsdam["gt"]:
        path = os.path.join(potsdam["gt_path"], f"gt%02i.png"%n)
        n = n+1
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
        
        # if m == 1:
        #     break
        # m = m+1
    return images, ground_truth

def load_potsdam_adap(resize_factor=0):
    images = []
    ground_truth = []
    #for i in hannover["images"]:
    print (len(potsdam["images"]))
    for i in range(0,len(potsdam["images"])):
        path = os.path.join(potsdam["recpath"], "top%02i.png"%i)
        images.append(load_image(path, resize_factor=resize_factor))
    #for i, j in itertools.product(range(4), range(4)):
        path = os.path.join(potsdam["recpath"], "gt%02i.png"%i)
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_potsdam_eval(resize_factor=0):
    images = []
    ground_truth = []
    #for i in hannover["images"]:
    path_list = os.listdir(potsdam["evalpath"])
    path_list.sort()
    # print (path_list)
    # print (len(path_list))
    for i in range(0,len(path_list)):
        #print(i)
        path = os.path.join(potsdam["evalpath"], "top%02i.png"%i)
        images.append(load_image(path, resize_factor=resize_factor))
    #for i, j in itertools.product(range(4), range(4)):
        # mog: nota, ele realmente pega os gts de recpath, ja que eles foram preparados antes
        path = os.path.join(potsdam["recpath"], "gt%02i.png"%i)
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_vaihingen_irg(resize_factor=0):
    images = []
    ground_truth = []
    n=0
    for i in vaihingen["images"]:
        path = os.path.join(vaihingen["path"], f"top_mosaic_09cm_area{i}.tif")
        images.append(load_image(path, resize_factor=resize_factor))
        path = os.path.join(vaihingen["gt_path"], f"top_mosaic_09cm_area{i}.tif")
        #path = os.path.join(vaihingen["gt_path"], f"gt0{n}.png")
        #n=n+1
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_vaihingen_irg2(resize_factor=0):
    images = []
    ground_truth = []
    for i in vaihingen["images"]:
        path = os.path.join(vaihingen["path"], f"top_mosaic_09cm_area{i}.tif")
        images.append(load_image(path, resize_factor=resize_factor))
        
    for i in vaihingen["imagesbak"]:
        path = os.path.join(vaihingen["gt_path"], f"gt{i}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
        
    return images, ground_truth

def load_vaihingenbak_irg(resize_factor=0):
    images = []
    ground_truth = []
    for i in vaihingen["imagesbak"]:
        path = os.path.join(vaihingen["path"], f"top_mosaic_09cm_area{i}.tif")
        images.append(load_image(path, resize_factor=resize_factor))
        path = os.path.join(vaihingen["gt_path"], f"top_mosaic_09cm_area{i}.tif")
        path = os.path.join(vaihingen["path_back"], f"gt{i}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth

def load_recvaihingen_irg(resize_factor=0):
    images = []
    ground_truth = []
    print(len(vaihingen["images"]))
    for i in range(0,len(vaihingen["images"])):
        # path = os.path.join(vaihingen["recpath"], f"{i}.png")
        path = os.path.join(vaihingen["recpath"], "top%02i.png"%i)
        images.append(load_image(path, resize_factor=resize_factor))
        # path = os.path.join(vaihingen["recpath"], f"gt{i}.png")
        path = os.path.join(vaihingen["recpath"], "gt%02i.png"%i)
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth
    
# def load_potsdam_adap(resize_factor=0):
#     images = []
#     ground_truth = []
#     #for i in hannover["images"]:
#     print (len(potsdam["images"]))
#     for i in range(0,len(potsdam["images"])):
#         path = os.path.join(potsdam["recpath"], "top%02i.png"%i)
#         images.append(load_image(path, resize_factor=resize_factor))
#     #for i, j in itertools.product(range(4), range(4)):
#         path = os.path.join(potsdam["recpath"], "gt%02i.png"%i)
#         ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
#     return images, ground_truth

def patch_extraction(img, gt, coor_batch, patch_size, aug_batch = [], tf = False):
    #extract in batches

    #if only_img

    batch_size = len(coor_batch)
    if tf:
        out_img = numpy.zeros((batch_size,patch_size,patch_size, img.shape[2]))
    else:
        out_img = numpy.zeros((batch_size,img.shape[0],patch_size,patch_size))
    # print(out_img.shape)
    out_gt = numpy.zeros((batch_size,patch_size,patch_size))
    # print(out_gt.shape)

    for i in range(batch_size):
        a = int(coor_batch[i][0])
        b = int(coor_batch[i][1])
        if tf:
            out_img[i] = img[a:a+patch_size, b:b+patch_size,:]
        else:
            out_img[i] = img[:,a:a+patch_size, b:b+patch_size]
        out_gt[i] = gt[a:a+patch_size, b:b+patch_size]

    # print(out_img.shape)
    # print(out_gt.shape)
    # print(type(aug_batch))

    if len(aug_batch) > 0:
        out_img = apply_augmentations(out_img, aug_batch, tf = tf)
        out_gt = apply_augmentations(out_gt, aug_batch)

    return out_img, out_gt

def draw_bootstrap_sets(sample_range, num_sets=10, samples_per_set=10):
    bootstrap = {}
    for i in range(num_sets):
        training_set = []
        for j in range(samples_per_set):
            training_set.append(random.randrange(sample_range))
        training_set = tuple(sorted(training_set))
        test_set = tuple([x for x in range(sample_range) if not x in training_set])
        bootstrap[i] = (training_set, test_set)
    return bootstrap


def load_bootstrap_sets(path):
    bootstrap = {}
    with open(path, "r") as f:
        temp = json.load(f)
        for key, values in temp.items():
            training_set = tuple([int(x) for x in values[0]])
            test_set = tuple([int(x) for x in values[1]])
            bootstrap[int(key)] = (training_set, test_set)
    return bootstrap


def get_transform(h_flip=False, v_flip=False, x_shear=0, y_shear=0, rotation=0, tx=0, ty=0):
    transform = numpy.eye(3, dtype=numpy.float64)
    #TODO: this is a very hacky way of implementing flipping and should probably not be used
    if h_flip: # horizontal flipping
        transform[0,2] = 1
    if v_flip: # vertical flipping
        transform[1,2] = 1
    if x_shear != 0: # horizontal shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,1] = numpy.tan(x_shear*numpy.pi/180)
        transform = numpy.matmul(temp, transform)
    if y_shear != 0: # vertical shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[1,0] = numpy.tan(y_shear*numpy.pi/180)
        transform = numpy.matmul(temp, transform)
    if rotation != 0: # rotation
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,2] = -0.5
        temp[1,2] = -0.5
        transform = numpy.matmul(temp, transform)
        alpha = rotation * numpy.pi / 180
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,0] = numpy.cos(alpha)
        temp[1,0] = numpy.sin(alpha)
        temp[0,1] = -temp[1,0]
        temp[1,1] = temp[0,0]
        transform = numpy.matmul(temp, transform)
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,2] = 0.5
        temp[1,2] = 0.5
        transform = numpy.matmul(temp, transform)
    if tx != 0 or ty != 0: # translation ("cropping")
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,2] = tx
        temp[1,2] = ty
        transform = numpy.matmul(temp, transform)
    if not transform.flags["C_CONTIGUOUS"]:
        transform = numpy.ascontiguousarray(transform)
    return transform


def get_random_transform(h_flip=False, v_flip=False, x_shear_range=0, y_shear_range=0, rotation_range=0, tx_range=0, ty_range=0):
    # best for Hannover (17.04.2019): no flipping, shearing range 16, rotation range 45, translation range 1
    params = {}
    params["h_flip"] = h_flip and random.random() < 0.5
    params["v_flip"] = v_flip and random.random() < 0.5
    params["x_shear"] = random.uniform(-x_shear_range, x_shear_range)
    params["y_shear"] = random.uniform(-y_shear_range, y_shear_range)
    params["rotation"] = random.uniform(-rotation_range, rotation_range)
    params["tx"] = random.uniform(0, tx_range)
    params["ty"] = random.uniform(0, ty_range)
    return get_transform(**params)
      

def get_nested_transform(x_shear=0, y_shear=0, rotation=0, tx=0, ty=0, scaling=0):
    transform = numpy.eye(3, dtype=numpy.float64)
    if x_shear != 0: # horizontal shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,1] = numpy.tan(x_shear*numpy.pi/180)
        transform = numpy.matmul(temp, transform)
    if y_shear != 0: # vertical shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[1,0] = numpy.tan(y_shear*numpy.pi/180)
        transform = numpy.matmul(temp, transform)
    if rotation != 0: # rotation
        alpha = rotation * numpy.pi / 180
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,0] = numpy.cos(alpha)
        temp[1,0] = numpy.sin(alpha)
        temp[0,1] = -temp[1,0]
        temp[1,1] = temp[0,0]
        transform = numpy.matmul(temp, transform)
    if tx != 0 or ty != 0: # translation ("cropping")
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,2] = tx
        temp[1,2] = ty
        transform = numpy.matmul(temp, transform)
    if scaling != 0: # scaling
        if scaling > 0:
            scaling += 1
        else:
            scaling = 1 / (1 - scaling)
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0,0] = scaling
        temp[1,1] = scaling
        transform = numpy.matmul(temp, transform)
    if not transform.flags["C_CONTIGUOUS"]:
        transform = numpy.ascontiguousarray(transform)
    return transform


def get_random_nested_transform(x_shear_range=0, y_shear_range=0, rotation_range=0, tx_range=0, ty_range=0, scaling_range=0):
    params = {}
    params["x_shear"] = random.uniform(-x_shear_range, x_shear_range)
    params["y_shear"] = random.uniform(-y_shear_range, y_shear_range)
    params["rotation"] = random.uniform(-rotation_range, rotation_range)
    params["tx"] = random.uniform(-tx_range, tx_range)
    params["ty"] = random.uniform(-ty_range, ty_range)
    params["scaling"] = random.uniform(-scaling_range, scaling_range);
    return get_nested_transform(**params)
      

def get_mini_batch(mini_batch_size, patch_size, image, ground_truth, num_classes, transforms, nested_transforms=None):
    assert image.shape[1] == ground_truth.shape[0]
    assert image.shape[2] == ground_truth.shape[1]
    assert transforms.shape[0] == mini_batch_size
    assert nested_transforms is None or nested_transforms.shape[0] == mini_batch_size
    shape_info = (mini_batch_size, patch_size, patch_size, image.shape[1], image.shape[2], image.shape[0], num_classes)
    shape_info = numpy.ascontiguousarray(shape_info, dtype=numpy.int64)
    mini_batch = numpy.empty((mini_batch_size, image.shape[0], patch_size, patch_size), dtype=numpy.float32)
    if not mini_batch.flags["C_CONTIGUOUS"]:
        mini_batch = numpy.ascontiguousarray(mini_batch)
    mini_batch_gt = numpy.zeros((mini_batch_size, patch_size, patch_size), dtype=numpy.int32)
    if not mini_batch_gt.flags["C_CONTIGUOUS"]:
        mini_batch_gt = numpy.ascontiguousarray(mini_batch_gt)
    if nested_transforms is None:
        get_batch = get_batch_uint8 if image.dtype == numpy.uint8 else get_batch_float32
        get_batch(mini_batch, mini_batch_gt, image, ground_truth, transforms, shape_info)
    else:
        get_batch = get_batch_ext_uint8 if image.dtype == numpy.uint8 else get_batch_ext_float32
        get_batch(mini_batch, mini_batch_gt, image, ground_truth, transforms, nested_transforms, shape_info)
    
    #print (mini_batch_gt)
    #print (numpy.array_equal(mini_batch_gt,numpy.zeros(mini_batch_gt.shape)))
    #print(numpy.array_equal(mini_batch_gt[0],mini_batch_gt[1]))
    #print (mini_batch_gt.shape)
    #print(mini_batch_gt[0][0:10,0:10])
    #print (mini_batch.shape)
    return mini_batch, mini_batch_gt


def get_validation_data(images, ground_truth, subset, patch_size, num_classes):
    num_patches = 0
    print (subset)
    for index in subset:
        image = images[index]
        hor_patches = ((image.shape[1]-1) // patch_size) + 1
        ver_patches = ((image.shape[2]-1) // patch_size) + 1
        num_patches += (hor_patches * ver_patches)
    val_images = numpy.empty((num_patches, images[0].shape[0], patch_size, patch_size), dtype=numpy.float32)
    val_gt = numpy.empty((num_patches, patch_size, patch_size), dtype=numpy.int32)
    offset = 0
    scale_factor = (1.0 / 255.0) if images[0].dtype == numpy.uint8 else 1.0 # assumes that either all images are of type uint8 or all images are of type float32
    print(f"scale_factor = {scale_factor}")
    for index in subset:
        image = images[index]
        gt = ground_truth[index]
        hor_patches = ((image.shape[1]-1) // patch_size) + 1
        ver_patches = ((image.shape[2]-1) // patch_size) + 1
        for x, y in itertools.product(range(hor_patches), range(ver_patches)):
            left = x * patch_size if (x+1)*patch_size < image.shape[1] else image.shape[1] - patch_size
            top = y * patch_size if (y+1)*patch_size < image.shape[2] else image.shape[2] - patch_size
            val_images[offset,:,:,:] = image[:,left:left+patch_size,top:top+patch_size] * scale_factor
            val_gt[:,:] = gt[left:left+patch_size,top:top+patch_size]
            offset += 1
    #print(val_gt.shape)
    #print(val_gt[2][0:10,0:10])
    #print(numpy.array_equal(val_gt, numpy.zeros(val_gt.shape)))
    #print(val_images.shape)
    return val_images, val_gt


def get_num_slices(data, slice_size):
    assert data.shape[0] > 0
    return ((data.shape[0]-1) // slice_size) + 1


def get_slice(data, slice_index, slice_size):
    if isinstance(data, tuple) or isinstance(data, list):
        l = len(data)
        for i in range(1, l):
            assert data[0].shape[0] == data[i].shape[0]
        return [get_slice(data[i], slice_index, slice_size) for i in range(l)]
    else:
        begin = slice_index * slice_size
        end = (slice_index + 1) * slice_size
        if end > data.shape[0]:
            end = data.shape[0]
        index = [slice(begin,end)]
        for i in range(1, len(data.shape)):
            index.append(slice(0, data.shape[i]))
        return data.__getitem__(tuple(index))
    
    
def slice_generator(data, slice_size):
    if isinstance(data, tuple) or isinstance(data, list):
        l = len(data)
        for i in range(1, l):
            assert data[0].shape[0] == data[i].shape[0]
        num_slices = get_num_slices(data[0], slice_size)
        for i in range(num_slices):
            yield [get_slice(data[j], i, slice_size) for j in range(l)]
    else:
        num_slices = get_num_slices(data, slice_size)
        for i in range(num_slices):
            yield get_slice(data, i, slice_size)


class ModelFitter:
    '''
        Class responsible for the definition of the training steps.
        It is used as an interface here and then better defined by each model training.
    '''
    def __init__(self, num_epochs, num_mini_batches, shuffle=True, output_path=None, history_filename="history.json", max_queue_size=4, do_not_clear_output=False):
        self.early_stop = False
        self.early_stop_count = 0
        self.start_epoch = 0

        self.early_skip = False
        
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.shuffle = shuffle
        self.output_path = output_path
        self.history_filename = history_filename
        self.max_queue_size = (max_queue_size - 1) if max_queue_size > 1 else 1
        self.do_not_clear_output = do_not_clear_output
        
        
    def initialize(self):
        '''
            Run 
        '''       
        pass
    
    def pre_epoch(self, epoch):
        pass
    
    def get_batch(self, epoch, batch, batch_data):
        pass
    
    def train(self, epoch, batch, batch_data, metrics, iteration):
        pass
    
    def post_epoch(self, epoch, metrics):
        pass
    
    def finalize(self):
        pass

    def fit(self):
        self._output = []
        self.history = {}
        self.initialize()
        for epoch in range(self.start_epoch, self.num_epochs):
            if self.early_stop:
                print("Early Stop by F1-score in epoch: ", epoch)
                break
            if self.early_skip:
                if (self.early_skip_count == self.early_skip_limit) and epoch < self.epoch_for_next_model:
                    print("Early Skipping...")
                    # epoch = self.epoch_for_next_model - 1
                    continue
            #self.scheduler.step()
            epoch_timestamp = time.perf_counter()
            self.pre_epoch(epoch)
            # mog: não é mais usado
            batch_ids = numpy.arange(self.num_mini_batches)
            if self.shuffle:
                numpy.random.shuffle(batch_ids)
            # mog: fim
            self._fit_epoch(epoch, batch_ids)
            metrics = {key: numpy.mean(self.history[key][-self.num_mini_batches:]) for key in self._metrics_keys}
            index = len(self._output)
            self.post_epoch(epoch, metrics)
            for key, value in metrics.items():
                if key in self._metrics_keys:
                    continue
                if not key in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            if self.output_path != None and self.history_filename != None:
                filename = f"{self.output_path}/{self.history_filename}"
                self.print(f"saving history to '{filename}'...")
                with open(filename, "w") as f:
                    json.dump(self.history, f)
            epoch_timestamp = time.perf_counter() - epoch_timestamp
            self._progress(epoch, -index if index > 0 else -1, epoch_timestamp, metrics)
            
        self.finalize()
        
    def print(self, s):
        self._output.append(s)
        print(s)
        
    def _fit_epoch(self, epoch, batch_ids):
        self._metrics_keys = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                q = queue.Queue(maxsize=self.max_queue_size)
                executor.submit(self._get_batches, epoch, batch_ids, q)
                for iteration, batch in enumerate(batch_ids):
                    batch_timestamp = time.perf_counter()
                    metrics = {}
                    self.train(epoch, batch, q.get(), metrics, iteration)
                    batch_timestamp = time.perf_counter() - batch_timestamp
                    for key, value in metrics.items():
                        if not key in self._metrics_keys:                            
                            self._metrics_keys.add(key)
                        if not key in self.history:
                            self.history[key] = []
                        values = self.history[key]
                        values.append(value)
                        metrics[key] = numpy.mean(values[-(iteration+1):])
                    self._progress(epoch, iteration, batch_timestamp, metrics)
            except:
                traceback.print_exc()
                                
    def _get_batches(self, epoch, batch_ids, q):
        try:
            for batch in batch_ids:
                data = []
                self.get_batch(epoch, batch, data)
                q.put(data)
        except:
            traceback.print_exc()

    def _progress(self, epoch, iteration, elapsed_time, metrics):
        s = f"epoch = {epoch+1}/{self.num_epochs}"
        if iteration >= 0:
            s = f"{s}, iteration = {iteration+1}/{self.num_mini_batches}"
        if elapsed_time < 1:            
            elapsed_time *= 1000
            unit = "ms"
            if elapsed_time < 1:
                elapsed_time *= 1000
                unit = "us"
            elapsed_time = round(elapsed_time)
            s = f"{s}, time = {elapsed_time}{unit}"
        elif elapsed_time < 60:
            s = f"{s}, time = {elapsed_time:.2f}s"
        else:
            elapsed_time /= 60
            unit = "min"
            if elapsed_time >= 60:
                elapsed_time /= 60
                unit = "h"
                if elapsed_time >= 24:
                    elapsed_time /= 24
                    unit = "d"
            s = f"{s}, time = {elapsed_time:.1f}{unit}"
        for key, value in metrics.items():
            s = f"{s}, {key} = {value}"
        if have_ipython and not self.do_not_clear_output:
            if iteration == 0:
                self._output.append(s)
            elif iteration < 0:
                self._output[(-iteration)-1] = s
            else:
                self._output[-1] = s
            clear_output(True)
            for s in self._output:
                print(s)
        else:
            print(s if iteration >= 0 else f"\n\n{s}\n\n")

            
def predict(net, data, mini_batch_size, as_float=True):
    results = []
    device = next(net.parameters()).device
    with torch.no_grad():
        for x in slice_generator(data, mini_batch_size):
            x = torch.from_numpy(x)
            if as_float:
                x = x.float()
            x = x.to(device)
            results.append(net(x).cpu())
    return torch.cat(results, 0).numpy()


def reduce_iterations_to_epochs(data, num_mini_batches, reduce_fn=numpy.mean):
    assert len(data) % num_mini_batches == 0
    num_epochs = len(data) // num_mini_batches
    return [reduce_fn(data[epoch*num_mini_batches:(epoch+1)*num_mini_batches]) for epoch in range(num_epochs)]


def load_deep_lab_v3p_weights(net, input_map=None):
    def collect_layers(net):
        result = []
        for layer in net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                result.append(layer)
            elif isinstance(layer, nn.ModuleList) or isinstance(layer, nn.Sequential):
                result.extend(collect_layers(layer))
        return result
    
    layers = collect_layers(net.part1)
    layers.extend(collect_layers(net.part2))
    
    mapping = {
        "MobilenetV2/Conv":0,
        "MobilenetV2/Conv/BatchNorm":1,
        "MobilenetV2/expanded_conv/depthwise":4,
        "MobilenetV2/expanded_conv/depthwise/BatchNorm":5,
        "MobilenetV2/expanded_conv/project":6,
        "MobilenetV2/expanded_conv/project/BatchNorm":7
    }
    
    for index in range(1, 17):
        target = len(mapping) + 2
        mapping[f"MobilenetV2/expanded_conv_{index}/expand"] = target
        mapping[f"MobilenetV2/expanded_conv_{index}/expand/BatchNorm"] = target + 1
        mapping[f"MobilenetV2/expanded_conv_{index}/depthwise"] = target + 2
        mapping[f"MobilenetV2/expanded_conv_{index}/depthwise/BatchNorm"] = target + 3
        mapping[f"MobilenetV2/expanded_conv_{index}/project"] = target + 4
        mapping[f"MobilenetV2/expanded_conv_{index}/project/BatchNorm"] = target + 5
        
    conv_shape_map = {2: 1, 3: 0, 0: 2, 1: 3}
    depthwise_conv_shape_map = {2: 0, 3: 1, 0: 2, 1: 3}
        
    # download URL for pre-trained weights: https://github.com/qixuxiang/deeplabv3plus/blob/master/model/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz
    # expected MD5: 4020e71ab31636648101b000cca3b6b4
    # expected SHA-256: 2b7fe43461c2d9d56b3ed6baed5547fc066361ba12ab84c8e85f0c31914c034f
    reader = tensorflow.compat.v1.train.NewCheckpointReader("/home/psoto/matheus/projeto/Projeto Mestrado/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000")    

    for key, target in mapping.items():
        layer = layers[target]
        if "BatchNorm" in key:
            for torch_name, tf_name in (("weight", "gamma"), ("bias", "beta"), ("running_mean", "moving_mean"), ("running_var", "moving_variance")):
                pre_trained_tensor = reader.get_tensor(f"{key}/{tf_name}")
                tensor = layer._parameters.get(torch_name, None)
                tensor = layer._buffers.get(torch_name, tensor)
                if pre_trained_tensor.shape[0] != tensor.shape[0]:
                    raise RuntimeError(f"tensor shape mismatch for {key}/{tf_name}: {pre_trained_tensor.shape} vs. {tensor.shape}")
                tensor = torch.from_numpy(pre_trained_tensor).float().to(tensor.device)
                layer.__setattr__(torch_name, tensor if "running" in torch_name else nn.Parameter(tensor))
        else:
            weights_string = "weights" if layer.groups == 1 else "depthwise_weights"
            shape_map = conv_shape_map if layer.groups == 1 else depthwise_conv_shape_map
            pre_trained_tensor = reader.get_tensor(f"{key}/{weights_string}")
            tensor = layer.weight
            # assume square kernels
            assert pre_trained_tensor.shape[0] == pre_trained_tensor.shape[1]
            assert tensor.shape[2] == tensor.shape[3]
            for i, j in shape_map.items():
                if pre_trained_tensor.shape[i] != tensor.shape[j]:
                    raise RuntimeError(f"tensor shape mismatch for {key}/{weights_string}: {pre_trained_tensor.shape} vs. {tensor.shape}")
            weights = numpy.empty(tensor.shape)
            for i, j in itertools.product(range(pre_trained_tensor.shape[2]), range(pre_trained_tensor.shape[3])):
                s = slice(0, tensor.shape[3])
                index = [s, s, s, s]
                index[shape_map[2]] = i if target != 0 or not input_map else input_map[i]
                index[shape_map[3]] = j
                weights.__setitem__(tuple(index), pre_trained_tensor.__getitem__((s, s, i, j)))
            layer.weight = nn.Parameter(torch.from_numpy(weights).float().to(tensor.device))

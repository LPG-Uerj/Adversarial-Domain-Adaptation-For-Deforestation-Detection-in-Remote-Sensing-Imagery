import argparse
import os
import torch

from model_architectures import DeepLabV3plus

from parameters.training_parameters import Evaluate_cyclegan, Global
from utilities import training_utils, utils

os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def parser_call():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-source_model", type=str, help="Source model to use for evaluation.", required=True)    
    
    return parser.parse_args()


def evaluate(source, source_parameters, parameters: Evaluate_cyclegan, global_parameters: Global, gan_models_path, source_model):
    
    best_source_model = source_model
    
    print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())
    device = torch.device("cuda:0")
    source_domain = global_parameters.source_domain
    IMAGE_SAVE_PATH = parameters.savedata_folder + source_domain + '/'

    path_pre = []
    path_pre.append(IMAGE_SAVE_PATH)
    print("make dir")
    for i in range (len(path_pre)):
        p = path_pre[i]
        if (os.path.isdir(p) == False):
            os.makedirs(p)
            print('\''+ p + '\'' + " path created")

    patch_size = parameters.patch_size

    starting_model_index = parameters.starting_model_index


    # dilation_rates = parameters.dilation_rates
    generator_prefix = source_parameters.generator_type
    output_path = gan_models_path
    model_names = [name for name in os.listdir(output_path) if
                   os.path.splitext(name)[1] == '.pt' and name.split('_')[1] == generator_prefix]
    model_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print(model_names)

    # epochs_per_model = parameters.epochs_per_model
    mini_batch_size = parameters.batch_size
    # num_mini_batches = parameters.num_batches
    # iteration_modifier = parameters.iteration_modifier
    # if iteration_modifier < 1:
    #     iteration_modifier = 1
    # # num_epochs = epochs_per_model*(len(model_names)-1 - starting_model_index)
    # # print("[*] Total number of epochs:", num_epochs)

    global best_gan_epoch
    best_gan_epoch = ''


    print("loading datasets...")
    # validation = source.central_pixels_coor_vl
    # num_mini_batches = int(train_set.shape[0]//mini_batch_size//iteration_modifier)
    # validation = torch.utils.data.DataLoader(source.central_pixels_coor_vl, 
    #             batch_size = mini_batch_size)
    

    print("loading source Network...")
    source_deep_lab_v3p = DeepLabV3plus.create(parameters)
    source_deep_lab_v3p = source_deep_lab_v3p.to(device) # source semantic model
    checkpoint1 = torch.load(best_source_model)
    source_deep_lab_v3p.load_state_dict(checkpoint1)
    source_deep_lab_v3p.eval()

    def changedataset(path_to_weights):
        print("loading adapted datasets...")
        source.Prepare_GAN_Set(source_parameters, path_to_weights, eval = True, 
            adapt_back_to_original = True)


    print("Starting evaluation...")
    metrics = {}
    best_f1 = -1
    for i in range (starting_model_index, len(model_names)):
        path_to_weights = os.path.join(output_path, model_names[i])
        running_model = model_names[i]
        running_model_epoch = model_names[i].split('_')[-1].split('.')[0]
        
        changedataset(path_to_weights)
        validation_set = torch.utils.data.DataLoader(source.central_pixels_coor_vl, 
            batch_size = mini_batch_size)
        source_image = utils.channels_last2first(source.adapted_image)
        source_gt = source.new_reference
        # img[source_domain] = utils.channels_last2first(source.adapted_image)
        # gt[source_domain] = source.new_reference
        # acc, f1, rec, prec, alert = 0, 0, 0, 0, 0
        f1_score = 0
        count = 0

        for coordinate in validation_set:
            count += 1
            x , y = training_utils.patch_extraction(source_image, source_gt, coordinate, patch_size)
            # batch_data.append(x)
            # batch_data.append(y)
            # print(x.shape)
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).long().to(device)
            x = source_deep_lab_v3p(x)
            
            x = x.argmax(1)
            x = x.cpu()
            y = y.cpu()
            
            acc, f1, rec, prec, alert = training_utils.return_metrics(y,x)
            # metrics["seg_acc"] += acc
            # metrics["seg_f1"] += f1
            # metrics["seg_rec"] += rec
            # metrics["seg_prec"] += prec
            # metrics["seg_alert"] += alert
            # acc += acc
            f1_score += f1
            # rec += rec
            # prec += prec
            # alert += alert

        f1_score = f1_score/count
        print(f1_score)
        if f1_score > best_f1:
            best_f1 = f1_score
            best_gan_epoch = running_model_epoch
            print("Updating best F1-score:", f1_score)
            print("Updating best epoch:", best_gan_epoch)
    
    print("Best F1-score:", best_f1)
    print("Best epoch:", best_gan_epoch)
    print("Best Domain Adaptation model:", path_to_weights)

    return path_to_weights


from dataset_preprocessing.dataset_select import select_domain
if __name__=='__main__':
    # def evalutate_cyclegan_model(source, source_args, train_args, global_args, gan_models_path):
    global_parameters = Global()
    eval_parameters = Evaluate_cyclegan()
    
    args = parser_call()
    assert os.path.isfile(args.source_model)
    
    source, source_params = select_domain(global_parameters.source_domain)
    target, target_params = select_domain(global_parameters.target_domain)
    
    gan_models_path = Global.cyclegan_models_path
    
    source_params.stride = source_params.train_source_stride
    training_utils.ready_domain(source, source_params, train_set = True, augmented = True)

    best_gan_path = evaluate(source, source_params, 
        eval_parameters, global_parameters, gan_models_path, args.source_model)
    
    txt = "Best Domain Adaptation Model:"
    utils.log_best_model(txt, best_gan_path)
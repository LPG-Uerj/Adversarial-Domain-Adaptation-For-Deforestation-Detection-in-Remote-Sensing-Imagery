import argparse
import json
import numpy as np
import os
from scipy.special import softmax
import skimage # para o prodes
import torch

from model_architectures import DeepLabV3plus
from utilities import training_utils, utils
from parameters.training_parameters import Test_models, Global

os.environ["OMP_NUM_THREADS"] = "7" # 7 OpenMP threads + 1 Python thread = 800% CPU util.
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("NUMBER OF CUDA DEVICES: ", torch.cuda.device_count())


def parser_call():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("-models", type=str, nargs='*', help="Classification models that you wanna test for later comparison", required=True)    
    parser.add_argument("-use_domain", type=str, nargs='*', help="Flag for domain test. \'source\' for source model, \'gt_target\' for supervised target model, \'adapted_target\' for model trained using the domain adaptation", required=True)    
    
    return parser.parse_args()

def check_args(args):
    models = args.models
    domains = args.use_domain
    
    for m in models: assert os.path.isfile(m)
    assert len(models) == len(domains)
    
    model_grouping = {}
    model_grouping['source_group'] = []
    model_grouping['adapted_target_group'] = []
    for i in range(len(models)):
        key = 'source_group' if domains[i] == 'source' else 'adapted_target_group'
        model_grouping[key].append(models[i])
        
    print(model_grouping)
    
    return model_grouping
    


def test(source, target, parameters: Test_models, global_args: Global, models, file_name=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print(models)

    output_path = parameters.output_path

    path_pre = []
    path_pre.append(output_path)
    print("make dir")
    for i in range (len(path_pre)):
        p = path_pre[i]
        if (os.path.isdir(p) == False):
            os.makedirs(p)
            print('\''+ p + '\'' + " path created")

    patch_size = parameters.patch_size
    num_classes = parameters.num_classes
    channels = parameters.channels

    mini_batch_size = parameters.batch_size

    # large_latent_space = True
    # dilation_rates = ()

    run_test = True
    prodes = parameters.prodes_polygon_removal
    # test_only_deforestation = False

    # for threshold
    change_threshold = parameters.change_threshold
    number_of_points = parameters.number_of_points

    if hasattr(parameters, 'FS_on_adapted_target_only'):
        FS_on_adapted_target_only = parameters.FS_on_adapted_target_only
    else:
        FS_on_adapted_target_only = False
    if hasattr(parameters, 'FS_on_target_only'):
        FS_on_target_only = parameters.FS_on_target_only
    else:
        FS_on_target_only = False
    if hasattr(parameters, 'FS_on_source_only'):
        FS_on_source_only = parameters.FS_on_source_only
    else:
        FS_on_source_only = False

    test_only_on_proper_domain = parameters.test_only_on_proper_domain
    test_FS_models = parameters.test_FS_models
    test_FS_on_both_domains = parameters.test_FS_on_both_domains



    # output_path = f"./class_output"
    # if (os.path.isdir(output_path) == False):
    # 	os.makedirs(output_path)

    # validation_check = False #Change Path

    # source_domain = global_args.source_domain
    # target_domain = global_args.target_domain


    # print("loading data...")

    # domains = {}
    # domains[source_domain] = source
    # domains[target_domain] = target

    file = open(os.path.join(output_path, file_name.split('.')[0] + '_' + 
        "test_metrics.txt"), 'w')
    print(file)



    domains = []
    domains.append(source)
    domains.append(target)

    if test_only_on_proper_domain:
        source_name = global_args.source_domain
        target_name = global_args.target_domain
        model_to_domain_table = {}
        model_to_domain_table["adapted_target_group"] = target_name
        domain_name = {}
        domain_name[str(type(target).__name__)] = target_name
        domain_name[str(type(source).__name__)] = source_name

        model_to_domain_table["source_group"] = 'dont'
        model_to_domain_table["gt_target_group"] = 'dont'	

        if test_FS_models:
            model_to_domain_table["source_group"] = source_name
            model_to_domain_table["gt_target_group"] = target_name			

            if  test_FS_on_both_domains:
                model_to_domain_table["source_group"] = 'both'
                model_to_domain_table["gt_target_group"] = 'both'

    if FS_on_adapted_target_only:
        model_to_domain_table = {}
        model_to_domain_table["source_group"] = target_name
        model_to_domain_table["gt_target_group"] = 'dont'
    if FS_on_target_only:
        model_to_domain_table["source_group"] = target_name
        model_to_domain_table["gt_target_group"] = 'dont'
    if FS_on_source_only:
        model_to_domain_table["source_group"] = source_name
        model_to_domain_table["gt_target_group"] = 'dont'	

    print("defining segmentation network...")


    def patch_data(img, isgt = False):
        cut = patch_size
        if (isgt):
            h = img.shape[0]//cut
            w = img.shape[1]//cut
            channel = 1 
            output_img = np.zeros((h*w, cut, cut))
            
        else:
            h = img.shape[1]//cut
            w = img.shape[2]//cut
            channel = channels
            output_img = np.zeros((h*w, channel, cut, cut))
            
        # print(output_img.shape)
        n = 0
        for i in range (h):
            for j in range (w):
                top = cut * i
                bottom = cut * (i+1)
                left = cut * j
                right = cut * (j+1)
                if isgt:
                    newImg = img[top:bottom,left:right]
                else:
                    newImg = img[:channel, top:bottom,left:right]
                output_img[n] = newImg
                n = n+1
        if(isgt):
            return output_img
        return output_img, h, w


    def rec_gt(patches, h, w):
        # print(patches[0].shape)
        cut = patch_size
        rec = np.zeros((h*cut, w*cut))
        n = 0
        # print(rec.shape)
        for i in range (h):
            for j in range(w):
                top = cut * i
                bottom = cut * (i+1)
                left = cut * j
                right = cut * (j+1)
                rec[top:bottom,left:right] = patches[n]
                n = n+1
        return rec

    def raster_gt(gt):
        new_gt = np.zeros((gt.shape[0],gt.shape[1],3))
        masks = []
        masks.append((gt == 0))
        masks.append((gt == 1))
        # masks.append((gt == 2))
        # masks.append((gt == 3))
        # masks.append((gt == 4))
        # masks.append((gt == 5))

        new_gt[:][masks[0]] = [0,0,0]
        new_gt[:][masks[1]] = [255,255,255]
        # new_gt[:][masks[2]] = [0,255,255]
        # new_gt[:][masks[3]] = [0,255,0]
        # new_gt[:][masks[4]] = [255,255,0]
        # new_gt[:][masks[5]] = [255,0,0] 

        return new_gt

    def calcula_metricass(ref, pred, patch_size):
      total = 0
      fp, tp = 0, 0
      fn, tn = 0, 0
      c2 = 2
      c1 = 1
      c0 = 0

    #  result_image = np.zeros((9, 368,520, 3), dtype = np.float32)

      for k in range(len(ref)):
        for i in range(patch_size): #patch_size ou tile_size
          for j in range(patch_size):  #patch_size ou tile_size
            if(ref[k,i,j]!=c2):
              if(ref[k,i,j]==c1):
                if(pred[k,i,j]==1):
                  tp = tp+1
          #        result_image[k,i,j] = [0, 255, 255 ] 
                else:
                  fn = fn+1
           #       result_image[k,i,j] = [255 ,0 , 0 ] 
              elif(ref[k,i,j]==c0):
                if(pred[k,i,j]==0):
                  tn = tn+1
             #     result_image[k,i,j] = [255, 255, 255 ] 
                else:
                  fp = fp+1
             #     result_image[k,i,j] = [0, 0 , 255 ] 
            else:
            #  result_image[k,i,j] = [255, 255, 255 ] 
              total = total+1
      #  cv2.imwrite("resultados/dissertacao/DLCD-4/4tiles/result"+str(k)+".png", result_image[k])
              
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


    # def test():
    # 	torch.utils.data.DataLoader(pa.central_pixels_coor_ts, batch_size = mini_batch_size)


    print("starting classification...")
    if run_test:
        print("Running test... ")
    if prodes:
        print("Prodes ON")
    else:
        print("Prodes OFF")

    if change_threshold:
        min_array = np.zeros((1 , ))
        threshold_list = np.arange(1.0, 0, -1.0/(number_of_points - 1))
        threshold_list = np.concatenate((threshold_list , min_array))
        print(threshold_list)
        print("list size:", len(threshold_list))


    metrics = {}
    for domain in (domains):
        this_domain_type = str(type(domain).__name__)

        print("domain: ", this_domain_type)
        

        images = {}
        if not FS_on_adapted_target_only:
            images["original"] = utils.channels_last2first(domain.conc_image)
        gt = domain.new_reference

        if parameters.test_transforms:
            images["adapted"] = utils.channels_last2first(domain.adapted_image)

        metrics[domain] = {}
        test_loader = domain.central_pixels_coor_ts
        total_patches = len(domain.central_pixels_coor_ts)
        test_loader = torch.utils.data.DataLoader(domain.central_pixels_coor_ts, batch_size = mini_batch_size)
        
        for image in images:
            print(f"--- {image} image ---")
            
            img = images[image]
            for model_group in models:
                print(f"model: {model_group}")
                if len(models[model_group]) < 1:
                    print(f'No models in {model_group}')
                    continue

                if test_only_on_proper_domain:
                    if model_to_domain_table[model_group] !=  domain_name[this_domain_type]:
                        if model_to_domain_table[model_group] != 'both':
                            print("[*] Skipping \'" + model_group + "\' on domain \'" + this_domain_type + '\'')
                            continue
                if prodes:
                    print("prodes 69 polygon removal")

                file.write('< ' + this_domain_type + ' >\n')
                file.write(f"{image} image\n")
                file.write('--------' + model_group + ':--------\n')
                number_of_rounds = len(models[model_group])
                print("number of rounds", number_of_rounds)
                metrics[domain][model_group] = {}

                my_test_x = np.zeros((total_patches,num_classes-1,patch_size,patch_size))
                my_test_x_temp = np.zeros((number_of_rounds, total_patches,num_classes-1,patch_size,patch_size))
                my_test_y = np.zeros((total_patches,patch_size,patch_size))

                model_count = 0
                for model in (models[model_group]):
                    print(f"round {model_count}: {model}")
                    # seg_loss_fn = nn.CrossEntropyLoss()
                    # seg_loss_fn = FocalLoss(weight = torch.FloatTensor(weights).to(device), gamma = gamma)
                    classifier = DeepLabV3plus.create(parameters)
                    # classifier = d3plus.DeepLab(num_classes=num_classes,
                    #                     backbone='mobilenet',
                    #                     output_stride=8
                    #                     # sync_bn=args.sync_bn,
                    #                     # freeze_bn=args.freeze_bn
                    #                     )
                    classifier.to(device)
                    classifier.load_state_dict(torch.load(model))
                    classifier.eval()

                    metrics[domain][model] = {}
                    # acc, loss, f1, rec, prec, alert  = 0, 0, 0, 0, 0, 0

                    with torch.no_grad():					
                        this_batch = 0
                        count = 0
                        for coor_batch in (test_loader):
                            # x = test_loader
                            count += 1
                            this_batch_size = coor_batch.shape[0]
                            # print(this_batch_size)
                            # coor_batch = x
                            x , y = training_utils.patch_extraction(img, gt, coor_batch, patch_size)
                            x = torch.from_numpy(x).float().to(device)
                            x = classifier(x)
                            y = torch.from_numpy(y).long().to(device)
                            # loss += seg_loss_fn(x, y).item()
                            
                            x = x.cpu().numpy()
                            y = y.cpu().numpy()
                            a = this_batch
                            b = this_batch + this_batch_size
                            
                            my_test_x_temp[model_count, a:b] = x[:,:2]
                            my_test_y[this_batch:this_batch + this_batch_size] = y
                            this_batch += this_batch_size
                    model_count += 1

                    del classifier

                for i in range(number_of_rounds):
                    my_test_x_temp[i] = softmax(my_test_x_temp[i], axis = 1)
                    my_test_x += my_test_x_temp[i]
                del my_test_x_temp
                
                my_test_x = my_test_x/number_of_rounds
                if change_threshold:
                    print("Threshold Start...")
                    metrics_threshold = {}
                    metrics_threshold['acc'] = []
                    metrics_threshold['f1'] = []
                    metrics_threshold['rec'] = []
                    metrics_threshold['prec'] = []
                    metrics_threshold['alert'] = []

                    for threshold in threshold_list:
                        temp_x = (my_test_x[:,1] >= threshold)*1

                        # if prodes:
                        # 	print(temp_x.shape)
                        # 	temp_x = skimage.morphology.area_opening(temp_x.astype('int'), area_threshold = 69, connectivity=1)

                        # print(temp_x.shape)
                        # print(np.unique(temp_x))
                        acc, f1, rec, prec, alert = training_utils.return_metrics(my_test_y, temp_x)
                        # metrics_threshold['loss'] = loss
                        metrics_threshold['acc'].append(acc)
                        metrics_threshold['f1'].append(f1)
                        metrics_threshold['rec'].append(rec)
                        metrics_threshold['prec'].append(prec)
                        metrics_threshold['alert'].append(alert)

                    with open(os.path.join(output_path, f"{type(domain).__name__}_{image}_image_{model_group}__metrics_threshold.json"), 'w') as f:
                        json.dump(metrics_threshold, f)


                my_test_x = my_test_x.argmax(1)
                if prodes:
                    new_test_x = skimage.morphology.area_opening(my_test_x.astype('int'), area_threshold = 69, connectivity=1)
                    dont_care_x = my_test_x - new_test_x
                    my_test_x = new_test_x
                    del new_test_x
                    
                    my_test_y += dont_care_x*2
                    # print("my_test_y before unique", np.unique(my_test_y))
                    my_test_y[:][(my_test_y == 3)] = 2
                    my_test_y[:][(my_test_y == 4)] = 2

                    # print("my_test_y AFTER unique", np.unique(my_test_y))

                # my_test_x = (my_test_x[:,1] >= 1.0)
                acc, f1, rec, prec, alert = training_utils.return_metrics(my_test_y, my_test_x)
                # metrics[model][domain]['loss'] = loss
                metrics[domain][model_group]['acc'] = acc
                metrics[domain][model_group]['f1'] = f1
                metrics[domain][model_group]['rec'] = rec
                metrics[domain][model_group]['prec'] = prec
                metrics[domain][model_group]["alert"] = alert
                        

                # # file.write('Loss: ' + str(metrics[domain][model_group]['loss'])+ '\n')
                # file.write('Accuracy: ' + str(metrics[model][domain]['acc'])+ '\n')
                # file.write('F1-score: ' +  str(metrics[domain][model_group]['f1'])+ '\n')
                # file.write('Recall: ' + str(metrics[domain][model_group]['rec'])+ '\n')
                # file.write('Precision: ' + str(metrics[domain][model_group]['prec'])+ '\n')
                # file.write('Alert Rate: ' + str(metrics[domain][model_group]['alert'])+ '\n')
                # file.write('\n')

                # file.write('Loss: ' + str(metrics[model][domain]['loss'])+ '\n')
                file.write(str(metrics[domain][model_group]['acc'])+ '\n')
                file.write(str(metrics[domain][model_group]['f1'])+ '\n')
                file.write(str(metrics[domain][model_group]['rec'])+ '\n')
                file.write(str(metrics[domain][model_group]['prec'])+ '\n')
                file.write(str(metrics[domain][model_group]['alert'])+ '\n')
                file.write('\n')

                # print(f"{domain} loss: {metrics[model][domain]['loss']}")
                print(f"{this_domain_type} acc: {metrics[domain][model_group]['acc']}")
                print(f"{this_domain_type} f1: {metrics[domain][model_group]['f1']}")
                print(f"{this_domain_type} rec: {metrics[domain][model_group]['rec']}")
                print(f"{this_domain_type} prec: {metrics[domain][model_group]['prec']}")
                print(f"{this_domain_type} alert: {metrics[domain][model_group]['alert']}")


from dataset_preprocessing.dataset_select import select_domain
if __name__=='__main__':
    # def run_test(source, source_args, target, target_args, test_args, global_args, 
    # models, file_name, last_gans):
    global_parameters = Global()
    test_parameters = Test_models()
    
    args = parser_call()
    models = args.models
    models = check_args(args)
    
    source, source_params = select_domain(global_parameters.source_domain)
    target, target_params = select_domain(global_parameters.target_domain)

    training_utils.ready_domain(source, source_params, train_set = False, test_set = True)
    training_utils.ready_domain(target, source_params, train_set = False, test_set = True)

    if test_parameters.test_transforms:
        # if global_parameters.inverse_cyclegan:
        #     source_params.best_generator = last_gans[1]
        #     target_params.best_generator = last_gans[0]
        # else:
        #     source_params.best_generator = last_gans[0]
        #     target_params.best_generator = last_gans[1]

        A2B_best_gan_model = source_params.best_generator
        B2A_best_gan_model = target_params.best_generator

        source.Prepare_GAN_Set(source_params, A2B_best_gan_model, eval = False, load = False)
        target.Prepare_GAN_Set(target_params, B2A_best_gan_model, eval = False, load = False)

    test(source, target, test_parameters, global_parameters, models)
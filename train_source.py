
import numpy as np
import os
from scipy.special import softmax
import torch

# Model Architecture
from model_architectures import DeepLabV3plus

# Utilies
from utilities import training_utils, utils

# Paraneters
from parameters.training_parameters import Train_source, Global

'''
    Code responsible for running the supervised training of deforestation classifier for the source
    domain. It is expected that you use parameters in the format 
'''

def train(source, target, parameters: Train_source, global_parameters: Global)-> str:
    '''
        Run the supervised training for the deforestation model on the source code.

        Returns the path where the best model was saved.
    '''
    
    # Select GPU for training
    device = torch.device("cuda:0")
 
    # Check if output path exists. If it doesn't, then create it
    output_path = parameters.output_path
    path_pre = []
    path_pre.append(output_path)
    print("make dir")
    for i in range (len(path_pre)):
        p = path_pre[i]
        if (os.path.isdir(p) == False):
            os.makedirs(p)
            print('\''+ p + '\'' + " path created")

    # patch_size = parameters.patch_size
    # num_classes = parameters.num_classes
    # channels = parameters.channels

    mini_batch_size = parameters.batch_size
    iteration_modifier = parameters.iteration_modifier
    # if iteration_modifier < 1:
    # 	iteration_modifier = 1
    num_epochs = parameters.num_epochs

    # weights = parameters.weights
    # gamma = parameters.gamma
    # learning_rate = parameters.learning_rate

    # dilation_rates = parameters.dilation_rates

    # use_early_stop = parameters.early_stop
    # early_stop_limit = parameters.early_stop_limit

    # source_domain = global_parameters.source_domain
    # target_domain = global_parameters.target_domain

    # global best_model_path
    # best_model_path = ''


    print("loading datasets...")
    train_set = source.augmented_train_set
    num_mini_batches = int(train_set.shape[0]//mini_batch_size//iteration_modifier)

    print("Defining network...")

    deep_lab_v3p = DeepLabV3plus.create(parameters)
    deep_lab_v3p = deep_lab_v3p.to(device)


    print("Starting train...")
    best_model_path = TrainFitter(num_epochs, num_mini_batches, source, target, parameters, global_parameters, train_set, deep_lab_v3p, device, output_path).fit()
    
    return best_model_path



class TrainFitter(utils.ModelFitter):
    def __init__(self, num_epochs, num_mini_batches, source, target, parameters, global_parameters, train_set, dlv3, device, output_path):
        super().__init__(num_epochs, num_mini_batches, output_path=output_path)        
        
        self.deep_lab_v3p = dlv3
        self.device = device
        
        self.patch_size = parameters.patch_size
        self.num_classes = parameters.num_classes
        self.channels = parameters.channels

        self.mini_batch_size = parameters.batch_size
        self.iteration_modifier = parameters.iteration_modifier
        if self.iteration_modifier < 1:
            self.iteration_modifier = 1
        self.num_epochs = parameters.num_epochs

        self.weights = parameters.weights
        self.gamma = parameters.gamma
        self.learning_rate = parameters.learning_rate

        self.dilation_rates = parameters.dilation_rates

        self.use_early_stop = parameters.early_stop
        self.early_stop_limit = parameters.early_stop_limit

        self.source_domain = global_parameters.source_domain
        self.target_domain = global_parameters.target_domain
        
        self.source = source
        self.target = target
        
        self.train_set = train_set
        
        self.best_model_path = ''
        
        self.txt = 'Best Source Classifier Model:'

        
    def initialize(self):
        # self.use_early_stop = use_early_stop
        # self.early_stop_limit = 0
        # if self.use_early_stop:
        # 	self.early_stop_limit = early_stop_limit
        # # mog: adding weights
        # # class 2 is dont care, weight 0
        # self.gamma = gamma
        # self.learning_rate = learning_rate

        print("weights:", self.weights)
        print("gamma:", self.gamma)
        print("learning rate: ", self.learning_rate)

        self.class_weights = torch.FloatTensor(self.weights).cuda()
        self.seg_loss_fn = training_utils.FocalLoss(weight = self.class_weights, gamma = self.gamma)		
        self.optim_seg = torch.optim.Adam(self.deep_lab_v3p.parameters(), lr = self.learning_rate)

        self.validation_set = {}
        self.validation_set_size = {}
        self.validation_set[self.source_domain] = torch.utils.data.DataLoader(self.source.central_pixels_coor_vl, 
            batch_size = self.mini_batch_size)
        self.validation_set_size[self.source_domain] = len(self.source.central_pixels_coor_vl)
        self.validation_set[self.target_domain] = torch.utils.data.DataLoader(self.target.central_pixels_coor_vl, 
            batch_size = self.mini_batch_size)
        self.validation_set_size[self.target_domain] = len(self.target.central_pixels_coor_vl)
        
        if len(source.central_pixels_coor_vl) < len(target.central_pixels_coor_vl):
            self.num_val_patches_per_domain = len(source.central_pixels_coor_vl) + 200
        else:
            self.num_val_patches_per_domain = len(target.central_pixels_coor_vl) + 200

        self.img = {}
        self.img[self.source_domain] = utils.channels_last2first(source.conc_image)
        self.img[self.target_domain] = utils.channels_last2first(target.conc_image)

        self.gt = {}
        self.gt[self.source_domain] = source.new_reference
        self.gt[self.target_domain] = target.new_reference
        
    def pre_epoch(self, epoch):
        self.epoch_train_set = torch.utils.data.DataLoader(self.train_set, 
            batch_size = self.mini_batch_size, shuffle = True)
        self.tr_coor = iter(self.epoch_train_set)

        self.deep_lab_v3p.train()
        
    
    def get_batch(self, epoch, batch, batch_data):
        batch = next(self.tr_coor)
        coor_batch = batch[:,:2]
        aug_batch = batch[:,2]
        x , y = training_utils.patch_extraction(self.img[self.source_domain], 
            self.gt[self.source_domain], coor_batch, self.patch_size, aug_batch = aug_batch)
        batch_data.append(x)
        batch_data.append(y)
        
    
    def train(self, epoch, batch, batch_data, metrics, iteration):
        self.optim_seg.zero_grad()
        x = torch.from_numpy(batch_data[0]).float().requires_grad_().to(self.device)
        x = self.deep_lab_v3p(x)
        y = torch.from_numpy(batch_data[1]).long().to(self.device)
        loss = self.seg_loss_fn(x, y)
        loss.backward()
        self.optim_seg.step()
        metrics["seg_loss"] = loss.item()
        # metrics["seg_acc"] = (x.argmax(1) == y).sum().item() / (x.shape[0] * patch_size * patch_size)
        x = x.argmax(1)
        x = x.cpu()
        y = y.cpu()
        # print(return_metrics(y,x))
        # print(calcula_metricas(y,x))

        acc, f1, rec, prec, alert = training_utils.return_metrics(y,x)
        metrics["seg_acc"] = acc
        metrics["seg_f1"] = f1
        metrics["seg_rec"] = rec
        metrics["seg_prec"] = prec
        metrics["seg_alert"] = alert


    def post_epoch(self, epoch, metrics):
        self.deep_lab_v3p.eval()
        with torch.no_grad():
            for domain in (self.validation_set):
                print("validation domain", domain)
                count = 0
                acc, loss, f1, rec, prec, alert  = 0, 0, 0, 0, 0, 0
                val_x = np.zeros((self.validation_set_size[domain], self.num_classes-1, self.patch_size, self.patch_size))
                val_y = np.zeros((self.validation_set_size[domain], self.patch_size, self.patch_size))
                this_batch = 0
                for x in (self.validation_set[domain]):
                    count += 1
                    this_batch_size = x.shape[0]
                    coor_batch = x
                    x , y = utils.patch_extraction(self.img[domain], self.gt[domain], coor_batch, self.patch_size)
                    x = torch.from_numpy(x).float().to(self.device)
                    x = self.deep_lab_v3p(x)
                    y = torch.from_numpy(y).long().to(self.device)
                    loss += self.seg_loss_fn(x, y).item()

                    x = x.cpu().numpy()
                    y = y.cpu().numpy()
                    val_x[this_batch:this_batch + this_batch_size] = x[:,:2]
                    val_y[this_batch:this_batch + this_batch_size] = y
                    this_batch += this_batch_size

                val_x = softmax(val_x, axis = 1)
                val_x = val_x.argmax(1)
                acc, f1, rec, prec, alert = training_utils.return_metrics(val_y, val_x)

                metrics[f"val_acc_{domain}"] = acc
                metrics[f"val_f1_{domain}"] = f1
                metrics[f"val_rec_{domain}"] = rec
                metrics[f"val_prec_{domain}"] = prec
                metrics[f"val_alert_{domain}"] = alert 
                metrics[f"val_loss_{domain}"] = loss / count

        f1 = metrics["val_f1_" + self.source_domain]
        # global best_model_path
        if self.use_early_stop and epoch > 0:
            if f1 < np.max(self.history["val_f1_" + self.source_domain]):
                self.early_stop_count += 1
            else:
                self.early_stop_count = 0
            
            print("Countdown to early stop: " + str(self.early_stop_count) + '/' + str(self.early_stop_limit))
            
            if self.early_stop_count == self.early_stop_limit:
                self.early_stop = True
                path = os.path.join(self.output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
                # path = f"{output_path}/model_{epoch:02d}_{f1:.4f}.pt"
                print(f"saving model weights to '{path}'")
                torch.save(self.deep_lab_v3p.state_dict(), path)

        if epoch > 0 and f1 > np.max(self.history["val_f1_" + self.source_domain]):
            path = os.path.join(self.output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
            self.best_model_path = path
            print("[*] Best model updated:", path)
            print(f"saving model weights to '{path}'")
            torch.save(self.deep_lab_v3p.state_dict(), path)

        elif epoch == 0 or epoch == (self.num_epochs-1) or epoch%10 == 0:
            path = os.path.join(self.output_path, f"model_{epoch:02d}_{f1:.4f}.pt")
            print(f"saving model weights to '{path}'")
            torch.save(self.deep_lab_v3p.state_dict(), path)
            if epoch == 0:
                self.best_model_path = path

        print("[*] Best model so far:", self.best_model_path)
        
        def finalize(self):
            utils.log_best_model(self.txt, self.best_model_path)
        


from dataset_preprocessing.dataset_select import select_domain
if __name__=='__main__':
    global_parameters = Global()
    train_parameters = Train_source()
    
    
    rounds = train_parameters.number_of_rounds
    this_models = []
    base_path = train_parameters.output_path
    
    source, source_params = select_domain(global_parameters.source_domain)
    target, target_params = select_domain(global_parameters.target_domain)
    
    # source_args.patches_dimension = train_args.patch_size
    source_params.stride = source_params.train_source_stride
    training_utils.ready_domain(source, source_params, train_set = True, augmented = True)

    # target_args.patches_dimension = train_args.patch_size
    target_params.stride = target_params.train_source_stride
    training_utils.ready_domain(target, target_params, train_set = True, augmented = False)
    print(len(target.central_pixels_coor_vl))
    print(len(source.central_pixels_coor_vl))
    
    for i in range(global_parameters.train_source_starting_round, rounds):
        train_parameters.output_path = os.path.join(base_path, f"round_{i}")
        print(train_parameters.output_path)
        best_source_model = train(source, target, train_parameters, global_parameters)
        this_models.append(best_source_model)
        
    print('[*] Source Training Finalized!')


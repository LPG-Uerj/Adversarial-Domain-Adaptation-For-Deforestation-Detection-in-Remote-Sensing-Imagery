import os
import sys
import numpy as np 

import json
import matplotlib.pyplot as plt


def Area_under_the_curve(X, Y):
    X = X[:]
    Y = Y[:]
    dx = np.diff(X)
    X_ = np.array([])
    Y_ = np.array([])
    
    eps = 5e-3
    for i in range(len(dx)):
        if dx[i] > eps:
            x0 = X[i]; x1 = X[i+1]
            y0 = Y[i]; y1 = Y[i+1]
            a = (y1 - y0) / (x1 - x0)
            b = y0 - a * x0
            x = np.arange(x0, x1, eps)
            y = a * x + b                
            X_ = np.concatenate((X_, x))
            Y_ = np.concatenate((Y_, y))
        else:
            X_ = np.concatenate((X_, X[i:i+1]))
            Y_ = np.concatenate((Y_, Y[i:i+1]))
                    
    X_ = np.concatenate((X_, X[-1:]))
    Y_ = np.concatenate((Y_, Y[-1:]))
    
    new_dx = np.diff(X_)
    area = 100 * np.inner(Y_[:-1], new_dx)
    
    return area

def create_mAP(results_folders, labels, colors, output_path, graph_title, file_name):
# if __name__ == '__main__':
    file_name += '.png'
    fig = plt.figure()
    # ax = plt.subplot(111)
#     Npoints = num_samples
    Interpolation = True
    Correct = True
    for rf in range(len(results_folders)):
        
#         print(results_folders[rf])
        with open(results_folders[rf], "r") as f:
            json_file = json.load(f)
        num_samples = len(np.array(json_file["rec"]))
        Npoints = num_samples
                
        recall = np.zeros((num_samples))
        precision = np.zeros((num_samples))
        
        MAP = 0
        
        recall_i = np.zeros((num_samples))
        precision_i = np.zeros((num_samples))
        
        AP_i = []
        AP_i_ = 0

        recall__ = np.array(json_file["rec"])
        precision__ = np.array(json_file["prec"])

        #print(precision__)
        #print(recall__)

        if np.size(recall__) > Npoints:
            recall__ = recall__[:]
        if np.size(precision__) > Npoints:
            precision__ = precision__[:-1]

        recall__ = recall__/100
        precision__ = precision__/100

        if Correct:

            if precision__[0] == 0: # Only when Precision is 0 as product of undefinitions
                precision__[0] = 2 * precision__[1] - precision__[2]                

            if Interpolation:
                precision = precision__[:]
                precision__[:] = np.maximum.accumulate(precision[::-1])[::-1]


            if recall__[0] > 0:
                recall = np.zeros((num_samples + 1))
                precision = np.zeros((num_samples + 1))
                # Replicating precision value
                precision[0] = precision__[0]
                precision[1:] = precision__
                precision__ = precision
                # Attending recall
                recall[1:] = recall__
                recall__ = recall


        #p = precision__[:,:-1]
        #dr = np.diff(recall__)

        recall_i = recall__
        precision_i = precision__

        #mAP = 100 * np.matmul(p , np.transpose(dr))
        mAP = Area_under_the_curve(recall__, precision__)
        ax.plot(recall_i[:], precision_i[:], color=colors[rf], label=labels[rf] + 'mAP:' + str(np.round(mAP,1)))
        
    ax.legend()
    #plt.show()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    # plt.grid(True)
    

    plt.ylabel('Precision')
    plt.xlabel('Recall')
    
    plt.title(graph_title)
    plt.savefig(os.path.join(output_path, file_name))


def check_folders(path):
    file_list = [f for f in os.listdir(path) if os.path.splitext(f)[1] == '.json']
#     print(len(file_list))
#     file_list = os.listdir(path)
    if len(file_list) > 1 or len(file_list) == 0:
        return 0
    else: 
        return os.path.join(path, file_list[0])

def check_labels(path):
    path = path.split('/')[-1]
    source = path.split('-')[0]
    target = path.split('-')[1].split('_')[0]
    if source[0] == 's':
        source = source[1:]
    if target[0] == 't':
        target = target[1:]
#     print("source", source)
#     print("target",target)
    ftype = path.split('_')[1]
    letter = ftype[0]
    number = ftype.split(letter)[1]
#     print (letter)
#     print(number)
    model = f'FT_{target}'
    if letter == 'A':
        model += "-CN"
    elif letter == 'B':
        model += "-CT"
    elif letter == 'D':
        model += "-CD"
        
    if number == '1':
        model += "-T"
    elif number == '2':
        model += "-P"
    elif number == '3':
        model += "-TP"
    elif number == '4':
        model += "-TPF"
    model += f"({target})"
    
    return source, target, model, path

def get_title(source, target, model):
    title = 'S:'
    if source == 'PA':
        title += ' Amazon PA,'
    elif source == 'RO':
        title += ' Amazon RO,'
    
    if target == 'PA':
        title += ' T: Amazon PA'
    if target == 'RO':
        title += ' T: Amazon RO'
    
    return title

def run_all(list_of_folders, colors):
    for i in range(len(list_of_folders)):
        results_folders = []
        labels = []
        source, target, model, file_name = check_labels(list_of_folders[i])

        results_folders.append(FS[f"{source}({target})"])
        results_folders.append(FS[f"{target}({target})"])

        labels.append(f"FS_{source}({target})")
        labels.append(f"FS_{target}({target})")
        labels.append(model)

        path = os.path.join(experiments_path, list_of_folders[i], default_results_folder)
        check = check_folders(path)
        if check !=0:
            results_folders.append(check)

        title = get_title(source, target, model)

        if len(results_folders) > 2:
            create_mAP(results_folders, labels, colors, output_path, title, file_name)
            


experiments_path = "./experimentos/"
output_path = "./graphs/experimentos/"
default_results_folder = "test_results"

FS = {}
FS["PA(RO)"] = "./experimentos/PA-RO_A1/test_results/AMAZON_RO_original_image_source_group__metrics_threshold.json"
FS["RO(RO)"] = "./experimentos/PA-RO_A1/test_results/AMAZON_RO_original_image_gt_target_group__metrics_threshold.json"

FS["RO(PA)"] = "./experimentos/PA-RO_A1/test_results/CERRADO_original_image_gt_target_group__metrics_threshold.json"
FS["PA(PA)"] = "./experimentos/PA-RO_A1/test_results/CERRADO_original_image_source_group__metrics_threshold.json"
# FSS_onT = "./experimentos/PA-RO_A1/test_results/AMAZON_RO_original_image_source_group__metrics_threshold.json"
# FST_onT = "./experimentos/PA-RO_A1/test_results/AMAZON_RO_original_image_gt_target_group__metrics_threshold.json"

list_of_folders = os.listdir(experiments_path)
# print(len(list_of_folders))
# print(list_of_folders)
list_of_folders.sort()
print(list_of_folders)


# results_folders = []
# labels = []
colors = []
colors.append('#ff0000')
colors.append('#0000ff')
colors.append('#00ff00')

colors.append('#ffc0cb')
colors.append('#a52a2a')
colors.append('#000000')

# run_all(list_of_folders, colors)

test = []
test.append('./experimentos/AM_PA_original_image_gt_target_group__metrics_threshold.json')
labels = []
labels.append('test')

create_mAP(test, labels, colors, './graphs', 'TESTE', 'test')
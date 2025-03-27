# Adversarial-Domain-Adaptation-For-Deforestation-Detection-in-Remote-Sensing-Imagery


## Introduction
Code and data for the paper "Adversarial Domain Adaptation For Deforestation Detection in Remote Sensing Imagery".
The object of this code is to experiment using a CycleGAN adapted model to aid in the classification of unlabeled data. More specifically, the experiment has the objective to evaluate results of this strategy on deforestation detection in satellite images.
```
Disclaimer: The original CycleGAN and overall strategy is not new to this code. What you see here is only those codes adapted for the target of deforestation. The original codes are used as the basis for this one.
```


## Considerations
This code was thought out to be used for semantic segmentation (pixel level classification) for deforestation on remote sensing (satellite images). Aside from that, the code also was planned to run as experiments to be tested and evaluated. Therefore, a lot of its code structure and parameterizations were made with this in mind.


## Code Organization
### Main Scripts
This code is divided into 5 different steps, which are expected to be run in a single order:


1. train_source.py - responsible to running the supervised training for deforestation on source data, resulting in a classifier model
2. train_cyclegan.py - responsible for running the unsupervised image-to-image adaptation from source to target and vice versa, resulting in two generative models
3. evaluate_cyclegan_inference - responsible to picking the best trained target-to-source generative model
4. train_adapted_target.py - responsible for training the source data again, but this time on the adapted image source-to-target, using one of the resulting models in step 3. At the end, results in a new deforestation classifier
5. test_models.py - tests classifier models on the dataset outputing the respective metrics


### Parameters
The parameters can be found in the "parameters" folder. Each parameter is organized in different classes. They are divided in 2 files:
 - training_parameters.py - where you will find training and script step related parameters, like batch size and number of epochs
 - dataset_parameters.py - parameters specific to used dataset
 
Aside from that, also related to the dataset handling, the "dataset_preprocessing" contains codes related to the processing of the dataset in different steps.


## How to run
### Environment
The main Python packages from the eviroment used for experiments:
```
python=3.7.9
scikit-image=0.17.2
scikit-learn=0.23.2
tensorflow=2.3.0
torch=1.11.0+cu115
```

### Scripts
Once you have the dataset and the parameters organized you can simply run:
 - step 1 can be run  independently of other steps and generated data:
    ```
    python train_source.py
    ```
    It will output its best model trained on terminal and on "best_models.txt"
  - same goes for step 2, it does not require any other step:
    ```
    python train_cyclegan.py
    ```
    Only runs training and save checkpoints, evaluation of best model is on the next step.
 - for step 3, it is expected that you have run step 1 and 2, it requires the trained source model path, because it will use this model to evaluate the image-to-image adaptations. This script will also look up the default save folder of the CycleGAN checkpoints. Example run:
    ```
    python evaluate_cyclegan_inference.py -source_model 'source\model\path'
    ```
    This script outputs the best CycleGAN model and its epoch terminal and on "best_models.txt". It only evaluates one model, but it should be used with its pair for the next step (G_A2B and G_B2A).
 - step 4 requires 3 arguments, the source model and the best CycleGAN pair models (G_A2B and G_B2A):
    ```
    python train_adapted_target.py -source_model 'source\model\path' -a2b_model 'A2B\model\path' -b2a_model 'B2A\model\path'
    ```
 - to run step 5 you need to input the models you wanna test, and then use a flag for the respective model. The flags are as follow:
   - 'source' - for the supervised trained model on the source data
   - 'gt_target' - for the supervised trained model on the target data, to use as baseline for comparisons
   - 'adapted_target' - for the model trained on the domain adapted manner for the target data
    ```
    python test_models.py -models 'supervised\target\path' 'source\model\path' 'adapted\target\path' -use_domain 'gt_target' 'source' 'adapted_target'
    ```
    Outputs results on the "test_results" folder


## Output Folder Structure
While running the code, by default, it step you run, will create the following folder structure, inside the root:
```
tmp/
├── best_models.txt
└── experiments/
    └── models/
        ├── cyclegan/
        ├── source_seg/
        └── target_adapted_seg/
        └── test_results/
```
The "best_models.txt" is updated every time you run a code that registers the best model. Every time you run said code, it will add the best model of that run to the end of it.


The following folders contain a root folder "experiments". You can change the name in each run if you want to run different experiments. Inside this folder you will find all the saved models created in the training code. The history with all metrics of each training are also there. They are divided in:
 - "cyclegan/" - where all your cyclegan checkpoints will be saved and the training history
 - "source_seg/" - for the supervised source training, all checkpoints and best models will be saved here, and also the training history
 - "target_adapted_seg/" - for the supervised adapted source training, all checkpoints and best models will be saved here, and also the training history
 - "test_results/" - contains the metrics of the models tested on "test.py"
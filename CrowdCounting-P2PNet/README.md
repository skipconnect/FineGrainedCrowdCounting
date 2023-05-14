# Guide to training

To be able to train the network on the provided datasets, follow the below steps:
* Download the vgg-16 pretrained weights here: 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
* Place the weights in the root folder (CrowdCounting-P2PNet) with the name 'vgg16_bn-6c64b313.pth'
* Run the following script

`CUDA_VISIBLE_DEVICES=0 python train.py --data_root DATA_ROOT_WAITING --dataset_file SHHA --epochs 1000 --lr_drop 3500 --output_dir ./logs --checkpoints_dir ./weights --tensorboard_dir ./logs --lr 0.0001 --lr_backbone 0.00001 --batch_size 8 --eval_freq 1 --gpu_id 0 --num_workers 0 --line 1 --row 1 `

# Guide to testing

Run the following script:

`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/best_mae.pth --output_dir ./testRes/ --line 1 --row 1 --dataset Waiting`

# Overview over files
All the datasets are collected in the folders "DATA_ROOT", "DATA_ROOT_TOWARDS", "DATA_ROOT_VIOLENT" and "DATA_ROOT_Waiting".

### crowd_datasets folder

Contains the following files:

* __init__.py (Function that loads the dataset, calls the function ´loading_data´ found in SHHA/loading_data.py)
* SHHA/SHHA.py (Function that builds the dataset and performs the data augmentation)
* SHHA/loading_data.py (Function that calls SHHA/SHHA.py to get the training and validation dataset and loads the data with normalization)

### models folder

Contains the following files:

* __init__.py (Function that builds the final-model)
* backbone.py (Script that defines and builds the backbone network VGG16, calling the script vgg_.py)
* matcher.py (Defines the Hungarian matching algorithm used in the P2P-network)
* p2pnet.py (Defines and builds the overall P2PNetwork) close to what is found in https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet, but modified to handle more classes as for the FGCC-task
*vgg_.py (Builds the backbone network used in the P2PNetwork)


### Utils folder
Contains the following file

* misc.py (Script that contains a lot of helper functions forinstance for the logging, and collate_fn for the dataloader. These functions are not really modified and is almost identical to the one found in: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)

### engine.py
This script handles the training and evaluating step as well as a function to visualize the results, almost identical to what is found in https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

### train.py

Script that performs the training of the network. Calls all the helper functions descriped above to perform training. Close to what is found in https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet

### run_test.py

Script that performs the evaluation of the network. Calls all the helper functions descriped above to perform evaluation. Close to what is found in https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet, but modified to calculate new metrics as relative CMAE and patch CMAE. Also some modifications is done to visualize and save the results.

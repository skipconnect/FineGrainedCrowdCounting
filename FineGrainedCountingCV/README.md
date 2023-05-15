# FineGrainedCountingCV

This repository contains scripts produced for carrying out experiments regarding the Fine Grained Crowd Counting
- **FG-MC-OC** is the folder that is a modification of (https://ieeexplore.ieee.org/document/9506384).

The Remainding of this folders contains scripts and data for carrying out experiments for our work build on top of work by: https://github.com/jia-wan/Fine-Grained-Counting
- **Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset** Contains the raw images and annotations for the datasets used. Our annotations are split in to test, train and val .json files to illustrate the data used for our splits.
- **models**, **ModelsAttention**, and **ModelsSkipConnection** represent the three modification of Base-FGCC from https://github.com/jia-wan/Fine-Grained-Counting. 
- Each modification can be trained by running *exp.py*, *expAttention.py*, and *expSkipConnection.py* and tested by running, *test.py*, *testAttention.py*, and *testSkipConnection.py*
- **utils** contains the script *utils.py* this is where our test-scripts are run. 

To reproduce these results:
| Dataset            | Modification    | MAE Gr. 1↓ | MAE Gr. 2↓ | CMAE↓  | Rel. CMAE↓ | Patch CMAE↓ | OMAE↓ |
|--------------------|-----------------|------------|------------|--------|-------------|-------------|-------|
| Towards/Away       | No modifications | 1.64       | 3.34       | 2.49   | 0.145       | 0.102       | 3.68  |
|                    | SkipConnections | **1.40**   | 3.25       | 2.33   | **0.137**   | **0.095**   | 2.94  |
|                    | CBAM            | 2.21       | **2.18**   | **2.20**| 0.150       | 0.096       | **2.69** |
| Standing/Sitting   | No modifications | 7.35       | **6.29**   | **6.82**| **0.578**   | **0.192**   | 7.75  |
|                    | SkipConnections | 8.27       | 6.31       | 7.31   | 0.650       | 0.198       | 8.07  |
|                    | CBAM            | **7.07**   | 7.60       | 7.33   | 0.792       | 0.203       | **7.67** |
| Violent/Nonviolent | No modifications | 5.40       | 3.35   | 4.38   |0.679   | **0.116**   | 6.85  |
|                    | SkipConnections | **4.02**   | 3.82       | **3.92**| **0.648**   | 0.135       | **5.02** |
|                    | CBAM            | 5.13       | **3.28**   | 4.20   | 0.740       | 0.127   | 5.13 |
| Waiting/Notwaiting | No modifications | **2.54**   | 2.59       | **2.57**| 0.368       | **0.077**   | 2.15  |
|                    | SkipConnections | 2.79       | **2.58**   | 2.68   | **0.361**   | 0.085       | 2.73  |
|                    | CBAM            | 2.56   | 2.59       | 2.58 | 0.383       | 0.078   | **2.07**  |


## Training
Follow the following steps:

1. Make sure you have followed the steps presented in the ReadMe in the main directory:
2. Create the GT-density map by running all cells in *CreateDensityMap.ipynb*
3. Train the model by calling the following script:

#### No modifications:
```
python exp.py --name FirstRun_Towards --att --prop --final_loss --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi
```
#### SkipConnections
```
python expSkipConnection.py --name FirstRun_SkipConnection_Towards --att --prop --final_loss --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi
```
#### CBAM
```
python expAttention.py --name FirstRun_Attention_Towards --att --prop --final_loss --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi
```

To train on different datasets change, *--train_json*, *--val_json*, *--test_json* to the path of the other datasets. The datasets are given below, Note, *--dmap_type* should be in acordance with the below table as well. *--lr* is in accordance with the table below for SkipConnections and CBAM, yet for no modifications *--lr*=0.00001:

| Dataset               | Towards_vs_Away | Standing_vs_Sitting | Violent_vs_Nonviolent | Waiting_vs_Notwaiting |
|-----------------------|-----------------|---------------------|-----------------------|-----------------------|
| *--dmap_type*       | fix4         | fix16                  | fix4                  | fix16                 |
| *--lr*              | 0.00001         | 0.000002               | 0.000002                 | 0.00001                 |


## Testing

1. From this root make a folder that corresponds to the name of the model in a new folder called TestResults e.g.:
```
./TestResults/FirstRun_Towards
```

2. Run the below scripts, each with a corresponding folder .
#### No modifications
```
python test.py --name FirstRun_Towards --att --final_loss --net vgg --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi --pre ./checkpoints/FirstRun_Towards/vgg_bestval.pth
```
#### SkipConnections
```
python testSkipConnection.py --name FirstRun_SkipConnection_Towards --att --final_loss --net vgg --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi --pre ./checkpoints/FirstRun_SkipConnection_Towards/vgg_bestval.pth
```
#### SkipConnections
```
python testAttention.py --name FirstRun_SkipConnection_Towards --att --final_loss --net vgg --count_loss --seg_loss --seg_w 100 --net vgg --soft --seg --model vgg --output_cn 2 --train_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json --val_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json --test_json ./Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json --display_freq 100 --lr 0.00001 --downsample 8 --dmap_type fix4 --input_cn 3 --seg_lr 0.00001 --train_counter --hourglass_iter 3 --weight 10 --seg_gt_act multi --pre ./checkpoints/FirstRun_Attention_Towards/vgg_bestval.pth
```

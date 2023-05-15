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
| Violent/Nonviolent | No modifications | 5.40       | **3.35**   | 4.38   | **0.679**   | **0.116**   | 6.85  |
|                    | SkipConnections | **4.02**   | 3.82       | **3.92**| **0.648**   | 0.135       | **5.02** |
|                    | CBAM            | 5.13       | **3.28**   | 4.20   | 0.740       | **0.127**   | **5.13** |
| Waiting/Notwaiting | No modifications | **2.54**   | 2.59       | **2.57**| 0.368       | **0.077**   | 2.15  |
|                    | SkipConnections | 2.79       | **2.58**   | 2.68   | **0.361**   | 0.085       | 2.73  |
|                    | CBAM            | **2.56**   | 2.59       | **2.58**| 0.383       | **0.078**   | 2.07  |


Follow the following steps:

1. Pull the full repository 

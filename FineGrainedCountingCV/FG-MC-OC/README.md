# Implementation of FG-MC-OC from (https://ieeexplore.ieee.org/document/9506384)

All the code for building the model and dataset are in main.py

OBS! This model utilizes the data in FineGrainedCountingCV/Fine-Grained-Counting-Dataset/.../
And in order to run the training, you need the generate the GT Density Maps. The code for generating these density maps are found here: FineGrainedCountingCV/DataExploration.ipynb

## Guide for training

To train this model run the following script:

`python main.py --lr 0.001 --weight_decay 0.01 --model_name "MyModel" --model_description "Description" --dataset "Towards" --gradbatches 10`

## Guide for testing the model

To test/evaluate the model run the following script.

` python main.py --model_name "MyModel" --model_description "Description" --dataset "Towards" --checkpoint_dir "MyModel/Description/savedCheckpoints/checkpoint.ckpt" --testing True`

## Hyperparameter setting for the results reported in paper

| Dataset            | Learning Rate   | Weight decay    | Grad Batches   |
|--------------------|-----------------|-----------------|----------------|
| Towards/Away       | $5 \cdot e^{-5}$ | $1 \cdot e^{-3}$ | 1              |
| Standing/Sitting   | $5 \cdot e^{-5}$ | $1 \cdot e^{-4}$ | 10             |
| Violent/Nonviolent | $5 \cdot e^{-5}$ | $1 \cdot e^{-3}$ | 1              |
| Waiting/Not waiting| $1 \cdot e^{-4}$ | $1 \cdot e^{-3}$ | 1              |
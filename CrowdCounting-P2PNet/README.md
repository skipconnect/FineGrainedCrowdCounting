# Guide to training

To be able to train the network on the provided datasets, follow the below steps:
* Download the vgg-16 pretrained weights here: 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
* Place the weights in the root folder (CrowdCounting-P2PNet) with the name 'vgg16_bn-6c64b313.pth'
* Run the following script

`CUDA_VISIBLE_DEVICES=0 python train.py --data_root DATA_ROOT_WAITING --dataset_file SHHA --epochs 1000 --lr_drop 3500 --output_dir ./logs --checkpoints_dir ./weights --tensorboard_dir ./logs --lr 0.0001 --lr_backbone 0.00001 --batch_size 8 --eval_freq 1 --gpu_id 0 --num_workers 0 --line 1 --row 1 `

# Guide to testing

Run the following script:

`CUDA_VISIBLE_DEVICES=0 python run_test.py --weight_path ./weights/best_mae.pth --output_dir ./testRes/ --line 1 --row 1 --dataset Waiting`

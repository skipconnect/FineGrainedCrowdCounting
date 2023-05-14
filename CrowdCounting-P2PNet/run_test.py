import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np
import json
import gc

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')


torch.cuda.empty_cache()

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
    
    parser.add_argument('--save_name', default='',
                        help='Name of image to save')
    
    parser.add_argument('--dataset', default='',
                        help='Name of dataset for testing')
    
    
    parser.add_argument('--cpu_test', type=bool, default=False, help='Wheter or not to run the test on gpu')

    parser.add_argument('--save_json', type=bool, default=False, help='Wheter or not to save json files and output predictions')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    if args.cpu_test:
        print("Running inference on CPU!")
        device = torch.device('cpu')
    else:
        print("Running inference on GPU!")
        device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    create_table = False
    if "Towards" in args.dataset:
        annot_path = "./DATA_ROOT_TOWARDS/eval_test_annotations"
        images_path = "./DATA_ROOT_TOWARDS/eval_test"
    elif "Waiting" in args.dataset:
        annot_path = "./DATA_ROOT_WAITING/eval_test_annotations"
        images_path = "./DATA_ROOT_WAITING/eval_test"
    elif "Standing" in args.dataset:
        create_table = True
        annot_path = "./DATA_ROOT/eval_test_annotations"
        images_path = "./DATA_ROOT/eval_test"
        
    elif "Violent" in args.dataset:
        annot_path = "./DATA_ROOT_VIOLENT/eval_test_annotations"
        images_path = "./DATA_ROOT_VIOLENT/eval_test"
    else:
        print("You didnt provide a vilad dataset")
        0/0
    
    with open(annot_path + "/annotations.json", 'r') as f:
        points_data = json.load(f)
        
    images = os.listdir(images_path)
    
    MAEs1 = []
    MAEs2 = []
    rMAEs1 = []
    rMAEs2 = []
    Patch_mae1_values = []
    Patch_mae2_values = []
    true_counts = []
    pred_counts = []
    sliced = images[0:]
    results = {}
    saving = False
    low1 = []
    low2 = []
    mid1 = []
    mid2 = []
    high1 = []
    high2 = []
    for i,img_ in enumerate(sliced):
        print(i+1, "/", len(sliced))
    
        gt_cnt_1 = len(points_data[img_][0]["x"])
        gt_cnt_2 = len(points_data[img_][1]["x"])
        gt_points_1 = np.array(list(zip(points_data[img_][0]["x"],points_data[img_][0]["y"])))
        gt_points_2 = np.array(list(zip(points_data[img_][1]["x"],points_data[img_][1]["y"])))
        
        true_counts.append(gt_cnt_1+gt_cnt_2)
    
        # set your image path here
        img_path = images_path + "/"+img_
        # load the images
        img_raw = Image.open(img_path).convert('RGB')
        
        # round the size
        width, height = img_raw.size
        new_width = width // 16 *16
        new_height = height // 16 *16
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        
        # pre-proccessing
        img = transform(img_raw)
        
        #show_cache()
        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        
        
        #SMALL SIZING GUIDE: width,height is the order when the image is in PIL, while height,width is the order in Tensor.
        # Width is equal to x while height is equal to y
        #Do equal scaling to gt points
        
        try:
            gt_points_1[:,0] = gt_points_1[:,0] * (new_width/width)
            gt_points_1[:,1] = gt_points_1[:,1] * (new_height/height)
        except:
            pass
        try:
            gt_points_2[:,0] = gt_points_2[:,0] * (new_width/width)
            gt_points_2[:,1] = gt_points_2[:,1] * (new_height/height)
        except:
            pass
        
        
        gt_1_dot = torch.zeros(img.shape[1:])
        gt_2_dot = torch.zeros(img.shape[1:])
        pred_1_dot = torch.zeros(img.shape[1:])
        pred_2_dot = torch.zeros(img.shape[1:])
        
        # run inference
        outputs = model(samples)
        
        pred_logits = outputs["pred_logits"].detach().cpu()
        pred_points = outputs["pred_points"].detach().cpu()
        samples = samples.detach().cpu()
        del samples
        del outputs
        #show_cache()
        

        #Scores for first group
        #outputs_scores_1 = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_scores_1 = torch.nn.functional.softmax(pred_logits, -1)[:, :, 1][0]
        #Scores for second group
        outputs_scores_2 = torch.nn.functional.softmax(pred_logits, -1)[:, :, 2][0]


        outputs_points = pred_points[0]

        threshold = 0.5
        # filter the predictions
        #For the first group
        points_1 = outputs_points[outputs_scores_1 > threshold].detach().cpu().numpy().tolist()
        #For the second group
        points_2 = outputs_points[outputs_scores_2 > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores_1 > threshold).sum()) + int((outputs_scores_2 > threshold).sum())
        predict_cnt_1 = int((outputs_scores_1 > threshold).sum())
        predict_cnt_2 = int((outputs_scores_2 > threshold).sum())
        pred_counts.append(predict_cnt_1 + predict_cnt_2)
        #print("FOR GR1 Predicted: ", predict_cnt_1, "GT:", gt_cnt_1)
        #print("FOR GR2 Predicted: ", predict_cnt_2, "GT:", gt_cnt_2)
        MAEs1.append(np.abs(gt_cnt_1 - predict_cnt_1))
        MAEs2.append(np.abs(gt_cnt_2 - predict_cnt_2))
        
        #Creating table
        if create_table:
            if gt_cnt_1 + gt_cnt_2 < 40:
                if gt_cnt_1>0:
                    low1.append(np.abs(gt_cnt_1 - predict_cnt_1)/gt_cnt_1)
                if gt_cnt_2>0:
                    low2.append(np.abs(gt_cnt_2 - predict_cnt_2)/gt_cnt_2)
                
            elif gt_cnt_1 + gt_cnt_2 <= 80:
                if gt_cnt_1>0:
                    mid1.append(np.abs(gt_cnt_1 - predict_cnt_1)/gt_cnt_1)
                if gt_cnt_2>0:
                    mid2.append(np.abs(gt_cnt_2 - predict_cnt_2)/gt_cnt_2)
                
            else:
                if gt_cnt_1>0:
                    high1.append(np.abs(gt_cnt_1 - predict_cnt_1)/gt_cnt_1)
                if gt_cnt_2>0:
                    high2.append(np.abs(gt_cnt_2 - predict_cnt_2)/gt_cnt_2)
        
        if gt_cnt_1>0:
            rMAEs1.append(np.abs(gt_cnt_1 - predict_cnt_1)/gt_cnt_1)
        if gt_cnt_2>0:
            rMAEs2.append(np.abs(gt_cnt_2 - predict_cnt_2)/gt_cnt_2)
        
        points_1_ = np.array(points_1)
        points_2_ = np.array(points_2)
        
        
        for x,y in gt_points_1:
            gt_1_dot[int(y),int(x)] = 1
        for x,y in gt_points_2:
            gt_2_dot[int(y),int(x)] = 1
        for x,y in points_1_:
            try:
                #pred_1_dot[int(y),int(x)] = 1
                pred_1_dot[min([int(y),pred_1_dot.shape[0]-1]),min([int(x),pred_1_dot.shape[1]-1])] = 1
            except:
                print("out class1", "(", min([int(y),pred_1_dot.shape[0]]),min([int(x),pred_1_dot.shape[1]]), ")", img_)
                print(img.shape)
                print(pred_1_dot.shape)
        for x,y in points_2_:
            try:
                #pred_2_dot[int(y),int(x)] = 1
                pred_2_dot[min([int(y),pred_2_dot.shape[0]-1]),min([int(x),pred_2_dot.shape[1]-1])] = 1
            except:
                print("out class2", "(", min([int(y),pred_2_dot.shape[0]]),min([int(x),pred_2_dot.shape[1]]), ")", img_)
                
        
        #PATH MAE
        pmae1 = []
        pmae2 = []
        for y in range(0, img.shape[1]-img.shape[1]%8, math.floor(img.shape[1]/8)):#
            for x in range(0,img.shape[2]-img.shape[2]%8, math.floor(img.shape[2]/8)):#
                y_true_patch_MAE1 = gt_1_dot[y:y +  math.floor(img.shape[1]/8), x:x + math.floor(img.shape[2]/8)].sum()
                y_true_patch_MAE2 = gt_2_dot[y:y +  math.floor(img.shape[1]/8), x:x + math.floor(img.shape[2]/8)].sum()
                y_pred_patch_MAE1 = pred_1_dot[y:y +  math.floor(img.shape[1]/8), x:x + math.floor(img.shape[2]/8)].sum()
                y_pred_patch_MAE2 = pred_2_dot[y:y +  math.floor(img.shape[1]/8), x:x + math.floor(img.shape[2]/8)].sum()

                Patch_mae1_values.append(np.abs(y_true_patch_MAE1 - y_pred_patch_MAE1 ))
                pmae1.append(np.abs(y_true_patch_MAE1 - y_pred_patch_MAE1 ))
                Patch_mae2_values.append(np.abs(y_true_patch_MAE2  - y_pred_patch_MAE2 ))
                pmae2.append(np.abs(y_true_patch_MAE2 - y_pred_patch_MAE2 ))
                
        
        
        
        if (np.abs(gt_cnt_1 - predict_cnt_1) + np.abs(gt_cnt_2 - predict_cnt_2))/2 > 20 or img_.startswith("351") or i == 10:
            print("Saving image")

            outputs_scores = torch.nn.functional.softmax(pred_logits, -1)[:, :, 1][0]

            outputs_points = pred_points[0]
            # draw the predictions
            # Blue color in BGR
            color_1 = (255, 0, 0)
            # Red color in BGR
            color_2 = (0, 0, 255)
            size = 2
            img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
            for p in points_1:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, color_2, -1)
            for p in points_2:
                img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, color_1, -1)



            # save the visualized image
            cv2.imwrite(os.path.join(args.output_dir, 'pred{}.jpg'.format(img_[:-4])), img_to_draw)
            
        results[img_] = {"CMAE": str((np.abs(gt_cnt_1 - predict_cnt_1)+np.abs(gt_cnt_2 - predict_cnt_2))/2),
                         "RMAE1": str(np.abs(gt_cnt_1 - predict_cnt_1)/gt_cnt_1 if gt_cnt_1>0 else -1),
                         "RMAE2": str(np.abs(gt_cnt_2 - predict_cnt_2)/gt_cnt_2 if gt_cnt_2>0 else -1),
                         "PMAE1": str(np.mean(pmae1)),
                         "PMAE2": str(np.mean(pmae2))}
        
        if args.save_name == img_:
            print("Preparing save..")
            saving = True
            the_img = img_
            save_group1 = points_1_
            save_group2 = points_2_
            
            
    avg_mae1 = np.mean(MAEs1)
    avg_mae2 = np.mean(MAEs2)
    avg_mae = (avg_mae1 + avg_mae2)/2
    
    avg_rmae1 = np.mean(rMAEs1)
    avg_rmae2 = np.mean(rMAEs2)
    avg_rmae = (avg_rmae1 + avg_rmae2)/2
    print("MAE1: ", avg_mae1)
    print("MAE2: ", avg_mae2)
    print("AVG.MAE: ", avg_mae)
    
    print("Rel.MAE1: ", avg_rmae1)
    print("Rel.MAE2: ", avg_rmae2)
    print("Rel.AVG.MAE: ", avg_rmae)
    
    print("patch MAE1", np.mean(Patch_mae1_values))
    print("patch MAE2", np.mean(Patch_mae2_values))
    print("patch MAE", (np.mean(Patch_mae1_values)+np.mean(Patch_mae2_values))/2)
    
    print("##########")
    print("Average Total True Count:", np.mean(true_counts))
    print("Average Total Pred Count:", np.mean(pred_counts))
    
    if create_table:
        print("##########")
        print("Low:", (np.mean(low1) + np.mean(low2))/2)
        print("Mid:", (np.mean(mid1) + np.mean(mid2))/2)
        print("High:", (np.mean(high1) + np.mean(high2))/2)

    
    #Saving metrics
    json_object = json.dumps(results)
    
    if args.save_json: 

        if "Towards" in annot_path:
            # Writing to sample.json
            with open("../TowardsResults/{name}.json".format(name=args.weight_path.split("/")[1]), "w") as outfile:
                outfile.write(json_object)
            if saving == True:
                np.save("../TowardsOutputs/{theimg}_P2P_Group1.npy".format(theimg=the_img[:-4]), save_group1)
                np.save("../TowardsOutputs/{theimg}_P2P_Group2.npy".format(theimg=the_img[:-4]), save_group2)
                
        if "Waiting" in annot_path:
            # Writing to sample.json
            with open("../WaitingResults/{name}.json".format(name=args.weight_path.split("/")[1]), "w") as outfile:
                outfile.write(json_object)
            if saving == True:
                np.save("../WaitingOutputs/{theimg}_P2P_Group1.npy".format(theimg=the_img[:-4]), save_group1)
                np.save("../WaitingOutputs/{theimg}_P2P_Group2.npy".format(theimg=the_img[:-4]), save_group2)
        
        if "Standing" in annot_path:
            # Writing to sample.json
            with open("../StandingResults/{name}.json".format(name=args.weight_path.split("/")[1]), "w") as outfile:
                outfile.write(json_object)
            if saving == True:
                np.save("../StandingOutputs/{theimg}_P2P_Group1.npy".format(theimg=the_img[:-4]), save_group1)
                np.save("../StandingOutputs/{theimg}_P2P_Group2.npy".format(theimg=the_img[:-4]), save_group2)

                
        if "Violent" in annot_path:
            # Writing to sample.json
            with open("../ViolentResults/{name}.json".format(name=args.weight_path.split("/")[1]), "w") as outfile:
                outfile.write(json_object)
            if saving == True:
                np.save("../ViolentOutputs/{theimg}_P2P_Group1.npy".format(theimg=the_img[:-4]), save_group1)
                np.save("../ViolentOutputs/{theimg}_P2P_Group2.npy".format(theimg=the_img[:-4]), save_group2)

    
    #print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
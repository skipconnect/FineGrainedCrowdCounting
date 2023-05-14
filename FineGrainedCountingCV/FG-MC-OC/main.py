import torch
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.models import vgg16
import json
from collections import OrderedDict
from PIL import Image
import h5py
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import argparse
from argparse import Namespace
import os



#Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--beta_1', type=float, default=0.9, help='decay rate 1')
parser.add_argument('--beta_2', type=float, default=0.98, help='decay rate 2')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay')
parser.add_argument('--factor', type=float, default=0.1, help='plateau factor')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--n_gpus', type=int, default=1, help='number of GPUs')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--gradbatches', type=int, default=1, help='Number of batches to accumululate before calculating gradients')
parser.add_argument('--model_description', type=str, help='Description of the model', required=True)



parser.add_argument('--testing', default=False, type=bool, help = "Testting or no testing")
parser.add_argument('--dataset', type=str, help='Name of dataset: Towards, Standing, Waiting, Violent', required=True)
parser.add_argument('--save_name', type=str, help='Name of test-image you want to save')
parser.add_argument('--save_json', default=False, type=bool, help = "Wheter to save json-file with results and output predictions")
parser.add_argument('--checkpoint_dir', type=str, default="", help='Checkpoint of saved weights')



opt = parser.parse_args()




class FineGrainedDataset():
    def __init__(self, data_json,dmap_type):
        
        with open(data_json, 'r') as outfile:
            self.data_list = json.load(outfile)
        #self.data_list = self.data_list[0:6]    
        self.dmap_type = dmap_type
        self.data_json = data_json
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        #data = OrderedDict()
        img_path = self.data_list[idx]
        img_path = "/".join(self.data_json.split("/")[0:4])+"/images/"+img_path
        # read image
        img = Image.open(img_path).convert('RGB')
        #print(img_path)
        #print("IMAGE-SIZE:", img.size)
        ratio = min(1080/img.size[0], 1080/img.size[1])
        l = 32
        w, h = int(ratio*img.size[0]/l)*l, int(ratio*img.size[1]/l)*l
        o_w, o_h = img.size
        img = img.resize([w, h])

        #img = img.resize([int(ratio*img.size[0]), int(ratio*img.size[1])])

        ## read density map
        # get ground-truth path, dot, fix4, fix16, or adapt
        if 'fix4' in self.dmap_type:
            temp = '_fix4.h5'
        elif 'fix16' in self.dmap_type:
            temp = '_fix16.h5'
        elif 'adapt' in self.dmap_type:
            temp = '_adapt.h5'
        elif 'dot' in self.dmap_type:
            temp = '_dot.h5'
        else:
            print('dmap type error!')
        suffix = img_path[-4:]
        # suppose the ground-truth density maps are stored in ground-truth folder
        gt_path = img_path.replace(suffix, temp).replace('images', 'ground-truths')
        
        gt_file = h5py.File(gt_path, 'r')
        den = np.asarray(gt_file['density'])


        # transformation
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
        transform = transforms.Resize((h,w))
        den = torch.from_numpy(den)
        #print("IMG:",img.shape)
        #print("DEN:",den.shape)
        #Downscale so GT is same 
        s = den.sum()
        
        den = nn.functional.interpolate(den.unsqueeze(0), [int(h/8),int(w/8)] ).squeeze()
        den = den/den.sum()*s if s > 0 else den
        #den = den/den.max()
        
        annot_path = "/".join(img_path.split("/")[0:-2])+"/annotations/annotations.json"
        with open(annot_path, 'r') as f:
            points_data = json.load(f)

        GT1 = len(points_data[img_path.split("/")[-1]][0]["y"])
        

        GT2 = len(points_data[img_path.split("/")[-1]][1]["y"])
        

        # return
        #data['img'] = img
        #data['den'] = den
        #data['path'] = img_path
        
        return img, den.type(torch.float32), img_path.split("/")[-1]

class FrontEndEncoder(pl.LightningModule):
    def __init__(self):
        super(FrontEndEncoder, self).__init__()
        self.layers = nn.Sequential(*list(vgg16(pretrained=True).features.children())[:23])

    def forward(self, x):
        x = self.layers(x)
        return x
    
class ClassMaskDecoder(pl.LightningModule):
    def __init__(self):
        super(ClassMaskDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, dilation=1, padding=1)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=3, dilation=1, padding=1)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=1, dilation=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = nn.Softmax(dim=1)(x)
        return x
    

class BackEndDecoder(pl.LightningModule):
    def __init__(self):
        super(BackEndDecoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2)
        self.conv7 = nn.Conv2d(64, 1, kernel_size=1, dilation=1, padding=0)
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        return x
    
    
class Model(pl.LightningModule):
    def __init__(self, save_name="", dataset=""):
        super(Model, self).__init__()
        self.frontend = FrontEndEncoder()
        self.ClassMaskDecoder =  ClassMaskDecoder()
        self.BackEndDecoder = BackEndDecoder()
        self.save_name = save_name
        self.dataset = dataset
        
    def forward(self,x):
        shared = self.frontend(x)
        classmask = self.ClassMaskDecoder(shared)
        density = self.BackEndDecoder(shared)
        att = torch.mul(classmask, density)
        return att, classmask, density
   
     
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=opt.lr, betas=(0.9, 0.999),weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 10, factor = 0.1)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "avg_val_loss"}

    def train_dataloader(self):
        if "Towards" in self.dataset:
            train_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/train.json", "fix4"), shuffle=True, 
                            batch_size=opt.batch_size, num_workers = 1)
        
        if "Standing" in self.dataset:
            train_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Standing_vs_Sitting/annotations/train.json", "fix16"), shuffle=True, 
                            batch_size=opt.batch_size, num_workers = 1)
            
        if "Waiting" in self.dataset:
            train_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Waiting_vs_Notwaiting/annotations/train.json", "fix16"), shuffle=True, 
                            batch_size=opt.batch_size, num_workers = 1)
            
        if "Violent" in self.dataset:
            train_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Violent_vs_Nonviolent/annotations/train.json", "fix4"), shuffle=True, 
                            batch_size=opt.batch_size, num_workers = 1)
        
        return train_loader
    
    def val_dataloader(self):
        
        if "Towards" in self.dataset:
            val_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/val.json", "fix4"), shuffle=False, 
                            batch_size=1, num_workers = 1)
            
        if "Standing" in self.dataset:
            val_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Standing_vs_Sitting/annotations/val.json", "fix16"), shuffle=False, 
                            batch_size=1, num_workers = 1)
        
        if "Waiting" in self.dataset:
            val_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Waiting_vs_Notwaiting/annotations/val.json", "fix16"), shuffle=False, 
                            batch_size=1, num_workers = 1)
        
        if "Violent" in self.dataset:
            val_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Violent_vs_Nonviolent/annotations/val.json", "fix4"), shuffle=False, 
                            batch_size=1, num_workers = 1)
        
            
        
        return val_loader
    
    def test_dataloader(self):
        
        if "Towards" in self.dataset:
            test_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Towards_vs_Away/annotations/test.json", "fix4"), shuffle=False, 
                            batch_size=1, num_workers = 1)
            
        if "Standing" in self.dataset:
            test_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Standing_vs_Sitting/annotations/test.json", "fix16"), shuffle=False, 
                            batch_size=1, num_workers = 1)
            
        if "Waiting" in self.dataset:
            test_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Waiting_vs_Notwaiting/annotations/test.json", "fix16"), shuffle=False, 
                            batch_size=1, num_workers = 1)
            
        if "Violent" in self.dataset:
            test_loader = DataLoader(FineGrainedDataset("../Fine-Grained-Counting-Dataset/Fine-Grained-Counting-Dataset/Violent_vs_Nonviolent/annotations/test.json", "fix4"), shuffle=False, 
                            batch_size=1, num_workers = 1)
        
        return test_loader
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_, target,path = batch
        att, classmask, density = self(input_)
        mse = nn.MSELoss(reduction='sum').cuda()
        loss_tot = mse(density,torch.sum(target,dim=1).unsqueeze(1))
        
        loss_cls = 0
        loss_mask = 0
        for i in range(2):
            loss_cls += mse(att[:,i,:,:], target[:,i,:,:])
            loss_mask += mse(torch.mul(classmask[:,i,:,:],torch.sum(target,dim=1)), target[:,i,:,:])
        loss = loss_tot + 0.5*loss_cls + 0.5*loss_mask
        
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train_Loss", avg_loss, self.current_epoch)
        #self.logger.experiment.add_scalar("LearningRate", self.lr, self.current_epoch)
        #   return {"avg_loss": avg_loss}
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_, target,path = batch
        mse = nn.MSELoss(reduction='sum').cuda()
        att, classmask, density = self(input_)
        loss_tot = mse(density,torch.sum(target,dim=1).unsqueeze(1))
        loss_cls = 0
        loss_mask = 0
        for i in range(2):
            loss_cls += mse(att[:,i,:,:], target[:,i,:,:])
            loss_mask += mse(torch.mul(classmask[:,i,:,:],torch.sum(target,dim=1)), target[:,i,:,:])
        loss = loss_tot + 0.5*loss_cls + 0.5*loss_mask
        
        MAE1 = abs(target[:,0,:,:].sum()-att[:,0,:,:].sum())
        MAE2 = abs(target[:,1,:,:].sum()-att[:,1,:,:].sum())
        #loss = loss_tot
       
        # Logging to TensorBoard by default
        #self.log("val_loss", loss)
        #tensorboard_logs = {'val_BCE_loss': loss}
        #return {"loss": loss, "log": tensorboard_logs}
        return {"val_loss": loss, "loss_tot": loss_tot, "loss_cls": loss_cls, "loss_mask": loss_mask, "MAE1": MAE1, "MAE2": MAE2}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_loss_tot = torch.stack([x["loss_tot"] for x in outputs]).mean()
        avg_loss_cls = torch.stack([x["loss_cls"] for x in outputs]).mean()
        avg_loss_mask = torch.stack([x["loss_mask"] for x in outputs]).mean()
        avg_MAE1 = torch.stack([x["MAE1"] for x in outputs]).mean()
        avg_MAE2 = torch.stack([x["MAE2"] for x in outputs]).mean()
        
        self.logger.experiment.add_scalar("Validation_Loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation_Loss_tot", avg_loss_tot, self.current_epoch)
        self.logger.experiment.add_scalar("Validation_Loss_cls", avg_loss_cls, self.current_epoch)
        self.logger.experiment.add_scalar("Validation_Loss_mask", avg_loss_mask, self.current_epoch)
        self.logger.experiment.add_scalar("Validation_MAE1", avg_MAE1, self.current_epoch)
        self.logger.experiment.add_scalar("Validation_MAE2", avg_MAE2, self.current_epoch)
        #used to monitor lr_scheduler
        self.log("avg_val_loss", avg_loss)
        return {'avg_val_loss': avg_loss}
    def test_step(self, batch, batch_idx):
        MAES_1=[]
        MAES_2=[]
        counts=[]
        RELATIVEMAE_1=[]
        RELATIVEMAE_2=[]
        Patch_mae1_values=[]
        Patch_mae2_values=[]
       
        input_, target,path = batch
        
        att, classmask, density = self(input_)
        MAE1 = abs(target[:,0,:,:].sum()-att[:,0,:,:].sum())
        MAE2 = abs(target[:,1,:,:].sum()-att[:,1,:,:].sum())
        MAETOT = abs(density.sum()-torch.sum(target,dim=1).sum())
        pred_count = att[:,0,:,:].sum() + att[:,1,:,:].sum()
        gt_count = target[:,0,:,:].sum() + target[:,1,:,:].sum()
        RELATIVEMAE_1 = torch.tensor([-1])
        RELATIVEMAE_2 = torch.tensor([-1])
        theoutput = torch.zeros((10,10))
        if path[0] == self.save_name:
            theoutput = att.cpu()
        
        if target[:,0,:,:].sum()>0:
            RELATIVEMAE_1= MAE1/target[:,0,:,:].sum()
        if target[:,1,:,:].sum()>0:
            RELATIVEMAE_2= MAE2/target[:,1,:,:].sum()
            
            
        
        for y in range(0, target.shape[2]-target.shape[2]%8, math.floor(target.shape[2]/8)):#
            for x in range(0,target.shape[3]-target.shape[3]%8, math.floor(target.shape[3]/8)):#
                
                y_true_patch_MAE1 = target[:,0, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()
                y_true_patch_MAE2 = target[:,1, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()
                y_pred_patch_MAE1 = att[:,0, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()
                y_pred_patch_MAE2  =att[:,1, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()
                            
                Patch_mae1_values.append(abs(y_true_patch_MAE1 - y_pred_patch_MAE1 ).item())
                
                
                Patch_mae2_values.append(abs(y_true_patch_MAE2  - y_pred_patch_MAE2 ).item())
                #if target [:,0, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()+target[:,1, y:y +  math.floor(target.shape[2]/8), x:x + math.floor(target.shape[3]/8)].sum()>0:
        
        
        
        Patch_mae1_values=np.mean(Patch_mae1_values)
        Patch_mae2_values=np.mean(Patch_mae2_values)
        
        
        return {'MAE1': MAE1, "MAE2": MAE2, "MAETOT":MAETOT, "patchMAE1":Patch_mae1_values,"patchMAE2":Patch_mae2_values, "MAE1_relative": RELATIVEMAE_1, "MAE2_relative": RELATIVEMAE_2, "pred_count": pred_count, "gt_count": gt_count, "theoutput": theoutput,"path":path[0]}
    def test_epoch_end(self, outputs):
        avg_MAE1 = torch.stack([x["MAE1"] for x in outputs]).mean()
        avg_MAE2 = torch.stack([x["MAE2"] for x in outputs]).mean()
        avg_MAETOT = torch.stack([x["MAETOT"] for x in outputs]).mean()
        avg_MAErelative1 = torch.stack([x["MAE1_relative"] for x in outputs if x["MAE1_relative"].sum()>0]).mean()
      #  avg_MAErelative2 = torch.stack([x["MAE2_relative"] for x in outputs]).mean()
        avg_Patch1 = np.mean([[x["patchMAE1"] for x in outputs]])
        avg_Patch2 = np.mean([[x["patchMAE2"] for x in outputs]])
     #   print("Acc", outputs["acc"])
        print("TESTRESULTS")
        print("MAE1:", avg_MAE1)
        print("MAE2:", avg_MAE2)
        print("MAE:", (avg_MAE1+avg_MAE2)/2)
        print("MAETOT:", avg_MAETOT)
        print("Patch_mae1", avg_Patch1)
        print("Patch_mae2", avg_Patch2)
        print("Patch_mae", (avg_Patch1+ avg_Patch2)/2)
        print("avg_MAErelative", avg_MAErelative1)
        print("NewRelMAE1:", torch.stack([x["MAE1_relative"] for x in outputs if x["MAE1_relative"].sum()>0]).mean())
        print("NewRelMAE2:", torch.stack([x["MAE2_relative"] for x in outputs if x["MAE2_relative"].sum()>0]).mean())
        print("NewRelCMAE:", (torch.stack([x["MAE1_relative"] for x in outputs if x["MAE1_relative"].sum()>0]).mean() + torch.stack([x["MAE2_relative"] for x in outputs if x["MAE2_relative"].sum()>0]).mean())/2)
        print("Average Total Count GT:", torch.stack([x["gt_count"] for x in outputs]).mean())
        print("Average Total Count Pred:", torch.stack([x["pred_count"] for x in outputs]).mean())
      #  print("avg_MAErelative2", avg_MAErelative2)
      #  print("avg_MAErelative", (avg_MAErelative1+ avg_MAErelative2)/2)
        results = {}
        saving = False
        create_table = False
        low1=[]
        low2=[]
        mid1=[]
        mid2=[]
        high1=[]
        high2=[]
        if "Standing" in self.dataset:
            create_table = True
        for i,x in enumerate(outputs):
            res_ = {"CMAE": str(((x["MAE1"] + x["MAE2"])/2).item()),
                    "RMAE1": str(x["MAE1_relative"].item()),
                    "RMAE2": str(x["MAE2_relative"].item()),
                    "PMAE1": str(np.mean(x["patchMAE1"])),
                    "PMAE2": str(np.mean(x["patchMAE1"]))}
            if create_table:
                if x["gt_count"] < 40:
                    if x["MAE1_relative"].item() >= 0:
                        low1.append(x["MAE1_relative"].item())
                    if x["MAE2_relative"].item() >= 0:
                        low2.append(x["MAE2_relative"].item())
                elif x["gt_count"] <= 80:
                    if x["MAE1_relative"].item() >= 0:
                        mid1.append(x["MAE1_relative"].item())
                    if x["MAE2_relative"].item() >= 0:
                        mid2.append(x["MAE2_relative"].item())
                    
                else:
                    if x["MAE1_relative"].item() >= 0:
                        high1.append(x["MAE1_relative"].item())
                    if x["MAE2_relative"].item() >= 0:
                        high2.append(x["MAE2_relative"].item())
            if x["path"] == self.save_name:
                theoutput_ = x["theoutput"].numpy()
                the_img = x["path"][:-4]
                saving = True
                
           
            results[x["path"]]=res_
        
        if create_table:
            print("#########")
            print("Low:", (np.mean(low1) + np.mean(low2))/2)
            print("Mid:", (np.mean(mid1) + np.mean(mid2))/2)
            print("High:", (np.mean(high1) + np.mean(high2))/2)
        json_object = json.dumps(results)
        if opt.save_json:
            if "Towards" in self.dataset:
                with open("../../TowardsResults/{name}.json".format(name="Alternative1"), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../../TowardsOutputs/{theimg}_Alternative1.npy".format(theimg=the_img), theoutput_)

            if "Standing" in self.dataset:
                with open("../../StandingResults/{name}.json".format(name="Alternative1"), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../../StandingOutputs/{theimg}_Alternative1.npy".format(theimg=the_img), theoutput_)
            
            if "Waiting" in self.dataset:
                with open("../../WaitingResults/{name}.json".format(name="Alternative1"), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../../WaitingOutputs/{theimg}_Alternative1.npy".format(theimg=the_img), theoutput_)
                    
            if "Violent" in self.dataset:
                with open("../../TowardsResults/{name}.json".format(name="Alternative1"), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../../ViolentOutputs/{theimg}_Alternative1.npy".format(theimg=the_img), theoutput_)



        
        return {"MAE1": avg_MAE1, "MAE2": avg_MAE2}
    

    torch.cuda.empty_cache()


def run_trainer():
    if not opt.testing:
        logger = TensorBoardLogger(opt.model_name, opt.model_description)
        epoch_callback = ModelCheckpoint(filename = opt.model_name+"-{epoch}", dirpath=opt.model_name+"/"+opt.model_description+"/savedCheckpoints/", every_n_epochs = 1, save_top_k=-1)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        trainer = Trainer(fast_dev_run = False, gpus=opt.n_gpus, max_epochs = opt.epochs, logger=logger, callbacks = [epoch_callback,lr_monitor], auto_lr_find = False, num_sanity_val_steps=0, accumulate_grad_batches=opt.gradbatches)
        model = Model(save_name = opt.save_name, dataset=opt.dataset)    
        trainer.fit(model)
    else:
      
        c_path = opt.checkpoint_dir
        model_test = Model.load_from_checkpoint(c_path,  dataset=opt.dataset, save_name = opt.save_name)
        
        trainer = Trainer(gpus = opt.n_gpus)
        trainer.test(model_test)


if __name__ == '__main__':
    run_trainer()
    


    

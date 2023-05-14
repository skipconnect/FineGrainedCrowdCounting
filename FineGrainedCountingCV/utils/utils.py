from __future__ import print_function
from collections import OrderedDict
from tensorboardX import SummaryWriter
import time
import os
import pdb
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import matplotlib.colors as mcolors
import torchvision.transforms as transforms
import math

class Logger():
    def __init__(self, opt):
        self.opt = opt
        self.log_dir = os.path.join(opt.log_dir, opt.name + '_' + opt.model)
        self.writer = SummaryWriter(self.log_dir)

    def log(self, log, step, prefix='Err'):
        for k in log.keys():
            self.writer.add_scalar(prefix + '/' + k, log[k], step)


class Validator():
    def __init__(self, suffix='val'):
        self.best_results = None # best group results
        self.best_p = 1000.0
        self.suffix = suffix
        self.results = None
        self.avg_results = None
    
    def validate(self, model, loader):
        self.global_err = 0
        self.acc = 0 # classification error for segmentation model
        self.relative_mae = np.zeros([1,2])
        self.prec = np.zeros([1,2])
        self.recall = np.zeros([1,2])
        #if 'back' in model.opt.model:
        #    self.prec = np.zeros([1,3])
        #    self.recall = np.zeros([1,3])

        self.p = 0.0
        self.results = None
        self.avg_results  = None
        p = 0
        MAES_1=[]
        MAES_2=[]
        MAES_1_round=[]
        MAES_2_round=[]
        RELATIVEMAE_1=[]
        RELATIVEMAE_2=[]
        Patch_mae1_values=[]
        Patch_mae2_values=[]
        output_sums = []
        target_sums = []
        results_ = {}
        saving = False
        create_table = False
        low1=[]
        low2=[]
        mid1=[]
        mid2=[]
        high1=[]
        high2=[]
        if "Standing" in model.opt.test_json:
            create_table = True
        with torch.no_grad():
            for i, data in enumerate(loader):
                
                p, results, rel_mae = model.validate(data)
                self.p += p # sum of error of all groups 
                self.relative_mae += rel_mae
                self.accumulate(results) # error of indivudual groups
                if 'det' in model.opt.model:
                    self.global_err += 0
                    acc, prec, recall = model.get_acc_prec_recall(model.output.detach(), model.target)
                    self.acc += acc
                    self.prec += prec
                    self.recall += recall
                elif 'direct' not in model.opt.model and 'sep' not in model.opt.model and 'localize' not in model.opt.model:
                    self.global_err += abs(model.dmap.detach().sum()-model.target.detach().sum())
                    acc, prec, recall = model.get_acc_prec_recall(model.seg_output.detach(), model.smap_target)
                    self.acc += acc
                    self.prec += prec
                    self.recall += recall
                else:
                    self.global_err += abs(model.dmap.detach().sum()-model.target.detach().sum())
                    #self.p = self.global_err 
                    
                #SELFMADE SAVING
                if self.suffix == "test":
                    #calculating evaluation metrics
                    
                    MAE1 = abs(model.target.detach()[:,0,:,:].sum()-model.output.detach()[:,0,:,:].sum())
                    MAE2 = abs(model.target.detach()[:,1,:,:].sum()-model.output.detach()[:,1,:,:].sum())
                    MAES_1.append(MAE1.item())
                    MAES_2.append(MAE2.item())
                    MAE1_round = abs(torch.round(model.target.detach()[:,0,:,:].sum())-torch.round(model.output.detach()[:,0,:,:].sum()))
                    MAE2_round = abs(torch.round(model.target.detach()[:,1,:,:].sum())-torch.round(model.output.detach()[:,1,:,:].sum()))
                    MAES_1_round.append(MAE1_round.item())
                    MAES_2_round.append(MAE2_round.item())
                    
                    if model.target.detach()[:,0,:,:].sum()>0:
                        RELATIVEMAE1=MAE1/model.target.detach()[:,0,:,:].sum()#
                        RELATIVEMAE_1.append(RELATIVEMAE1.item())
                    if  model.target.detach()[:,1,:,:].sum() >0:
                        RELATIVEMAE2=MAE2/model.target.detach()[:,1,:,:].sum()#
                        RELATIVEMAE_2.append(RELATIVEMAE2.item())
                    
                    #RELATIVEMAE_1.append(RELATIVEMAE1.item())#
                    #RELATIVEMAE_2.append(RELATIVEMAE2.item())#
                    
                    output_sum = model.output.detach()[:,:,:,:].sum()
                    target_sum = model.target.detach()[:,:,:,:].sum()
                    
                    #Creating table
                    if create_table:
                        if model.target.detach()[:,0,:,:].sum() + model.target.detach()[:,1,:,:].sum() < 40:
                            if model.target.detach()[:,0,:,:].sum()>0:
                                low1.append(RELATIVEMAE1.item())
                            if model.target.detach()[:,1,:,:].sum() >0:
                                low2.append(RELATIVEMAE2.item())
                        elif model.target.detach()[:,0,:,:].sum() + model.target.detach()[:,1,:,:].sum() <= 80:
                            if model.target.detach()[:,0,:,:].sum()>0:
                                mid1.append(RELATIVEMAE1.item())
                            if model.target.detach()[:,1,:,:].sum() >0:
                                mid2.append(RELATIVEMAE2.item())
                            
                        else:
                            if model.target.detach()[:,0,:,:].sum()>0:
                                high1.append(RELATIVEMAE1.item())
                            if model.target.detach()[:,1,:,:].sum() >0:
                                high2.append(RELATIVEMAE2.item())
                            
                   
                    output_sums.append(output_sum.item())
                    target_sums.append(target_sum.item())
                    pmae1 = []
                    pmae2 = []
                    for y in range(0, model.target.shape[2]-model.target.shape[2]%8, math.floor(model.target.shape[2]/8)):#
                        for x in range(0,model.target.shape[3]-model.target.shape[3]%8, math.floor(model.target.shape[3]/8)):#
                            
                            y_true_patch_MAE1 = model.target.detach()[:,0, y:y +  math.floor(model.target.shape[2]/8), x:x + math.floor(model.target.shape[3]/8)].sum()
                            y_true_patch_MAE2 = model.target.detach()[:,1, y:y +  math.floor(model.target.shape[2]/8), x:x + math.floor(model.target.shape[3]/8)].sum()
                            y_pred_patch_MAE1 = model.output.detach()[:,0, y:y +  math.floor(model.target.shape[2]/8), x:x + math.floor(model.target.shape[3]/8)].sum()
                            y_pred_patch_MAE2  =model.output.detach()[:,1, y:y +  math.floor(model.target.shape[2]/8), x:x + math.floor(model.target.shape[3]/8)].sum()
                            
                            Patch_mae1_values.append(np.abs(y_true_patch_MAE1.cpu() - y_pred_patch_MAE1.cpu() ))
                            pmae1.append(np.abs(y_true_patch_MAE1.cpu() - y_pred_patch_MAE1.cpu() ))
                            Patch_mae2_values.append(np.abs(y_true_patch_MAE2.cpu()  - y_pred_patch_MAE2.cpu() ))
                            pmae2.append(np.abs(y_true_patch_MAE2.cpu()  - y_pred_patch_MAE2.cpu() ))
                    
                    #for saving outputs
                    img_path = data["path"][0]
                    annot_path = "/".join(img_path.split("/")[0:-2])+"/annotations/annotations.json"
                    
                    
                    img = mpimg.imread(img_path)
                    with open(annot_path, 'r') as f:
                        points_data = json.load(f)

                    y_cord1 = points_data[img_path.split("/")[-1]][0]["y"]
                    x_cord1 = points_data[img_path.split("/")[-1]][0]["x"]

                    y_cord2 = points_data[img_path.split("/")[-1]][1]["y"]
                    x_cord2 = points_data[img_path.split("/")[-1]][1]["x"]
                    # Define the colormap to modify
                    cmap = plt.get_cmap('plasma')

                    # Define the alpha values to use for different intensity levels
                    alphas = np.linspace(0.3, 0.7, cmap.N//4)
                    alphas2 = np.linspace(0.7, 1.0, cmap.N - cmap.N//4)
                    cmap_colors = cmap(np.arange(cmap.N))
                    cmap_colors[:, -1] = np.concatenate((alphas,alphas2))

                    # Create the modified colormap
                    new_cmap = mcolors.ListedColormap(cmap_colors)

                    transform = transforms.Resize((img.shape[0],img.shape[1]))


                    output_0 = np.round(np.sum(model.output[0,0].cpu().detach().numpy()))
                    target_0 = np.round(np.sum(model.target[0,0].cpu().detach().numpy()))
                    gt_0 = len(x_cord1)



                    output_1 = np.round(np.sum(model.output[0,1].cpu().detach().numpy()))
                    target_1 = np.round(np.sum(model.target[0,1].cpu().detach().numpy()))
                    gt_1 = len(x_cord2)


                    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
                    fig.suptitle('ImagePath:{img_path} \n Acc: {acc}'.format(img_path=img_path,acc=acc))
                    axs[0,0].imshow(img)
                    #axs[0,0].scatter(x_cord1,y_cord1, color="red")
                    axs[0,0].imshow(transform(model.output)[0,0,].cpu().detach().numpy(), cmap=new_cmap)
                    axs[0,0].set_title('OUTPUT (class 0)\n OutputCount: {output_0}'.format(output_0=output_0))

                    axs[0,1].imshow(img)
                    #axs[0,1].scatter(x_cord2,y_cord2, color="red")
                    axs[0,1].imshow(transform(model.output)[0,1,].cpu().detach().numpy(), cmap=new_cmap)
                    axs[0,1].set_title('OUTPUT (class 1)\n OutputCount: {output_1}'.format(output_1=output_1))

                    axs[1,0].imshow(img)
                    #axs[1,0].scatter(x_cord1,y_cord1, color="red")
                    axs[1,0].imshow(transform(model.target)[0,0,].cpu().detach().numpy(), cmap=new_cmap)
                    axs[1,0].set_title('TARGET (class 0) \n TargetCount: {target_0}, GT: {gt_0}'.format(target_0=target_0,gt_0=gt_0))

                    axs[1,1].imshow(img)
                    #axs[1,1].scatter(x_cord2,y_cord2, color="red")
                    axs[1,1].imshow(transform(model.target)[0,1,].cpu().detach().numpy(), cmap=new_cmap)
                    axs[1,1].set_title('TARGET (class 1)\n TargetCount: {target_1}, GT: {gt_1}'.format(target_1=target_1,gt_1=gt_1))
                    plt.tight_layout()
                    plt.savefig("TestResults/" + model.opt.name +"/"+ img_path.split("/")[-1])
                    plt.close()
                    
                    if model.opt.save_name == img_path.split("/")[-1]:
                        theoutput = model.output.cpu().detach().numpy()
                        the_img = img_path.split("/")[-1]
                        saving = True
                    
                    
                    results_[img_path.split("/")[-1]] = {"CMAE": str((MAE1.item()+MAE2.item())/2),
                         "RMAE1": str((MAE1/model.target.detach()[:,0,:,:].sum()).item() if model.target.detach()[:,0,:,:].sum() > 0 else -1),
                         "RMAE2": str((MAE2/model.target.detach()[:,1,:,:].sum()).item() if model.target.detach()[:,1,:,:].sum() > 0 else -1),
                         "PMAE1": str(np.mean(pmae1)),
                         "PMAE2": str(np.mean(pmae2))}
                    
                    
                    
        print("output_sum", np.mean(output_sums))
        print("target_sum", np.mean(target_sums))
        
        print("MAE1:", np.mean(MAES_1))
        print("MAE2:", np.mean(MAES_2))
        print("CMAE:", (np.mean(MAES_1)+np.mean(MAES_2))/2)
        
        print("MAE1_round:", np.mean(MAES_1_round))
        print("MAE2_round:", np.mean(MAES_2_round))
        print("CMAE_round:", (np.mean(MAES_1_round)+np.mean(MAES_2_round))/2)
        
        print("RELATIVE MAE", (np.mean(RELATIVEMAE_1)+ np.mean(RELATIVEMAE_2))/2)#
        print("Relative MAE1", np.mean(RELATIVEMAE_1))#
        print("Relative MAE2", np.mean(RELATIVEMAE_2))#
        
        print("patch MAE1", np.mean(Patch_mae1_values))
        print("patch MAE2", np.mean(Patch_mae2_values))
        print("patch MAE", (np.mean(Patch_mae1_values)+np.mean(Patch_mae2_values))/2)
        
        if create_table:
            print("#########")
            print("Low:", (np.mean(low1) + np.mean(low2))/2)
            print("Mid:", (np.mean(mid1) + np.mean(mid2))/2)
            print("High:", (np.mean(high1) + np.mean(high2))/2)
            
        #Saving metrics
        json_object = json.dumps(results_)

        if self.suffix == "test":
            if "Towards" in annot_path:
                # Writing to sample.json
                with open("../TowardsResults/{name}.json".format(name=model.opt.name), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../TowardsOutputs/"+the_img[:-4]+ "_" + model.opt.name+".npy",theoutput)

            if "Waiting" in annot_path:
                # Writing to sample.json
                with open("../WaitingResults/{name}.json".format(name=model.opt.name), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../WaitingOutputs/"+the_img[:-4]+ "_" + model.opt.name+".npy",theoutput)

            if "Standing" in annot_path:
                # Writing to sample.json
                with open("../StandingResults/{name}.json".format(name=model.opt.name), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../StandingOutputs/"+the_img[:-4]+ "_" + model.opt.name+".npy",theoutput)


            if "Violent" in annot_path:
                # Writing to sample.json
                with open("../ViolentResults/{name}.json".format(name=model.opt.name), "w") as outfile:
                    outfile.write(json_object)
                if saving == True:
                    np.save("../ViolentOutputs/"+the_img[:-4]+ "_" + model.opt.name+".npy",theoutput)
    
        
        self.p = self.p/len(loader)/model.opt.output_cn
        self.relative_mae = self.relative_mae/len(loader)

        #pdb.set_trace()
        avg_results = self.avg(len(loader))
        if self.p < self.best_p:
            self.best_p = self.p
            self.best_results = avg_results
            model.save('best' + self.suffix)
            best = True
        else:
            best = False
        print('#' + self.suffix, end=': ')
        for k in avg_results.keys():
            print("%s: %.4f" % (k, avg_results[k]), end=', ')
        print('Avg %s MAE: %.4f, Relative Avg.: %.4f ' % (self.suffix, self.p, self.relative_mae.mean()))
        if 'direct' not in model.opt.model and 'sep' not in model.opt.model and 'localize' not in model.opt.model:
            print('#' + self.suffix, end=': ')
            print('total:  %.4f, Acc: %.2f' % (self.global_err / len(loader), (self.acc / len(loader))))
            print('#' + self.suffix, end=': ')
            print('Precsion and recall', end=', ')
            print(self.prec/len(loader), self.recall/len(loader))
        elif 'localize' in model.opt.model:
            print('total:  %.4f' % (self.global_err / len(loader)))
        if self.best_results is not None:
            print('#' + self.suffix, end=': ')
            for k in self.best_results.keys():
                print("Best %s: %.4f" % (k, self.best_results[k]), end=', ')
        print('best %s MAE: %.4f' % (self.suffix, self.best_p))
        return best

    def get_info(self):
        info = OrderedDict()
        info['performance'] = self.p
        info['best_performance'] = self.best_p
        for k in self.avg_results.keys():
            info[k] = self.avg_results[k]
        return info

    def accumulate(self, results):
        if self.results is None:
            self.results = results
        else:
            for k in results.keys():
                self.results[k] += results[k]

    def avg(self, n):
        self.avg_results = self.results
        for k in self.results.keys():
            self.avg_results[k] /= n
        return self.avg_results
        
class Printer():

    def display(self, counter, timer, model):
        counter.display()
        timer.display_steps()
        model.display()

class Counter():
    def __init__(self):
        self.curr_epochs = 0
        self.curr_steps = 0
        self.total_steps = 0

    def update_epoch(self):
        self.curr_epochs += 1
        self.curr_steps = 0
    
    def update_step(self):
        self.curr_steps += 1
        self.total_steps += 1

    def get_epochs(self):
        return self.curr_epochs

    def get_total_steps(self):
        return self.total_steps

    def get_steps(self):
        return self.curr_steps

    def display(self):
        print("Epoch: %d, steps: %d" % (self.get_epochs(), self.get_steps()), end=', ')

class Timer():
    def __init__(self):
        self.steps = 0
        self.epochs = 0

        self.start_time = time.time()

        self.total_time = 0
        self.total_data_time = 0
        self.total_opt_time = 0

        self.last_time = time.time()

    def get_epoch_time(self):
        if self.epochs == 0:
            return 0
        return (time.time()-self.start_time)/self.epochs

    def get_data_time(self):
        if self.steps == 0:
            return 0
        return self.total_data_time/self.steps

    def get_opt_time(self):
        if self.steps == 0:
            return 0
        return self.total_opt_time/self.steps

    def update_epoch(self):
        self.epochs += 1

    def update_data(self):
        curr_time = time.time()
        self.total_data_time += (curr_time-self.last_time)
        self.last_time = curr_time

    def update_step(self):
        curr_time = time.time()
        self.total_opt_time += (curr_time-self.last_time)
        self.last_time = curr_time
        self.steps += 1

    def get_times(self):
        return self.get_epoch_time(), self.get_data_time(), self.get_opt_time()

    def display_steps(self):
        print("Data time: %.4f, optimize time: %.4f" % (self.get_data_time(), self.get_opt_time()), end=', ')

    def display_epochs(self):
        print("Epoch time: %.4f" % (self.get_epoch_time()))


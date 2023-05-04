import numpy as np
import torch
from vit import VisionTransformer
import torch.nn as nn
from torchsummary import summary
from torchvision import transforms, datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
import sys
import torchvision
from collections import Counter
from tqdm import tqdm
import time
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from sklearn.metrics import accuracy_score
import json
from torch.nn import DataParallel
import json
import os
import argparse
import yaml
from imagenet32_dataloader import ImageNet32

parser = argparse.ArgumentParser()
parser.add_argument("--config", help = "path of the training configuartion file", required = True)
args = parser.parse_args()

#Reading the configuration file
with open(args.config, 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


torch.manual_seed(123) 
def download_data(apply_transforms = True, valid_ratio = config['valid_ratio'], path=config['dataset_path']):
    """
    This funtion downloads the dataset in the given path. Applies the transformations if bool True.
    The train set is split into valid and an train set, depending on the validation ratio.
    Returns --> Dataset object, that can be passed to a dataloader.
    """
    if apply_transforms:
        if config['dataset'] == "cifar":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                #Normalization values picked up from a discussion @ https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    if config['dataset'] == 'cifar':
        trainset = datasets.CIFAR100(root = path + '/train', train = True, download=True, transform=transform)

        valid_ratio = valid_ratio
        n_train_samples = int(len(trainset) * (1-valid_ratio))
        n_valid_samples = len(trainset) - n_train_samples
        print(f"There are {n_train_samples} Train samples, and {n_valid_samples} in the Dataset.")
        trainset, validset = data.random_split(trainset, [n_train_samples, n_valid_samples])
        print(type(trainset))
        return trainset, validset
    else:
        trainset = ImageNet32(root = config['imagenet_path'], train = True, transform =transform)
        print("-----------------------0-0-0-0-0-0-0---------------------0-0-0-0-0-0-0-0---------")
        valid_ratio = valid_ratio
        n_train_samples = int(len(trainset) * (1-valid_ratio))
        n_valid_samples = len(trainset) - n_train_samples
        print(f"There are {n_train_samples} Train samples, and {n_valid_samples} in the Dataset.")
        trainset, validset = data.random_split(trainset, [n_train_samples, n_valid_samples])
        print(type(trainset))
        return trainset, validset

trainset, validset = download_data()

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size = config['train_batch_size'],
                                          shuffle = True)

validloader = torch.utils.data.DataLoader(validset,
                                          batch_size = config['valid_batch_size'],
                                          shuffle = True)


def visualize_data(dataloader):
    def imshow(img):
        npimg = img.numpy()
        plt.figure(figsize= (10, 10))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images))

def check_validation_split(validloader):
    all_labels = []
    for _, label in validloader:
        all_labels.append(label.detach().numpy().item())
    
    labels_dict = Counter(all_labels)
    assert len(labels_dict.keys()) == 100, "All classes not present in validation set."

device = "cuda" if torch.cuda.is_available() else "cpu"

custom_config = config['custom_config']

model = VisionTransformer(**custom_config).to(device=device)

if(config["dataset"] == 'cifar'):
    model = nn.Sequential(model,
                        nn.Linear(1000,100, bias=True))


print(summary(model, (3, 32, 32)))


def train(epochs, optimizer, model, train_data_loader, val_data_loader, save_name, scheduler):
    start_time = time.time()
    train_epochs = epochs
    train_loss_track = {}
    valid_loss_track = {}

    criterion = nn.CrossEntropyLoss()
    #prog_bar = tqdm(range(train_epochs))
    mb = master_bar(range(train_epochs))
    for epoch in mb:
        model.train()
        tot_train_loss = 0
        
        preds = []
        gt = []
        
        for train_batch, (data, target) in enumerate(progress_bar(train_data_loader, parent=mb)):
            #Training Batch

            data = data.to(device=device)
            target = target.to(device=device)
           
            #fwd pass to the data
            scores = model(data)
            
            loss = criterion(scores, target)
            
            tot_train_loss+=loss.item()

            #Backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            #print(f"Train loss after Epoch{epoch}: {tot_train_loss/batch}")

            with torch.no_grad():
                #fwd pass to the data
                model.eval()
                scores = model(data)
                _ , prediction = scores.max(1)
                preds.extend(list(prediction.cpu().detach().numpy()))
                gt.extend(list(target.cpu().detach().numpy()))
            
            # if(train_batch == 1):
            #     break
        train_epoch_accuracy = accuracy_score(preds, gt)

        train_loss_track[epoch] = tot_train_loss/len(train_data_loader)
        
        tot_val_loss = 0
        preds = []
        gt = []
        for val_batch, (data, target) in enumerate(val_data_loader):
            #Validation Batch
            model.eval()
            data = data.to(device=device)
            target = target.to(device=device)
            with torch.no_grad():
                #fwd pass to the data
                scores = model(data)
                _ , prediction = scores.max(1)
                preds.extend(list(prediction.cpu().detach().numpy()))
                gt.extend(list(target.cpu().detach().numpy()))
                

            loss = criterion(scores, target)
            tot_val_loss+=loss.item()
        #print(len(preds))
        #print(preds)
        #print(len(gt))
        #print(gt)
        val_epoch_accuracy = accuracy_score(preds, gt)
        
        val_batch += 1
        if(epoch==0):
            min_val_loss = tot_val_loss/val_batch
            best_epoch = epoch
            if not os.path.exists(config['save_model_path']):
                os.makedirs(config['save_model_path'])
            torch.save(model,config['save_model_path']+save_name+f'_{epoch}.h5')
            log_best_train_loss = tot_train_loss/train_batch
            with open(config["config_write_path"]+"config_ViT.json", 'w') as outfile:
                json.dump(custom_config, outfile)
        else:
            if(tot_val_loss/val_batch < min_val_loss):
                min_val_loss = tot_val_loss/val_batch
                best_epoch = epoch
                torch.save(model,config['save_model_path']+save_name+f'_{epoch}.pth')
                log_best_train_loss = tot_train_loss/train_batch
        
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]

        print(f'Before LR Scheduler --> {before_lr} || After LR Scheduler --> {after_lr}')
        #prog_bar.set_description(f"Epoch {epoch}/{epochs} : Train Loss {epoch}: {tot_train_loss/train_batch} | Val Loss {tot_val_loss/val_batch}")
        train_loss = tot_train_loss/train_batch
        val_loss = tot_val_loss/val_batch
        #print(f"Valid loss after Epoch{epoch}: {tot_val_loss/batch}")    
        valid_loss_track[epoch] = tot_val_loss/val_batch
        print(f'--------- After Epoch {epoch} : Train Loss {train_loss} ||  Validation Loss {val_loss} || Validation Accuracy {val_epoch_accuracy} || Train Accuracy {train_epoch_accuracy}---------')
        # wandb.log({"train/Train_Loss":train_loss,
        #            "val/Valid_Loss": val_loss,
        #            "val/Valid_Accuracy": val_epoch_accuracy,
        #            "train/Train_Accuracy": train_epoch_accuracy})
    tot_time = time.time()-start_time
    
    return train_loss_track, valid_loss_track, tot_time, min_val_loss, log_best_train_loss, best_epoch, val_epoch_accuracy

optimizer = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
scheduler_expoLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)


#wandb.init(project=config['wandb_project'], config=config, name = config['wandb_run_name'])
vit_train_log, vit_val_log, vit_train_time, vit_min_val_loss, vit_log_best_train_loss, vit_best_epoch, vit_val_acc = train(epochs = config['epochs'], 
                                                                                                              optimizer=optimizer,
                                                                                                              model=model,
                                                                                                              train_data_loader=trainloader,
                                                                                                              val_data_loader=validloader,
                                                                                                              save_name=config['save_model_name'],
                                                                                                              scheduler=scheduler_expoLR)



#print(summary(model_custom, (3, 384, 384)))


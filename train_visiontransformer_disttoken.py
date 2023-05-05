import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import os
from pickle import *
from sklearn.metrics import *
import torch.optim as optim
import time
import numpy as np
from vit import VisionTransformer
from deit import DataEfficientImageTransformer as DEIT
import wandb
import yaml
import argparse
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

if config['transformations']:
    if config['dataset'] == 'cifar':
        transforms = transforms.Compose([transforms.ToTensor(),
				      					 transforms.RandomHorizontalFlip(p=0.1),
									     transforms.RandomVerticalFlip(p=0.1),
									     transforms.RandomRotation(degrees=(0,10)),
									     transforms.Normalize((0.2675, 0.2565, 0.2761),(0.5071, 0.4867, 0.4408))])
    else:
        transforms = transforms.Compose([transforms.ToTensor(),
				     					 transforms.RandomHorizontalFlip(p=0.1),
									     transforms.RandomVerticalFlip(p=0.1),
									     transforms.RandomRotation(degrees=(0,10)),
									     transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    if config['dataset'] == 'cifar':
        trainset = datasets.CIFAR100(root = config['paths']['dataset_download_path'], train = True, transform =transforms ,download = True)
    else:
        trainset = ImageNet32(root = config['paths']['dataset_download_path'], train = True, transform =transforms)
else:
    if config['dataset'] == 'cifar':
        trainset =datasets.CIFAR100(root = config['dataset_download_path'], train = True, transform =None ,download = True)
    else:
        trainset =ImageNet32(root = config['paths']['dataset_download_path'], train = True, transform =None)

trainset, valset = torch.utils.data.random_split(trainset, [round(config['val_split']*len(trainset)), len(trainset)-round(config['val_split']*len(trainset))])

train_loader = DataLoader(trainset, batch_size = config['batch_size'], shuffle = True)
val_loader = DataLoader(valset, batch_size = config['batch_size'], shuffle = True)

# Setting Up the training Optimizer
def getOptimizer(model, lr, mode, momentum = 0.09, weight_decay = 1e-4):
  """
  returning the specific training optimizer
  """
  if mode == "sgd":
    optimizer = optim.SGD(model.parameters(), lr = lr)
  elif mode == "sgd_m":
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
  elif mode == "sgd_l2":
    optimizer = optim.SGD(model.parameters(), lr = lr , weight_decay = weight_decay)
  elif mode == "rms":
    optimizer =optim.RMSprop(model.parameters(), lr=lr)
  elif mode == "adam":
    optimizer = optim.Adam(model.parameters(), lr = lr)
  return optimizer, mode

def computeKL(student_out, teacher_out, target, T = 3.0):
	ce_loss = F.cross_entropy(student_out, target)
	kl_loss = F.kl_div(F.log_softmax(student_out/T, dim = 1), 
								F.softmax(teacher_out/T, dim = 1))

	return ce_loss, kl_loss

def eval_model(teacher_model, student_model, val_loader,T = 3.0, device = "cuda", distillation_type = "soft"):
	ce_val = 0
	kl_val = 0
	student_model.eval()

	for batch_index, (data, target) in enumerate(val_loader):
		data = data.to(device = device)
		target = target.to(device = device)

		with torch.no_grad():
			score_student_cls, score_student_dist = student_model(data)
			score_teacher = teacher_model(data)
		# loss = F.cross_entropy(score, target)
		if distillation_type == 'soft':
			ce_loss_val, kl_loss_val = computeKL(score_student_cls, score_teacher,target,T)
			ce_val += ce_loss_val.item()
			kl_val += kl_loss_val.item()
		elif distillation_type == 'hard':
			ce_loss_val, _ = computehardLabel(score_student_cls, score_student_dist, score_teacher, target)
			ce_val += ce_loss_val.item()
		# if batch_index % 10 == 0:
		# 	print(f"validationBatchLoss:{batch_index}\t ce_loss :{ce_val/(batch_index+1)}\t kl_loss:{kl_val/(batch_index+1)}", flush = True)
	if (distillation_type == 'soft'):
		return ce_val, kl_val
	elif (distillation_type == 'hard'):
		return ce_val

def inference_model(student_model, loader, device= "cuda"):

	top1_acc_cls_token = 0
	top1_acc_dist_token = 0
	top1_acc_cls_dist_token = 0
	

	student_model.eval()
	with torch.no_grad():
		for batch_index, (data, gt) in enumerate(loader):
			data = data.to(device = device)
			gt = gt.to(device = device)

			scores_cls_token, scores_dist_token = student_model(data)
			scores_cls = F.softmax(scores_cls_token, dim =1)
			scores_cls = scores_cls.cpu().detach().numpy()

			scores_dist = F.softmax(scores_dist_token, dim =1)
			scores_dist = scores_dist.cpu().detach().numpy()

			scores_cls_dist = scores_cls_token + scores_dist_token
			scores_cls_dist = F.softmax(scores_cls_dist, dim =1)
			scores_cls_dist = scores_cls_dist.cpu().detach().numpy()

			gt = gt.cpu().detach().numpy()
			labels = np.arange(0,100)

			top1_cls = top_k_accuracy_score(gt,scores_cls, k=1, labels = labels)*100
			top1_acc_cls_token += top1_cls

			top1_dist = top_k_accuracy_score(gt,scores_dist, k=1, labels = labels)*100
			top1_acc_dist_token += top1_dist

			top1_cls_dist = top_k_accuracy_score(gt,scores_cls_dist, k=1, labels = labels)*100
			top1_acc_cls_dist_token += top1_cls_dist

		
		return (top1_acc_cls_token/len(loader), top1_acc_dist_token/len(loader), top1_acc_cls_dist_token/len(loader))

def computehardLabel(student_out_cls, student_out_dist, teacher_out, target):
	ce_loss = F.cross_entropy(student_out_cls,target)
	hard_distil = F.cross_entropy(student_out_dist, teacher_out.argmax(dim = 1))

	return ce_loss, hard_distil


def train(teacher_model, student_model,num_epochs,train_loader,val_loader,optimizer,criterion,T = 3.0, alpha =0.1, model_name = config['model_name'], model_path = config['paths']['model_save_path'], device = "cuda", distillation_type = 'soft'):
	train_loss_ce = {}
	train_loss_kl = {}
	train_loss_distil = {}
	top_1_acc = {}
	lr_scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)

	dur = []
	val_ce_loss = {}

	for epoch in range(num_epochs):
		t0 = time.time()
		ce_loss = 0
		distill_loss = 0
		kl_loss = 0
		student_model.train()
		teacher_model.eval()

		for batch_index ,(data, target) in enumerate(train_loader):
			optimizer.zero_grad()
			data = data.to(device = device)
			target = target.to(device = device)
			with torch.no_grad():
				out_teacher = teacher_model(data)
			out_student_cls_token, out_student_dist_token = student_model(data)

			# crossEntropy_loss = criterion(out_student, target)
			if (distillation_type == 'soft'):
				
				ce_loss , kl_div_loss = computeKL(out_student_cls_token, out_teacher,target, T)

				loss = (1-alpha) * ce_loss + (alpha * T **2) * kl_div_loss
			
			elif (distillation_type == 'hard'):
				ce_loss, hard_distill = computehardLabel(out_student_cls_token, out_student_dist_token, out_teacher, target)
				loss = (0.5*ce_loss) + (0.5*hard_distill)

			
			loss.backward()
			optimizer.step()

			ce_loss += ce_loss.item()
			distill_loss += loss.item()
			if distillation_type == 'soft':
				kl_loss += kl_div_loss.item()

			# if batch_index % 100 == 0:
			# 	print(f"train_batch : {batch_index}\t ce_loss: {ce_loss/(batch_index+1)}\t kl_loss: {kl_loss/(batch_index+1)} \t distill_loss : {distill_loss/(batch_index+1)}")

		ce_loss_val, kl_loss_val = eval_model(teacher_model, student_model, val_loader,T)
		val_acc_cls, val_acc_dist, val_acc_cls_dist = inference_model(student_model, val_loader)
		train_acc_cls, train_acc_dist, train_acc_cls_dist = inference_model(student_model, train_loader)
		
		

		val_ce_loss[epoch+1] =  ce_loss_val/(len(val_loader))
		dur.append(time.time() - t0)
		curr_lr = optimizer.param_groups[0]['lr']

		cross_entropy_loss_train_log = ce_loss / len(train_loader)
		
		dist_loss_log = distill_loss/len(train_loader)
		cross_entropy_loss_val_log = ce_loss_val/len(val_loader)
		if distillation_type == 'soft':
			kl_loss_train_log = kl_loss/len(train_loader)
			kl_loss_val_log = kl_loss_val/(len(val_loader))

	
		lr_log = curr_lr
		

		if distillation_type == "soft":
			if(config["wandb"]["enable"]):
				wandb.log({"epoch": epoch+1,
						"train/CE_train:": cross_entropy_loss_train_log,
						"train/KL_Train": kl_loss_train_log,
						"train/Dist_Train": dist_loss_log,
						"val/CE_val": cross_entropy_loss_val_log,
						"val/KL_Val": kl_loss_val_log,
						"val/Val_Acc": val_acc_cls,
						"lr": lr_log,
						"train/Train_Acc": train_acc_cls})
		elif distillation_type == 'hard':
			if(config["wandb"]["enable"]):
				wandb.log({"epoch": epoch+1,
						"train/CE_train:": cross_entropy_loss_train_log,
						"train/Dist_Train": dist_loss_log,
						"val/CE_val": cross_entropy_loss_val_log,
						"val/Val_Acc_cls_token": val_acc_cls,
						"lr": lr_log,
						"train/Train_Acc_cls_token": train_acc_cls,
						"train/Train_Acc_dist_token": train_acc_dist,
						"val/Val_Acc_dist_token": val_acc_dist,
						"train/Train_Acc_cls_dist_token": train_acc_cls_dist,
						"val/Val_Acc_cls_dist_token": val_acc_cls_dist })
		
		if distillation_type == "soft":

			print(f'Epoch {epoch+1} \t CE@train: {ce_loss / len(train_loader)} \t KL@train : {kl_loss/len(train_loader)} \t distill@train: {distill_loss/len(train_loader)}\t CE@val: {ce_loss_val/len(val_loader)} kl@val :{kl_loss_val/(len(val_loader))} \t val_acc@1: {val_acc_cls} \t LR:{curr_lr} \t Time(s):{np.mean(dur)}', flush = True)
		elif distillation_type == 'hard':

			print(f'Epoch {epoch+1} \t CE@train: {ce_loss / len(train_loader)} \t distill@train: {distill_loss/len(train_loader)}\t CE@val: {ce_loss_val/len(val_loader)} \t val_acc@1: {val_acc_cls_dist} \t LR:{curr_lr} \t Time(s):{np.mean(dur)}', flush = True)

		lr_scheduler.step()

		torch.save(student_model.state_dict(),model_path+"/"+model_name +"_" +str(epoch+1)+'.h5')


device = "cuda" if torch.cuda.is_available() else "cpu"

if config["transormer_config"]["use_default_vit_B"]:
	
	custom_config = {
        "img_size": 32,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 768,
        "depth": 12,
        "n_heads": 12,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

elif config["transormer_config"]["use_default_vit_S"]:
	
	custom_config = {
        "img_size": 32,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 384,
        "depth": 12,
        "n_heads": 6,
        "qkv_bias": True,
        "mlp_ratio": 4,
}

elif config["transormer_config"]["use_default_vit_Ti"]:
	
	custom_config = {
        "img_size": 32,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 192,
        "depth": 12,
        "n_heads": 3,
        "qkv_bias": True,
        "mlp_ratio": 4,
}
else:
	
    custom_config = {
            "img_size": 32,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
    }

if config['dataset'] == 'cifar':
	parent_model = models.regnet_y_16gf()
else:
	parent_model = models.regnet_y_16gf('DEFAULT')
num_ftrs = parent_model.fc.in_features
if config['dataset'] == 'cifar':
	parent_model.fc = nn.Linear(num_ftrs, 100)
	parent_model.load_state_dict(torch.load(config['paths']['teacher_model_path']))

print("++++++++ PARENT MODEL +++++++++")
print(parent_model)
parent_model.to(device=device)


student_model = DEIT(**custom_config)

print("++++++++ STUDENT TRANSFORMER MODEL +++++++++")
print(student_model)
student_model.to(device=device)


criterion = nn.CrossEntropyLoss()
optimizer, _ = getOptimizer(student_model, config['initial_lr'], config['optimizer'])
num_epochs = 100

if(config["wandb"]["enable"]):
    wandb.init(project=config['wandb']['project_name'],
            name = config['wandb']['run_name'],
            config = config)
    
train(parent_model, student_model, num_epochs, train_loader, val_loader, optimizer, criterion, distillation_type=config['distillation_type'])

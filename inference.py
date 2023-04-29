import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
# import timm
import torch.nn.functional as F
import os
import torchvision.models as models
# from teacher_model_18 import teacher_model
from sklearn.metrics import *
from sklearn.metrics import top_k_accuracy_score
import numpy as np
from vit import VisionTransformer
from deit import DataEfficientImageTransformer as DEIT
torch.manual_seed(123)



def getInferenceHardToken(model, loader, acc_mode, per_batch, isPrintBatchAcc , device):
	
	model.eval()
	cls_token_acc_log = 0
	distill_token_acc_log = 0
	cls_distill_token_acc_log = 0
	with torch.no_grad():
		for batch_index , (data, gt) in enumerate(loader):
			data = data.to(device)
			gt = gt.to(device)
			
			scores_cls_token, scores_distill_token = model(data)
			scores_cls_distill_token = scores_cls_token + scores_distill_token

			scores_cls_token =F.softmax(scores_cls_token, dim = 1)
			scores_distill_token = F.softmax(scores_distill_token, dim = 1)
			scores_cls_distill_token = F.softmax(scores_cls_distill_token, dim = 1)

			scores_cls_token = scores_cls_token.cpu().detach().numpy()
			scores_distill_token = scores_distill_token.cpu().detach().numpy()
			scores_cls_distill_token = scores_cls_distill_token.cpu().detach().numpy()

			gt = gt.cpu().detach().numpy()
			labels = np.arange(0,100)
			
			if acc_mode == "top1":
				cls_token_acc = top_k_accuracy_score(gt,scores_cls_token, k=1, labels = labels)*100
				distill_token_acc = top_k_accuracy_score(gt, scores_distill_token, k = 1, labels = labels)*100
				cls_distill_token_acc = top_k_accuracy_score(gt, scores_cls_distill_token,k = 1, labels = labels)*100

			if acc_mode == "top3":
				cls_token_acc = top_k_accuracy_score(gt,scores_cls_token, k=3, labels = labels)*100
				distill_token_acc = top_k_accuracy_score(gt, scores_distill_token, k = 3, labels = labels)*100
				cls_distill_token_acc = top_k_accuracy_score(gt, scores_cls_distill_token,k = 3, labels = labels)*100

			if acc_mode == "top5":
				cls_token_acc = top_k_accuracy_score(gt,scores_cls_token, k=5, labels = labels)*100
				distill_token_acc = top_k_accuracy_score(gt, scores_distill_token, k = 5, labels = labels)*100
				cls_distill_token_acc = top_k_accuracy_score(gt, scores_cls_distill_token,k = 5, labels = labels)*100
			
			if isPrintBatchAcc == True:
				if batch_index % per_batch == 0:
					print(f"batch_index: {batch_index}\t acc_cls_token :{cls_token_acc}\t acc_distill_token :{distill_token_acc}\t acc_cls_distill_token :{cls_distill_token_acc}")

			cls_token_acc_log += cls_token_acc
			distill_token_acc_log += distill_token_acc
			cls_distill_token_acc_log += cls_distill_token_acc

		return (cls_token_acc_log/(batch_index+1), distill_token_acc_log/(batch_index+1), cls_distill_token_acc_log/(batch_index+1), acc_mode)



def getInference(model, loader, acc_mode,per_batch, isPrintBatchAcc, device):
	model.eval()
	out_acc = 0

	for batch_index ,(data, gt) in enumerate(loader):
		data = data.to(device)
		gt = gt.to(device)

		scores = model(data)
		scores = F.softmax(scores, dim = 1)
		scores = scores.cpu().detach().numpy()
		gt = gt.cpu().detach().numpy()

		labels = np.arange(0,100)

		if acc_mode == "top1":
			acc = top_k_accuracy_score(gt,scores, k=1, labels = labels)*100
			
		if acc_mode == "top3":
			acc = top_k_accuracy_score(gt,scores, k=3, labels = labels)*100
			

		if acc_mode == "top5":
			acc = top_k_accuracy_score(gt,scores, k=5, labels = labels)*100
			
		if isPrintBatchAcc == True:
			if batch_index % per_batch == 0:
				print(f"batch_index: {batch_index}\t acc :{acc}")

		out_acc += acc
	return (out_acc/(batch_index+1), acc_mode)

def doInference(loader, mode = "distilled_token", model= None,  acc_mode = "top1",per_batch = 3, isPrintBatchAcc = True, device="cuda"):
	if mode == "distilled_token":

		final_model = model
		
		loader = loader
		device = device
		acc_mode = acc_mode

		
		if final_model == None:
			print("No model is provided!!")
		else:
			clsTokenAcc, distillTokenAcc, clsDistillTokenAcc, acc_mode = getInferenceHardToken(final_model,loader,acc_mode,per_batch, isPrintBatchAcc, device)
			print("Final Accuracy" + acc_mode)
			print(f"cls_token_acc:{clsTokenAcc}\t distill_token_acc:{distillTokenAcc}\t cls_distill_token_acc:{clsDistillTokenAcc}")

	else:
		final_model = model
		
		loader = loader
		device = device
		
		if final_model == None:
			print("No model is provided!!")
		else:
			final_acc,acc_mode = getInference(final_model,loader, acc_mode, per_batch,isPrintBatchAcc,  device)
			print("Final Accuracy" + acc_mode)
			print(f"acc:{final_acc}")


	

#apply transformation
transforms = transforms.Compose([transforms.ToTensor(), 
							transforms.Normalize((0.2675, 0.2565, 0.2761),(0.5071, 0.4867, 0.4408))])
#load the test set
testset = datasets.CIFAR100(root = "/ssd_scratch/cvit/varun", train = False, transform = transforms, download = True)
batch_size = 1024
test_dataloaders = DataLoader(testset, batch_size = batch_size, shuffle = "False", num_workers = 4)

#check the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initializing the model
# model = models.regnet_y_16gf()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 100)

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

# model = VisionTransformer(**custom_config)
# model = nn.Sequential(model, nn.Linear(1000,100, bias=True))

model = DEIT(**custom_config)


#calculating the total number of parameters
total_num_param = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters for:{total_num_param}")

#loading the  model
saved_model_path = "/ssd_scratch/cvit/varun/distill_hard_token/selected/vit_b_reg16gf_hard_dist_token_18"
model.load_state_dict(torch.load(saved_model_path))
model.to(device)
doInference(mode = "distilled_token",model= model,loader= test_dataloaders)

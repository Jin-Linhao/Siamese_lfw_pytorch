#!/usr/bin/env python
__author__ = 'LH'

import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
import os

from siamese_19_BCE import SiameseNetwork_BCE
from testnet import testNet

parser = argparse.ArgumentParser(description='PyTorch_Siamese_lfw')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
					help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
					metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=0.00001, type=float,
					metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
					help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_list', default='../data/train.txt', type=str, metavar='PATH',
					help='path to training list (default: ../data/train.txt)')
parser.add_argument('--test_list', default='../data/train.txt', type=str, metavar='PATH',
					help='path to validation list (default: ../data/train.txt)')
parser.add_argument('--save_path', default='../data/', type=str, metavar='PATH',
					help='path to save checkpoint (default: ../data/)')
parser.add_argument('--cuda', default="off", type=str, 
					help='switch on/off cuda option (default: off)')

def imshow(img,text,should_save=False):
	npimg = img.numpy()
	plt.axis("off")
	if text:
		plt.text(75, 8, text, style='italic',fontweight='bold',
			bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()    


def show_plot(iteration,loss):
	plt.plot(iteration,loss)
	plt.show()
	

def default_loader(path):
	img = Image.open(path).convert('L')
	return img


def default_list_reader(fileList):
	imgList = []
	with open(fileList, 'r') as file:
		for line in file.readlines():
			imgshortList = []
			imgPath1, imgPath2, label = line.strip().split(' ')
			
			imgshortList.append(imgPath1)
			imgshortList.append(imgPath2)
			imgshortList.append(label)
			imgList.append(imgshortList)
			
	return imgList


class ImageList(data.Dataset):
	def __init__(self, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
		# self.root      = root
		self.imgList   = list_reader(fileList)
		self.transform = transform
		self.loader    = loader

	def __getitem__(self, index):
		final = []
		[imgPath1, imgPath2, target] = self.imgList[index]
		img1 = self.loader(os.path.join(args.lfw_path, imgPath1))
		img2 = self.loader(os.path.join(args.lfw_path, imgPath2))
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
#         img1 = self.loader(os.path.join(os.path.splitext(imgPath1)[0],imgPath1))
#         img2 = self.loader(os.path.join(os.path.splitext(imgPath2)[0],imgPath2))
#         print target
		return img1, img2, torch.from_numpy(np.array([target],dtype=np.float32))

	def __len__(self):
		return len(self.imgList)


def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = 0.01 * (0.1 ** (epoch//3))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def save_checkpoint(state, filename):
	torch.save(state, filename)


def train(train_dataloader, forward_pass, criterion, optimizer, epoch):
	plot_counter = []
	loss_history = [] 
	iteration_number= 0

	for i, data in enumerate(train_dataloader,0):
		img0, img1 , label = data
		if args.cuda == "off":
			img0, img1 , label = Variable(img0), Variable(img1) , Variable(label)
		else:
			img0, img1 , label = Variable(img0, requires_grad=False).cuda(), Variable(img1, requires_grad=False).cuda() , Variable(label, requires_grad=False).cuda()
		output= forward_pass(img0,img1)
		optimizer.zero_grad()
		forward_pass.zero_grad()
		loss = F.binary_cross_entropy(output, label)
		loss.backward()	
		optimizer.step()
		print("Epoch: {}, current iter: {}/{}\n Current loss {}\n".format(epoch, i, len(train_dataloader), loss.data[0]))
		iteration_number +=1
		# plot_counter.append(iteration_number)
		# loss_history.append(loss.data[0])


def validate(test_dataloader, forward_pass, criterion):
	for i, data in enumerate(test_dataloader,0):
		img0, img1 , label = data
		concatenated = torch.cat((img0, img1),0)	
		if args.cuda == "off":
			img0, img1 , label = Variable(img0), Variable(img1) , Variable(label)
		else:
			img0, img1 , label = Variable(img0.cuda()), Variable(img1.cuda()) , Variable(label.cuda())
		output= forward_pass(img0,img1)

		# imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}, ground truth'.format(output.cpu().data.numpy()[0][0], label))



def main():
	global args     
	args = parser.parse_args()
	train_dataloader = torch.utils.data.DataLoader(
						ImageList(fileList=args.train_list, 
								transform=transforms.Compose([ 
								transforms.Scale((128,128)),
								transforms.ToTensor(),            ])),
						shuffle=True,
						num_workers=args.workers,
						batch_size=args.batch_size)

	test_dataloader = torch.utils.data.DataLoader(
						ImageList(fileList=args.train_list, 
								transform=transforms.Compose([ 
								transforms.Scale((128,128)),
								transforms.ToTensor(),            ])),
						shuffle=True,
						num_workers=args.workers,
						batch_size=args.batch_size)

	if args.cuda == "off":
		forward_pass = testNet()
	else:
		forward_pass = testNet().cuda()
	# forward_pass = SiameseNetwork()
	criterion = nn.BCELoss()
	optimizer = optim.Adam(forward_pass.parameters(), lr = args.learning_rate )

	for epoch in range(args.start_epoch, args.epochs):

		adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train(train_dataloader, forward_pass, criterion, optimizer, epoch)
		validate(test_dataloader, forward_pass, criterion)
		# evaluate on validation set
		# prec1 = validate(val_loader, model, criterion)
		save_name = args.save_path + str(epoch) + '_checkpoint.pth.tar'
		save_checkpoint({
			'epoch': epoch + 1,
	#         'arch': args.arch,
	#         'state_dict': model.state_dict(),
			# 'prec1': prec1,
			}, save_name)


if __name__ == '__main__':
	main()

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import torch
# import torchvision
# from torchvision import utils
# import torchvision.datasets as dset
# from torch.utils.data import Dataset,DataLoader
# import torchvision.transforms as transforms
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import numpy as np
# from torch.autograd import Variable
# import torch.nn as nn
# import time
# import os
# from skimage import io,transform
# import pandas as pd
# import numpy as np
# import torch.nn.functional as F


# #with open("test.txt") as f:
# #    for line in f:
# #        print (line.split()[1])
# #        a = io.imread(os.path.join('lfw\lfw',line.split()[1]))
# #        plt.figure()
# #        plt.imshow(a)
# #        plt.show()



# class Dataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, txt_file, root_dir, transform=None):
#         """
#         Args:
#             txt_file (string): Path to the txt file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """

#         self.pairs = pd.read_csv(txt_file, header=None, sep=r"\s+")
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
        
#         img_name1 = os.path.join(self.root_dir, self.pairs.ix[idx, 0])
#         im1 = io.imread(img_name1)
        
#         img_name2 = os.path.join(self.root_dir,self.pairs.ix[idx, 1])
#         im2 = io.imread(img_name2)

#         labels = self.pairs.ix[idx, 2]
        

        
#         if self.transform is not None:
#             image1 = self.transform(im1)
#             image2 = self.transform(im2)
            
#         sample = {'image1': image1, 'image2': image2, 'labels': float(labels)}
#             #sample = self.transform(sample)

#         return sample


# class Rescale(object):
#     """Rescale the image in a sample to a given size.

#     Args:
#         output_size (tuple or tuple): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, image):

#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size

#         new_h, new_w = int(new_h), int(new_w)

#         img = transform.resize(image, (new_h, new_w))
#         return img  
    
    
# class Flatten(nn.Module):
#     def __init__(self, size):
#         super(Flatten, self).__init__()
#         self.size = size
        
#     def forward(self, tensor):
#         flat = tensor.view(N,self.size)
#         return flat
        
  
# print("ddd")
# time.sleep(2)
# trans = transforms.Compose([Rescale(128),transforms.ToTensor()])
# train_set = Dataset(txt_file = "../data/train.txt", root_dir="../lfw", transform=trans)
# N = 32



# D_in = 128 * 128
# C_in = 3

# Conv1_C_out = 64
# Conv1_Ker = 5
# Conv1_stride = 1
# Conv1_padding = 2

# Batch1_f = 64

# Max1_ker = 2
# Max1_stride = 2

# Conv2_C_out = 128
# Conv2_Ker = 5
# Conv2_stride = 1
# Conv2_padding = 2

# Batch2_f = 128

# Max2_ker = 2
# Max2_stride = 2

# Conv3_C_out = 256
# Conv3_Ker = 3
# Conv3_stride = 1
# Conv3_padding = 1

# Batch3_f = 256

# Max3_ker = 2
# Max3_stride = 2

# Conv4_C_out = 512
# Conv4_Ker = 3
# Conv4_stride = 1
# Conv4_padding = 1

# Batch4_f = 512

# W1 = 1024

# Batch5_f = 1024

# D_out = 1

# model1 = nn.Sequential(
#         nn.Conv2d(3, Conv1_C_out, Conv1_Ker, stride=Conv1_stride, padding=Conv1_padding),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(Batch1_f),
#         nn.MaxPool2d(Max1_ker, stride=Max1_stride),
#         nn.Conv2d(Conv1_C_out, Conv2_C_out, Conv2_Ker, stride=Conv2_stride, padding=Conv2_padding),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(Batch2_f),
#         nn.MaxPool2d(Max2_ker, stride=Max2_stride),
#         nn.Conv2d(Conv2_C_out, Conv3_C_out, Conv3_Ker, stride=Conv3_stride, padding=Conv3_padding),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(Batch3_f),
#         nn.MaxPool2d(Max3_ker, stride=Max3_stride),
#         nn.Conv2d(Conv3_C_out, Conv4_C_out, Conv4_Ker, stride=Conv4_stride, padding=Conv4_padding),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm2d(Batch4_f),
#         Flatten(16*16*512),
#         nn.Linear(16*16*512, W1),
#         nn.ReLU(inplace=True),
#         nn.BatchNorm1d(Batch5_f)
# )


# model2 = nn.Sequential(
#         nn.Linear(1024*2, D_out),
#         nn.Sigmoid()
# )

# model1.cuda()
# model2.cuda()

# learning_rate = 1e-6 # learning rate
# iters = 50
# optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

# for epoch in range(iters):
#     train_loader = DataLoader(train_set, batch_size=N, shuffle=True)
#     for i,data in enumerate(train_loader):
#         print(epoch)
#         print(i)
#         x1 = Variable(data['image1'], requires_grad=False).cuda()
#         x2 = Variable(data['image2'], requires_grad=False).cuda()
#         f1 = model1(x1)
#         f2 = model1(x2)
#         x = torch.cat((f1, f2), 1)
#         y = Variable(data['labels'].float(), requires_grad=False).cuda()
#         y_pred = model2(x)
#         loss = F.binary_cross_entropy(y_pred, y)
#         print("y", y)
#         print("y_pred", y_pred)
#         print(loss)

        
#         model2.zero_grad()
#         optimizer.zero_grad()
        
#         l = loss.expand(8,1024)
        
#         loss.backward()


# #        
#         for param in model2.parameters():
#             param.data -= learning_rate * param.grad.data
            
#         optimizer.step()

# ##        
# ##        loss.backward(retain_graph=True)
# ##        
# #        loss1.backward()
# ##        
#   #      loss2.backward()
# ##
# #        for param in model1.parameters():
# #            param.data -= learning_rate * param.grad.data       


# time.sleep(1000)

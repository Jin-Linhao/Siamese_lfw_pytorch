#!/usr/bin/env python
__author__ = 'LH'

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

from siamese_net_19 import SiameseNetwork

fileList = "../data/train.txt"
train_dir = "../lfw"


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

class Config():
    training_dir = "../lfw"
    testing_dir = "../lfw"
    train_list = "../data/train.txt"
    fileList = "../data/train.txt"
    train_batch_size = 64
    train_number_epochs = 100
    
def default_loader(path):
    img = Image.open(path)
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
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        final = []
        [imgPath1, imgPath2, target] = self.imgList[index]
        img1 = self.loader(os.path.join(train_dir, imgPath1))
        img2 = self.loader(os.path.join(train_dir, imgPath2))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
#         img1 = self.loader(os.path.join(os.path.splitext(imgPath1)[0],imgPath1))
#         img2 = self.loader(os.path.join(os.path.splitext(imgPath2)[0],imgPath2))
#         print target
        return img1, img2, torch.from_numpy(np.array([target],dtype=np.float32))

    def __len__(self):
        return len(self.imgList)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
#         print euclidean_distance, "label: ", label
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

def train():
	train_dataloader = torch.utils.data.DataLoader(
                    ImageList(root=fileList, fileList=Config.train_list, 
                                transform=transforms.Compose([ 
                                transforms.Scale((128,128)),
                                transforms.ToTensor(),            ])),
                    shuffle=False,
                    num_workers=8,
                    batch_size=1)

	net = SiameseNetwork().cuda()
	criterion = ContrastiveLoss()
	optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
	counter = []
	loss_history = [] 
	iteration_number= 0

	for i, data in enumerate(train_dataloader,0):
	    img0, img1 , label = data
	    img0, img1 , label = Variable(img0, volatile = true).cuda(), Variable(img1, volatile = true).cuda() , Variable(label, volatile = true).cuda()
	    output1,output2 = net(img0,img1)
	    optimizer.zero_grad()
	    loss_contrastive = criterion(output1,output2,label)
	    loss_contrastive.backward()
	    optimizer.step()
	    print("Iteration: {}\n Current loss {}\n".format(i,loss_contrastive.data[0]))
	    iteration_number +=10
	    counter.append(iteration_number)
	    loss_history.append(loss_contrastive.data[0])
	show_plot(counter,loss_history)


if __name__ == '__main__':
    train()
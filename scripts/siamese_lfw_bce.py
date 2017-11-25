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
import scipy
from scipy import ndimage
import scipy.misc

from siamese_19_BCE import SiameseNetwork_BCE


parser = argparse.ArgumentParser(description='PyTorch_Siamese_lfw')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
					help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
					help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
					metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=1e-6, type=float,
					metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
					help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_list', default='../data/train.txt', type=str, metavar='PATH',
					help='path to training list (default: ../data/train.txt)')
parser.add_argument('--test_list', default='../data/test.txt', type=str, metavar='PATH',
					help='path to validation list (default: ../data/train.txt)')
parser.add_argument('--save_path', default='../data/', type=str, metavar='PATH',
					help='path to save checkpoint (default: ../data/)')
parser.add_argument('--aug', default='off', type=str,
					help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="off", type=str, 
					help='switch on/off cuda option (default: off)')

plot_x = []
plot_y = []

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
	img = Image.open(path)
	if args.aug != "off":
		pix = np.array(img)
		pix_aug = img_augmentation(pix)
		img = Image.fromarray(np.uint8(pix_aug))
	# print pix
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


def img_augmentation(img):
	if random.random()>0.7:

		h, w, c= np.shape(img)
		# scale
		# if random.random() > 0.5:
		# 	s = (random.random() - 0.5) / 1.7 + 1
		# 	img = scipy.misc.imresize(img, (int(h * s), int(w * s)))
		# translation
		if random.random() > 0.5:
			img = scipy.ndimage.shift(img, (int(random.random() * 20 - 10), int(random.random() * 20 - 10), 0))
		# rotation
		if random.random() > 0.5:
			img = scipy.ndimage.rotate(img, random.random() * 60 - 30)
		# flipping
		if random.random() > 0.5:
			img = np.flip(img, 1)
		# crop and padding
		h_c, w_c = img.shape[:2]
		if h_c > h:
			top = int(h_c / 2 - h / 2)
			left = int(w_c / 2 - w / 2)
			img_out = img[top: top + h, left: left + w]
		else:
			pad_size = int((h - h_c) / 2)
			pads = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
			img_out = np.pad(np.array(img), pads, 'constant', constant_values=0)
	else:
		img_out = img
	# print np.shape(img_out)
	return img_out

	
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

		# 
		# img2 = self.img_augmentation(img2)
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
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
	
	running_loss = 0.0
	iteration_number= 0

	for i, data in enumerate(train_dataloader,0):
		img0, img1 , label = data

		if args.cuda == "off":
			img0, img1 , label = Variable(img0), Variable(img1) , Variable(label)
		else:
			img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()

		optimizer.zero_grad()
		output= forward_pass(img0,img1)
		# print output
		# forward_pass.zero_grad()
		loss = criterion(output, label)
		loss.backward()	
		optimizer.step()
		print("Epoch: {}, current iter: {}/{}\n Current loss {}\n".format(epoch, i, len(train_dataloader), loss.data[0]))
		

		running_loss += loss.data[0]
		# print i, loss.data[0], "/", len(train_dataloader)

		plot_x.append(len(plot_x)+1)
		plot_y.append(loss.data[0])

			
	# 	iteration_number +=1
	# 	plot_counter.append(iteration_number)
	# 	loss_history.append(loss.data[0])
	return running_loss


def validate(test_dataloader, forward_pass, criterion):
	cnt = 0
	total = 0
	forward_pass = SiameseNetwork_BCE().cuda()
	for i, data in enumerate(test_dataloader,0):
		img0, img1, label = data
		# concatenated = torch.cat((img0, img1),0)	
		if args.cuda == "off":
			img0, img1 , label = Variable(img0, volatile = True).cuda(), Variable(img1, volatile = True).cuda(), Variable(label, volatile = True).cuda()
		else:
			img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
		output= forward_pass(img0,img1)
		# print "output", output.data
		for j in range(0, label.size(0)):
			predicted = output.data[j] > 0.5
			predicted = predicted.type('torch.LongTensor')
			label_data = label.data
			label_data = label.data.type('torch.LongTensor')
			cnt += torch.sum(predicted == label_data[j])
			total = total+1

		# print "predicted: ", predicted.data[0], "output: ", output.data[0], "label: ", label.data[0]
		# imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}, ground truth'.format(output.cpu().data.numpy()[0][0], label))
	print('Accuracy of the network on the all test images: %d %%' % (100 * cnt / total))

	return cnt, total


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
						ImageList(fileList=args.test_list, 
								transform=transforms.Compose([ 
								transforms.Scale((128,128)),
								transforms.ToTensor(),            ])),
						shuffle=True,
						num_workers=args.workers,
						batch_size=8)

	if args.cuda == "off":
		forward_pass = SiameseNetwork_BCE()
	else:
		forward_pass = SiameseNetwork_BCE().cuda()
	# forward_pass = SiameseNetwork()
	criterion = nn.BCELoss()
	# criterion = F.binary_cross_entropy()
	optimizer = optim.Adam(forward_pass.parameters(), lr = args.learning_rate )
	

	correct = 0
	total = 0
	
	training_plot = "p1a_trainloss.txt"
	for epoch in range(args.start_epoch, args.epochs):

		# adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		running_loss = train(train_dataloader, forward_pass, criterion, optimizer, epoch)
		correct, total = validate(test_dataloader, forward_pass, criterion)
		print "correct matches: ", correct, "total matches: ", total
		print "total accuracy = ", float(correct)/total
		# evaluate on validation set
		# prec1 = validate(val_loader, model, criterion)
		save_name = args.save_path + str(epoch) + '_checkpoint.pth.tar'
		save_checkpoint({
			'epoch': epoch + 1,
	#         'arch': args.arch,
	#         'state_dict': model.state_dict(),
			# 'prec1': prec1,
			}, save_name)

		print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss/138))
	with open(training_plot, 'w') as f:
		for i in range(0,len(plot_x)):
			f.write(" ".join([str(plot_x[i]),str(plot_y[i])]))
			f.write('\n')
	print "done"		

def plot_training_loss():
	txt_file = 'p1a_trainloss.txt'
	plot_x = []
	plot_y = []
	with open(txt_file, 'r') as f:
		for line in f:
			data = line.strip()
			data = data.split(' ')
			plot_x.append(int(data[0]))
			plot_y.append(float(data[1]))
	plt.plot(plot_x, plot_y, 'b')
	plt.show()



if __name__ == '__main__':
	main()
	plot_training_loss()
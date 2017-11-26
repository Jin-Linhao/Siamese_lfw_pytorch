# A pytorch implementation of Siamese network using Labeled-Faces-in-the-Wild dataset.


### Table of Contents
- <a href='#updates'>Updates</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training'>Training</a>
- <a href='#references'>References</a>

## Updates
- Nov 11, 2017
	- First version, training on cpu works

## Installation
- Install [pytorch](http://pytorch.org/) following the website.
- Clone this repository.
	- Note: The current version is dependent on Python 2.7, not tested in Python3.

## Datasets
- Download Labeled Faces in the Wild dataset under main folder.
	- http://vis-www.cs.umass.edu/lfw/

## Training 
- Go to the script folder and run:
```
	  python siamese_lfw_train.py
```
or
```
	  python siamese_lfw_bce.py
```

- To load the model and test:
```
	  python siamese_lfw_train.py --load PATH/TO/YOUR/OWN/MODEL.pth.tar

```
or
```
	  python siamese_lfw_bce.py --load PATH/TO/YOUR/OWN/MODEL.pth.tar
```

For example:
```
	  python siamese_lfw_train.py --load /home/lh/cv_ws/src/Siamese_lfw_pytorch/data/9_checkpoint1.pth.tar

```

- For cuda version, run:
```
	  python siamese_lfw_train.py --cuda on
```
or
```
	  python siamese_lfw_bce.py --cuda on
```
## References
- [Pytorch](https://github.com/pytorch/pytorch). 
- https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/tree/master
- https://github.com/AlfredXiangWu/LightCNN#citation
- https://github.com/delijati/pytorch-siamese

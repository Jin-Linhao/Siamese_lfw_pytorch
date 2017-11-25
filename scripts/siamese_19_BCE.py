# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

class SiameseNetwork_BCE(nn.Module):
    def __init__(self):
        super(SiameseNetwork_BCE, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 64, 5, padding=(2,2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=(2,2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=(1,1))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=(1,1))
        self.bn4 = nn.BatchNorm2d(512)
        self.fc5 = nn.Linear(131072, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc = nn.Linear(2048, 1)

    def net(self,x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.bn5(F.relu(self.fc5(x)))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, y):
        output1 = self.net(x)
        output2 = self.net(y)
        output_c = torch.cat((output1, output2), 1)
        output = self.fc(output_c)
        output = torch.sigmoid(output)
        return output



# class SiameseNetwork_BCE(nn.Module):
#     def __init__(self):
#         super(SiameseNetwork_BCE, self).__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(64, 128, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(2, stride=2),

#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(512)
#             )
            

#         self.fc = nn.Sequential(
#             nn.Linear(131072, 1024),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(1024))

#         self.linear = nn.Sequential(
#             nn.Linear(2048, 1))


#     def forward_once(self, x):
#         output = self.cnn(x)
#         output = output.view(output.size()[0], -1)
#         output = self.fc(output)
#         return output

#     def forward(self, input1, input2):
#         output1 = self.forward_once(input1)
#         output2 = self.forward_once(input2)
#         output_c = torch.cat((output1, output2), 1)
#         output = self.linear(output_c)
#         output = torch.sigmoid(output)
#         # print "sigmoid"
#         return output
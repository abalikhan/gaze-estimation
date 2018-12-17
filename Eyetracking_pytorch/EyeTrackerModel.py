import argparse
import os
import shutil
import time, math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

'''
Pytorch model for the iTracker.

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''


class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=4, padding=0),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2, groups=2),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=0),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=384),
            nn.MaxPool2d(kernel_size=1, stride=3),
            nn.Conv2d(384, 64, kernel_size=3, stride=1, padding=2),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=64)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.face_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Dropout2d(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),
            nn.Dropout2d(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1)


        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 256),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            # nn.Dropout(0.3),
            # nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        x = self.face_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(num_features=256),
            nn.Linear(256, 128),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(num_features=128, affine=False)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ITrackerModel(nn.Module):

    def __init__(self):
        super(ITrackerModel, self).__init__()
        self.eyeModel = ItrackerImageModel()
        self.faceModel = FaceImageModel()
        self.gridModel = FaceGridModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(2 * 7 * 7 * 64, 128),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(num_features=128)
            # nn.Dropout(0.2),
            # nn.BatchNorm1d(num_features=128, affine=False)
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ELU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, faces, eyesLeft, eyesRight, faceGrids):
        # Eye nets
        xEyeL = self.eyeModel(eyesLeft)
        xEyeR = self.eyeModel(eyesRight)
        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesFC(xEyes)

        # Face net
        xFace = self.faceModel(faces)
        xGrid = self.gridModel(faceGrids)

        # Cat all
        x = torch.cat((xEyes, xFace, xGrid), 1)
        # x = nn.Dropout(0.3)(x)
        x = self.fc(x)

        return x

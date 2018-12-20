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
from torchvision import models
import torchvision.models as models
from torchsummary import summary
import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd.variable import Variable

class ItrackerImageModel(nn.Module):
    # Used for both eyes (with shared weights) and the face (with unqiue weights)
    def __init__(self):
        super(ItrackerImageModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=96),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=384),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)

        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.Vgg16 = models.vgg16_bn(pretrained=True)

        for param in self.Vgg16.features.parameters():
            param.require_grad = False


        self.fc = (
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        features = list(self.Vgg16.classifier.children())[:2]
        features.extend(self.fc)
        self.Vgg16.classifier = nn.Sequential(*features)

    def forward(self, x):
        x = self.Vgg16.features(x)
        x = x.view(x.size(0), -1)
        x = self.Vgg16.classifier(x)
        return x

class FaceGridModel(nn.Module):
    # Model for the face grid pathway
    def __init__(self, gridSize=25):
        super(FaceGridModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(gridSize * gridSize, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
        )
        # Joining everything
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
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
        x = self.fc(x)

        return x

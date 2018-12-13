import torch.utils.data as data
# import scipy.io as sio
from PIL import Image
from os.path import join
# import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
# import re
import cv2
import json
import deepdish as dd


DATASET_PATH = '../data'
meta_file = '../Eyetracking_pytorch/meta_file.h5'

# normalize a single image
def image_normalization(img):

    img = img.astype('float32') / 255.
    img = img - np.mean(img)

    return img
# load h5 file containing all information required
def loadData(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading data from %s...' % filename)
        metadata = dd.io.load(filename)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class ITrackerData(data.Dataset):
    def __init__(self, split= 'train', imSize = (224, 224), gridSize = (25, 25)):
        self.imSize = imSize
        self.gridSize = gridSize

        # loading dataset
        print ('loading dataset.... ')
        self.data = dd.io.load (meta_file)

        # applying transformation
        self.transformFace = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])
        self.transformEyeL = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])
        self.transformEyeR = transforms.Compose([
            transforms.Resize(self.imSize),
            transforms.ToTensor()
        ])
        self.split = split
        if split == 'train':
            mask = self.data['maskTr']
        elif split == 'val':
            mask = self.data['maskVl']
        elif split == 'test':
            mask = self.data['maskTs']

        self.indices = np.argwhere(mask)[:,0]
        print('dataset is split with %s having %d images' %(split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = cv2.imread(path)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)
            #im = Image.new("RGB", self.imSize, "white")

        return im

    def makeGrid(self, params):
        gridLen = self.gridSize[0] * self.gridSize[1]
        grid = np.zeros([gridLen, ], np.float32)

        indsY = np.array([i // self.gridSize[0] for i in range(gridLen)])
        indsX = np.array([i % self.gridSize[0] for i in range(gridLen)])
        condX = np.logical_and(indsX >= params[0], indsX < params[0] + params[2])
        condY = np.logical_and(indsY >= params[1], indsY < params[1] + params[3])
        cond = np.logical_and(condX, condY)

        grid[cond] = 1
        return grid
    def __getitem__(self, index):
        index = self.indices[index]

        # get the images
        image_name = join(DATASET_PATH + r'/' + self.data[self.split][index])
        idx = int(image_name[-8:-4])
        recNum = self.data[self.split][index][:5]

        # opening json files
        facefile = open(join(DATASET_PATH + r'/' + recNum  + r'/appleFace.json'))
        leyefile = open(join(DATASET_PATH + r'/' + recNum + r'/appleLeftEye.json'))
        reyefile = open(join(DATASET_PATH + r'/' + recNum + r'/appleRightEye.json'))

        # load json files to memory
        face_json = json.load(facefile)
        left_json = json.load(leyefile)
        right_json = json.load(reyefile)

        try:
            img = self.loadImage(image_name)
        except OSError:
            raise RuntimeError('image loading failed at:  ' + image_name)

        # get face
        tl_x_face = int(face_json["X"][idx])
        tl_y_face = int(face_json["Y"][idx])
        w = int(face_json["W"][idx])
        h = int(face_json["H"][idx])
        br_x = tl_x_face + w
        br_y = tl_y_face + h
        face = img[tl_y_face:br_y, tl_x_face:br_x]
        # cv2.imwrite(r'D:\facefile.jpg', face)

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]
        # cv2.imwrite(r'D:\Leyefile.jpg', left_eye)

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]
        # cv2.imwrite(r'D:\righteyefile.jpg', right_eye)

        # normalize the images
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

        # conversion from ndarry to PIL image
        face = Image.fromarray(np.uint8(face))
        left_eye = Image.fromarray(np.uint8(left_eye))
        right_eye = Image.fromarray(np.uint8(right_eye))

        # transformation applied on face, left and right eyes images
        imFace = self.transformFace(face)
        imEyeL = self.transformEyeL(left_eye)
        imEyeR = self.transformEyeR(right_eye)

        # open facegrid and gaze labels
        dot_file = open(join(DATASET_PATH + r'/' + recNum + r'/dotInfo.json'))
        fg_file = open(join(DATASET_PATH + r'/' + recNum + r'/faceGrid.json'))

        # load the json content
        dot_json = json.load(dot_file)
        fg_json = json.load(fg_file)

        # get labels
        y_x = dot_json["XCam"][idx]
        y_y = dot_json["YCam"][idx]
        gaze = [y_x, y_y]

        # face grid making
        fg_X = fg_json['X'][idx]
        fg_Y = fg_json['Y'][idx]
        fg_H = fg_json['H'][idx]
        fg_W = fg_json['W'][idx]
        param = [fg_X, fg_Y, fg_H, fg_W]

        # facegrid for face
        faceGrid = self.makeGrid(param)

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)
        return row, imFace, imEyeL, imEyeR, faceGrid, gaze

    def __len__(self):
        return len(self.indices)

if __name__ == '__main__':
    pass

    # batch_size = 50
    # imSize = (64, 64)
    # workers = 2
    # dataTrain = ITrackerData(split='train', imSize = imSize)
    # train_loader = torch.utils.data.DataLoader(
    #     dataTrain,
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=workers, pin_memory=True)
    #
    # for i, (row, imFace, imEyeL, imEyeR, faceGrid, gaze) in enumerate(train_loader):
    #     print('images loaded successfully....')
    #



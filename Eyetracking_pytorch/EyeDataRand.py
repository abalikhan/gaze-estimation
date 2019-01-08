import torch.utils.data as data
import scipy.io as sio
from os.path import join
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import json
import deepdish as dd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#
# DATASET_PATH = r'D:\gazecapture'
# meta_file = r'C:\Users\Aliab\PycharmProjects\Implement_pytorch\meta_small.h5'
# MEAN_PATH = r'C:\Users\Aliab\PycharmProjects\Implement_pytorch'

DATASET_PATH = '../data'
meta_file = '../Eyetracking_pytorch/meta_file.h5'
MEAN_PATH = '../Eyetracking_pytorch'

# normalize a single image
def image_normalization(img):

    img = img.astype('float32')/ 255.0

    return img
class SubtractMean(object):
    """Normalize a tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = meanImg.astype('float32') / 255.0
        self.meanImg = transforms.ToTensor()(self.meanImg)


    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)

# load h5 file containing all information required
def loadData(filename, silent = False):
    try:
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
        self.faceMean = sio.loadmat(join(MEAN_PATH, 'mean_face_224.mat'), squeeze_me=True, struct_as_record=False)['image_mean']
        self.eyeLeftMean = sio.loadmat(join(MEAN_PATH, 'mean_left_224.mat'), squeeze_me=True, struct_as_record=False)['image_mean']
        self.eyeRightMean = sio.loadmat(join(MEAN_PATH, 'mean_right_224.mat'), squeeze_me=True, struct_as_record=False)['image_mean']

        #     # applying transformation
        self.transformFace = transforms.Compose([
            transforms.ToTensor(),
            SubtractMean(meanImg=self.faceMean),

        ])

        self.transformEyeL = transforms.Compose([
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean)
        ])
        self.transformEyeR = transforms.Compose([
            transforms.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
        ])

        tdata = self.data['train'] + self.data['test'] + self.data['val']
        self.data['train'], self.data['test'] = train_test_split(tdata, test_size=0.2, shuffle=True)
        self.data['test'], self.data['val'] = train_test_split(self.data['test'], test_size=0.5, shuffle=True)
        # sklearn(valSet + testSet + trainingSet, )
        # valSet = []
        # testSet = []
        # trainingSet = []
        #
        # totalLength = len(valSet) + len(testSet) + len(trainingSet)
        #
        self.data['maskTr'] = np.ones(len(self.data['train']))
        self.data['maskVl'] = np.ones(len(self.data['val']))
        self.data['maskTs'] = np.ones(len(self.data['test']))

        self.split = split
        if split == 'train':
            mask = self.data['maskTr']
            # indx = len(train_data)
        elif split == 'val':
            mask = self.data['maskVl']
            # indx = len(val_data)
        elif split == 'test':
            mask = self.data['maskTs']
            # indx = len(test_data)

        self.indices = np.argwhere(mask)[:, 0]
        # self.indices[0] = valData
        print('dataset is split with %s having %d images' %(split, len(self.indices)))

    def loadImage(self, path):
        try:
            im = cv2.imread(path)
        except OSError:
            raise RuntimeError('Could not read image: ' + path)

        return im

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

        # get left eye
        tl_x = tl_x_face + int(left_json["X"][idx])
        tl_y = tl_y_face + int(left_json["Y"][idx])
        w = int(left_json["W"][idx])
        h = int(left_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        left_eye = img[tl_y:br_y, tl_x:br_x]

        # get right eye
        tl_x = tl_x_face + int(right_json["X"][idx])
        tl_y = tl_y_face + int(right_json["Y"][idx])
        w = int(right_json["W"][idx])
        h = int(right_json["H"][idx])
        br_x = tl_x + w
        br_y = tl_y + h
        right_eye = img[tl_y:br_y, tl_x:br_x]

        # resize images
        face = cv2.resize(face, self.imSize)
        left_eye = cv2.resize(left_eye, self.imSize)
        right_eye = cv2.resize(right_eye, self.imSize)
        #
        # normalize image to range [0, 1]
        face = image_normalization(face)
        left_eye = image_normalization(left_eye)
        right_eye = image_normalization(right_eye)

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
        face_grid = np.zeros(shape=(self.gridSize[0], self.gridSize[1]))
        fg_X = int(fg_json['X'][idx])
        fg_Y = int(fg_json['Y'][idx])
        fg_H = int(fg_json['H'][idx])
        fg_W = int(fg_json['W'][idx])
        br_x = fg_X + fg_W
        br_y = fg_Y + fg_H
        face_grid[fg_Y:br_y, fg_X:br_x] = 1
        face_grid = np.transpose(face_grid)
        faceGrid = face_grid.flatten()

        # to tensor
        row = torch.LongTensor([int(index)])
        faceGrid = torch.FloatTensor(faceGrid)
        gaze = torch.FloatTensor(gaze)

        return row, imFace, imEyeL, imEyeR, faceGrid, gaze

    def __len__(self):
        return len(self.indices)

#
if __name__ == '__main__':
    # pass
    batch_size = 10
    imSize = (224, 224)
    workers = 2
    dataTrain = ITrackerData(split='val', imSize = imSize)
    train_loader = torch.utils.data.DataLoader(
        dataTrain,
        batch_size=10, shuffle=True,
        num_workers=workers, pin_memory=True)
    for i, (row, imface, imL, imR, fg, gaze) in enumerate(train_loader):
        print(i)
    print('images loaded successfully....')




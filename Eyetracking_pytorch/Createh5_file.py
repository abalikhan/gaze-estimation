import json
import os
from os.path import join
import glob
import argparse
import pandas as pd
import numpy as np
import tables
import deepdish as dd
import cv2

# face dictionary
# face = {}
# face['X'] = []
# face['Y'] = []
# face['H'] = []
# face['W'] = []

# left eye dictionary
# leye = {}
# leye['X'] = []
# leye['Y'] = []
# leye['H'] = []
# leye['W'] = []
# right eye dictionary
# reye = {}
# reye['X'] = []
# reye['Y'] = []
# reye['H'] = []
# reye['W'] = []

# dictinoary for dotinfo file
# label = {}
# label['X'] = []
# label['Y'] = []

#face grid dictionary
# fg = {}
# fg['X'] = []
# fg['Y'] = []
# fg['H'] = []
# fg['W'] = []

# saving all dictionaries in h5 file
def makeh5file(path):

    path = path
    train, val, test, total_frames, recNum, maskTrain, maskTest, maskVal = create_dataset_lists(path)

    dd.io.save('meta_file.h5', {'train':train, 'val':val, 'test':test, 'recNum':recNum, 'Validframes':total_frames, 'maskTr':maskTrain, 'maskTs': maskTest, 'maskVl':
                                 maskVal}, compression=('blosc', 9))

# making the dictionary by passing each jason file to a proper dictionary part
# def makeDict(jsonfile, i, flag=0):
#
#     # flag = 0 for face
#     if (flag ==0):
#         face['X'].append(jsonfile['X'][i])
#         face['Y'].append((jsonfile['Y'][i]))
#         face['H'].append(jsonfile['H'][i])
#         face['W'].append((jsonfile['W'][i]))
#         # return face
#     # flag == 1 for left eye
#     if (flag == 1):
#         leye['X'].append(jsonfile['X'][i])
#         leye['Y'].append((jsonfile['Y'][i]))
#         leye['H'].append(jsonfile['H'][i])
#         leye['W'].append((jsonfile['W'][i]))
#         # return leye
#     # flag == 2 for right eye
#     if (flag == 2):
#         reye['X'].append(jsonfile['X'][i])
#         reye['Y'].append((jsonfile['Y'][i]))
#         reye['H'].append(jsonfile['H'][i])
#         reye['W'].append((jsonfile['W'][i]))
#         # return reye
#     # flag == 3 for lables
#     if (flag == 3):
#         label['X'].append(jsonfile['XCam'][i])
#         label['Y'].append((jsonfile['YCam'][i]))
#         # return fg
#     # flag = 4 for face grid
#     if (flag == 4):
#         fg['X'].append(jsonfile['X'][i])
#         fg['Y'].append((jsonfile['Y'][i]))
#         fg['H'].append(jsonfile['H'][i])
#         fg['W'].append((jsonfile['W'][i]))
#         # return fg

# main programing module
def create_dataset_lists(path):
    # read args from main (input and output)
    dataset_path = path
    dirs = sorted(glob.glob(join(dataset_path, "0*")))

    # count how many frames are finally selected
    tot_valid_frame = 0

    # dataset creating
    train= []
    val = []
    test = []
    recNum = []

    # total number of valid frames
    tframe = []
    nonvalid = 0
    # main loop
    for dir in dirs[:]:

        print("analyzing {}".format(dir))

        # open json files
        face_file = open(join(dataset_path, dir, "appleFace.json"))
        left_file = open(join(dataset_path, dir, "appleLeftEye.json"))
        right_file = open(join(dataset_path, dir, "appleRightEye.json"))
        frames_file = open(join(dataset_path, dir, "frames.json"))
        info_file = open(join(dataset_path, dir, "info.json"))
        dotinfo_file = open(join(dataset_path, dir, 'dotInfo.json'))
        fg_file = open(join(dataset_path, dir, 'faceGrid.json'))

        # read json content
        face_json = json.load(face_file)
        left_json = json.load(left_file)
        right_json = json.load(right_file)
        frames_json = json.load(frames_file)
        info_json = json.load(info_file)
        dotinfo_json = json.load(dotinfo_file)
        fg_json = json.load(fg_file)

        # as reported in the original paper, a sanity check is conducted and to avoid negative coordinates
        for i in range(0, int(info_json["TotalFrames"])):
            if left_json["IsValid"][i] and right_json["IsValid"][i] and face_json["IsValid"][i] \
                    and int(face_json["X"][i]) > 0 \
                    and int(face_json["Y"][i]) > 0 and \
                    int(left_json["X"][i]) > 0 and int(left_json["Y"][i]) > 0 and \
                    int(right_json["X"][i]) > 0 and int(right_json["Y"][i]) > 0:


# testing and debugging
#                 if int(face_json["X"][i]) < 0 or int(face_json["Y"][i]) < 0 or \
#                     int(left_json["X"][i]) < 0 or int(left_json["Y"][i]) < 0 or \
#                     int(right_json["X"][i]) < 0 or int(right_json["Y"][i]) < 0:
#                     img_path = join(dir + r'\frames\%05d.jpg' %i)
#                     img = cv2.imread(img_path)
#                     cv2.imwrite(r'D:\image%05d.jpg' %i, img)
#                     frame = dir[-4:]
#                     idx = int(frame)
#                     # get face
#                     if int(face_json['X'][i] < 0 ):
#                         face_json["X"][i] = face_json["X"][i]*-1
#
#                     if int(face_json['Y'][i] < 0):
#                         face_json["Y"][i] = face_json["Y"][i]*-1
#
#                     t1_x_face = int(face_json['X'][i])
#                     tl_y_face = int(face_json["Y"][i])
#                     w = int(face_json["W"][i])
#                     h = int(face_json["H"][i])
#                     br_x = t1_x_face + w
#                     br_y = tl_y_face + h
#                     face = img[tl_y_face:br_y, t1_x_face:br_x]
#                     cv2.imwrite(r'D:\face%05d.jpg' %i, face)
#
#                     if int(left_json['X'][i] <0 ):
#                         left_json["X"][i] = left_json["X"][idx]*-1
#
#                     if int(face_json['Y'][i] < 0):
#                         left_json["Y"][i] = left_json["Y"][idx]*-1
#
#                     t1_x_leye = int(left_json['X'][i])
#                     tl_y_leye = int(left_json["Y"][i])
#                     w = int(left_json["W"][i])
#                     h = int(left_json["H"][i])
#                     br_x = t1_x_leye + w
#                     br_y = tl_y_leye + h
#                     leye = img[tl_y_leye:br_y, t1_x_leye:br_x]
#                     cv2.imwrite(r'D:\leye%05d.jpg' %i, leye)
#
#                     if int(right_json['X'][i] <0 ):
#                         right_json["X"][i] = right_json["X"][i]*-1
#
#                     if int(face_json['Y'][i] < 0):
#                         right_json["Y"][i] = right_json["Y"][i]*-1
#
#                     tl_x_reye = int(right_json["X"][i])
#                     tl_y_reye = int(right_json["Y"][i])
#                     w = int(right_json["W"][i])
#                     h = int(right_json["H"][i])
#                     br_x = tl_x_reye + w
#                     br_y = tl_y_reye + h
#                     reye = img[tl_y_reye:br_y, tl_x_reye:br_x]
#                     cv2.imwrite(r'D:\reye%05d.jpg' %i, reye)

# debugging ended

                if info_json["Dataset"] == "train":
                    train.append(os.path.basename(dir) + '/frames/' + frames_json[i])
                    recNum.append(os.path.basename(dir))
                    # makeDict(face_json, i, flag=0)
                    # makeDict(left_json, i, flag=1)
                    # makeDict(right_json, i, flag=2)
                    # makeDict(dotinfo_json, i, flag=3)
                    # makeDict(fg_json, i, flag=4)

                if info_json["Dataset"] == "test":
                    test.append(os.path.basename(dir) + '/frames/' + frames_json[i])
                    recNum.append(os.path.basename(dir))
                    # makeDict(face_json, i, flag=0)
                    # makeDict(left_json, i, flag=1)
                    # makeDict(right_json, i, flag=2)
                    # makeDict(dotinfo_json, i, flag=3)
                    # makeDict(fg_json, i, flag=4)

                if info_json["Dataset"] == "val":
                    val.append(os.path.basename(dir) + '/frames/' + frames_json[i])
                    recNum.append(os.path.basename(dir))
                    # makeDict(face_json, i, flag=0)
                    # makeDict(left_json, i, flag=1)
                    # makeDict(right_json, i, flag=2)
                    # makeDict(dotinfo_json, i, flag=3)
                    # makeDict(fg_json, i, flag=4)

                # increase the number of valid frame
                tframe.append(tot_valid_frame)
                tot_valid_frame += 1
            else:
                nonvalid = nonvalid + 1

    print('total valid frames are : {}'.format(tot_valid_frame))
    print(' non valid files are {}'.format(nonvalid))
    print('rec numbers are {}'.format(recNum))

    mask_train = np.zeros(len(tframe))
    mask_test = np.zeros(len(tframe))
    mask_val = np.zeros(len(tframe))

    for i in range(0, len(train)):
        mask_train[i] = 1
    for i in range(0, len(test)):
        mask_test[i] = 1
    for i in range(0, len(val)):
        mask_val[i] = 1

    print('train mask is {}'.format(mask_train))
    print('test mask is {}'.format(mask_test))
    print('val mask is {}'.format(mask_val))
    return train, val, test , tframe, recNum, mask_train, mask_test, mask_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset names for train, validation and test.")


    args = parser.parse_args()
    i = r"D:\gazecapture"
    makeh5file(i)
    # create_dataset_lists(path=i)
    print(' file created')

####################################################################################
#   UnetSequence.py
#       Script for implementing a custom generator for the unet model defined in
#       Train_unet.py. This script is responsible for efficiently loading and preprocessing data
#       to save on memory during training.
#   Name: Yuqi Yang
#   Student Number: 20150516
#   Date: 03/21/2023
#####################################################################################
import numpy as np
from numpy import newaxis
import os
import cv2
import math
import gc
from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence, to_categorical

class UnetSequence(Sequence):
    def __init__(self,images_and_segmentations,batchSize=8,shuffle=True):
        self.inputs = images_and_segmentations[0]
        self.targets = images_and_segmentations[1]
        self.shuffle = shuffle
        self.batch_size=batchSize
        self.on_epoch_end()

    def __len__(self):
        length = len(self.inputs)/self.batch_size
        return math.ceil(length)

    def on_epoch_end(self):
        if self.shuffle:
            self.inputs,self.targets = shuffle(self.inputs,self.targets)
        gc.collect()

    #############################################################################################################
    # Question 4:
    #    Complete the following function that will read in your image. Include any preprocessing that you wish to
    #    perform on your images here. Document in your PDF what enhancement techniques you chose (or why you chose
    #    not to use any), and why.
    #############################################################################################################
    def readImage(self,fileName):
        '''
        Args:
            fileName: The path to an image file
        Returns:
            img: your image as a np array. shape=(128,128,1)
        '''
        # since for this assignment, we will mainly be discussing greyscale images
        # I will not consider the constraint of colored images 
        img_arr = cv2.imread(fileName, cv2.IMREAD_ANYDEPTH)
        # sharpening filter 
        sharpen_kernel = np.array([[0,-1,0], [-1,6,-1], [0,-1,0]])
        sharpen = cv2.filter2D(img_arr, -1, sharpen_kernel)

        # thresholding (contrast enhancement)
        img_thresh = np.copy(sharpen)
        img_thresh[ img_thresh > 230 ] = 255
        img_thresh[ img_thresh < 230 ] = 0
        img = img_thresh[:, :, newaxis]
        return img

    #############################################################################################################
    # Question 4:
    #    Complete the following function that will read in your ground truth segmentation.
    #############################################################################################################
    def readSegmentation(self,fileName):
        '''
        Args:
            fileName: The path to a segmentation file
        Returns:
            one_hot_img: your segmentation as a np array. shape=(128,128,2)
        '''
        seg = cv2.imread(fileName, cv2.IMREAD_ANYDEPTH)
        # one-hot encoding, from the perspective I'm viewing the np array, this
        # means that I will have one layer of image just for the background, and one
        # layer of image just for the object. First layer is background, second layer
        # is object 
        obj_seg = np.copy(seg)
        bg_seg = np.copy(seg)
        for row in range((seg.shape[0])):
            for col in range((seg.shape[1])):
                # if location is marked as object, not background
                if seg[row][col] != 0:
                    obj_seg[row][col] = 1
                    bg_seg[row][col] = 0
                else:
                    obj_seg[row][col] = 0
                    bg_seg[row][col] = 1
        # stack the one hot encoded image to desired shape 
        one_hot_img = np.stack((bg_seg, obj_seg), axis=-1)
        # print("one_hot_img:\n",one_hot_img.shape)
        return one_hot_img 

    def __getitem__(self, index):
        startIndex = index*self.batch_size
        index_of_next_batch = (index+1)*self.batch_size
        inputBatch = [self.readImage(x) for x in self.inputs[startIndex:index_of_next_batch]]
        outputBatch = [self.readSegmentation(x) for x in self.targets[startIndex:index_of_next_batch]]
        return (np.array(inputBatch),np.array(outputBatch))
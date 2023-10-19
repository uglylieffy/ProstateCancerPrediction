####################################################################################
#   Train_unet.py
#       Script for implementing and training a unet model for segmentation
#   Name: Yuqi Yang
#   Student Number: 20150516
#   Date: 03/21/2023
#####################################################################################

import cv2
import numpy as np
from numpy import newaxis
import os

import scipy.spatial.distance
from scipy.spatial.distance import directed_hausdorff
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Average
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *
import math
import random
from UnetSequence import UnetSequence
from sklearn.model_selection import train_test_split


def define_UNet_Architecture(imageSize,numClasses,filterMultiplier=10):
    input_ = layers.Input(imageSize)
    skips = []
    output = input_

    num_layers = int(np.floor(np.log2(imageSize[0])))
    down_conv_kernel_sizes = np.zeros([num_layers],dtype=int)
    up_conv_kernel_sizes = np.zeros([num_layers], dtype=int)

    down_filter_numbers = np.zeros([num_layers],dtype=int)
    up_filter_numbers = np.zeros([num_layers],dtype=int)

    for layer_index in range(num_layers):
        up_conv_kernel_sizes[layer_index]=int(4)
        down_conv_kernel_sizes[layer_index] = int(3)
        down_filter_numbers[layer_index] = int((layer_index+1)*filterMultiplier + numClasses)
        up_filter_numbers[layer_index] = int((num_layers-layer_index-1)*filterMultiplier + numClasses)
        
    #Create contracting path
    for kernel_shape,num_filters in zip(down_conv_kernel_sizes,down_filter_numbers):
        skips.append(output)
        output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                               strides=2,
                               padding="same",
                               activation="relu",
                               bias_regularizer=l1(0.))(output)

    #Create expanding path
    lastLayer = len(up_conv_kernel_sizes)-1
    layerNum = 0
    for kernel_shape,num_filters in zip(up_conv_kernel_sizes,up_filter_numbers):
        output = layers.UpSampling2D()(output)
        skip_connection_output = skips.pop()
        output = layers.concatenate([output,skip_connection_output],axis=3)
        if layerNum!=lastLayer:
            output = layers.Conv2D(num_filters,(kernel_shape,kernel_shape),
                                   padding="same",
                                   activation="relu",
                                   bias_regularizer=l1(0.))(output)
        else: #Final output layer
            output = layers.Conv2D(num_filters, (kernel_shape, kernel_shape),
                                   padding="same",
                                   activation="softmax",
                                   bias_regularizer=l1(0.))(output)
        layerNum+=1
    return Model([input_],[output])

#############################################################################################################
# Question 2:
#    Complete the following function to generate your simulated images and segmentations. You may implement
#    many helper functions as necessary to do so.
#############################################################################################################

# CISC472 Homework 1 Question 7
# Generate a simulated image
def ellipse_bg_generator(dimension_x, dimension_y):
    # 0 for background and 255 for ellipse 
    # first generate an array filled with 0s with a given dimension 
    bg_arr = np.zeros((dimension_x, dimension_y))
    return bg_arr

def ellipse_generator(dimension_x, dimension_y, depth):
    ellipse_bg_arr = ellipse_bg_generator(dimension_x, dimension_y)
    ellipse_arr = np.copy(ellipse_bg_arr)
    # the size and location of ellipse is random every time.
    # the location of the ellipse is considered its conter point here.
    # in case of some extreme random location generated, I will leave a 
    # padding of around 15 pixel for room. where my ellipse
    # has a y or x value of minimum 1/10 pixel value of the background dimension
    # rounding to the nearest int 
    pixel_val = math.ceil((min(dimension_x, dimension_y))/10)
    ellipse_loc_x = random.randint(15,(dimension_x-16))
    ellipse_loc_y = random.randint(15,(dimension_y-16))

    # The edges of the ellipse may not be cut off by the boundaries of the image,
    # which means depending on the location of the ellipse, the size of the ellipse
    # will be restrained accordingly 
    ellipse_x = random.randint(15,(min(ellipse_loc_x, dimension_x-ellipse_loc_x)))
    ellipse_y = random.randint(15,(min(ellipse_loc_y, dimension_y-ellipse_loc_y)))

    # generate a rectangle with x and y value
    for i in range(ellipse_x):
        for j in range(ellipse_y):
            # generate at all 4 direction 
            ellipse_arr[ellipse_loc_x+i][ellipse_loc_y+j] = (2**depth)-1
            ellipse_arr[ellipse_loc_x-i][ellipse_loc_y-j] = (2**depth)-1
            ellipse_arr[ellipse_loc_x-i][ellipse_loc_y+j] = (2**depth)-1
            ellipse_arr[ellipse_loc_x+i][ellipse_loc_y-j] = (2**depth)-1

    # determine if point is in ellipse range
    for row in range(ellipse_arr.shape[0]):
        for col in range(ellipse_arr.shape[1]):
            if ellipse_arr[row][col] == (2**depth)-1:
                flag = (((row-ellipse_loc_x)**2)/(ellipse_x**2))+(((col-ellipse_loc_y)**2)/(ellipse_y**2))
                if flag > 1:
                    ellipse_arr[row][col] = 0
                else:
                    ellipse_arr[row][col] = (2**depth)-1

    # Now we have a narray of every points in the ellipsoid covering every pixel value        
    # return the simulated image as an array
    return ellipse_arr

# smoothing filter 
def smooth(arr, count):
    for i in range(count):
        arr = cv2.GaussianBlur(arr,(5,5),cv2.BORDER_DEFAULT)
    return arr

def generateDataset(datasetDirectory,num_images,imageSize):
    '''
    Args:
        datasetDirectory: the path to the directory where your images and segmentations will be stored
        num_images: the number of images that you wish to generate
        imageSize: the shape of your images and segmentations
    Returns:
        None: Saves all images and segmentations to the dataset directory
    '''
    # generate 100 sample images of a white ellipse on a black background with 
    # a bit-depth of 8 (max value 255) with shape (128,128)
    for i in range(num_images):
        random_ellipse = ellipse_generator(imageSize[0], imageSize[1], 8)
        modified_image = np.copy(random_ellipse)
        # To obtain the ground truth segmentations rescale the images to a bit-depth of 1 (max value 1)
        # resized dim is still (128,128), only add 1 then divide the whole set by 128, then minus 1
        # and round value as 0 or 1
        for row in range((random_ellipse.shape[0])):
            for col in range((random_ellipse.shape[1])):        
                temp_val = round(((random_ellipse[row,col]+1)/128)-1)
                modified_image[row][col] = temp_val

       
        if i < 10:
            rescaledFilename = datasetDirectory + "\segmentation_0" + str(i) + ".png"
            smoothFilename = datasetDirectory + "\image_0" + str(i) + ".png"
        else:
            rescaledFilename = datasetDirectory + "\segmentation_" + str(i) + ".png"
            smoothFilename = datasetDirectory + "\image_" + str(i) + ".png"
        # save rescaled image files 
        plt.imsave(rescaledFilename, modified_image, cmap = 'gray')

        # apply the smoothing filter 1-100 times for each images
        count = random.randint(1, 100)
        random_ellipse = smooth(random_ellipse, count)
        # save smoothed image files
        plt.imsave(smoothFilename, random_ellipse, cmap = 'gray')
        


#############################################################################################################
# Question 3:
#    Complete the following function so that it returns your data divided into 3 non-overlapping sets
# You do not need to read the images at this stage
#############################################################################################################
def splitDataIntoSets(images,segmentations):
    '''
    Args:
        images: list of all image filepaths in the dataset
        segmentations: list of all segmentation filepaths in the dataset

    Returns:
        trainImg: list of all image filepaths to be used for training
        trainSeg: list of all segmentation filepaths to be used for training
        valImg: list of all image filepaths to be used for validation
        valSeg: list of all segmentation filepaths to be used for validation
        testImg: list of all image filepaths to be used for testing
        testSeg: list of all segmentation filepaths to be used for testing
    '''
    # Divide data randomly into 3 distinct sets for training, validation and testing
    # ideally training data should be around 70%, where validation and test split the 30%
    # now test only consist of 10%, we can split the data into 70% training, 20% validation, 10% test. 
    train_scalar = 70
    valid_scalar = 20
    test_scalar = 10
    # set seed, so randomnization is reproducible 
    random.seed(10)
    data_index = list(range(0,100))
    # shuffle the index list to obtain a randomized effect 
    random.shuffle(data_index)
    trainId, valID, testID = data_index[:train_scalar],data_index[train_scalar:valid_scalar+train_scalar],data_index[(valid_scalar+train_scalar):]
    # split images and segmentations sets accordingly 
    trainImg = [ images[i] for i in trainId]
    valImg = [ images[i] for i in valID]
    testImg = [ images[i] for i in testID]
    trainSeg = [ segmentations[i] for i in trainId]
    valSeg = [ segmentations[i] for i in valID]
    testSeg = [ segmentations[i] for i in testID]
    return (trainImg,trainSeg),(valImg,valSeg),(testImg,testSeg)


#############################################################################################################
# Question 5:
#    Complete the following function so that it will create a plot for the training and validation loss/metrics.
#    Training and validation should be shown on the same graph so there should be one plot per loss/metric
#############################################################################################################
def plotLossAndMetrics(trainingHistory):
    '''
    Args:
        trainingHistory: The dictionary containing the progression of the loss and metrics for training and validation
    Returns:
        None: should save each graph as a png
    '''
    # summarize history for accuracy
    plt.plot(trainingHistory.history['accuracy'])
    plt.plot(trainingHistory.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(trainingHistory.history['loss'])
    plt.plot(trainingHistory.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # below code should be commented out accordingly 
    #  IOU coefficient
    # plt.plot(trainingHistory.history['IOU'])
    # plt.plot(trainingHistory.history['val_IOU'])
    # plt.title('IOU')
    # plt.ylabel('IOU Coefficient')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    #  hausdorff coefficient
    # plt.plot(trainingHistory.history['hausdorffDistance'])
    # plt.plot(trainingHistory.history['val_hausdorffDistance'])
    # plt.title('hausdorff')
    # plt.ylabel('hausdorff Distance')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()

    #  for dice coefficient
    # plt.plot(trainingHistory.history['diceCoefficient'])
    # plt.plot(trainingHistory.history['val_diceCoefficient'])
    # plt.title('dice Coefficient')
    # plt.ylabel('dice Distance')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    
    return

def display_image(img_arr, tittle):
    # Displays the image in a GUI window. Press ESC key and the window is removed automatically
    # when color image is displayed 
    if len(img_arr.shape) == 3:
        cv2.imshow(tittle, img_arr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # every non color image is considered grayscale here
    else:
        plt.imshow(img_arr, cmap='gray')
        plt.title(tittle)
        plt.show()
    
def mean_precision(y_true,y_pred):
    '''
    Computes the mean precision between predicted segmentation and ground truth. All values are original
    image array.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize, imgHeight,imgWidth)
        ypred: predicted segmentation, shape = (batchSize, imgHeight,imgWidth)

    Returns:
        mean_precision: the mean precision of the prediction 
    '''
    # TP
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # TP+FP
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    # precision = TP/(TP+FP)
    mean_precision = true_positives / (possible_positives + K.epsilon())
    return mean_precision

def mean_recall(y_true,y_pred):
    '''
    Computes the mean recall between predicted segmentation and ground truth. All values are original
    image array.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize, imgHeight,imgWidth)
        ypred: predicted segmentation, shape = (batchSize, imgHeight,imgWidth)

    Returns:
        mean_recall: the mean recall of the prediction 
    '''
    # TP
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    # TP+FN
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    # recall = TP/(TP+FN)
    mean_recall = true_positives / (predicted_positives + K.epsilon())
    return mean_recall
    
def main():
    dataSetPath = os.path.join(os.getcwd(), "CISC_472_dataset")
    # generateDataset(dataSetPath, 100,imageSize = (128,128,1)) #this line only needs to be run once
    images = sorted([os.path.join(dataSetPath,x) for x in os.listdir(dataSetPath) if "image" in x])
    segmentations = sorted([os.path.join(dataSetPath,x) for x in os.listdir(dataSetPath) if "segmentation" in x])

    trainData,valData,testData = splitDataIntoSets(images,segmentations)

    
    trainSequence = UnetSequence(trainData)
    valSequence = UnetSequence(valData)
    testSequence = UnetSequence(testData,shuffle=False)

    unet = define_UNet_Architecture(imageSize=(128,128,1),numClasses=2)
    unet.summary()

    # #############################################################################################################
    # # Set the values of the following hyperparameters
    # #############################################################################################################
    
    # Q5,6,7
    # Compile the model using the loss function “categorical_crossentropy” and IOU
    lossFunction_ce="categorical_crossentropy"
    lossFunction_IOU = [IOU_Loss]

    # I wanted to add precision and recall just for fun
    # metrics for different model mofication 
    metrics_ce=["accuracy",
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
    metrics_IOU=[IOU, "accuracy"]
    metrics_dice=[diceCoefficient, "accuracy"]
    metrics_hausdorff=[hausdorffDistance, "accuracy"]

    # adjust epoch size here 
    numEpochs=20

    # #############################################################################################################
    # # Create model checkpoints here, and add the variable names to the callbacks list in the compile command
    # #############################################################################################################
    
    # Early Stopping
    callbacks = [EarlyStopping(monitor='val_loss', patience=3)] 

    # model checkpoints parameter to save best model per epoch, for both question 5 and 6
    # model_filepath = "../a3/unet_ce_loss_accuracy_.h5"
    model_filepath = "../a3/unet_ce_loss_accuracy.h5"
    model_filepath_IOUloss = "../a3/unet_iou_loss_accuracy.h5"
    model_filepath_IOU = "../a3/unet_ce_IOU.h5"
    model_filepath_hausdorff = "../a3/unet_ce_hausdorffDistance.h5"
    model_filepath_dice = "../a3/unet_ce_dice.h5"
    monitor_ce = 'val_accuracy'
    monitor_IOU = 'val_IOU'
    monitor_haus= 'val_hausdorffDistance'
    monitor_dice = 'val_diceCoefficient'
    # implement check point 
    checkpoint = ModelCheckpoint(
        filepath=model_filepath,
        monitor=monitor_ce,
        mode='max', # or min for hausdorff distance
        save_best_only=True,
        save_weights_only=True
    )
    
    # change learning rate here !!
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss=lossFunction_ce,
                 metrics=metrics_ce,
                 run_eagerly=True)

    history = unet.fit(x=trainSequence,
                       validation_data=valSequence,
                       epochs=numEpochs,
                       batch_size = 16,
                       verbose=2,
                       callbacks=[checkpoint, callbacks])

    plotLossAndMetrics(history)

    unet.evaluate(testSequence)
    predictions = unet.predict(testSequence)

    #############################################################################################################
    # Add additional code for generating predictions and evaluations here
    #############################################################################################################
    
    ########### generating predictions, visulization ###########
    # saved model weight file path 
    model_filepath = "unet_ce_loss_accuracy.h5"
    model_filepath_IOUloss = "../a3/unet_iou_loss_accuracy.h5"
    model_filepath_IOU = "../a3/unet_ce_IOU.h5"
    model_filepath_hausdorff = "../a3/unet_ce_hausdorffDistance.h5"
    model_filepath_dice = "../a3/unet_ce_dice.h5"

    # standard ce model
    unet.load_weights(model_filepath)
    standard_predictions = unet.predict(testSequence)
    for batchSize in range(0, (standard_predictions.shape[0])):
        org_std_predict = convert(standard_predictions, batchSize)
        display_image(org_std_predict, 'standard model - q5')
    unet.evaluate(testSequence)

    # # IOU loss model
    # unet.load_weights(model_filepath_IOUloss)
    # IOUloss_predictions = unet2.predict(testSequence)
    # for batchSize in range(0, (IOUloss_predictions.shape[0])):
    #     org_IOUloss_predict = convert(IOUloss_predictions, batchSize)
    #     display_image(org_IOUloss_predict, 'IOU loss model - q6')
    # unet.evaluate(testSequence)

    # # IOU ce model
    # unet.load_weights(model_filepath_IOU)
    # IOU_predictions = unet.predict(testSequence)
    # for batchSize in range(0, (IOU_predictions.shape[0])):
    #     org_IOU_predict = convert(IOU_predictions, batchSize)
    #     display_image(org_IOU_predict, 'IOU ce model - q7')
    # unet.evaluate(testSequence)

    # # hausdorff distance ce model
    # unet.load_weights(model_filepath_hausdorff)
    # haus_predictions = unet.predict(testSequence)
    # for batchSize in range(0, (haus_predictions.shape[0])):
    #     org_haus_predict = convert(haus_predictions, batchSize)
    #     display_image(org_haus_predict, 'hausdorff distance ce model - q7')
    # unet.evaluate(testSequence)

    # # dice coefficient ce model   
    # unet.load_weights(model_filepath_dice)
    # dice_predictions = unet.predict(testSequence)
    # for batchSize in range(0, (dice_predictions.shape[0])):
    #     org_dice_predict = convert(dice_predictions, batchSize)
    #     display_image(org_dice_predict, 'dice coefficient ce model - q7')
    # unet.evaluate(testSequence)

    ########### evaluation ###########
    # ground truth array 
    all_arr = []
    for i in range(len(testData[1])):
        # read in ground truth image 
        seg = cv2.imread(testData[1][i], cv2.IMREAD_ANYDEPTH)
        all_arr.append(seg[None,:])
    groundTruth = np.vstack(all_arr)

    def quantitative_evaluation(prediction):
        hot_encode_arr = []
        # prediction array 
        for batchSize in range(0, (prediction.shape[0])):
            temp = convert(prediction, batchSize)
            hot_encode_arr.append(temp[None,:])
        predict = np.vstack(hot_encode_arr)

        # q5 ce standard 
        model_precision = mean_precision(groundTruth, predict)
        model_recall = mean_recall(groundTruth, predict)
        print(model_precision, model_recall)
    
    # evaluation of models in order 
    quantitative_evaluation(standard_predictions)
    # quantitative_evaluation(IOUloss_predictions)
    # quantitative_evaluation(IOU_predictions)
    # quantitative_evaluation(haus_predictions)
    # quantitative_evaluation(dice_predictions)


def IOU(y_true,y_pred):
    y_true_f = K.flatten(y_true[:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,1])
    intersection = K.sum(y_true_f*y_pred_f)
    return(intersection)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection)

def IOU_Loss(y_true,y_pred):
    return 1-IOU(y_true,y_pred)

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean hausdorff distance
#############################################################################################################
def convert(img_arr, batchSize):
        oneHot_y = img_arr[batchSize,:,:,:]
        # 0 = background, 1=object
        oneHot_object = oneHot_y[:,:,1]
        return oneHot_object
def hausdorffDistance(ytrue,ypred):
    '''
    Computes the mean hausdorff distance between predicted segmentation and ground truth. All values are one-hot encoded.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)
        ypred: predicted segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)

    Returns:
        mean_hausdorff_distance: the mean hausdorff distance across all samples in the batch
    '''
    distance = []
    for batchSize in range(0, (ytrue.shape[0])):
        ytrue_h = convert(ytrue,batchSize)
        ypred_h = convert(ypred,batchSize)
        hausdorff_distance = directed_hausdorff(ytrue_h, ypred_h)[0]
        distance.append(hausdorff_distance)
    mean_hausdorff_distance = sum(distance) / len(distance)
    return mean_hausdorff_distance

#############################################################################################################
# Question 7:
#    Complete the following function to compute the mean dice coefficient
#############################################################################################################

def diceCoefficient(ytrue,ypred):
    '''
    Computes the mean dice coefficient between predicted segmentation and ground truth. All values are one-hot encoded.
    Args:
        ytrue: ground truth segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)
        ypred: predicted segmentation, shape = (batchSize,imgHeight,imgWidth,numClasses)

    Returns:
        mean_dice_coefficient: the mean dice coefficient across all samples in the batch
    '''
    # dice coefficient = 2*TP/((TP+FP)+(TP+FN))
    ytrue = ytrue.astype(ypred.dtype)
    y_true_f = K.flatten(ytrue[:,:,1])
    y_pred_f = K.flatten(ypred[:,:,1])
    intersection = K.sum(y_true_f*y_pred_f)
    mean_dice_coefficient = (2*intersection)/(K.sum(y_true_f)+K.sum(y_pred_f))
    return mean_dice_coefficient

main()




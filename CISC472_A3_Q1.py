# Author: Yuqi Yang, 200516
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# CISC472 Homework 3 Question 1 
# Data Exploration

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

# reslice function from Assignment 2
def volume_reslice(arr):
    # sagittal has the same x in every slices 
    # the middle slice will be 224*224 too, but is each 1*224 line of the x middle line 
    # for all 224 slices 
    z_val, row_val, col_val = arr.shape

    # initialize saigital array 
    sagittal_arr = np.zeros((row_val,z_val,col_val))
    # shape is [0] layers, [1] rows, and [2] columns.
    for col in range(arr.shape[2]):
        temp_arr = np.zeros((arr.shape[0], arr.shape[2]))
        for z in range(arr.shape[0]):
            # one slice is composed of same x and its corresponding y values along the line 
            temp_arr[z] = arr[z][:,col]
        sagittal_arr[col] = temp_arr
                 
    # coronal has the same y value
    coronal_arr = np.zeros((col_val,z_val,row_val))
    for row in range(arr.shape[1]):
        temp_arr = np.zeros((arr.shape[0], arr.shape[1]))
        for z in range(arr.shape[0]):
            # one slice is composed of same y and its corresponding x values along the line 
            temp_arr[z] = arr[z][row]
        coronal_arr[row] = temp_arr 

    # axil is slicing through the middle, its middle slice will just be the array that
    # has an middle index 
    axial_arr = np.copy(arr)
    return axial_arr, coronal_arr, sagittal_arr

def q1Test():
    # Scan 00
    test_arr = np.load('TrainingData/Case00.npy')
    axial_arr, coronal_arr, sagittal_arr = volume_reslice(test_arr)
    mid_axial = axial_arr[int((test_arr.shape[0])/2)]
    mid_coronal = coronal_arr[int((test_arr.shape[1])/2)]
    mid_sagittal = sagittal_arr[int((test_arr.shape[1])/2)]
    display_image(mid_axial, 'mid slice of axial')
    display_image(mid_coronal, 'mid slice of coronal')
    display_image(mid_sagittal, 'mid slice of sagittal')
    # Ground Truth of Scan 00
    test_arrGT = np.load('TrainingData/Case00_segmentation.npy')
    axial_arrGT, coronal_arrGT, sagittal_arrGT = volume_reslice(test_arrGT)
    mid_axialGT = axial_arrGT[int((test_arrGT.shape[0])/2)]
    mid_coronalGT = coronal_arrGT[int((test_arrGT.shape[1])/2)]
    mid_sagittalGT = sagittal_arrGT[int((test_arrGT.shape[1])/2)]
    display_image(mid_axialGT, 'mid slice of axial (GT)')
    display_image(mid_coronalGT, 'mid slice of coronal (GT)')
    display_image(mid_sagittalGT, 'mid slice of sagittal (GT)')

def main():
    # q1 test 
    q1Test()
    return
main()
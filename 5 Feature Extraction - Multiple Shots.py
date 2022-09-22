"""
Feature-based forest data segmentation using
    Random Forest:
        Application with multiple training images

Seven steps involved:
    1. Read training images and extract features
    2. Read labeled images (masks) and create
        another dataframe
    3. Get data ready for random forest
    4. Define classifier and fit model with training
        data
    5. Check model accuracy
    6. Save model for future use
    7. Make prediction on new images
"""
import numpy as np
import cv2
import pandas as pd
import glob
import pickle
import os
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage.filters import roberts, sobel, scharr, prewitt

"""
Step 1: Read the training images
        and extract features
"""
images_dataset = pd.DataFrame()  # dataframe to capture image features
images_path = 'C:/DATA_many/Train_Images_many/'

for image in os.listdir(images_path):  # iterate through each file
    print(image)

    df = pd.DataFrame()  # Temporary dataframe to capture features from each loop
    # Reset dataframe to blank after each loop

    input_image = cv2.imread(images_path + image)   # Read images

    # Check if input image is RGB or gray, and convert
    # to gray if RGB

    if input_image.ndim == 3 and input_image.shape[-1] == 3:
        image1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    elif input_image.ndim == 2:
        image1 = input_image

    else:
        raise Exception('The module works only with grayscale and RGB images!')

    """
    Start adding data to the dataframe
        - Pixel values
        - Image name
        - Extract Gabor features
    """
    pixel_values = image1.reshape(-1)
    df['Pixel_Values'] = pixel_values
    df['Image_Name'] = image

    # num=1 serves to count numbers up in order to give
    # gabor features a label in the dataframe
    num = 1
    kernel_size = 5
    kernels = list()
    for theta in range(2):  # Define number of thetas
        theta = theta / 4. * np.pi
        for sigma in (1, 3):  # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi / 4):  # Range of wavelengths
                for gamma in (0.05, 0.5):  # Gamma values in 0.05 and 0.5
                    gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc
                    # print(gabor_label)
                    kernel = cv2.getGaborKernel(
                        (kernel_size, kernel_size),
                        sigma, theta, lamda, gamma,
                        0, ktype=cv2.CV_32F)
                    kernels.append(kernel)

                    """
                    Now filter image and add 
                    values to a new column
                    """
                    filter_image = cv2.filter2D(
                        image1,
                        cv2.CV_8UC3,
                        kernel)
                    filtered_image = filter_image.reshape(
                        -1)
                    df[gabor_label] = filtered_image  # Labels columns as Gabor1, Gabor2
                    num += 1  # Increment for Gabor column label

    """
    - Apply 'Canny edge' detector 
    """
    edges = cv2.Canny(image1, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1

    """
    Add more Filters/Features:
        - Roberts, Sobel, Scharr, Prewitt
    """
    more_filters = [roberts, sobel, scharr, prewitt]
    for philter in more_filters:
        df[f'{philter}'] = philter(image1).reshape(-1)

    """
    Add GAUSSIAN based filters/feature 
        - with different sigma values
    """
    for num in range(1, 34, 2):
        gaussian = 'Gaussian s' + str(num)
        df[gaussian] = nd.gaussian_filter(
            image1, sigma=num).reshape(-1)

    """
    Add MEDIAN based filters/feature 
        - with different size values
    """
    for num in range(1, 34, 2):
        median = 'Median s' + str(num)
        df[median] = nd.median_filter(
            image1, size=num).reshape(-1)

    images_dataset = images_dataset.append(df)


"""
Step 2: Read labeled images and create another dataframe
"""
masks_dataset = pd.DataFrame()  # Create dataframe to capture mask info

masks_path = 'C:/DATA_many/Train_Masks_many/'

for mask in os.listdir(masks_path):  # iterate through each file
    print(mask)

    df2 = pd.DataFrame()  # Temporary dataframe to capture features from each loop
    # Reset dataframe to blank after each loop

    input_mask = cv2.imread(masks_path + mask)   # Read masks

    # Check if input mask is RGB or gray, and convert
    # to gray if RGB

    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)

    elif input_mask.ndim == 2:
        label = input_mask

    else:
        raise Exception('The module works only with grayscale and RGB images!')

    """
    Start adding data to the dataframe
        - Pixel values
        - Image name
        - Extract Gabor features
    """
    label_values = label.reshape(-1)
    df2['Label_Values'] = label_values
    df2['Mask_Name'] = mask

    masks_dataset = masks_dataset.append(df2)


final_dataset = pd.concat(
    [images_dataset, masks_dataset], axis=1)

# Remove pixels which have a value of 0
final_dataset = final_dataset[
    final_dataset.Label_Values != 0]

final_dataset.to_csv('segment_many.csv',
                     index=False)

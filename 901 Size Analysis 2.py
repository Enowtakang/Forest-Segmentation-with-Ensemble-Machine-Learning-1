from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
import numpy as np
import pandas as pd
import os
import glob


"""
Create a master properties dataframe
"""
master_properties_df = pd.DataFrame()

"""
Path to all image groups
"""
# Ground_truth_segments
path1 = 'C:/DATASET + Results/Test_Images/Masks/*.jpg'
# Single_shot_segments
path2 = 'C:/DATASET + Results/Results_Single_Shot/*.jpg'
# Multiple_shot_segments
path3 = 'C:/DATASET + Results/Results_Multiple_Shots/*.jpg'


"""
Extract properties per image, 
and put in master_properties_df
"""
for file in glob.glob(path3):
    image = img_as_ubyte(rgb2gray(io.imread(file)))
    threshold = threshold_otsu(image)
    # Label image
    label_image = measure.label(
        # Parts (pixels) of the image above the threshold
        # refer to the white regions, which mask
        # the forest parts of the images
        image > threshold,
        connectivity=image.ndim)
    properties = ['area',
                  'euler_number', 'extent',
                  'perimeter', 'solidity']
    props = measure.regionprops_table(
        label_image,
        image,
        properties=properties)
    df = pd.DataFrame(props)
    master_properties_df = master_properties_df.append(
        df)


master_properties_df.to_csv(
    'Multiple_Shots_properties.csv',
    index=False)

from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.color import label2rgb, rgb2gray
from skimage.filters import threshold_otsu
import numpy as np
import pandas as pd


"""
Load the image
"""
path = 'C:/DATASET + Results/Test_Images/Masks/855_mask_01.jpg'
image = img_as_ubyte(rgb2gray(io.imread(path)))


"""
Threshold the image
    - There is a binary thresholding problem
    - There is also multilineary otsu thresholding
    
We use the default otsu threshold (single threshold)
"""
threshold = threshold_otsu(image)
# print(threshold)


"""
Label the image:
    Below threshold, 
    Above threshold
"""
label_image = measure.label(
    # Parts (pixels) of the image above the threshold
    # refer to the white regions, which mask
    # the forest parts of the images
    image > threshold,
    connectivity=image.ndim)

# Visualize labeled image
image_label_overlay = label2rgb(
    label_image,
    image=image)
plt.imshow(image_label_overlay)
# plt.show()
# plt.imsave()


"""
Compute image properties
"""
properties = ['area',
              'euler_number', 'extent',
              'perimeter', 'solidity']

props = measure.regionprops_table(
    label_image,
    image,
    properties=properties)


"""
Store table in pandas dataframe
    - 'props' above is pandas compatible
"""
df = pd.DataFrame(props)
# print(df.head())

import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd


"""
- Import Image (2D, RGB)
- Convert image to gray channel (2D, G)
- reshape image to 1D (1D, G) 
"""
path = 'D:/DATASETS/Forest Segmented/images/855_sat_01.jpg'
image1 = cv2.imread(path)
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = image1.reshape(-1)
# print(image1.shape)
# print(image2.shape)


"""
- Create empty dataframe 
- Add original pixel values to the dataframe 
    as feature number 1
"""
df = pd.DataFrame()
df['Original Pixels'] = image2
# print(df.head())


"""
- Add Gabor features as feature number 2
"""
# num=1 serves to count numbers up in order to give
# gabor features a label in the dataframe
num = 1
kernel_size = 5
kernels = list()
for theta in range(2):  # Define number of thetas
    theta = theta / 4. * np.pi
    for sigma in (1, 3):    # Sigma with 1 and 3
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
                    image2,
                    cv2.CV_8UC3,
                    kernel)
                filtered_image = filter_image.reshape(
                    -1)
                df[gabor_label] = filtered_image  # Labels columns as Gabor1, Gabor2
                # print(gabor_label,
                #       ': theta', theta,
                #       ': sigma', sigma,
                #       ': lamda', lamda,
                #       ': gamma', gamma)
                num += 1  # Increment for Gabor column label


# print(df.head())


"""
- Apply 'Canny edge' detector 
- Reshape it and use as feature number 3
"""
edges = cv2.Canny(image1, 100, 200)
edges1 = edges.reshape(-1)
df['Canny Edge'] = edges1
# print(df.head())


"""
Add more Filters/Features:
    - Roberts, Sobel, Scharr, Prewitt
    - Rename columns to: Roberts, Sobel, Scharr, Prewitt
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


"""
Now, provide the last feature: The ground truth/Mask:

- Import Image (2D, RGB)
- Convert image to gray channel (2D, G)
- reshape image to 1D (1D, G) 

- 
"""
path2 = 'D:/DATASETS/Forest Segmented/masks/855_mask_01.jpg'
labeled_image = cv2.imread(path2)
labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2GRAY)
labeled_image1 = labeled_image.reshape(-1)
# print(labeled_image.shape)
# print(labeled_image1.shape)

df['Labels'] = labeled_image1


df.rename(
    columns={
        '<function roberts at 0x0000023CA6FB30D0>': 'Roberts',
        '<function sobel at 0x0000023CA6F48B80>': 'Sobel',
        '<function scharr at 0x0000023CA6F48D30>': 'Scharr',
        '<function prewitt at 0x0000023CA6F48EE0>': 'Prewitt'},
    inplace=True)


# print(df.shape)


# df.to_csv('segment.csv', index=False)

"""
Step 7:
Here, we are going to use the RF pickled model
to AUTOMATE image segmentation for multiple images
"""
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
import glob
import pickle
import matplotlib.pyplot as plt


"""
Define feature extraction function
"""


def feature_extraction(image1):
    image2 = image1.reshape(-1)
    """
    create empty dataframe. add original pixel 
    values as feature number 1
    """
    df = pd.DataFrame()
    df['Original Pixels'] = image2

    """
    Add 32 gabor features
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
                    df[gabor_label] = filtered_image  # Labels columns as Gabor1, Gabor2...
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

    return df


"""
- load pickled model
- for each image, extract features
"""
file_name = 'segment_model_many'
load_model = pickle.load(open(file_name, 'rb'))

path = 'C:/DATA/Train_Images/*.jpg'
save_image_path = 'C:/DATA_many/Segmented_many/'
cmap = 'Accent_r'    # cividis
digit = 1


for file in glob.glob(path):
    name = f'prediction_{digit}.jpg'
    image3 = cv2.imread(file)
    image1 = cv2.cvtColor(image3,
                          cv2.COLOR_BGR2GRAY)
    X = feature_extraction(image1)
    result = load_model.predict(X)
    segmented = result.reshape(image1.shape)
    plt.imsave(save_image_path + name,
               segmented,
               cmap=cmap)
    digit += 1


"""
supported cmap values are: 
'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 
'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 
'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 
'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 
'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 
'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 
'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 
'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 
'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 
'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 
'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 
'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 
'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 
'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 
'afmhot_r', 'autumn', 'autumn_r', 'binary', 
'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 
'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 
'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 
'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 
'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 
'gist_gray_r', 'gist_heat', 'gist_heat_r', 
'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 
'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 
'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 
'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 
'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 
'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 
'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 
'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 
'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 
'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 
'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 
'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 
'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 
'winter_r'
"""


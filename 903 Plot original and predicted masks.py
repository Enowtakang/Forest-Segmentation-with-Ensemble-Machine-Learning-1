import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


images = list()
folder = 'C:/DATASET + Results/Plots/*.jpg'

for image_path in glob.glob(folder):
    images.append(mpimg.imread(image_path))

plt.figure()
columns = 10

for i, image in enumerate(images):
    plt.subplot(
        columns,
        4,
        i + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

plt.show()

# """
# Plot 40 disease photos
# """
# from matplotlib import pyplot as plt
# from matplotlib.image import imread
#
#
# """
# Specify location of dataset
# """
# folder = 'C:/DATASET + Results/Plots/'
#
#
# """
# Plot first 40 images
# """
# for i in range(1, 1):
#     """
#     Define subplot
#     """
#     plt.subplot(540 + i)
#
#     """
#     Define filename
#     """
#     ends = str(i) + '.jpg'
#     filename = folder + ends
#
#     """
#     Load image pixels
#     """
#     image = imread(filename)
#
#     """
#     Plot raw pixel data
#     """
#     plt.imshow(image)
#
#
# """
# Show the figure
# """
# plt.show()

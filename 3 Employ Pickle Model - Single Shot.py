import pickle
import pandas as pd
from matplotlib import pyplot as plt


"""
Load data
"""
df = pd.read_csv('segment.csv')
# print(df.shape)


"""
Define dependent variable (Y) and 
    - Independent Variables (X)
"""
Y = df['Labels'].values
# print(Y)
X = df.drop(labels=['Labels'], axis=1)
# print(X)


"""
Load model
"""
load_model = pickle.load(open('segment_model', 'rb'))


"""
Find out if model can reconstruct original image
"""
cmap = 'summer'   # 'bone', 'summer'
result = load_model.predict(X)
segmented_image = result.reshape((256, 256))
plt.imshow(segmented_image,
           cmap=cmap)
plt.imsave('segmented_forest.jpg',
           segmented_image,
           cmap=cmap)

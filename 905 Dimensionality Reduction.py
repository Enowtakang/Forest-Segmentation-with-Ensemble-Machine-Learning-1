"""
Dimensionality reduction with:
    - PCA
    - LDA
    - t-SNE
    - UMAP
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap import UMAP   # after installing 'umap-learn' package


"""
Load datasets
"""
data_single = pd.read_csv(
    'dim_reduc_features_segment_single.csv')
data_multiple = pd.read_csv(
    'dim_reduc_features_segment_multiple.csv')

"""
- Define X and y
"""
# 'Labels' for data_single,
# 'Label_Values' for data_multiple
X = data_multiple.drop(
    labels=['Label_Values'], axis=1)

# 'Labels' for data_single,
# 'Label_Values' for data_multiple
y = data_multiple['Label_Values'].values


"""
- Set PCA
- Set LDA
- Set t-SNE
- Set UMAP
"""


def set_pca(X):
    x_pca = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(x_pca[:, 0], x_pca[:, 1],
                          # colors
                          c=y)
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(range(10)))
    plt.show()


# set_pca(X)


def set_lda(X, y):
    x_lda = LDA(n_components=2).fit_transform(X, y)
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(x_lda[:, 0], x_lda[:, 1],
                          # colors
                          c=y)
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(range(10)))
    plt.show()


# set_lda(X, y)


def set_tsne(X):
    x_tsne = TSNE(n_jobs=-1).fit_transform(X[:10000])
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(x_tsne[:, 0], x_tsne[:, 1],
                          # colors
                          c=y[:10000])
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(range(10)))
    plt.show()


# set_tsne(X)


def set_umap(X):
    x_umap = UMAP(
        n_neighbors=10,
        min_dist=0.1,
        metric='correlation').fit_transform(X)
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(x_umap[:, 0], x_umap[:, 1],
                          # colors
                          c=y)
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=list(range(10)))
    plt.show()


# set_umap(X)

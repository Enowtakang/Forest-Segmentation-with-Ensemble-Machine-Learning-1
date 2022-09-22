import pandas as pd


"""
Load datasets
"""
dataset_single = pd.read_csv('segment.csv')
dataset_multiple = pd.read_csv('segment_many.csv')


"""
Extract features
"""
ds_imp_features = ['Gaussian s27', 'Gaussian s23',
                   'Gaussian s25', 'Gaussian s19',
                   'Gaussian s15', 'Gaussian s17',
                   'Gaussian s11', 'Median s29',
                   'Prewitt', 'Sobel',
                   'Roberts', 'Scharr',
                   'Gaussian s29', 'Median s5',
                   'Gaussian s1', 'Median s19',
                   'Gabor31', 'Median s3',
                   'Median s1', 'Gabor29', 'Labels']

dm_imp_features = ['Scharr', 'Prewitt',
                   'Roberts', 'Sobel',
                   'Gabor15', 'Gabor16',
                   'Gabor31', 'Gabor23',
                   'Gabor32', 'Gabor5',
                   'Gabor29', 'Gabor30',
                   'Gabor24', 'Gabor6',
                   'Gabor21', 'Gabor4',
                   'Median s5', 'Pixel_Values',
                   'Median s3', 'Gabor8',
                   'Label_Values']

feat_ds_single = dataset_single[ds_imp_features]
feat_dm_multiple = dataset_multiple[dm_imp_features]

feat_ds_single.to_csv(
    'dim_reduc_features_segment_single.csv',
    index=False)
feat_dm_multiple.to_csv(
    'dim_reduc_features_segment_multiple.csv',
    index=False)

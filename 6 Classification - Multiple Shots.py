"""
Step 3: Get data ready for Random Forest
"""
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


"""
Load dataset
"""
dataset = pd.read_csv('segment_many.csv')

"""
Define X and Y
"""
labels = ['Image_Name', 'Mask_Name', 'Label_Values']
X = dataset.drop(labels=labels, axis=1)
Y = dataset['Label_Values'].values

"""
Train-Test Splitting
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2022)

# print('X_train.shape = ', X_train.shape,
#       'X_test.shape = ', X_test.shape,
#       'Y_train.shape = ', y_train.shape,
#       'Y_test.shape = ', y_test.shape)

"""
Step 4: Define the classifier and fit the 
        model with training data
"""
model = RandomForestClassifier(n_estimators=50,
                               random_state=2022)
model.fit(X_train, y_train)

"""
Step 5: Evaluate model - check model accuracy
"""
predict_test = model.predict(X_test)

accuracy = round(
    metrics.accuracy_score(
        y_test, predict_test) * 100, 2)

# Check Accuracy on test dataset
# print("Accuracy = ", accuracy)

"""
Step 6: Save model for future use
"""
model_name = 'segment_model_many'
# pickle.dump(model, open(model_name, 'wb'))


"""
See feature ranking (which feature contributed most
    to the success of the algorithm?)
    - make bar plot of first 9 features
"""
fimp = [round(val*100, 2) for val in model.feature_importances_]
feature_list = list(X.columns)
feature_imps = pd.Series(
    fimp,
    index=feature_list).sort_values(ascending=False)

# print(feature_imps.head(30))

feature_imps[:20].plot.barh()
# plt.show()

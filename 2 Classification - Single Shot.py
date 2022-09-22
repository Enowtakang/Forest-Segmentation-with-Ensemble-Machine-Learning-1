import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle


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
Split data into train and test subsets
    - 80% training, 20% testing
"""
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2022)

# print('X_train.shape = ', X_train.shape,
#       'X_test.shape = ', X_test.shape,
#       'Y_train.shape = ', Y_train.shape,
#       'Y_test.shape = ', Y_test.shape)


"""
Train an ML model
    -Start with RF classier, to classify/predict the 
        segment label (stored in the 'Labels' column.
    
    - Repeat this phase with many more ensemble 
        classifiers
"""
model1 = RandomForestClassifier(
    n_estimators=10, random_state=2022)
model1.fit(X_train, Y_train)

prediction_test = model1.predict(X_test)
accuracy = round(
    metrics.accuracy_score(Y_test, prediction_test) * 100,
    2)

# print("Accuracy = ", accuracy)


"""
See feature ranking (which feature contributed most
    to the success of the algorithm?)
    - make bar plot of first 9 features
"""
fimp = [round(val*100, 2) for val in model1.feature_importances_]
feature_list = list(X.columns)
feature_imps = pd.Series(
    fimp,
    index=feature_list).sort_values(ascending=False)

# print(feature_imps.head(30))

feature_imps[:20].plot.barh(color='brown')
# plt.show()


"""
Save the trained model to pickle
"""
file_name = 'segment_model'
# pickle.dump(model1, open(file_name, 'wb'))

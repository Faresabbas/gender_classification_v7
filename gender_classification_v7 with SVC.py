#laod data
import pandas as pd 
dataset = pd.read_csv('E:\ml\projects\gender_classification_v7\gender_classification_v7.csv')
X = dataset.iloc[ : , : -1]
y = dataset.iloc[ : , -1 ]


#Information about the data columns
print(dataset.info())

#checking to see if there's any null variables
print(dataset.isnull().sum())

# listing the unique values for the wine quality
print(dataset['gender'].unique())
print(dataset['gender'].value_counts())


# Cleaning data

from sklearn.impute import SimpleImputer
import numpy as np
#----------------------------------------------------
'''
impute.SimpleImputer(missing_values=nan, strategy='mean’, fill_value=None, verbose=0, copy=True)
'''
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
#X Data
print('X Data is \n' , X[:10])
#y Data
print('y Data is \n' , y[:10])

#feature_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
FeatureSelection = SelectFromModel( estimator=LogisticRegression()).fit(X, y) # make sure that thismodel is well-defined
X = FeatureSelection.fit_transform(X, y)

#showing X Dimension 
#print('X Shape is ' , X.shape)
#print('Selected Features are : ' , FeatureSelection.get_support())


#Data Scaling 
from sklearn.preprocessing import PolynomialFeatures

#Polynomial the Data
scaler = PolynomialFeatures(degree=3, include_bias=True, interaction_only=False)
X = scaler.fit_transform(X)

#showing data
print('X \n' , X[:10])
print('y \n' , y[:10])



#TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train \n' , X_train)
    print('X_test \n' , X_test)
    print('y_train \n' ,y_train)
    print('y_test \n' , y_test)

   
#Applying SVC Model 
from sklearn.svm import SVC

#Applying SVC Model 

SVCModel = SVC(kernel= 'rbf',# it can be also linear,poly,sigmoid,precomputed
               max_iter=100,C=1.0,gamma='auto')
SVCModel.fit(X_train, y_train)

#Calculating Details
print('SVCModel Train Score is : ' , SVCModel.score(X_train, y_train))
print('SVCModel Test Score is : ' , SVCModel.score(X_test, y_test))
print('----------------------------------------------------')

#Calculating Prediction
y_pred = SVCModel.predict(X_test)
print('Predicted Value for SVCModel is : ' , y_pred[:10])


#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))
from sklearn.metrics import accuracy_score
AccScore = accuracy_score(y_test, y_pred, normalize=False)
print('Accuracy Score is : ', AccScore)


#Calculating F1 Score  : 2 * (precision * recall) / (precision + recall)
from sklearn.metrics import f1_score
F1Score = f1_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
print('F1 Score is : ', F1Score)


#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))   1 / 1+2 
from sklearn.metrics import recall_score
RecallScore = recall_score(y_test, y_pred, average='micro') #it can be : binary,macro,weighted,samples
print('Recall Score is : ', RecallScore)

#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



#Confusion Matrix shape 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.heatmap(cm,center=True)
plt.show()

#analysing gender variable
sns.countplot(y)
plt.show()
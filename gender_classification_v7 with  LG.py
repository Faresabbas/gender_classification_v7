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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 , f_classif 
#----------------------------------------------------
#Feature Selection by KBest 
print('Original X Shape is ' , X.shape)
FeatureSelection = SelectKBest(score_func= chi2 ,k=4) # score_func can = f_classif 
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension 
print('X Shape is ' , X.shape)
print('Selected Features are : ' , FeatureSelection.get_support())


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.3, random_state=1)




# Fitting Logistic Regression to the Training set
'''
#linear_model.LogisticRegression(penalty='l2’,dual=False,tol=0.0001,C=1.0,fit_intercept=True,intercept_scaling=1,
#                                class_weight=None,random_state=None,solver='warn’,max_iter=100,
#                                multi_class='warn’, verbose=0,warm_start=False, n_jobs=None)
'''
from sklearn.linear_model import LogisticRegression
clss = LogisticRegression(penalty='l2',solver='sag',random_state = 15)
clss.fit(X_train, y_train)


# Predicting the Test set results
y_pred = clss.predict(X_test)
print(list(y_pred[:10]))
print(list(y_test[:10]))
print(clss.n_iter_)
print(clss.classes_)



#probability of all values
pr = clss.predict_proba(X_test)[0:10,:]
print(pr)



#  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#Calculating Details
print('LogisticRegressionModel Train Score is : ' , clss.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , clss.score(X_test, y_test))


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



#Confusion Matrix shape 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.heatmap(cm,center=True)
plt.show()

#analysing gender variable
sns.countplot(y)
plt.show()
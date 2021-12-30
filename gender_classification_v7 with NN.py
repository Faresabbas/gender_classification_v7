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


#Feature Selection by KBest 
# print('Original X Shape is ' , X.shape)
FeatureSelection = SelectKBest(score_func= chi2 ,k=3) # score_func can = f_classif 
X = FeatureSelection.fit_transform(X, y)

# showing X Dimension 
# print('X Shape is ' , X.shape)
# print('Selected Features are : ' , FeatureSelection.get_support())



#Data Scaling 

from sklearn.preprocessing import StandardScaler

#Standard Scaler for Data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)

# #showing data
# print('X \n' , X[:10])
# print('y \n' , y[:10])



#Data Split
from sklearn.model_selection import KFold

#KFold Splitting data

kf = KFold(n_splits=4, random_state=44, shuffle =True)

#KFold Data
for train_index, test_index in kf.split(X):
    print('Train Data is : \n', train_index)
    print('Test Data is  : \n', test_index)
    print('-------------------------------')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print('X_train Shape is  ' , X_train.shape)
    print('X_test Shape is  ' , X_test.shape)
    print('y_train Shape is  ' ,y_train.shape)
    print('y_test Shape is  ' , y_test.shape)
   
   
   
#MLPClassifier
from sklearn.neural_network import MLPClassifier

#Applying MLPClassifier Model 


MLPClassifierModel = MLPClassifier(activation='tanh', # can be also identity , logistic , relu
                                   solver='lbfgs',  # can be also sgd , adam
                                   learning_rate='constant', # can be also invscaling , adaptive
                                   early_stopping= False,
                                   alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)
MLPClassifierModel.fit(X_train, y_train)

#Calculating Details
print('MLPClassifierModel Train Score is : ' , MLPClassifierModel.score(X_train, y_train))
print('MLPClassifierModel Test Score is : ' , MLPClassifierModel.score(X_test, y_test))
print('MLPClassifierModel loss is : ' , MLPClassifierModel.loss_)
print('MLPClassifierModel No. of iterations is : ' , MLPClassifierModel.n_iter_)
print('MLPClassifierModel No. of layers is : ' , MLPClassifierModel.n_layers_)
print('MLPClassifierModel last activation is : ' , MLPClassifierModel.out_activation_)
print('----------------------------------------------------')

#Calculating Prediction
y_pred = MLPClassifierModel.predict(X_test)
y_pred_prob = MLPClassifierModel.predict_proba(X_test)
print('Predicted Value for MLPClassifierModel is : ' , y_pred[:10])
print('Prediction Probabilities Value for MLPClassifierModel is : ' , y_pred_prob[:10])



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
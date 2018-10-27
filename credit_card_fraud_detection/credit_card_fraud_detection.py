#!/usr/bin/env python
# coding: utf-8
# In[1]:
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
# In[4]:
data = pd.read_csv("creditcard.csv")
print(data.columns)


# In[5]:
print(data.shape)
print(data.describe())


# In[6]:
data = data.sample(frac = 0.1,random_state=1)
print(data.shape)
# In[7]:
#plot histogram of each parameter
data.hist(figsize=(20,20))
plt.show()
# In[8]:
#determine number of fraud cases n dataset
fraud = data[data['Class']==1]
valid = data[data['Class']==0]
outlier_fraction = len(fraud)/ float(len(valid))
print(outlier_fraction)
print("# of fraud cases:{}".format(len(fraud)))
print("# of valid cases:{}".format(len(valid)))


# In[9]:
#correlation matrix
corrmat = data.corr()
fit = plt.figure(figsize =(12,9))
sns.heatmap(corrmat, vmax=0.8, square = True)
plt.show()
# In[10]:
#get all the columns from the dataframe
columns = data.columns.tolist()
#filter the columns to remove data we dont want
columns = [c for c in columns if c not  in ['Class']]
#store the var we'll be predicting for
target = "Class"
X = data[columns]
Y = data[target]
#print the shapes of x and y
print(X.shape)
print(Y.shape)
# In[11]:
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1
#define the outlier detection methods
classifiers={
    "Isolation forest": IsolationForest(max_samples=len(X),
                                       contamination = outlier_fraction,
                                       random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
    n_neighbors =20,
    contamination = outlier_fraction)
}
# In[13]:
#fit the model
n_outliers = len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred=clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred =clf.decision_function(X)
        y_pred=clf.predict(X)
    #reshape the prediction values to 0  for valid and 1 for fake
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_errors = (y_pred !=Y).sum()
    #run classification metrics
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))




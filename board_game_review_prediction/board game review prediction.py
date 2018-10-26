#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)


# In[2]:


x = 4
print x


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import  train_test_split


# In[4]:


games = pandas.read_csv("scrapers.csv")


# In[5]:


print(games.columns)
print("------------------")
print(games.shape)


# In[6]:


plt.hist(games["average_rating"])
plt.show()


# In[7]:


#print the first row of all games with zero scores
print(games[games["average_rating"]==0].iloc[0])
#print the first row of games with  scores > 0
print(games[games["average_rating"]>0].iloc[0])


# In[8]:


#remove rows without any reviews
games = games[games["users_rated"] > 0]
#remove any rows with missing values
games = games.dropna(axis=0)
# make a histogram of all avg ratings
plt.hist(games["average_rating"])
plt.show()


# In[9]:


print(games.columns)


# In[10]:


#correlation matrix
cormat = games.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(cormat,vmax=.8,square =True)
plt.show()


# In[11]:


#get all columns from the dataframe
columns = games.columns.tolist()

#filter  the columns to remove data we dont want
columns = [c for  c in columns if c not in ["bayes_average_rating","average_rating", "type", "name", "id"]]
#store the var we ll be predicting on
target = "average_rating"
#generate training & test datasets
from sklearn.model_selection import train_test_split
train = games.sample(frac=0.8, random_state = 1)

#select anything  not in the training set and put  it in test
test = games.loc[~games.index.isin(train.index)]
#print shapes
print(train.shape)
print(test.shape)




# In[12]:


#import linear regression model first
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error
#initlize model class
LR = LinearRegression()
LR.fit(train[columns], train[target])


# In[13]:


#generate predictions for testing set
predictions = LR.predict(test[columns])
#compute error btw our test predictions and actual values
mean_squared_error(predictions,test[target])


# In[14]:


#import the random forest model
from sklearn.ensemble import RandomForestRegressor
#init model
RFR =  RandomForestRegressor(n_estimators = 100,min_samples_leaf =10,random_state=1)
#fit to the data
RFR.fit(train[columns],train[target])



# In[17]:


predictions  = RFR.predict(test[columns])


# In[22]:


#compute the error btw our  test predictions and actual values
mean_squared_error(predictions, test[target])
#make prediction with both models
rating_lr = LR.predict(test[columns].iloc[0].values.reshape(1,-1))
rating_rfr = RFR.predict(test[columns].iloc[0].values.reshape(1,-1))
#print out the predictions
print(rating_lr)
print(rating_rfr)

print("actual value from the test dataset: " + str(test[target].iloc[0] ) )


# In[ ]:





# In[ ]:





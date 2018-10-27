#!/usr/bin/env python
# coding: utf-8

# In[17]:


from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np
import sys, keras
from PIL import Image
# In[19]:
#load the data
(X_train,y_train),(X_test,y_test) = cifar10.load_data()


# In[20]:
#lets determine the dataset characteristics
print("training images:{}".format(X_train.shape))
print("testing images:{}".format(X_test.shape))
#A single image
print(X_train[0].shape)
# In[21]:
for i in range(0,9):
    plt.subplot(330+1+i)
    img = X_train[i].transpose([1,2,0])
    plt.imshow(img)
    
#show the plot
plt.show()
#preprocessing the data
#fix the random seed for reproducibility
seed = 6
np.random.seed(seed)
#load the data
(X_train,y_train), (X_test,y_test) = cifar10.load_data()
#normalize the inputs from 0-255 to 0.0-1.0

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train/255.0
X_test = X_test/255.0
print(X_train[0])


# In[22]:
#class lables shape
print(y_train.shape)
print(y_train[0])


# In[23]:
#hot encode the outputs
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_class = Y_test.shape[1]
print(num_class)
print(Y_train.shape)
print(Y_train[0])




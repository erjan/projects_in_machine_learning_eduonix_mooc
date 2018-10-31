#!/usr/bin/env python
# coding: utf-8
# In[28]:
import sys, sklearn,matplotlib,numpy as np
# In[29]:
from keras.datasets import mnist
# In[30]:
(x_train,y_train), (x_test,y_test) = mnist.load_data()
# In[31]:
print('training data:{}'.format(x_train.shape))
print('training labels:{}'.format(y_train.shape))
# In[32]:
print('testing data: {}'.format(x_test.shape))
print('testing labels: {}'.format(y_test.shape))
# In[33]:
import matplotlib.pyplot as plt
#python magic function
get_ipython().magic(u'matplotlib inline')
# In[34]:
#create a fgure 3x3 subplots using pyplot
fig,axs = plt.subplots(3,3,figsize=(12,12))
plt.gray()
#loop thru subplots and add mnist images
for i, ax in enumerate(axs.flat):
    ax.matshow(x_train[i])
    ax.axis('off')
    ax.set_title('number {}'.format(y_train[i]))
#display the figure
fig.show()
# In[35]:
#preprocessing images
#convert each image to 1 dim array
X = x_train.reshape(len(x_train), -1)
Y = y_train
# In[36]:
print X.shape
# In[37]:
print x_train.shape
# In[38]:
#normalize the data to 0-1
X[0]
# In[39]:
X = X.astype(float)/255.
print X.shape
# In[40]:
print(X[0].shape)
print(X[0])
# In[41]:
from sklearn.cluster import MiniBatchKMeans
n_digits = len(np.unique(y_test))
print(n_digits)
#init kmeans model
kmeans = MiniBatchKMeans(n_clusters =n_digits)
kmeans.fit(X)
# In[42]:
kmeans.labels_[:20]
# In[43]:
def infer_cluster_labels(kmeans,actual_labels):
    '''
    associate most probable label with each  cluster in kmeans model
    returns: dictionary  of clusters assigned to each label
    '''
    inferred_labels = {}
    for i in range(kmeans.n_clusters):
        #find the index of points in cluster
        labels=[]
        index = np.where(kmeans.labels_ ==i)
        #append actual labels for each  point in cluster
        labels.append(actual_labels[index])
        #determine most common label
        if len(labels) == 0:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
        #assign cluster to a value in the inferred labels dictionary
        if np.argmax(counts) in inferred_labels:
            inferred_labels[np.argmax(counts)].append(i)
        else:
            #create new array for this key
            inferred_labels[np.argmax(counts)] = [i]
        #print(labels)
        #print('cluster:{}, label : {}'.format(i,np.argmax(counts)))
    return inferred_labels
    
        
# In[44]:
array = np.ones((1,3))
print(array.shape)
# In[45]:
np.bincount(np.squeeze(array).astype(np.uint8))
# In[46]:
def infer_data_labels(X_labels, cluster_labels):
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    for i , cluster in enumerate(X_labels):
        for key,value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i]= key
    return predicted_labels
# In[47]:
#test the infer cluster labels
cluster_labels = infer_cluster_labels(kmeans,Y)
# In[48]:
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print predicted_labels[:20]
print Y[:20]
# In[49]:
#optimizing & evaluating the clustering algo
from sklearn import metrics
def calculate_metrics(estimator,data,labels):
    #calculate and print metrics
    print("number of clusters:{}".format(estimator.n_clusters))
    print("inertia:{}".format(estimator.inertia_))
    print("homogeneity: {}".format(metrics.homogeneity_score(labels,estimator.labels_)))
# In[50]:
clusters  = [10,16,36,64,144,256]
#test diff numbers of clusters
for n_clusters in clusters:
    estimator = MiniBatchKMeans(n_clusters = n_clusters)
    estimator.fit(X)
    
    #print cluster metrics
    calculate_metrics(estimator,X,Y)
    #determine predicted labels
    cluster_labels =  infer_cluster_labels(estimator, Y)
    predicted_Y= infer_data_labels(estimator.labels_,cluster_labels)
    #calculate and print accuracy
    print('accuracy:{}\n'.format(metrics.accuracy_score(Y,predicted_Y)))
# In[51]:
#test kmeans algo on testing dataset
#convert each image to 1 dim array
X_test = x_test.reshape(len(x_test),-1)
X_test = X_test.astype(float)/255.
#initialize and fit kmeans algo on training data
kmeans = MiniBatchKMeans(n_clusters=256)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans,Y)
#predict labels fro testing data
test_clusters = kmeans.predict(X_test)
predicted_labels = infer_data_labels(test_clusters,cluster_labels)

#calc  and print accuracy
print("testing accuracy: {}".format(metrics.accuracy_score(y_test, predicted_labels)))
# In[ ]:
#visualize cluster centroids
#initialize and fit Kmeans algo
kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X)
#record centroid values
centroids = kmeans.cluster_centers_
#reshape centroids into imgs
images = centroids.reshape(36,28,28)
images +=255
images = images.astype(np.uint8)

#determine cluster labels
cluster_labels =  infer_cluster_labels(kmeans,Y)

#create fig with subplots using plt
fig,axs = plt.subplots(6,6,figsize = (20,20))
plt.gray()

#loop thru  subplots and add centroid images
for i , ax in enumerate(axs.flat):
    for key,value in cluster_labels.items():
        if i in value:
            ax.set_title('inferred label:{}'.format(key))
    #add image to subplot
    ax.matshow(images[i])
    ax.axis('off')
#display the figure
fig.show()

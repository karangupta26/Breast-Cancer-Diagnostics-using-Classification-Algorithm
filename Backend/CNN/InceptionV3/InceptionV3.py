#!/usr/bin/env python
# coding: utf-8

# # Importing System, Data Wrangling, Deeplearning Libaries

# In[ ]:


from PIL import Image
import itertools
import numpy as np
import pandas as pd
import warnings
import nltk
import math
import time
import re
import os
import pickle
#Deep-Learning Library
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


# # Importing Google Drive for Mounting

# In[3]:


from google.colab import drive 
drive.mount('/content/drive')


# The Below Variables are used for Saving paths in Google Drive

# In[ ]:


path = "/content/drive/My Drive/Major Project B.Tech Sem 8/Breast Cancer Dataset/"
savingpath="/content/drive/My Drive/Major Project B.Tech Sem 8/"


# In[ ]:


train_data_dir=path+"Train"
test_data_dir=path+"Test"

#Training paths for saving
data_cnn_InceptionV3_train=savingpath+"InceptionV3/train/data_cnn_InceptionV3_train.npy"
data_filenames_InceptionV3_train=savingpath+"InceptionV3/train/data_filenames_InceptionV3_train.npy"
data_cancerclass_InceptionV3_train=savingpath+"InceptionV3/train/data_cancerclass_InceptionV3_train.npy"
data_cancertype_InceptionV3_train=savingpath+"InceptionV3/train/data_cancertype_InceptionV3_train.npy"
data_mag_InceptionV3_train=savingpath+"InceptionV3/train/data_mag_InceptionV3_train.npy"

#testing paths for saving
data_cnn_InceptionV3_test=savingpath+"InceptionV3/test/data_cnn_InceptionV3_test.npy"
data_filenames_InceptionV3_test=savingpath+"InceptionV3/test/data_filenames_InceptionV3_test.npy"
data_cancerclass_InceptionV3_test=savingpath+"InceptionV3/test/data_cancerclass_InceptionV3_test.npy"
data_cancertype_InceptionV3_test=savingpath+"InceptionV3/test/data_cancertype_InceptionV3_test.npy"
data_mag_InceptionV3_test=savingpath+"InceptionV3/test/data_mag_InceptionV3_test.npy"


# ###### The below block is for genrating Training Data for the Machine learning models

# In[8]:


# Start Clock for time tracking
start_time = time.clock() 
# dimensions of our images.
img_width, img_height = 299, 299
epochs = 50
batch_size = 1
#print("Started")
# Util function for Checking the Substring like Cancer  class or Cancer Subtype or Magnification
def check(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return "NO" 
    else: 
        return "YES" 
    
# The Below Function Genrates and Return Numpy 2-D Array for Training Images
def save_bottlebeck_features_train(train_data=train_data_dir,nb_train_samples=5000,batch_size=1):
   """
    Function to compute InceptionResNetV2 CNN for image feature extraction.
    
    Parameters:
        train_data:-            It cantains the traingin data path
        nb_train_samples:-      Total number of Samples as Training Images
        batch_size:-            This Parameter ensures how Many batchhes to be created for precessing the images
                                Default is 1.
    """
    
    filenames = []                                                   # Array for Storing the Filenames
    cancerclass=[]                                                   # Array for Storing the Cancer Class for Particular Tensor
    cancertype=[]                                                    # Array for Storing the Cancer Subtype for Particular Tensor
    mag=[]                                                           # Array for Storing the Magnification lense for particular tensor
    datagen = ImageDataGenerator(rescale=1. / 255)                   # Variable to Perform Pre-processing of the Images by scaling them.
    #print("Initialized Required data and datagen done")                                                
    # build the InceptionV3 network
    # Setting up the include_top as False to not include Final Dense Layer
    # Including the weights of ImageNet Dataset
    model = applications.InceptionV3(include_top=False, weights='imagenet')
    print("Model settled")
    generator = datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    #print("Model settled")
    """
        Excecuting the Genrator Function to Genrate the Images from the Directory
        Process their Filenames
        start_time = time.clock()
    """
    for i in generator.filenames:
        
        # Array for File Names
        filenames.append(i)
        
        # Array For Cancer Class
        X=re.search("_B_", i)
        if (X):
          cancer=1
          cancerclass.append(cancer)
        else:
          cancer=2
          cancerclass.append(cancer)
        
        # Array For Cancer Type
        cancersubtypes=['_A-','_F-','_PT-','_TA-','_DC-','_LC-','_MC-','_PC-']
        codeforcancersubtypes=[11,12,13,14,21,22,23,24]                           # Numerical Coding of the Cancer Sutype CLasses
        k=0
        for types in cancersubtypes:          
          flagforcancersubtypes=check(i,types)         
          #print(type(pattern.match(i)))
          #print(obtainedpattern)
          if(flagforcancersubtypes=="YES"):
            cancertype.append(codeforcancersubtypes[k])
          k=k+1
         
        # Array for Magnification
        magnificationsformat=["-40-","-100-","-200-","-400-"]
        codeformagnifications=[40,100,200,400]
        j=0
        for types in magnificationsformat:
          
          flagformagnifications=check(i,types)         
          if(flagformagnifications=="YES"):
            mag.append(codeformagnifications[j])
          j=j+1
          
    """
        print(cancerclass)    
        print(cancertype)
        print(mag)
        print("For loop ended")
        print(time.clock() - start_time, "seconds, time for loop")
        print(generator)
        print("training start")
    """
    # Time for the Neural Network To genrate the Features
    start_time = time.clock()
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    print(time.clock() - start_time, "seconds, time for training")
    #print("training finised")
    #print("reshape the trained vector")
    # ReShaping the Data 
    bottleneck_features_train = bottleneck_features_train.reshape((5000,131072))
    print("Trainning done")
    np.save(open(data_cnn_InceptionV3_train, 'wb'), bottleneck_features_train)
    np.save(open(data_filenames_InceptionV3_train, 'wb'), np.array(filenames))
    np.save(open(data_cancerclass_InceptionV3_train, 'wb'), np.array(cancerclass))
    np.save(open(data_cancertype_InceptionV3_train, 'wb'), np.array(cancertype))
    np.save(open(data_mag_InceptionV3_train, 'wb'), np.array(mag))
    print("npy file saved")
    

save_bottlebeck_features_train()  

print(time.clock() - start_time, "seconds")


# In[12]:


# Start Clock for time tracking
start_time = time.clock() 
# dimensions of our images.
img_width, img_height = 299, 299
epochs = 50
batch_size = 1
#print("Started")
# Util function for Checking the Substring like Cancer  class or Cancer Subtype or Magnification
def check(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return "NO" 
    else: 
        return "YES" 
    
# The Below Function Genrates and Return Numpy 2-D Array for Testing Images
def save_bottlebeck_features_test(train_data=test_data_dir,nb_train_samples=2900,batch_size=1):
    """
    Function to compute InceptionResNetV2 CNN for image feature extraction.
    
    Parameters:
        train_data:-            It cantains the testing data path
        nb_train_samples:-      Total number of Samples as Testing Images
        batch_size:-            This Parameter ensures how Many batchhes to be created for precessing the images
                                Default is 1.
    """
    
    filenames = []                                                   # Array for Storing the Filenames
    cancerclass=[]                                                   # Array for Storing the Cancer Class for Particular Tensor
    cancertype=[]                                                    # Array for Storing the Cancer Subtype for Particular Tensor
    mag=[]                                                           # Array for Storing the Magnification lense for particular tensor
    datagen = ImageDataGenerator(rescale=1. / 255)                   # Variable to Perform Pre-processing of the Images by scaling them.
    #print("Initialized Required data and datagen done")                                                
    # build the InceptionV3 network
    # Setting up the include_top as False to not include Final Dense Layer
    # Including the weights of ImageNet Dataset
    model = applications.InceptionV3(include_top=False, weights='imagenet')
    #print("Model settled")
    """
        Excecuting the Genrator Function to Genrate the Images from the Directory
        Process their Filenames
    """
    generator = datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    #print("Genrator work finish")
    """
        Loop for Ganrating the Above arrays with some data about the particualr image.
        New Clock Function for total Time Consumption.
        start_time = time.clock()
    """
    for i in generator.filenames:
        
        # Array for File NAmes
        filenames.append(i)
        
        # Array For Cancer Class
        X=re.search("_B_", i)
        if (X):
          cancer=1
          cancerclass.append(cancer)
        else:
          cancer=2
          cancerclass.append(cancer)
        
        # Array For Cancer Type
        cancersubtypes=['_A-','_F-','_PT-','_TA-','_DC-','_LC-','_MC-','_PC-']
         codeforcancersubtypes=[11,12,13,14,21,22,23,24]                           # Numerical Coding of the Cancer Sutype CLasses
        k=0
        for types in cancersubtypes:          
          flagforcancersubtypes=check(i,types)         
          #print(type(pattern.match(i)))
          #print(obtainedpattern)
          if(flagforcancersubtypes=="YES"):
            cancertype.append(codeforcancersubtypes[k])
          k=k+1
         
        # Array for Magnification
        magnificationsformat=["-40-","-100-","-200-","-400-"]
        codeformagnifications=[40,100,200,400]
        j=0
        for types in magnificationsformat:
          
          flagformagnifications=check(i,types)         
          if(flagformagnifications=="YES"):
            mag.append(codeformagnifications[j])
          j=j+1
    """
        print(cancerclass)    
        print(cancertype)
        print(mag)
        print("For loop ended")
        print(time.clock() - start_time, "seconds, time for loop")
        print(generator)
        print("training start")
    """
    # Time for the Neural Network To genrate the Features
    start_time = time.clock()
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)
    print(time.clock() - start_time, "seconds, time for training")
    #print("training finised")
    # ReShaping the Data
    bottleneck_features_train = bottleneck_features_train.reshape((2900,131072)) 
    #print("Trainning done")
    np.save(open(data_cnn_InceptionV3_test, 'wb'), bottleneck_features_train)
    np.save(open(data_filenames_InceptionV3_test, 'wb'), np.array(filenames))
    np.save(open(data_cancerclass_InceptionV3_test, 'wb'), np.array(cancerclass))
    np.save(open(data_cancertype_InceptionV3_test, 'wb'), np.array(cancertype))
    np.save(open(data_mag_InceptionV3_test, 'wb'), np.array(mag))
    print("npy file saved")
    

save_bottlebeck_features_test()

print(time.clock() - start_time, "seconds , Testing data also saved")


# In[ ]:





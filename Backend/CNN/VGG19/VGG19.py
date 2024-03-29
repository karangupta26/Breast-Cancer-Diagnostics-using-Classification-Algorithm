#!/usr/bin/env python
# coding: utf-8

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
import glob


# In[2]:


from google.colab import drive 
drive.mount('/content/drive')


# In[ ]:


path = "/content/drive/My Drive/Major Project B.Tech Sem 8/Breast Cancer Dataset/"
savingpath="/content/drive/My Drive/Major Project B.Tech Sem 8/"


# In[ ]:


train_data_dir=path+"Train"
test_data_dir=path+"Test"

#Training paths
data_cnn_VGG19_train=savingpath+"VGG19/train/data_cnn_VGG19_train.npy"
data_filenames_VGG19_train=savingpath+"VGG19/train/data_filenames_VGG19_train.npy"
data_cancerclass_VGG19_train=savingpath+"VGG19/train/data_cancerclass_VGG19_train.npy"
data_cancertype_VGG19_train=savingpath+"VGG19/train/data_cancertype_VGG19_train.npy"
data_mag_VGG19_train=savingpath+"VGG19/train/data_mag_VGG19_train.npy"

#Testing paths
data_cnn_VGG19_test=savingpath+"VGG19/data_cnn_VGG19_test.npy"
data_filenames_VGG19_test=savingpath+"VGG19/data_filenames_VGG19_test.npy"
data_cancerclass_VGG19_test=savingpath+"VGG19/data_cancerclass_VGG19_test.npy"
data_cancertype_VGG19_test=savingpath+"VGG19/data_cancertype_VGG19_test.npy"
data_mag_VGG19_test=savingpath+"VGG19/data_mag_VGG19_test.npy"


# In[14]:


from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

#model = VGG19(weights='imagenet', include_top=True)

base_model = VGG19(weights='imagenet',include_top=True)
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
image_size = (224, 224)


# In[12]:


start_time = time.clock()
# dimensions of our images.
img_width, img_height = 224, 224
#top_model_weights_path = 'bottleneck_fc_model.h5'
#epochs = 50
#batch_size = 1
print("Started")

def check(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return "NO" 
    else: 
        return "YES" 

def save_bottlebeck_features_train(train_data=train_data_dir,nb_train_samples=5000,batch_size=1):
    
    #Function to compute VGG-16 CNN for image feature extraction.
        
    filenames = []
    cancerclass=[]
    cancertype=[]
    mag=[]
    features=[]
    datagen = ImageDataGenerator(rescale=1. / 255)
    print("Initialized Required data and datagen done")                                                
    # build the VGG19 network
    #model = applications.VGG19(include_top=False, weights='imagenet')
    #print("Model settled")
    generator = datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("Genrator work finish")
    start_time = time.clock()
    a=1
    for i in generator.filenames:
        #print(i)
        #print(type)
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
        codeforcancersubtypes=[11,12,13,14,21,22,23,24]
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
 
        image_path=train_data_dir+"/"+i
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
    
    print(features)
        
    print(cancerclass)    
    print(cancertype)
    print(mag)
    print("For loop ended")
    print(time.clock() - start_time, "seconds, time for loop")
    print(generator)
    
    features=np.array(features)
    print(features.shape)
     
    
    print("reshape the trained vector")
    #features = features.reshape((5000,25088))
    print("Trainning done")
    np.save(open(data_cnn_VGG19_train, 'wb'), features)
    np.save(open(data_filenames_VGG19_train, 'wb'), np.array(filenames))
    np.save(open(data_cancerclass_VGG19_train, 'wb'), np.array(cancerclass))
    np.save(open(data_cancertype_VGG19_train, 'wb'), np.array(cancertype))
    np.save(open(data_mag_VGG19_train, 'wb'), np.array(mag))
    print("npy file saved")

save_bottlebeck_features_train()

print(time.clock() - start_time, "seconds")


# In[16]:


start_time = time.clock()
# dimensions of our images.
img_width, img_height = 224, 224
top_model_weights_path = 'bottleneck_fc_model.h5'
epochs = 50
batch_size = 1
print("Started")

def check(string, sub_str): 
    if (string.find(sub_str) == -1): 
        return "NO" 
    else: 
        return "YES" 

def save_bottlebeck_features_test(train_data=test_data_dir,nb_train_samples=2900,batch_size=1):
    
    #Function to compute VGG-16 CNN for image feature extraction.
    
    features=[]
    filenames = []
    cancerclass=[]
    cancertype=[]
    mag=[]
    datagen = ImageDataGenerator(rescale=1. / 255)
    print("Initialized Required data and datagen done")                                                
    # build the VGG19 network
    #model = applications.VGG19(include_top=False, weights='imagenet')
    #print("Model settled")
    generator = datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("Genrator work finish")
    start_time = time.clock()
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
        codeforcancersubtypes=[11,12,13,14,21,22,23,24]
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
          
        image_path=test_data_dir+"/"+i
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        features.append(flat)
    
    features=np.array(features)
    print(features.shape)
    
    print(cancerclass)    
    print(cancertype)
    print(mag)
    print("For loop ended")
    print(time.clock() - start_time, "seconds, time for loop")
    print(generator)
    
    print("Trainning done")
    np.save(open(data_cnn_VGG19_test, 'wb'), features)
    np.save(open(data_filenames_VGG19_test, 'wb'), np.array(filenames))
    np.save(open(data_cancerclass_VGG19_test, 'wb'), np.array(cancerclass))
    np.save(open(data_cancertype_VGG19_test, 'wb'), np.array(cancertype))
    np.save(open(data_mag_VGG19_test, 'wb'), np.array(mag))
    print("npy file saved")
    

save_bottlebeck_features_test()

print(time.clock() - start_time, "seconds , Testing data also saved")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





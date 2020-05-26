#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
import numpy as np


# In[2]:


train_path="A:\\Projects\\Major Project\\CNN New\\InceptionResNetV2\\train"
test_path="A:\\Projects\\Major Project\\CNN New\\InceptionResNetV2\\test"


# In[3]:


# Training Paths 
data_train=np.load(train_path+"\\data_cnn_InceptionResNetV2_train.npy")
filenames_train=np.load(train_path+"\\data_filenames_InceptionResNetV2_train.npy")
mag_train=np.load(train_path+"\\data_mag_InceptionResNetV2_train.npy")
cancerclass_train=np.load(train_path+"\\data_cancerclass_InceptionResNetV2_train.npy")
cancertype_train=np.load(train_path+"\\data_cancertype_InceptionResNetV2_train.npy")

# Testing Paths 
data_test=np.load(test_path+"\\data_cnn_InceptionResNetV2_test.npy")
filenames_test=np.load(test_path+"\\data_filenames_InceptionResNetV2_test.npy")
mag_test=np.load(test_path+"\\data_mag_InceptionResNetV2_test.npy")
cancerclass_test=np.load(test_path+"\\data_cancerclass_InceptionResNetV2_test.npy")
cancertype_test=np.load(test_path+"\\data_cancertype_InceptionResNetV2_test.npy")


# In[4]:


# New Array as List
new_data_test=[]
new_filenames_test=[]
new_mag_test=[]
new_cancerclass_test=[]
new_cancertype_test=[]

#Counter
C4011=0
C10011=0
C20011=0
C40011=0

C4012=0
C10012=0
C20012=0
C40012=0

C4013=0
C10013=0
C20013=0
C40013=0

C4014=0
C10014=0
C20014=0
C40014=0

C4021=0
C10021=0
C20021=0
C40021=0

C4022=0
C10022=0
C20022=0
C40022=0

C4023=0
C10023=0
C20023=0
C40023=0

C4024=0
C10024=0
C20024=0
C40024=0



index=[]

#List as Set
cancertype40=[11,12,13,14,21,22,23,24]
cancertype100=[11,12,13,14,21,22,23,24]
cancertype200=[11,12,13,14,21,22,23,24]
cancertype400=[11,12,13,14,21,22,23,24]


# In[5]:


for i in range(0,len(cancertype_test)):
    #print(type(cancertype40))
    
    if ((mag_test[i]==40) and (cancertype_test[i] in cancertype40) and (C4011<50 or C4012<50 or C4013<50 or C4014<50 or C4021<50 or C4022<50 or C4023<50 or C4024<50)):
        #C40=C40+1
        if (cancertype_test[i]==11):
            C4011=C4011+1
            
        elif (cancertype_test[i]==12):
            C4012=C4012+1
            
        elif (cancertype_test[i]==13):
            C4013=C4013+1
            
        elif (cancertype_test[i]==14):
            C4014=C4014+1
            
        elif (cancertype_test[i]==21):
            C4021=C4021+1
            
        elif (cancertype_test[i]==22):
            C4022=C4022+1
            
        elif (cancertype_test[i]==23):
            C4023=C4023+1
            
        elif (cancertype_test[i]==24):
            C4024=C4024+1
            
        new_data_test.append(data_test[i])
        new_filenames_test.append(filenames_test[i])
        new_mag_test.append(mag_test[i])
        new_cancerclass_test.append(cancerclass_test[i])
        new_cancertype_test.append(cancertype_test[i])
        index.append(i)
        
    elif ((mag_test[i]==100) and (cancertype_test[i] in cancertype100) and (C10011<50 or C10012<50 or C10013<50 or C10014<50 or C10021<50 or C10022<50 or C10023<50 or C10024<50)):
        #C40=C40+1
        if (cancertype_test[i]==11):
            C10011=C10011+1
            
        elif (cancertype_test[i]==12):
            C10012=C10012+1
            
        elif (cancertype_test[i]==13):
            C10013=C10013+1
            
        elif (cancertype_test[i]==14):
            C10014=C10014+1
            
        elif (cancertype_test[i]==21):
            C10021=C10021+1
            
        elif (cancertype_test[i]==22):
            C10022=C10022+1
            
        elif (cancertype_test[i]==23):
            C10023=C10023+1
            
        elif (cancertype_test[i]==24):
            C10024=C10024+1
            
        new_data_test.append(data_test[i])
        new_filenames_test.append(filenames_test[i])
        new_mag_test.append(mag_test[i])
        new_cancerclass_test.append(cancerclass_test[i])
        new_cancertype_test.append(cancertype_test[i])
        index.append(i)
        
    if ((mag_test[i]==200) and (cancertype_test[i] in cancertype200) and (C20011<50 or C20012<50 or C20013<50 or C20014<50 or C20021<50 or C20022<50 or C20023<50 or C20024<50)):
        #C40=C40+1
        if (cancertype_test[i]==11):
            C20011=C20011+1
            
        elif (cancertype_test[i]==12):
            C20012=C20012+1
            
        elif (cancertype_test[i]==13):
            C20013=C20013+1
            
        elif (cancertype_test[i]==14):
            C20014=C20014+1
            
        elif (cancertype_test[i]==21):
            C20021=C20021+1
            
        elif (cancertype_test[i]==22):
            C20022=C20022+1
            
        elif (cancertype_test[i]==23):
            C20023=C20023+1
            
        elif (cancertype_test[i]==24):
            C20024=C20024+1
            
        new_data_test.append(data_test[i])
        new_filenames_test.append(filenames_test[i])
        new_mag_test.append(mag_test[i])
        new_cancerclass_test.append(cancerclass_test[i])
        new_cancertype_test.append(cancertype_test[i])
        index.append(i)
        
    if ((mag_test[i]==400) and (cancertype_test[i] in cancertype400) and (C40011<50 or C40012<50 or C40013<51 or C40014<50 or C40021<50 or C40022<50 or C40023<50 or C40024<50)):
        #C40=C40+1
        if (cancertype_test[i]==11):
            C40011=C40011+1
            
        elif (cancertype_test[i]==12):
            C40012=C40012+1
            
        elif (cancertype_test[i]==13):
            C40013=C40013+1
            
        elif (cancertype_test[i]==14):
            C40014=C40014+1
            
        elif (cancertype_test[i]==21):
            C40021=C40021+1
            
        elif (cancertype_test[i]==22):
            C40022=C40022+1
            
        elif (cancertype_test[i]==23):
            C40023=C40023+1
            
        elif (cancertype_test[i]==24):
            C40024=C40024+1
            
        new_data_test.append(data_test[i])
        new_filenames_test.append(filenames_test[i])
        new_mag_test.append(mag_test[i])
        new_cancerclass_test.append(cancerclass_test[i])
        new_cancertype_test.append(cancertype_test[i])
        index.append(i)
        
    if (C4011==50 and 11 in cancertype40):
        cancertype40.remove(11)
        
    if (C4012==50 and 12 in cancertype40):
        cancertype40.remove(12)
        
    if (C4013==50 and 13 in cancertype40): 
        cancertype40.remove(13)
        
    if (C4014==50 and 14 in cancertype40):
        cancertype40.remove(14)
        
    if (C4021==50 and 21 in cancertype40):
        cancertype40.remove(21)
        
    if (C4022==50 and 22 in cancertype40):
        cancertype40.remove(22)
        
    if (C4023==50 and 23 in cancertype40): 
        cancertype40.remove(23)
        
    if (C4024==50 and 24 in cancertype40):
        cancertype40.remove(24)
        
    
    if (C10011==50 and 11 in cancertype100):
        cancertype100.remove(11)
        
    if (C10012==50 and 12 in cancertype100):
        cancertype100.remove(12)
        
    if (C10013==50 and 13 in cancertype100): 
        cancertype100.remove(13)
        
    if (C10014==50 and 14 in cancertype100):
        cancertype100.remove(14)
        
    if (C10021==50 and 21 in cancertype100):
        cancertype100.remove(21)
        
    if (C10022==50 and 22 in cancertype100):
        cancertype100.remove(22)
        
    if (C10023==50 and 23 in cancertype100): 
        cancertype100.remove(23)
        
    if (C10024==50 and 24 in cancertype100):
        cancertype100.remove(24)
        

    if (C20011==50 and 11 in cancertype200):
        cancertype200.remove(11)
        
    if (C20012==50 and 12 in cancertype200):
        cancertype200.remove(12)
        
    if (C20013==50 and 13 in cancertype200): 
        cancertype200.remove(13)
        
    if (C20014==50 and 14 in cancertype200):
        cancertype200.remove(14)
        
    if (C20021==50 and 21 in cancertype200):
        cancertype200.remove(21)
        
    if (C20022==50 and 22 in cancertype200):
        cancertype200.remove(22)
        
    if (C20023==50 and 23 in cancertype200): 
        cancertype200.remove(23)
        
    if (C20024==50 and 24 in cancertype200):
        cancertype200.remove(24)
        
        
    if (C40011==50 and 11 in cancertype400):
        cancertype400.remove(11)
        
    if (C40012==50 and 12 in cancertype400):
        cancertype400.remove(12)
        
    if (C40013==51 and 13 in cancertype400): 
        cancertype400.remove(13)
        
    if (C40014==50 and 14 in cancertype400):
        cancertype400.remove(14)
        
    if (C40021==50 and 21 in cancertype400):
        cancertype400.remove(21)
        
    if (C40022==50 and 22 in cancertype400):
        cancertype400.remove(22)
        
    if (C40023==50 and 23 in cancertype400): 
        cancertype400.remove(23)
        
    if (C40024==50 and 24 in cancertype400):
        cancertype400.remove(24)
        
        


# In[6]:


stats.itemfreq(new_cancertype_test)


# In[7]:


stats.itemfreq(new_mag_test)


# In[ ]:





# In[8]:


print(len(new_data_test))
print(len(new_filenames_test))
print(len(new_mag_test))
print(len(new_cancerclass_test))
print(len(new_cancertype_test))


# In[9]:


print(len(index))


# In[10]:


data_test=np.delete(data_test,index,axis=0)
filenames_test=np.delete(filenames_test,index)
mag_test=np.delete(mag_test,index)
cancerclass_test=np.delete(cancerclass_test,index)
cancertype_test=np.delete(cancertype_test,index)


# In[11]:


print(len(data_test))
print(len(filenames_test))
print(len(mag_test))
print(len(cancerclass_test))
print(len(cancertype_test))


# In[12]:


#Saving Paths
save_train_path="A:\\Projects\\Major Project\\Extracted CNN Features\\InceptionResNetV2\\train"
save_test_path="A:\\Projects\\Major Project\\Extracted CNN Features\\InceptionResNetV2\\test"


# In[13]:


data_test=np.array(data_test)
filenames_test=np.array(filenames_test)
mag_test=np.array(mag_test)
cancerclass_test=np.array(cancerclass_test)
cancertype_test=np.array(cancertype_test)


# In[14]:


data_train=np.concatenate((data_train,data_test),axis=0)
filenames_train=np.concatenate((filenames_train,filenames_test))
mag_train=np.concatenate((mag_train,mag_test))
cancerclass_train=np.concatenate((cancerclass_train,cancerclass_test))
cancertype_train=np.concatenate((cancertype_train,cancertype_test))


# In[15]:


print(len(data_train))
print(len(filenames_train))
print(len(mag_train))
print(len(cancerclass_train))
print(len(cancertype_train))


# In[16]:


np.save(save_train_path+"\\data_cnn_InceptionResNetV2_train.npy",data_train)
np.save(save_train_path+"\\data_filenames_InceptionResNetV2_train.npy",filenames_train)
np.save(save_train_path+"\\data_mag_InceptionResNetV2_train.npy",mag_train)
np.save(save_train_path+"\\data_cancerclass_InceptionResNetV2_train.npy",cancerclass_train)
np.save(save_train_path+"\\data_cancertype_InceptionResNetV2_train.npy",cancertype_train)


# In[17]:


np.save(save_test_path+"\\data_cnn_InceptionResNetV2_test.npy",new_data_test)
np.save(save_test_path+"\\data_filenames_InceptionResNetV2_test.npy",new_filenames_test)
np.save(save_test_path+"\\data_mag_InceptionResNetV2_test.npy",new_mag_test)
np.save(save_test_path+"\\data_cancerclass_InceptionResNetV2_test.npy",new_cancerclass_test)
np.save(save_test_path+"\\data_cancertype_InceptionResNetV2_test.npy",new_cancertype_test)


# In[ ]:





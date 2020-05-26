#!/usr/bin/env python
# coding: utf-8

# # Library Import

# In[1]:


import os
import pandas as pd
import numpy as np
import pickle
import time
# Machine Learning Algorithms
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import validation_curve,learning_curve
from joblib import dump


# # Magnification Identification

# In[2]:


train_path="A:\\Projects\\Major Project\\Extracted CNN Features\\VGG19\\train"
test_path="A:\\Projects\\Major Project\\Extracted CNN Features\\VGG19\\test"


# In[3]:


# Training Paths
X_train=np.load(train_path+"\\data_cnn_VGG19_train.npy")
Y_train=np.load(train_path+"\\data_mag_VGG19_train.npy")
# Cancer class
cancerclass_train=np.load(train_path+"\\data_cancerclass_VGG19_train.npy")
# Cancer type
cancertype_train=np.load(train_path+"\\data_cancertype_VGG19_train.npy")
# Testing Paths
X_test=np.load(test_path+"\\data_cnn_VGG19_test.npy")
Y_test=np.load(test_path+"\\data_mag_VGG19_test.npy")
# Cancer class
cancerclass_test=np.load(test_path+"\\data_cancerclass_VGG19_test.npy")
# Cancer type
cancertype_test=np.load(test_path+"\\data_cancertype_VGG19_test.npy")


# In[4]:


param_grid={'C':[.001,.01,.1,1,10]}


# In[5]:


start_time=time.clock()
gs1=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs1.fit(X_train,Y_train)
print(time.clock() - start_time, "seconds")

print(gs1.best_score_)
print(gs1.best_params_)


# In[6]:


clf=gs1.best_estimator_
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))


# In[7]:


clf2=LogisticRegression(C=.001)
clf2.fit(X_train,Y_train)


# In[8]:


print(clf2.score(X_test,Y_test))


# In[9]:


pred=clf2.predict(X_test)


# In[10]:


con=confusion_matrix(Y_test,pred)


# In[11]:


print(con)


# In[12]:


precision_score(Y_test, pred, average='micro') 


# In[13]:


recall_score(Y_test, pred, average='micro') 


# In[14]:


f1_score(Y_test, pred, average='micro') 


# In[15]:


precision_recall_fscore_support(Y_test,pred)


# # CancerClass Identification

# In[7]:



Y_train_40=[]
X_train_40=[]

Y_train_100=[]
X_train_100=[]

Y_train_200=[]
X_train_200=[]

Y_train_400=[]
X_train_400=[]

for i in range(0,len(Y_train)):
    if(Y_train[i]==40):
        Y_train_40.append(cancerclass_train[i])
        X_train_40.append(X_train[i])
    if(Y_train[i]==100):
        Y_train_100.append(cancerclass_train[i])
        X_train_100.append(X_train[i])
    if(Y_train[i]==200):
        Y_train_200.append(cancerclass_train[i])
        X_train_200.append(X_train[i])
    if(Y_train[i]==400):
        Y_train_400.append(cancerclass_train[i])
        X_train_400.append(X_train[i])


# In[8]:


X_train_40=np.array(X_train_40)
X_train_100=np.array(X_train_100)
X_train_200=np.array(X_train_200)
X_train_400=np.array(X_train_400)
Y_train_40=np.array(Y_train_40)
Y_train_100=np.array(Y_train_100)
Y_train_200=np.array(Y_train_200)
Y_train_400=np.array(Y_train_400)
print(Y_train_40.size)


# In[9]:



Y_test_40=[]
X_test_40=[]

Y_test_100=[]
X_test_100=[]

Y_test_200=[]
X_test_200=[]

Y_test_400=[]
X_test_400=[]

for i in range(0,len(Y_test)):
    if(Y_test[i]==40):
        Y_test_40.append(cancerclass_test[i])
        X_test_40.append(X_test[i])
    if(Y_test[i]==100):
        Y_test_100.append(cancerclass_test[i])
        X_test_100.append(X_test[i])
    if(Y_test[i]==200):
        Y_test_200.append(cancerclass_test[i])
        X_test_200.append(X_test[i])
    if(Y_test[i]==400):
        Y_test_400.append(cancerclass_test[i])
        X_test_400.append(X_test[i])


# In[10]:


X_test_40=np.array(X_test_40)
X_test_100=np.array(X_test_100)
X_test_200=np.array(X_test_200)
X_test_400=np.array(X_test_400)
Y_test_40=np.array(Y_test_40)
Y_test_100=np.array(Y_test_100)
Y_test_200=np.array(Y_test_200)
Y_test_400=np.array(Y_test_400)


# # CancerClass Magnification classification 40

# In[20]:


param_grid={'C':[.001,.01,.1,1,10]}
gs1=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs1.fit(X_train_40,Y_train_40)
print(time.clock() - start_time, "seconds")

print(gs1.best_score_)
print(gs1.best_params_)


# In[21]:


clf3=gs1.best_estimator_
clf3.fit(X_train_40,Y_train_40)
clf3.score(X_test_40,Y_test_40)


# In[22]:


clf=LogisticRegression(C=.01)
clf.fit(X_train_40,Y_train_40)
clf.score(X_test_40,Y_test_40)


# In[23]:


pred=clf.predict(X_test_40)


# In[24]:


con=confusion_matrix(Y_test_40,pred)


# In[25]:


print(con)


# In[26]:


precision_score(Y_test_40,pred)


# In[27]:


recall_score(Y_test_40,pred)


# In[28]:


f1_score(Y_test_40,pred)


# In[29]:


precision_recall_fscore_support(Y_test_40,pred)


# # CancerClass Magnification classification 100

# In[30]:


gs2=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs2.fit(X_train_100,Y_train_100)
print(time.clock() - start_time, "seconds")

print(gs2.best_score_)
print(gs2.best_params_)


# In[31]:


c=LogisticRegression(C=.01)
c.fit(X_train_100,Y_train_100)
c.score(X_test_100,Y_test_100)


# # CancerClass Magnification classification 200

# In[32]:


gs3=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_200,Y_train_200)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[33]:


c=LogisticRegression(C=.001)
c.fit(X_train_200,Y_train_200)
c.score(X_test_200,Y_test_200)


# # CancerClass Magnification classification 400

# In[34]:


gs4=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs4.fit(X_train_400,Y_train_400)
print(time.clock() - start_time, "seconds")

print(gs4.best_score_)
print(gs4.best_params_)


# In[35]:


c=LogisticRegression(C=.001)
c.fit(X_train_400,Y_train_400)
c.score(X_test_400,Y_test_400)


# ## Benign Sub-Classification Using Cancer Classification

# In[24]:


Y_train_1=[]
X_train_1=[]

for i in range(0,len(Y_train)):
    if(cancerclass_train[i]==1):
        Y_train_1.append(cancertype_train[i])
        X_train_1.append(X_train[i])
    
X_train_1=np.array(X_train_1)
Y_train_1=np.array(Y_train_1)
print(Y_train_1.size)

Y_test_1=[]
X_test_1=[]

for i in range(0,len(Y_test)):
    if(cancerclass_test[i]==1):
        Y_test_1.append(cancertype_test[i])
        X_test_1.append(X_test[i])
    
X_test_1=np.array(X_test_1)
Y_test_1=np.array(Y_test_1)


# In[39]:


classes=[11,12,13,14]


# In[ ]:





# In[40]:


from sklearn.utils.class_weight import compute_class_weight


# In[41]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_1)


# In[42]:


print(class_weight) 


# In[43]:


print(np.unique(Y_train_1))


# In[44]:


print(len(X_train_1))


# In[45]:


print(len(Y_test_1))


# In[46]:


d = dict(enumerate(class_weight, 1))


# In[47]:


print(d)


# In[48]:


d1={1:11,2:12,3:13,4:14}


# In[49]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[50]:


d


# In[51]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_1,Y_train_1)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[52]:


clf4=gs3.best_estimator_
clf4.fit(X_train_1,Y_train_1)
print(clf4.score(X_test_1,Y_test_1))


# In[53]:


pred=clf4.predict(X_test_1)


# In[54]:


precision_recall_fscore_support(Y_test_1,pred)


# In[55]:


confusion_matrix(Y_test_1,pred)


# ## Malignant Sub-Classification Using Cancer Classification

# In[28]:


Y_train_2=[]
X_train_2=[]

for i in range(0,len(Y_train)):
    if(cancerclass_train[i]==2):
        Y_train_2.append(cancertype_train[i])
        X_train_2.append(X_train[i])
    
X_train_2=np.array(X_train_2)
Y_train_2=np.array(Y_train_2)
print(Y_train_2.size)

Y_test_2=[]
X_test_2=[]

for i in range(0,len(Y_test)):
    if(cancerclass_test[i]==2):
        Y_test_2.append(cancertype_test[i])
        X_test_2.append(X_test[i])
    
X_test_2=np.array(X_test_2)
Y_test_2=np.array(Y_test_2)


# In[69]:


classes=[21,22,23,24]


# In[70]:


from sklearn.utils.class_weight import compute_class_weight


# In[71]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_2)


# In[72]:


print(class_weight) 


# In[73]:


print(np.unique(Y_train_2))


# In[74]:


print(len(X_train_2))


# In[75]:


print(len(Y_test_2))


# In[76]:


d = dict(enumerate(class_weight, 1))


# In[77]:


print(d)


# In[78]:


d1={1:21,2:22,3:23,4:24}


# In[79]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[80]:


d


# In[81]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_2,Y_train_2)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[82]:


clf4=gs3.best_estimator_
clf4.fit(X_train_2,Y_train_2)
print(clf4.score(X_test_2,Y_test_2))


# In[83]:


pred=clf4.predict(X_test_2)


# In[84]:


precision_recall_fscore_support(Y_test_2,pred)


# In[85]:


confusion_matrix(Y_test_2,pred)


# # Dumping Models

# In[14]:


clf=LogisticRegression(C=.001)
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)


# In[15]:


dump(clf,'models/LR/LR_Models_VGG19_Magnification.joblib')


# In[ ]:





# In[16]:


clf=LogisticRegression(C=.001)
clf.fit(X_train_40,Y_train_40)
clf.score(X_test_40,Y_test_40)


# In[17]:


dump(clf,'models/LR/LR_Models_VGG19_Magnification_40.joblib')


# In[ ]:





# In[18]:


clf=LogisticRegression(C=.01)
clf.fit(X_train_100,Y_train_100)
clf.score(X_test_100,Y_test_100)


# In[19]:


dump(clf,'models/LR/LR_Models_VGG19_Magnification_100.joblib')


# In[ ]:





# In[ ]:





# In[32]:


clf=LogisticRegression(C=.001)
clf.fit(X_train_200,Y_train_200)
clf.score(X_test_200,Y_test_200)


# In[33]:


dump(clf,'models/LR/LR_Models_VGG19_Magnification_200.joblib')


# In[22]:


clf=LogisticRegression(C=.001)
clf.fit(X_train_400,Y_train_400)
clf.score(X_test_400,Y_test_400)


# In[23]:


dump(clf,'models/LR/LR_Models_VGG19_Magnification_400.joblib')


# In[ ]:





# In[25]:


clf=LogisticRegression(C=.01)
clf.fit(X_train_1,Y_train_1)
clf.score(X_test_1,Y_test_1)


# In[26]:


dump(clf,'models/LR/LR_Models_VGG19_CancerType_Benign.joblib')


# In[ ]:





# In[29]:


clf=LogisticRegression(C=.01)
clf.fit(X_train_2,Y_train_2)
clf.score(X_test_2,Y_test_2)


# In[31]:


dump(clf,'models/LR/LR_Models_VGG19_CancerType_Malignant.joblib')


# In[ ]:





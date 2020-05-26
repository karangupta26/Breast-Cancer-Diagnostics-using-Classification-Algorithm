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


train_path="A:\\Projects\\Major Project\\Extracted CNN Features\\ResNet50\\train"
test_path="A:\\Projects\\Major Project\\Extracted CNN Features\\ResNet50\\test"


# In[3]:


# Training Paths
X_train=np.load(train_path+"\\data_cnn_ResNet50_train.npy")
Y_train=np.load(train_path+"\\data_mag_ResNet50_train.npy")
# Cancer class
cancerclass_train=np.load(train_path+"\\data_cancerclass_ResNet50_train.npy")
# Cancer type
cancertype_train=np.load(train_path+"\\data_cancertype_ResNet50_train.npy")
# Testing Paths
X_test=np.load(test_path+"\\data_cnn_ResNet50_test.npy")
Y_test=np.load(test_path+"\\data_mag_ResNet50_test.npy")
# Cancer class
cancerclass_test=np.load(test_path+"\\data_cancerclass_ResNet50_test.npy")
# Cancer type
cancertype_test=np.load(test_path+"\\data_cancertype_ResNet50_test.npy")


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


dump(clf,'models/LR/LR_Models_ResNet50_Magnification.joblib')


# In[8]:


clf2=LogisticRegression(C=.001)
clf2.fit(X_train,Y_train)


# In[9]:


print(clf2.score(X_test,Y_test))


# In[10]:


pred=clf2.predict(X_test)


# In[11]:


con=confusion_matrix(Y_test,pred)


# In[12]:


print(con)


# In[13]:


precision_score(Y_test, pred, average='micro') 


# In[14]:


recall_score(Y_test, pred, average='micro') 


# In[15]:


f1_score(Y_test, pred, average='micro') 


# In[16]:


precision_recall_fscore_support(Y_test,pred)


# # CancerClass Identification

# In[17]:



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


# In[18]:


X_train_40=np.array(X_train_40)
X_train_100=np.array(X_train_100)
X_train_200=np.array(X_train_200)
X_train_400=np.array(X_train_400)
Y_train_40=np.array(Y_train_40)
Y_train_100=np.array(Y_train_100)
Y_train_200=np.array(Y_train_200)
Y_train_400=np.array(Y_train_400)
print(Y_train_40.size)


# In[19]:



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


# In[20]:


X_test_40=np.array(X_test_40)
X_test_100=np.array(X_test_100)
X_test_200=np.array(X_test_200)
X_test_400=np.array(X_test_400)
Y_test_40=np.array(Y_test_40)
Y_test_100=np.array(Y_test_100)
Y_test_200=np.array(Y_test_200)
Y_test_400=np.array(Y_test_400)


# # CancerClass Magnification classification 40

# In[21]:


param_grid={'C':[.001,.01,.1,1,10]}
gs1=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs1.fit(X_train_40,Y_train_40)
print(time.clock() - start_time, "seconds")

print(gs1.best_score_)
print(gs1.best_params_)


# In[22]:


clf3=gs1.best_estimator_
clf3.fit(X_train_40,Y_train_40)
clf3.score(X_test_40,Y_test_40)


# In[23]:


dump(clf3,'models/LR/LR_Models_ResNet50_Magnification_40.joblib')


# In[24]:


clf=LogisticRegression(C=.01)
clf.fit(X_train_40,Y_train_40)
clf.score(X_test_40,Y_test_40)


# In[25]:


pred=clf.predict(X_test_40)


# In[26]:


con=confusion_matrix(Y_test_40,pred)


# In[27]:


print(con)


# In[28]:


precision_score(Y_test_40,pred)


# In[29]:


recall_score(Y_test_40,pred)


# In[30]:


f1_score(Y_test_40,pred)


# In[31]:


precision_recall_fscore_support(Y_test_40,pred)


# # CancerClass Magnification classification 100

# In[32]:


gs2=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs2.fit(X_train_100,Y_train_100)
print(time.clock() - start_time, "seconds")

print(gs2.best_score_)
print(gs2.best_params_)


# In[33]:


c=LogisticRegression(C=.01)
c.fit(X_train_100,Y_train_100)
c.score(X_test_100,Y_test_100)


# In[34]:


dump(c,'models/LR/LR_Models_ResNet50_Magnification_100.joblib')


# # CancerClass Magnification classification 200

# In[80]:


gs3=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_200,Y_train_200)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[81]:


c=gs3.best_estimator_
c.fit(X_train_200,Y_train_200)
c.score(X_test_200,Y_test_200)


# In[82]:


dump(c,'models/LR/LR_Models_ResNet50_Magnification_200.joblib')


# # CancerClass Magnification classification 400

# In[38]:


gs4=GridSearchCV(LogisticRegression(),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs4.fit(X_train_400,Y_train_400)
print(time.clock() - start_time, "seconds")

print(gs4.best_score_)
print(gs4.best_params_)


# In[39]:


c=LogisticRegression(C=.001)
c.fit(X_train_400,Y_train_400)
c.score(X_test_400,Y_test_400)


# In[40]:


dump(c,'models/LR/LR_Models_ResNet50_Magnification_400.joblib')


# ## Benign Sub-Classification Using Cancer Classification

# In[41]:


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


# In[42]:


classes=[11,12,13,14]


# In[ ]:





# In[43]:


from sklearn.utils.class_weight import compute_class_weight


# In[44]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_1)


# In[45]:


print(class_weight) 


# In[46]:


print(np.unique(Y_train_1))


# In[47]:


print(len(X_train_1))


# In[48]:


print(len(Y_test_1))


# In[49]:


d = dict(enumerate(class_weight, 1))


# In[50]:


print(d)


# In[51]:


d1={1:11,2:12,3:13,4:14}


# In[52]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[53]:


d


# In[54]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_1,Y_train_1)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[55]:


clf4=gs3.best_estimator_
clf4.fit(X_train_1,Y_train_1)
print(clf4.score(X_test_1,Y_test_1))


# In[56]:


dump(clf4,'models/LR/LR_Models_ResNet50_CancerType_Benign.joblib')


# In[57]:


pred=clf4.predict(X_test_1)


# In[58]:


precision_recall_fscore_support(Y_test_1,pred)


# In[59]:


confusion_matrix(Y_test_1,pred)


# ## Malignant Sub-Classification Using Cancer Classification

# In[60]:


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


# In[61]:


classes=[21,22,23,24]


# In[62]:


from sklearn.utils.class_weight import compute_class_weight


# In[63]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_2)


# In[64]:


print(class_weight) 


# In[65]:


print(np.unique(Y_train_2))


# In[66]:


print(len(X_train_2))


# In[67]:


print(len(Y_test_2))


# In[68]:


d = dict(enumerate(class_weight, 1))


# In[69]:


print(d)


# In[70]:


d1={1:21,2:22,3:23,4:24}


# In[71]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[72]:


d


# In[73]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_2,Y_train_2)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[74]:


clf4=gs3.best_estimator_
clf4.fit(X_train_2,Y_train_2)
print(clf4.score(X_test_2,Y_test_2))


# In[75]:


dump(clf4,'models/LR/LR_Models_ResNet50_CancerType_Malignant.joblib')


# In[76]:


pred=clf4.predict(X_test_2)


# In[77]:


precision_recall_fscore_support(Y_test_2,pred)


# In[78]:


confusion_matrix(Y_test_2,pred)


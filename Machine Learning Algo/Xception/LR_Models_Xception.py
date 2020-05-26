#!/usr/bin/env python
# coding: utf-8

# # Library Import

# In[6]:


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
#Dumping Model Library
from joblib import dump,load


# # Magnification Identification

# In[2]:


train_path="A:\\Projects\\Major Project\\Extracted CNN Features\\Xception\\train"
test_path="A:\\Projects\\Major Project\\Extracted CNN Features\\Xception\\test"


# In[3]:


# Training Paths
X_train=np.load(train_path+"\\data_cnn_Xception_train.npy")
Y_train=np.load(train_path+"\\data_mag_Xception_train.npy")
# Cancer class
cancerclass_train=np.load(train_path+"\\data_cancerclass_Xception_train.npy")
# Cancer type
cancertype_train=np.load(train_path+"\\data_cancertype_Xception_train.npy")
# Testing Paths
X_test=np.load(test_path+"\\data_cnn_Xception_test.npy")
Y_test=np.load(test_path+"\\data_mag_Xception_test.npy")
# Cancer class
cancerclass_test=np.load(test_path+"\\data_cancerclass_Xception_test.npy")
# Cancer type
cancertype_test=np.load(test_path+"\\data_cancertype_Xception_test.npy")


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

# In[9]:



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


# In[10]:


X_train_40=np.array(X_train_40)
X_train_100=np.array(X_train_100)
X_train_200=np.array(X_train_200)
X_train_400=np.array(X_train_400)
Y_train_40=np.array(Y_train_40)
Y_train_100=np.array(Y_train_100)
Y_train_200=np.array(Y_train_200)
Y_train_400=np.array(Y_train_400)
print(Y_train_40.size)


# In[11]:



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


# In[12]:


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

# In[20]:


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


# In[37]:


classes=[11,12,13,14]


# In[ ]:





# In[38]:


from sklearn.utils.class_weight import compute_class_weight


# In[39]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_1)


# In[40]:


print(class_weight) 


# In[41]:


print(np.unique(Y_train_1))


# In[42]:


print(len(X_train_1))


# In[43]:


print(len(Y_test_1))


# In[44]:


d = dict(enumerate(class_weight, 1))


# In[45]:


print(d)


# In[46]:


d1={1:11,2:12,3:13,4:14}


# In[47]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[48]:


d


# In[49]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_1,Y_train_1)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[50]:


clf4=gs3.best_estimator_
clf4.fit(X_train_1,Y_train_1)
print(clf4.score(X_test_1,Y_test_1))


# In[51]:


pred=clf4.predict(X_test_1)


# In[52]:


precision_recall_fscore_support(Y_test_1,pred)


# In[53]:


confusion_matrix(Y_test_1,pred)


# ## Malignant Sub-Classification Using Cancer Classification

# In[21]:


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


# In[55]:


classes=[21,22,23,24]


# In[56]:


from sklearn.utils.class_weight import compute_class_weight


# In[57]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_2)


# In[58]:


print(class_weight) 


# In[59]:


print(np.unique(Y_train_2))


# In[60]:


print(len(X_train_2))


# In[61]:


print(len(Y_test_2))


# In[62]:


d = dict(enumerate(class_weight, 1))


# In[63]:


print(d)


# In[64]:


d1={1:21,2:22,3:23,4:24}


# In[65]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[66]:


d


# In[67]:


gs3=GridSearchCV(LogisticRegression(class_weight=d),param_grid=param_grid,scoring="accuracy",cv=10,n_jobs=-1)

start_time = time.clock()
#Training of Model
gs3.fit(X_train_2,Y_train_2)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[68]:


clf4=gs3.best_estimator_
clf4.fit(X_train_2,Y_train_2)
print(clf4.score(X_test_2,Y_test_2))


# In[69]:


pred=clf4.predict(X_test_2)


# In[70]:


precision_recall_fscore_support(Y_test_2,pred)


# In[71]:


confusion_matrix(Y_test_2,pred)


# # Dumping Models

# In[5]:


clf=LogisticRegression(C=0.1)
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)


# In[7]:


dump(clf,'models/LR/LR_Models_Xception_Magnification.joblib')


# In[ ]:





# In[13]:


clf=LogisticRegression(C=1)
clf.fit(X_train_40,Y_train_40)
clf.score(X_test_40,Y_test_40)


# In[14]:


dump(clf,'models/LR/LR_Models_Xception_Magnification_40.joblib')


# In[ ]:





# In[15]:


clf=LogisticRegression(C=0.1)
clf.fit(X_train_100,Y_train_100)
clf.score(X_test_100,Y_test_100)


# In[16]:


dump(clf,'models/LR/LR_Models_Xception_Magnification_100.joblib')


# In[ ]:





# In[27]:


clf=LogisticRegression(C=0.1)
clf.fit(X_train_200,Y_train_200)
clf.score(X_test_200,Y_test_200)


# In[28]:


dump(clf,'models/LR/LR_Models_Xception_Magnification_200.joblib')


# In[ ]:





# In[18]:


clf=LogisticRegression(C=10)
clf.fit(X_train_400,Y_train_400)
clf.score(X_test_400,Y_test_400)


# In[19]:


dump(clf,'models/LR/LR_Models_Xception_Magnification_400.joblib')


# In[ ]:





# In[25]:


clf=LogisticRegression(C=1)
clf.fit(X_train_1,Y_train_1)
clf.score(X_test_1,Y_test_1)


# In[26]:


dump(clf,'models/LR/LR_Models_Xception_CancerType_Benign.joblib')


# In[ ]:





# In[22]:


clf=LogisticRegression(C=1)
clf.fit(X_train_2,Y_train_2)
clf.score(X_test_2,Y_test_2)


# In[24]:


dump(clf,'models/LR/LR_Models_Xception_CancerType_Malignant.joblib')


# In[ ]:





# In[ ]:





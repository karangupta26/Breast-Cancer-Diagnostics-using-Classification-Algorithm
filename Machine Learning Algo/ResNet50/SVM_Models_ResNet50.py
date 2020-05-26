#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import os
import pandas as pd
import numpy as np
import pickle
import time
# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import validation_curve,learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
from joblib import dump,load


# # Loading Paths

# In[2]:


train_path="A:\\Projects\\Major Project\\Extracted CNN Features\\ResNet50\\train"
test_path="A:\\Projects\\Major Project\\Extracted CNN Features\\ResNet50\\test"


# In[3]:


param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': ['auto'], 
               'svc__kernel': ['rbf']}]
pipe_svc = make_pipeline(SVC(random_state=1))


# In[5]:


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


# ## Magnification classification

# In[5]:


start_time = time.clock()


param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': ['auto'], 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, Y_train)
print(gs.best_score_)
print(gs.best_params_)
print(time.clock() - start_time, "seconds")



clf = gs.best_estimator_
clf.fit(X_train, Y_train)
print('Test accuracy: %.3f' % clf.score(X_test, Y_test))


# In[6]:


clf = gs.best_estimator_
clf.fit(X_train, Y_train)
print('Test accuracy: %.3f' % clf.score(X_test, Y_test))
print(time.clock() - start_time, "seconds")


# In[7]:


dump(clf,'models/SVM/SVM_Models_ResNet50_Magnification.joblib')


# In[8]:


pred=clf.predict(X_test)


# In[9]:


con=confusion_matrix(Y_test,pred)


# In[10]:


print(con)


# In[11]:


precision_recall_fscore_support(Y_test,pred)


# ## CancerClass Magnification Classification

# In[6]:


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

X_train_40=np.array(X_train_40)
X_train_100=np.array(X_train_100)
X_train_200=np.array(X_train_200)
X_train_400=np.array(X_train_400)
Y_train_40=np.array(Y_train_40)
Y_train_100=np.array(Y_train_100)
Y_train_200=np.array(Y_train_200)
Y_train_400=np.array(Y_train_400)
print(Y_train_40.size)

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

X_test_40=np.array(X_test_40)
X_test_100=np.array(X_test_100)
X_test_200=np.array(X_test_200)
X_test_400=np.array(X_test_400)
Y_test_40=np.array(Y_test_40)
Y_test_100=np.array(Y_test_100)
Y_test_200=np.array(Y_test_200)
Y_test_400=np.array(Y_test_400)


# ## CancerClass Magnification Classification-40

# In[13]:


start_time = time.clock()
pipe_svc = make_pipeline(SVC(random_state=1))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': ['auto'], 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train_40, Y_train_40)
print(gs.best_score_)
print(gs.best_params_)
print(time.clock() - start_time, "seconds")


clf = gs.best_estimator_
clf.fit(X_train_40, Y_train_40)
print('Test accuracy: %.3f' % clf.score(X_test_40, Y_test_40))


# In[14]:


dump(clf,'models/SVM/SVM_Models_ResNet50_Magnification_40.joblib')


# ## CancerClass Magnification Classification-100

# In[15]:


start_time = time.clock()
pipe_svc = make_pipeline(SVC(random_state=1))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': ['auto'], 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train_100, Y_train_100)
print(gs.best_score_)
print(gs.best_params_)
print(time.clock() - start_time, "seconds")


clf = gs.best_estimator_
clf.fit(X_train_100, Y_train_100)
print('Test accuracy: %.3f' % clf.score(X_test_100, Y_test_100))


# In[16]:


print(gs.best_estimator_)


# In[17]:


dump(clf,'models/SVM/SVM_Models_ResNet50_Magnification_100.joblib')


# ## CancerClass Magnification Classification-200

# In[18]:


tart_time = time.clock()
pipe_svc = make_pipeline(SVC(random_state=1))

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': ['auto'], 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train_200, Y_train_200)
print(gs.best_score_)
print(gs.best_params_)
print(time.clock() - start_time, "seconds")


clf = gs.best_estimator_
clf.fit(X_train_200, Y_train_200)
print('Test accuracy: %.3f' % clf.score(X_test_200, Y_test_200))


# In[19]:


dump(clf,'models/SVM/SVM_Models_ResNet50_Magnification_200.joblib')


# ## CancerClass Magnification Classification-400

# In[8]:


start_time = time.clock()
pipe_svc = make_pipeline(SVC(random_state=1))



gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train_400, Y_train_400)
print(gs.best_score_)
print(gs.best_params_)
print(time.clock() - start_time, "seconds")


clf = gs.best_estimator_
clf.fit(X_train_400, Y_train_400)
print('Test accuracy: %.3f' % clf.score(X_test_400, Y_test_400))


# In[ ]:


dump(clf,'models/SVM/SVM_Models_ResNet50_Magnification_400.joblib')


# ## Benign Sub-Classification Using Cancer Classification

# In[22]:


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


# In[ ]:





# In[23]:


classes=[11,12,13,14]


# In[ ]:





# In[24]:


from sklearn.utils.class_weight import compute_class_weight


# In[25]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_1)


# In[26]:


print(class_weight) 


# In[27]:


print(np.unique(Y_train_1))


# In[28]:


print(len(X_train_1))


# In[29]:


print(len(Y_test_1))


# In[30]:


d = dict(enumerate(class_weight, 1))


# In[31]:


print(d)


# In[32]:


d1={1:11,2:12,3:13,4:14}


# In[33]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[34]:


d


# In[ ]:





# In[ ]:





# In[35]:


pipe_svc = make_pipeline(SVC(random_state=1,class_weight=d))
gs3=GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
start_time = time.clock()
#Training of Model
gs3.fit(X_train_1,Y_train_1)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[36]:


clf4=gs3.best_estimator_
clf4.fit(X_train_1,Y_train_1)
print(clf4.score(X_test_1,Y_test_1))


# In[37]:


dump(clf4,'models/SVM/SVM_Models_ResNet50_CancerType_Benign.joblib')


# In[ ]:





# ## Malignant Sub-Classification Using Cancer Classification

# In[38]:


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


# In[39]:


classes=[21,22,23,24]


# In[40]:


from sklearn.utils.class_weight import compute_class_weight


# In[41]:


class_weight=compute_class_weight(class_weight='balanced', classes=classes,y=Y_train_2)


# In[42]:


print(class_weight) 


# In[43]:


print(np.unique(Y_train_2))


# In[44]:


print(len(X_train_2))


# In[45]:


print(len(Y_test_2))


# In[46]:


d = dict(enumerate(class_weight, 1))


# In[47]:


print(d)


# In[48]:


d1={1:21,2:22,3:23,4:24}


# In[49]:


d=dict((d1[key], value) for (key, value) in d.items())


# In[50]:


d


# In[51]:


pipe_svc = make_pipeline(SVC(random_state=1,class_weight=d))
gs3=GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  cv=10,
                  n_jobs=-1)
start_time = time.clock()
#Training of Model
gs3.fit(X_train_2,Y_train_2)
print(time.clock() - start_time, "seconds")

print(gs3.best_score_)
print(gs3.best_params_)


# In[52]:


clf4=gs3.best_estimator_
clf4.fit(X_train_2,Y_train_2)
print(clf4.score(X_test_2,Y_test_2))


# In[53]:


pred=clf4.predict(X_test_2)


# In[54]:


precision_recall_fscore_support(Y_test_2,pred)


# In[55]:


confusion_matrix(Y_test_2,pred)


# In[56]:


dump(clf4,'models/SVM/SVM_models_ResNet50_CancerType_Malignant.joblib')


# In[ ]:





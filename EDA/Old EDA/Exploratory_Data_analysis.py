#!/usr/bin/env python
# coding: utf-8

# # <center>Exploratory Data Analysis-BreakHis Dataset</center>
# #### <center>February 23,2019</center>

# # <b>[1] About Data</b>
# 
# <p>The dataset which we are going to use is BreakHis dataset caontainin 7909 histopathical breast cancer sample images from 82 patients respectively.</p>
# <b> REPRESENTATION OF DATASET IN PROJECT IS AS FOLLOWS-</b>
# <ol>
#     <dl><b><i>1. Cancer Class</i></b>
#               <dt>1.1. Benign</dt>
#               <dd>This Class is represented by Integer-1</dd>
#               <dt>1.2. Malignant</dt>
#               <dd>This Class is represented by Integer-2</dd>
#     </dl>
#     <dl><i><b>2. Cancer Type</b></i>
#               <dt>2.1 Benign-A</dt>
#               <dd>Benign-A represents Adenosis.This Class is represented by Integer-11</dd>
#               <dt>2.2 Benign-FA</dt>
#               <dd>Benign-FA represents Fibro Adenoma.This Class is represented by Integer-12</dd>
#               <dt>2.3 Benign-TA</dt>
#               <dd>Benign-TA represents Tubulor Adenoma.This Class is represented by Integer-13</dd>
#               <dt>2.4 Benign-PT</dt>
#               <dd>Benign-PT represents Phyllodes Tumor.This Class is represented by Integer-14</dd>
#               <dt>2.5. Malignant-DC</dt>
#               <dd>Malignant-DC represents Ductol Carinoma.This Class is represented by Integer-21</dd>
#               <dt>2.6. Malignant-LC</dt>
#               <dd>Malignant-LC represents Lobular Carinoma.This Class is represented by Integer-22</dd>
#               <dt>2.7. Malignant-MC</dt>
#               <dd>Malignant-Mc represents Mucious Carinoma.This Class is represented by Integer-23</dd>
#               <dt>2.8. Malignant-PC</dt>
#               <dd>Malignant-PC represents Pappillary Carinoma.This Class is represented by Integer-24</dd>
#     </dl>
#     <dl><b><i>3. Magnification</i></b>
#               <dt>3.1. 40X  - 40</dt>
#               <dt>3.2. 100X - 100</dt>
#               <dt>3.3. 200X - 200</dt>
#               <dt>3.4. 400X - 400</dt>
#     </dl>
# </ol>
# 
# <b>Note -</b><p>After Each visualization some counts are represented for elaborations of plots which are used for distribution.</p>
# 

# # <b>Pre-Exploratory Data Analysis<b>

# <b>Import Library</b>

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
sb.set(style="darkgrid")
import matplotlib.pyplot as plt


# <b>Loading Numpy Array</b>

# In[2]:


# Train Arrays
data_cancerclass_train=np.load("train/data_cancerclass_train.npy")
data_cancertype_train=np.load("train/data_cancertype_train.npy")
data_mag_train=np.load("train/data_mag_train.npy")
# Test Arrays
data_cancerclass_test=np.load("test/data_cancerclass_test.npy")
data_cancertype_test=np.load("test/data_cancertype_test.npy")
data_mag_test=np.load("test/data_mag_test.npy")


# # <b>[2] Train Arrays Visualization</b>

# In[3]:


train_df=pd.DataFrame({'Cancer Class':data_cancerclass_train,
                      'Cancer Type':data_cancertype_train,
                      'Magnification':data_mag_train})


# ### <b>[2.1] Cancer Class</b>

# In[4]:


ax = sb.countplot(x="Cancer Class", data=train_df)
fig=ax.get_figure()
fig.savefig("Train Cancer Class.png")
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate(y,(x.mean(), y),ha='center', va='bottom')


# In[5]:


print(train_df.groupby("Cancer Class").count())


# ### <b>[2.3] Cancer Type</b>

# In[6]:


ax = sb.catplot(x="Cancer Class",hue="Cancer Type", data=train_df,kind="count")
ax.savefig("Train Cancer Class with cancer type.png")


# In[7]:


print(train_df.groupby("Cancer Type").count())


# ### <b>[2.3] Magnification</b>

# In[8]:


ax=sb.countplot(y="Magnification", data=train_df)
fig=ax.get_figure()
fig.savefig("Train Magnification.png")


# In[9]:


print(train_df.groupby("Magnification").count())


# ### <b>[2.4] Cancer Class Data Distribution</b>

# In[10]:


ax=sb.countplot(x="Magnification",hue="Cancer Class", data=train_df)
fig=ax.get_figure()
fig.savefig("Train Magnification in Train Numpy.png")


# In[11]:


print(train_df.groupby(["Cancer Class","Magnification"]).count())


# ### <b>[2.5] Train Data Distribution</b>

# In[12]:


ax= sb.catplot(x="Magnification", hue="Cancer Type", col="Cancer Class",data=train_df, kind="count")
ax.savefig("Train Cancer Type with Magnification using Cancer Class.png")


# In[13]:


print(train_df.groupby(["Cancer Type","Magnification"]).count())


# # <b>[3] Test Arrays Visualization</b>

# In[14]:


test_df=pd.DataFrame({'Cancer Class':data_cancerclass_test,
                      'Cancer Type':data_cancertype_test,
                      'Magnification':data_mag_test})


# ### <b>[3.1] Cancer Class</b>

# In[15]:


ax = sb.countplot(x="Cancer Class", data=test_df)
fig=ax.get_figure()
fig.savefig("Test Cancer Class.png")
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate(y,(x.mean(), y),ha='center', va='bottom')


# In[16]:


print(test_df.groupby("Cancer Class").count())


# ### <b>[3.2] Cancer Type</b>

# In[17]:


ax = sb.catplot(x="Cancer Class",hue="Cancer Type", data=test_df,kind="count")
ax.savefig("Test Cancer Class with cancer type.png")


# In[18]:


print(test_df.groupby("Cancer Type").count())


# ### <b>[3.3] Cancer Class Maginification</b>

# In[19]:


ax=sb.countplot(y="Magnification", data=test_df)
fig=ax.get_figure()
fig.savefig("Test Magnification.png")


# In[20]:


print(test_df.groupby("Magnification").count())


# ### <b>[3.4] Cancer Class Data Distribution</b>

# In[21]:


ax=sb.countplot(x="Magnification",hue="Cancer Class", data=test_df)
fig=ax.get_figure()
fig.savefig("Test Magnification in Test Numpy.png")


# In[22]:


print(test_df.groupby(["Cancer Class","Magnification"]).count())


# ### <b>[3.5] Test Data Distribution</b>

# In[23]:


ax= sb.catplot(x="Magnification", hue="Cancer Type", col="Cancer Class",
                data=test_df, kind="count");
ax.savefig("Test Cancer Type with Magnification using Cancer Class.png")


# In[24]:


print(test_df.groupby(["Cancer Type","Magnification"]).count())


# # <b>Post-Exploratory Data Analysis<b>

# <p>After the dataset is retrived, it was passed through some Deep-Learning Algorithms for feature Extraction.The Algorithms are known as Deep Convolution Neural Networks.</p>
# <dl><b><i>The Used CNN's are as follows</i></b>
#               <dt>1. VGG16</dt>
#               <dt>2. VGG19</dt>
#               <dt>3. Xception</dt>
#               <dt>4. ResNet50</dt>
#               <dt>5. InceptionV3</dt>
#               <dt>6. InceptionResNetV2</dt>
#     </dl>
# The Dataset was distributed as 5000 Train Samples,2900 Test Samples and Randomly 9 Images were removed for the checking of model.

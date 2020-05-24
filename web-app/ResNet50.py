#!/usr/bin/env python
# coding: utf-8

# In[1]:

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras import backend as k
import sys
sys.modules['Image'] = image
import numpy as np
import time

def resnet50_extractor(img):
    weights='imagenet'
    start=time.clock()
    base_model = ResNet50(weights=weights,include_top=True)
    model = Model(inputs=base_model.input, output=base_model.get_layer('avg_pool').output)
    image_size = (224, 224)
    img = image.load_img(img, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    print(time.clock()-start)
    k.clear_session()
    return flat

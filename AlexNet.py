#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd

import numpy as np
from tensorflow import keras
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D,BatchNormalization


# In[44]:


import os

import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = 'E:/'

class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, base_dir, output_size, shuffle=False, batch_size=10):
        self.base_dir = base_dir
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.df = None  # Add the dataset dataframe initialization here
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        X = np.empty((self.batch_size, *self.output_size, 3))  # Change the channel dimension to 3 for RGB images
        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

        for i, data_index in enumerate(indices):
            img_path = os.path.join(self.base_dir, self.df.iloc[data_index, 0])
            img = mpimg.imread(img_path)
            img = cv2.resize(img, self.output_size)
            X[i, ] = img

        return X


# In[45]:


train_directory='E:/OCT2017/train'
test_directory = 'E:/OCT2017/test'

from PIL import Image
import os, sys

path = ('E:\OCT2017\test')

def resize():
    for item in os.listdir(path):
        if os.path.isfile(item):
            im = Image.open(item)
            f, e = os.path.splitext(item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

    resize()
    


path = ('E:/OCT2017/train')

def resize():
    for item in os.listdir(path):
        if os.path.isfile(item):
            im = Image.open(item)
            f, e = os.path.splitext(item)
            imResize = im.resize((224,224), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

    resize()



# In[46]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,                   
                                   validation_split=0.2)


train_generator = train_datagen.flow_from_directory(
train_directory, 
target_size=(224,224), 
color_mode='rgb', 
batch_size=32, 
class_mode='categorical', 
subset='training',
shuffle=True,
seed=42
)
validation_generator = train_datagen.flow_from_directory(
train_directory,
target_size=(224,224), 
color_mode='rgb', 
batch_size=32, 
class_mode='categorical', 
subset='validation', 
shuffle=False
)
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
test_directory,
target_size=(224,224), 
color_mode='rgb', 
batch_size=32, 
class_mode='categorical', 
shuffle=False
)


# In[47]:


# (3) Create a sequential model
model = Sequential()
#model.add(layers.Flatten(input_shape=(224,224,3)))

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11),\
 strides=(4,4), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# Passing it to a dense layer
model.add(Flatten())

# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())


# Output Layer
model.add(Dense(4))
model.add(Activation('softmax'))


# In[48]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005

optimizer = Adam(
    learning_rate=learning_rate,
    beta_1=momentum
)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'],
)


# In[49]:


model.fit(train_generator,
steps_per_epoch=10,
epochs=400,
validation_data=validation_generator,
validation_steps=10,
)


# In[50]:


model.evaluate(test_generator)


# In[1]:





# In[ ]:





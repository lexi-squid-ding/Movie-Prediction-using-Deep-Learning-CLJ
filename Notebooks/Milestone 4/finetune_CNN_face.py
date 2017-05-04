# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 18:04:46 2017

@author: Cynthia9109
"""
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras import backend as K
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input

import h5py

#load libraries needed by CNN
from vgg16 import VGG16
import numpy as np
import argparse
import cv2
import sys

#import general libraries
import numpy as np
import pandas as pd
import pickle
from random import shuffle
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")   

import matplotlib.pyplot as plt

f = open('output.txt', 'w')
sys.stdout = f

#load movie id from 2005-2010
data = pd.read_csv("final_poster_label_2005_2010.csv")
#store movie ID
movieID = data["id"].values

###### read each image and resize 

# total number of posters
n = len(movieID)

# target image size
target_length = 224
target_width = 224

# np.array to store the whole matrix in the order of the movie id
img_matrix = np.zeros([n, target_length, target_width, 3])

for i in range(n):
    if i % 1000 == 0:    
        print (i)    
    
    movie_id = movieID[i]
    
    path = 'posters_2005_2010/' + str(movie_id) + '.jpg'
    
    #read image
    img = cv2.imread(path)
    
    # normalize the image[-1,1], center at 0
    # so that the zero padding does not affect the image
    im = img.astype('float32')
    img = (img - 255/2)/255
    
    # extract the length and width of the image
    img_length, img_width = np.asarray(img.shape[:2])

    
    ## put the image to the center of the array
    # calcuate location of the left edge
    left_loc = int(0.5*(target_width - img_width))
    # calcuate location of the top edge
    top_loc = int(0.5*(target_length - img_length))
    
    # store the image
    img_matrix[i, top_loc:(top_loc+img_length), 
               left_loc:(left_loc+img_width), :] = img
    
print ("Final data shape:")
img_matrix.shape

# reformat the data shape
if K.image_data_format() == 'channels_first':
    img_matrix = img_matrix.reshape(n, 3, target_length, target_width)
    input_shape = (3, target_length, target_width)
else:
    img_matrix = img_matrix.reshape(n, target_length, target_width, 3)
    input_shape = (target_length, target_width, 3)

print('data shape:', img_matrix.shape)

# y values: title position
y_face = data["face_count"].values

#remove nan from both x and y
index = np.where(np.isnan(y_face) == False)
y_face = y_face[index]
print (len(y_face))

img_matrix = img_matrix[index]
print (img_matrix.shape)

### split train and test
# shuffle the samples
n = len(y_face)
index = range(n)
shuffle(index)

y_face = y_face[index]
img_matrix = img_matrix[index]

# one-hot-encode y
encoder = LabelEncoder()
encoder.fit(y_face)
y_encoded = encoder.transform(y_face)

# split train and test data
ratio = 0.9
split_num = int(n*ratio)

x_train = img_matrix[:split_num]
x_test = img_matrix[split_num:]

y_train = y_encoded[:split_num]
y_test = y_encoded[split_num:]

# smaller batch size means noisier gradient, but more updates per epoch
batch_size = 64
# this is fixed, we have 10 digits in our data set
num_classes = 11
# number of iterations over the complete training data
epochs = 20

# convert y values to the format for Keras
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#load pretrained model - VGG16
base_model =VGG16(weights='imagenet', include_top=False)

x = base_model.output 
x = GlobalAveragePooling2D()(x) # let's add a fully-connected layer 
x = Dropout(0.3)(x)
x = Dense(1024, activation='relu')(x) # and a logistic layer -- let's say we have 200 classes 
predictions = Dense(num_classes, activation='softmax')(x)  # this is the model we will train 
model = Model(inputs=base_model.input, outputs=predictions)

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False
    
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

model.summary()

#earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

# fine-tune the model
nnet_mod = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
#                     callbacks=[earlyStopping], 
                    validation_split=0.1)
                    

plt.plot(nnet_mod.history['acc'])
plt.xlabel("epoch")
plt.ylabel("train accuracy")
plt.title("Face Count CNN Model Accuracy Plot by Epoch")
plt.savefig("Accuracy_Plot.png")
plt.clf()

plt.plot(nnet_mod.history['loss'])
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Face Count CNN Model Loss Plot by Epoch")
plt.savefig("Loss_Plot.png")
plt.clf()

plt.plot(nnet_mod.history['val_acc'])
plt.xlabel("epoch")
plt.ylabel("validation accuracy")
plt.title("Face Count CNN Model Validation Accuracy Plot by Epoch")
plt.savefig("Val_Accuracy_Plot.png")
plt.clf()

plt.plot(nnet_mod.history['val_loss'])
plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.title("Face Count CNN Model Validation Loss Plot by Epoch")
plt.savefig("Val_Loss_Plot.png")
plt.clf()



                    

                    
                
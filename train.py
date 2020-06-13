# -*- coding: utf-8 -*-

import sys
import os

import glob
import argparse
from keras import __version__
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import *
from keras.layers import *
from keras.activations import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import optimizers
from keras import callbacks
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.regularizers import l2,l1

import pandas as pd

K.tensorflow_backend._get_available_gpus()

train_data_path = '........training_path.....'
validation_data_path = '........validation_path.....'

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


batch_size = 16
image_size=(224,224)

nb_train_samples = get_nb_files(train_data_path)
nb_validation_samples = get_nb_files(validation_data_path)

classes_num = len(glob.glob(train_data_path + "/*"))

print("number of classe is :"+str(classes_num))

train_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.1,
                height_shift_range=0.1,
                preprocessing_function=preprocess_input,
                horizontal_flip=False,
                fill_mode='nearest')

def get_train_generator(train_datagen):
    train_generator = train_datagen.flow_from_directory(
                train_data_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical'
                )
    while True:
      Xi,Yi=train_generator.next()
      yield Xi, [Yi, Yi]
      
validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def get_validaion_generator(validation_datagen):
    validation_generator = validation_datagen.flow_from_directory(
                validation_data_path,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical'
                ) 
    while True:
      Xi,Yi= validation_generator.next()
      yield Xi, [Yi, Yi]

      
# Teacher/Student graph.
# Teacher ---> Decoder ---> Student
def build_graph(input_shape = (224,224,3),nbr_of_classes=38,view_summary=False):
#Teacher's graph.
    base_model1 = VGG16(include_top=False, weights='imagenet',input_shape = input_shape)
    x1_0 = base_model1.output
    x1_0 = Flatten(name='Flatten1')(x1_0)
    x1_1 = Dense(256, name='fc1',activation='relu')(x1_0) 
    x1_2 = classif_out_encoder1 = Dense(nbr_of_classes, name='out1', activation = 'softmax')(x1_1)  
#Decoder's graph.	
	#Get Teacher's tensors for skip connection.
    pool5 = base_model1.get_layer('block5_pool').output
    conv5 = base_model1.get_layer('block5_conv3').output
    conv4 = base_model1.get_layer('block4_conv3').output
    conv3 = base_model1.get_layer('block3_conv3').output
    conv2 = base_model1.get_layer('block2_conv2').output
    conv1 = base_model1.get_layer('block1_conv2').output
	#Inverse fully connected Teacher's layers. 
    inv_x1_1 = Dense(256, name='inv_x1_1',activation='relu')(x1_2)
    merge_x1_1 = Add(name='merge_x1_1')([inv_x1_1,x1_1])
    inv_x1_0 = Dense(7*7*512, name='x1_1',activation='relu')(merge_x1_1)
    reshaped_inv_x1_0 = Reshape((7, 7,512), name='')(inv_x1_0)
    inv_x1_0 = Add(name='merge_x1_0')([reshaped_inv_x1_0,pool5])
    #DECONV Block1
    up7 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(inv_x1_0))
    merge7 = concatenate([conv5,up7], axis = 3)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #DECONV Block2
    up8 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv4,up8], axis = 3)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #DECONV Block13
    up9 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv3,up9], axis = 3)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #DECONV Block14
    up10 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv2,up10], axis = 3)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv10 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
	#DECONVBlock15
    up11 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv10))
    merge11 = concatenate([conv1,up11], axis = 3)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge11)
    conv11 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    #Reconstructed image refinement
    conv11 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    mask = conv11 = Conv2D(3, 1, activation = 'sigmoid',name='Mask')(conv11)
    
#Graphe of Student
    base_model2 = VGG16(include_top=False, weights='imagenet',input_shape = (224,224,3))
    x2_0 = base_model2(mask)
    x2_0 = Flatten(name='Flatten2')(x2_0)
    x2_1 = Dense(256, name='fc2',activation='relu')(x2_0) 
    classif_out_encoder2  = Dense(nbr_of_classes, name='out2',activation='softmax')(x2_1)
  
#Get Teacher/Student Model
    model = Model(input = base_model1.input, output = [classif_out_encoder1,classif_out_encoder2])
    if(view_summary):
	    print(mode.summary())
#Compile the mode to use multi-task learning
    losses = {
            "out1": 'categorical_crossentropy',
            "out2": 'categorical_crossentropy'
            }
    alpha=0.4
    lossWeights = {"out1": alpha, "out2": (1.0-alpha)}
    model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss=losses, loss_weights=lossWeights,metrics = ['accuracy'])
  
    return model

model = build_graph(view_summary=True)

#Compile the mode to use multi-task learning
losses = {
        "out1": 'categorical_crossentropy',
        "out2": 'categorical_crossentropy'
        }
alpha=0.4
lossWeights = {"out1": alpha, "out2": (1.0-alpha)}
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), loss=losses, loss_weights=lossWeights,metrics = ['accuracy'])
    
nb_epoch = 20
history = model.fit_generator(get_train_generator(train_datagen),
                    nb_epoch=nb_epoch,
                    validation_data=get_validaion_generator(validation_datagen),
                    steps_per_epoch=nb_train_samples//batch_size,
                    validation_steps=nb_validation_samples//batch_size)


target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)

 
hist_df = pd.DataFrame(history.history) 
model.save_weights('./models/model_weights')
hist_csv_file = './models/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
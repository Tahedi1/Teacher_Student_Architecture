import sys
import os
import numpy as np
import glob
import argparse
from keras import backend as K
from keras import __version__
import cv2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import optimizers
from keras import callbacks
from keras.regularizers import l2,l1
from keras.preprocessing import image
from teacher_student import*
from PIL import Image

images_folder = './images/'
out_folder= './visualizations/'
model_weights_path='./model/black_models_15epochs_weights.h5'


def preprocess_image(image_path,image_size = (224,224)):
    img = image.load_img(image_path , target_size=image_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def build_visualization(model_weights_path):
    model = build_graph()
    model.load_weights(model_weights_path)
    layer_name ='Mask'
    NewInput = model.get_layer(layer_name).output
    visualization = K.function([model.input], [NewInput])
    return visualization

def reduce_channels_sequare(heatmap):
    channel1 = heatmap[:,:,0]
    channel2 = heatmap[:,:,1]
    channel3 = heatmap[:,:,2]
    new_heatmap = np.sqrt((channel1*channel1)+(channel2*channel2)+(channel3*channel3)) 
    return new_heatmap
	
def postprocess_vis(heatmap1,threshould = 0.9):
    heatmap=heatmap1.copy()
    heatmap = (heatmap - heatmap.min())/(heatmap.max() - heatmap.min())
    heatmap = reduce_channels_sequare(heatmap)
    heatmap = (heatmap - heatmap.min())/(heatmap.max() - heatmap.min())
    heatmap[heatmap<=threshould] = 0
    heatmap = heatmap*255
    
    return heatmap

def visualize_image(visualization,image_path,out_folder):
    base=os.path.basename(image_path)
    image_name= os.path.splitext(base)[0]
    img = preprocess_image(image_path)
    vis = visualization([img])[0][0]*255
    heatmap = postprocess_vis(vis)
    vis_path = os.path.join(out_folder,image_name+'_vis.jpg')
    cv2.imwrite(vis_path,vis)

    heatmap_path = os.path.join(out_folder,image_name+'_heatmap.jpg')
    cv2.imwrite(heatmap_path,heatmap)	

def visualize_folder(visualization,images_folder,out_folder):	
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for path in os.listdir(images_folder):
		    print(path)
		    image_path = os.path.join(images_folder,path)
		    visualize_image(visualization,image_path,out_folder) 
	

visualization = build_visualization(model_weights_path)
visualize_folder(visualization,images_folder,out_folder)

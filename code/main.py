import os
from xml_parse import xml_parse_training
from vgg_output import store_vgg_output
from svm_classification import read_vgg_output_data, trainSVM
from maximum_region_selector import maximum_region_selector
from region_selection import select_regions
import cv2
import pickle

classes = ['aeroplane','bicycle','bird','boat','car','person','horse','dog','cat']

#main directory path having train file name 
main_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/'

#xml files dir path
xml_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'

#all images dir path
images_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'

#cropped images dir path
cropped_img_dir_path = '../train_images/'

vgg_features_output = '../vgg_features/'
'''
# Crop Images
xml_parse_training(classes,main_dir_path,xml_dir_path,images_dir_path,cropped_img_dir_path)

# Store VGG outputs

for c in classes:
    store_vgg_output(os.path.join(cropped_img_dir_path,c),os.path.join(vgg_features_output,c))
'''
# Read VGG outputs
X,y = read_vgg_output_data(vgg_features_output,classes)

# Train SVM
# svm_classifier = trainSVM(X,y,0.2)
file = open('svm','rb')
svm_classifier = pickle.load(file)
file.close()
img = cv2.imread('C:\\Users\\chira\\Desktop\\RCNN_Project\\VOC2012test\\VOCdevkit\\VOC2012\\JPEGImages\\2008_000017.jpg')
rects_list = select_regions(img)
out = maximum_region_selector(svm_classifier,img,rects_list)

    
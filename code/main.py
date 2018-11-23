import os
from xml_parse import xml_parse_training
from vgg_output import store_vgg_output
from svm_classification import read_vgg_output_data, trainSVM

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

# Crop Images
xml_parse_training(classes,main_dir_path,xml_dir_path,images_dir_path,cropped_img_dir_path)

# Store VGG outputs
for c in classes:
    store_vgg_output(os.path.join(cropped_img_dir_path,c),os.path.join(vgg_features_output,c))

# Read VGG outputs
X,y = read_vgg_output_data(vgg_features_output,classes)

# Train SVM
trainSVM(X,y,0.2)
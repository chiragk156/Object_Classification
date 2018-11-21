import os
import sys
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def read_images(dataset_path):
    # It will return 4d array of n(images) x h x w x c(rgb)
    n = len(os.listdir(dataset_path))
    images = np.zeros(shape=(n,224,224,3))
    count = 0
    for img_path in os.listdir(dataset_path):
        full_img_path = os.path.join(dataset_path,img_path)
        image = load_img(full_img_path, target_size=(224, 224))
        image = img_to_array(image)
        images[count,:,:,:] = image
        count += 1
    return images


dataset_path = sys.argv[1]
output_path = sys.argv[2]

# VGG model
model = VGG16()
# Output of fc2 layer = 4096 dimesion
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer('fc2').output)

# image = load_img(r'C:\Users\chira\Downloads\face3.jpg', target_size=(224, 224))
# image = img_to_array(image)
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
images = preprocess_input(read_images(dataset_path))

vgg_output = intermediate_layer_model.predict(images)

# open a file to store the data
file = open(output_path, 'wb')
pickle.dump(vgg_output,file)
file.close()
import xml.etree.ElementTree as ET
import cv2
import os


def xml_parse_training(classes, main_dir_path, xml_dir_path, images_dir_path, cropped_img_dir_path):
    #making cropped images directory if not exists
    directory = os.path.dirname(cropped_img_dir_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for obj in classes:
        #making cropped images object directory if not exists for saving cropped images
        directory = os.path.dirname(cropped_img_dir_path + obj + '/')
        if not os.path.exists(directory):
            os.mkdir(directory)
           
        train_file = main_dir_path + obj + '_train.txt'             #train images text file name 
        file = open(train_file)                                     #opening training text file
        for line in file:                                           #reading text file and finding images with required object in them
            line = line.strip('\n')
            tokens = line.split(" ")
            if (len(tokens)==3 and tokens[2]=='1'):                 #if object image name is found
                xml_file =  xml_dir_path + tokens[0] +   '.xml'     #xml file name
                tree = ET.parse(xml_file)                           #parsing xml file and extracting required object coordinates
                root = tree.getroot()
                for child in root:
                    if child.tag == "object":
                        if(child.find('name').text==obj):
                            bndbox = child.find('bndbox')
                            xmin = int(bndbox.find('xmin').text)
                            ymin = int(bndbox.find('ymin').text)
                            xmax = int(bndbox.find('xmax').text)
                            ymax = int(bndbox.find('ymax').text)
                            print(images_dir_path+tokens[0]+'.jpg',obj)
                            img = cv2.imread(images_dir_path+tokens[0]+'.jpg')          #reading object image
                            cropped_img = img[ymin:ymax,xmin:xmax]                      #cropping with required dimensions
                            cv2.imwrite(cropped_img_dir_path + obj + '/' + tokens[0]+'.jpg',cropped_img)        #saving in required folder

if __name__=="__main__":
    classes = ['person','aeroplane','cat','bird']


    #main directory path having train file name 
    main_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Main/'

    #xml files dir path
    xml_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/'

    #all images dir path
    images_dir_path = '../VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/'

    #cropped images dir path
    cropped_img_dir_path = os.getcwd() + '/../train_images/'

    xml_parse_training(classes,main_dir_path,xml_dir_path,images_dir_path,cropped_img_dir_path)

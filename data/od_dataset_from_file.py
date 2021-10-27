import numpy as np
from PIL import Image
import glob
import os
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
import json

from tqdm import tqdm
import pickle
import xml.etree.ElementTree as ET
#import image_augmentation as img_aug
import cv2

#classes_map['background'] = 0

class DatasetFromFile(Dataset):
    def __init__(self, image_path,xml_path,imageset_list,classes,dataset_name,phase='train',difficultie = True,ext_img = ['jpg','bmp'],ext_anno = ['xml','json'],ori_classes_name=None):
        
        # Get image list
        #self.img_folder_list = glob.glob(folder_path+'*')
        
        self.item_list = list()
        self.phase = phase
        self.difficultie = difficultie
        self.classes = classes
        self.classes_map = {k: v for v, k in enumerate(classes)}
        self.ext_img = ext_img
        self.ext_anno = ext_anno
        im_list = list()
        if ori_classes_name!=None:
            self.ori_classes_name = ori_classes_name
        else:
            self.ori_classes_name = classes
        #print(type(image_path))
        self.list_name = 'data/%s.txt'%dataset_name
        
        if os.path.isfile(self.list_name):
            print(self.list_name)
            with open(self.list_name, "rb") as fp:   # Unpickling
                self.item_list = pickle.load(fp)
        else:            
            if type(imageset_list) is str and type(image_path) is str and type(xml_path) is str:
                with open(imageset_list,'r') as f:
                    for line in f:
                        for word in line.split():
                           im_list.append(word)
                self.parse_list(image_path,xml_path,im_list)
            elif type(imageset_list) is list :
                assert len(imageset_list) == len(image_path) == len(xml_path)
                for idx in range(len(imageset_list)) :
                    set = imageset_list[idx]
                    im_list.clear()
                    with open(set,'r') as f:
                        for line in f:
                            for word in line.split():
                               im_list.append(word)
                    self.parse_list(image_path[idx],xml_path[idx],im_list)
                                
            with open(self.list_name, "wb") as fp:   #Pickling
                pickle.dump(self.item_list, fp)
        self.data_len = len(self.item_list)
        print('total files of %s : %d'%(dataset_name,self.data_len))
        #print(self.item_list)
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path, single_anno_path = self.item_list[index]
        # Open image
        im = cv2.imread(single_image_path)
        
        boxes, labels, difficulties = self.parse_annotation(single_anno_path)
        yolo_labels = list()
        height, width, channels = im.shape
        im = cv2.imencode('.jpg', im,[int(cv2.IMWRITE_JPEG_QUALITY), 98])
        yolo_labels = self.to_yolo_label(boxes,labels,difficulties,width,height)      
        return (im, yolo_labels)

    def __len__(self):
        return self.data_len
    def to_yolo_label(self,boxes,labels,difficulties,width = 0,height = 0):
        yolo_labels = list()
        float = width == 0 and height == 0
        
        for index,box in enumerate(boxes):            
            if self.difficultie or not difficulties[index]:
                #print(box)
                yolo_label = list()
                yolo_label.clear()
                #print(box,labels[index])
                x = (box[0] + box[2])/2 
                y = (box[1] + box[3])/2 
                w = box[2] - box[0]
                h = box[3] - box[1] 
                if not float :
                    x = x / width
                    y = y / height
                    w = w / width
                    h = h / height
                yolo_label.append(labels[index])
                yolo_label.append(x)
                yolo_label.append(y)
                yolo_label.append(w)
                yolo_label.append(h)
                yolo_labels.append(yolo_label)
        return yolo_labels

    def parse_list(self,image_path,xml_path,im_list):    
        image_list = list()
        image_list.clear()
        for i in self.ext_img :
            pbar = tqdm(glob.glob(image_path+'/*.%s'%i))
            for f in pbar:
                path, filename = os.path.split(f)
                if any(filename[:-4] in s for s in im_list):
                    image_list.append(f)
                    pbar.set_description("Processing %s" % filename)   
        for i in self.ext_anno :
                pbar = tqdm(glob.glob(xml_path+'/*.%s'%i))
                for f in pbar:
                    path, filename = os.path.split(f)
                    #if any(filename[:-4] in s for s in im_list):
                    for img_f in image_list:
                        if filename[:-4] in img_f :
                            self.item_list.append([img_f,f])
                            #print([img_f,f])
                            pbar.set_description("Processing %s" % filename) 
    def parse_annotation(self,annotation_path):
        filename, file_extension = os.path.splitext(annotation_path)
        boxes = list()
        labels = list()       
        difficulties = list()   
        # VOC format xml
        if file_extension == '.xml':
            source = open(annotation_path)
            tree = ET.parse(source)
            root = tree.getroot()

            for object in root.iter('object'):
                difficult = int(object.find('difficult').text == '1')
                label = object.find('name').text.lower().strip()

                if label not in self.classes:
                    continue
                bbox = object.find('bndbox')
                xmin = int(bbox.find('xmin').text) - 1
                ymin = int(bbox.find('ymin').text) - 1
                xmax = int(bbox.find('xmax').text) - 1
                ymax = int(bbox.find('ymax').text) - 1
                boxes.append([xmin, ymin, xmax, ymax])
                #print(label)
                labels.append(self.classes_map[label])
                difficulties.append(difficult)
            source.close()
            return boxes, labels, difficulties 
        # COCO format json
        elif file_extension == '.json':
            with open(annotation_path, 'r') as f:
                data=json.load(f)        
            width = int(data['image']['width'])-1
            height = int(data['image']['height'])-1
            object_number = len(data['annotation'])
            for j in range(object_number):
                class_id = int(data['annotation'][j]['category_id'])-1
                category_name = self.ori_classes_name[class_id]
                if category_name in self.classes:
                    new_class_id = self.classes.index(category_name)
                    xmin = int(float(data['annotation'][j]['bbox'][0])+0.5)            
                    ymin = int(float(data['annotation'][j]['bbox'][1])+0.5)
                    if xmin<0:
                        xmin = 0
                    if ymin<0:
                        ymin = 0                    
                    xmax = int(float(data['annotation'][j]['bbox'][0])+float(data['annotation'][j]['bbox'][2])+0.5)
                    ymax = int(float(data['annotation'][j]['bbox'][1])+float(data['annotation'][j]['bbox'][3])+0.5)
                    if xmax>width:
                        xmax = width
                    if ymax>height:
                        ymax = height    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(new_class_id)
                    difficulties.append(0)
                    #print(xmin,ymin,class_id)
            return boxes, labels, difficulties    
    def collate_fn(self, batch):

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        
        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, H, W), 3 lists of N tensors each

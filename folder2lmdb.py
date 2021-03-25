import os
import sys
import six
import string
import argparse

import lmdb
import pickle
import msgpack
import tqdm
from PIL import Image

import torch
import torch.utils.data as data
from utils.image_augmentation import Image_Augmentation
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
from data.od_dataset_from_file import DatasetFromFile
import cv2
import numpy as np
import shutil
import random
trainval_dataset_path =  { \
	"imgs":['data/VOC2007/JPEGImages/','data/VOC2012/JPEGImages/'],  \
    "annos":['data/VOC2007/Annotations/','data/VOC2012/Annotations/'],  \
    "lists":['data/VOC2007/ImageSets/Main/trainval.txt','data/VOC2012/ImageSets/Main/trainval.txt'], \
    "name" :'voc_trainval',  \
    "lmdb" :'train-lmdb'  \
}

test_dataset_path =  { \
	"imgs":['data/VOC2007/JPEGImages/'],  \
	"annos":['data/VOC2007/Annotations/'],  \
    "lists":['data/VOC2007/ImageSets/Main/test.txt'],  \
    "name" :'voc_test',  \
    "lmdb" :'test-lmdb'  \
}
class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path,  transform=None, phase=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])            
        
        self.transform = transforms.Compose([
                transforms.Resize(size=(352,352), interpolation=2),
                transforms.ToTensor(),
                self.normalize,
            ])  
        
        self.phase = phase
        self.img_aug = Image_Augmentation()

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        #unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf[1])
        buf.seek(0)
        X_str= np.fromstring(buf.read(), dtype=np.uint8)
        img = cv2.imdecode(X_str, cv2.IMREAD_COLOR)       

        # load label
        target = unpacked[1]
        
        #if self.phase == 'train':
        target2 = torch.Tensor(target)           
        boxes = target2[...,1:5]

        x1 = (boxes[...,0] - boxes[...,2]/2).unsqueeze(1)
        y1 = (boxes[...,1] - boxes[...,3]/2).unsqueeze(1)
        x2 = (boxes[...,0] + boxes[...,2]/2).unsqueeze(1)
        y2 = (boxes[...,1] + boxes[...,3]/2).unsqueeze(1)
        boxes2 = torch.cat((x1*img.shape[1],y1*img.shape[0],x2*img.shape[1],y2*img.shape[0]),1)
        #if boxes.size(0) :
        labels = target2[...,0]
        #print(boxes2)
        difficulties = torch.zeros_like(labels)
        #for cls,x,y,w,h in target:
        #cls = target
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
        new_img, new_boxes, new_labels, new_difficulties = self.img_aug.transform_od(image, boxes2, labels, difficulties, (352,352), mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225],phase = self.phase)

        old_dims = torch.FloatTensor([new_img.width, new_img.height, new_img.width, new_img.height]).unsqueeze(0)
        new_boxes2 = new_boxes / old_dims  # percent coordinates
        
        w = (new_boxes2[...,2] - new_boxes2[...,0])
        h = (new_boxes2[...,3] - new_boxes2[...,1])
        x = (new_boxes2[...,0] + w/2).unsqueeze(1)
        y = (new_boxes2[...,1] + h/2).unsqueeze(1)
        #print(x.shape,y.shape,w.shape,h.shape,new_boxes.shape)
        new_boxes2 = torch.cat((x,y,w.unsqueeze(1),h.unsqueeze(1)),1)

        new_target = torch.cat((new_labels.unsqueeze(1),new_boxes2),1)
        #print(len(target),new_target.shape)
        #print(torch.Tensor(target).shape)
        #new_img,new_target = self.aug_image(img,target)
        #else :
        #    new_img,new_target = img , target
        #image = Image.fromarray(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))  
        #if self.transform is not None:
        #    image = self.transform(image)

        return (new_img,new_target)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    def set_transform(self,transform):
        self.transform = transform
    def collate_fn(self, batch):

        images = list()
        labels = list()
        if self.phase == 'train':
            random_size = random.choice([(352,352),(320,320),(384,384),(288,288),(416,416)])
            self.transform = transforms.Compose([
                    transforms.Resize(size=random_size, interpolation=2),
                    transforms.ToTensor(),
                    self.normalize,
                ])
        else :
            self.transform = transforms.Compose([
                    transforms.Resize(size=(352,352), interpolation=2),
                    transforms.ToTensor(),
                    self.normalize,
                ])  
        
        for b in batch:
            images.append(self.transform(b[0]))
            labels.append(b[1])
        
        images = torch.stack(images, dim=0)

        return images, labels  
         
def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def folder2lmdb(path, write_frequency=5000):
    directory = os.path.expanduser(path)
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)

    #print(trainval_dataset_path['imgs'])    
    trainval_dataset =  \
        DatasetFromFile(trainval_dataset_path['imgs'],trainval_dataset_path['annos'],trainval_dataset_path['lists'], \
        dataset_name=trainval_dataset_path['name'],phase = 'test',difficultie=False)
        
    test_dataset =  \
        DatasetFromFile(test_dataset_path['imgs'],test_dataset_path['annos'],test_dataset_path['lists'], \
        dataset_name=test_dataset_path['name'],phase = 'test',difficultie=False)
    outpath = trainval_dataset_path['lmdb'],test_dataset_path['lmdb']
    total_set = trainval_dataset,test_dataset
    for i in range(len(total_set)) :        
        data_loader = DataLoader(total_set[i], num_workers=4, collate_fn=lambda x: x)
        lmdb_path = os.path.expanduser(outpath[i])
        
        if os.path.exists(lmdb_path) and os.path.isdir(lmdb_path):
            shutil.rmtree(lmdb_path)
        #print(lmdb_path)
        os.mkdir(lmdb_path)
        print("Generate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, subdir=True,
	               map_size=1099511627776 * 2, readonly=False,
	               meminit=False, map_async=True)

        txn = db.begin(write=True)
        sum = 0
        for idx, data in enumerate(data_loader):
	        image, label = data[0][0],data[0][1]
	        sum += len(label)
	        txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
	        #txn.put(u'{}'.format(idx).encode('ascii'), pa.serialize((image, label)).to_buffer())
	        if idx % write_frequency == 0:
	            print("[%d/%d]" % (idx, len(data_loader)))
	            txn.commit()
	            txn = db.begin(write=True)

        print('total box : %d'%sum)
        # finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
	        txn.put(b'__keys__', pickle.dumps(keys))
	        txn.put(b'__len__', pickle.dumps(len(keys)))
	        #txn.put(b'__keys__', pa.serialize(keys).to_buffer())
	        #txn.put(b'__len__', pa.serialize(len(keys)).to_buffer())

        print("Flushing database ...")
        db.sync()
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Path to original image dataset folder")
    args = parser.parse_args()
    folder2lmdb(args.dataset)

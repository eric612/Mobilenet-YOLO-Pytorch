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
from torchvision import transforms, datasets
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
from data.od_dataset_from_file import DatasetFromFile
import cv2
import numpy as np
import shutil
import random
import yaml
from utils.box import wh_to_x2y2
import imgaug.augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    sometimes(iaa.SomeOf((1, 2),
        [
            #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                iaa.MedianBlur(k=(3,5)), # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 0.1), lightness=(0.9, 1.1)), # sharpen images
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.3), # add gaussian noise to images
        ],
        random_order=True
    ))
])

if torch.__version__> '1.8':
    from torchvision.transforms import InterpolationMode
    interp = InterpolationMode.BILINEAR
else :
    interp = 2
CLASSES = (#'__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path,batch_size,transform_size = [[352,352]], phase=None,expand_scale=1.5,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],has_seg = False, classes_name = CLASSES, seg_num_classes = 0):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))
        self.normalize = transforms.Normalize(mean=mean,std=std)            
        self.mean = mean
        self.std = std
        self.transform_size = transform_size
        self.phase = phase
        self.img_aug = Image_Augmentation()
        self.batch_size = batch_size
        self.count = 0
        self.expand_scale = expand_scale
        self.has_seg = has_seg
        self.classes_name = classes_name
        self.seg_num_classes = seg_num_classes
        
    def get_single_image(self,index,expand=False,expand_scale=1.5):
    
        img, target,img2 = None, None, None
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
        
        if self.has_seg:
            # load segmentation id
            imgbuf = unpacked[2]
            buf = six.BytesIO()
            buf.write(imgbuf[1])
            buf.seek(0)
            X_str= np.fromstring(buf.read(), dtype=np.uint8)
            img2 = cv2.imdecode(X_str, cv2.IMREAD_COLOR)     
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            seg_id = Image.fromarray(img2)
        else :
            seg_id = None
        
        #if self.phase == 'train':
        target2 = torch.Tensor(target)           
        boxes = target2[...,1:5]
        if boxes.shape[0] == 0 :
            #print(target2.shape)
            boxes2 = torch.zeros(0,4)
            labels = torch.zeros(0)
        else :
            x1 = (boxes[...,0] - boxes[...,2]/2).unsqueeze(1)
            y1 = (boxes[...,1] - boxes[...,3]/2).unsqueeze(1)
            x2 = (boxes[...,0] + boxes[...,2]/2).unsqueeze(1)
            y2 = (boxes[...,1] + boxes[...,3]/2).unsqueeze(1)
            boxes2 = torch.cat((x1*img.shape[1],y1*img.shape[0],x2*img.shape[1],y2*img.shape[0]),1)
            #if boxes.size(0) :
            labels = target2[...,0]
            #print(boxes2)
        #if labels == 7 :

        difficulties = torch.zeros_like(labels)
        img = seq(image=img)  # done by the library
        image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #print(seg_id)
        
        new_img, new_boxes, new_labels, new_difficulties, new_seg_id = self.img_aug.transform_od(image, boxes2, labels, difficulties,seg_id=seg_id, mean = [0.5, 0.5, 0.5],std = [1, 1, 1],phase = self.phase,expand = expand,expand_scale = self.expand_scale)

        array = np.array(new_seg_id)
        maps = list()
        if self.has_seg:
            for c in range(1,self.seg_num_classes+1):
                maps.append(Image.fromarray(array==c))
        old_dims = torch.FloatTensor([new_img.width, new_img.height, new_img.width, new_img.height]).unsqueeze(0)
        new_boxes2 = new_boxes / old_dims  # percent coordinates
        
        w = (new_boxes2[...,2] - new_boxes2[...,0])
        h = (new_boxes2[...,3] - new_boxes2[...,1])
        x = (new_boxes2[...,0] + w/2).unsqueeze(1)
        y = (new_boxes2[...,1] + h/2).unsqueeze(1)
        #print(x.shape,y.shape,w.shape,h.shape,new_boxes.shape)
        new_boxes2 = torch.cat((x,y,w.unsqueeze(1),h.unsqueeze(1)),1)
        new_target = torch.cat((new_labels.unsqueeze(1),new_boxes2),1)


        return (new_img,new_target,maps)
    def __getitem__(self, index):
        #print(index)
        
        
        if type(index) == list:

            group = []
            s = len(index)
            
            for idx in index:
                img,tar,seg_id = self.get_single_image(idx,s==1)
                group.append([img,tar,seg_id])   
            
            if s == 1 :
                #self.show_image(img,tar[...,1:5],tar[...,0],convert=True)
                return group[0][0],group[0][1],1,group[0][2]     
            else :
                b = self.img_aug.Mosaic(group,[1000,1000])
                #self.show_image(b[0],b[1][...,1:5].clone(),b[1][...,0].clone(),convert=True)
                return b[0],b[1],len(index)
        else:
            img,tar,_ = self.get_single_image(index)
            return img,tar,1
    
    def show_image(self,image,boxes=None,labels=None,convert=False,seg_id = False,gray_img_only = False,resize = None): 
        if gray_img_only == True :
            #print(image)
            cv_img = np.array(image.convert('L'))
            print(cv_img.shape)
            if resize!=None :
                cv_img = cv2.resize(cv_img, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 640, 480)        
            cv2.imshow('frame', cv_img)
            key = cv2.waitKey(3) 
        else :
            cv_img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            seg_id = (np.asarray(seg_id)!=0)*0.5
            #print(seg_id)
            #print(cv_img.shape,seg_id.shape)
            #cv_img = cv2.bitwise_and(cv_img,cv_img,mask = seg_id)
            cv_img[...,0] = cv_img[...,0]*seg_id + cv_img[...,0]*(seg_id==0)
            cv_img[...,2] = cv_img[...,2]*seg_id + cv_img[...,2]*(seg_id==0)
            for idx,box in enumerate(boxes) : 
                if convert :
                    #print(box,cv_img.shape)
                    wh_to_x2y2(box)
                    #print(box,cv_img.shape)
                    box[0],box[2] = box[0]*cv_img.shape[1],box[2]*cv_img.shape[1]
                    box[1],box[3] = box[1]*cv_img.shape[0],box[3]*cv_img.shape[0]
                    
                cv2.rectangle(cv_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
                text=self.classes_name[int(labels[idx])].lower()
                cv2.putText(cv_img, text, (int(box[0]),int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)
                
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 480, 480)        
            cv2.imshow('frame', cv_img)
            key = cv2.waitKey(0) 
            #cv2.imwrite('images//frame%04d.jpg'%self.count, cv_img)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    def set_transform(self,transform):
        self.transform = transform
    def collate_fn(self, batch):
        images = list()
        labels = list()
        seg_maps = list()
        random_size = random.choice(self.transform_size)
        seg_random_size = [int(number / 16) for number in random_size]
        #print(seg_random_size)
        self.transform = transforms.Compose([
                transforms.Resize(size=random_size, interpolation=interp),
                transforms.ToTensor(),
                self.normalize,
            ])  
        self.transform_seg = transforms.Compose([
                transforms.Resize(size=seg_random_size, interpolation=interp),
                transforms.ToTensor(),
            ])  
             
        count = 0
        
        for b in batch:
            if self.has_seg:
                maps = torch.zeros(seg_random_size[0],seg_random_size[1],self.seg_num_classes)
                for i,m in enumerate(b[3]):
                    cv_img = np.array(m.convert('L'))
                    cv_img = cv2.resize(cv_img, (seg_random_size[0], seg_random_size[1]), interpolation=cv2.INTER_AREA)
                    maps[...,i] = torch.Tensor(cv_img)/255.0 
                    #self.show_image(m,gray_img_only=True,resize=seg_random_size)
                seg_maps.append(maps)
                    
            images.append(self.transform(b[0]))
            labels.append(b[1])                
            count = b[2] + count
        images = torch.stack(images, dim=0)

        
        if self.phase == 'train':
            if self.has_seg:
                seg_maps = torch.stack(seg_maps, dim=0)
                return images, labels, count, seg_maps
            else:
                return images, labels, count, None
        else :
            return images, labels  
def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def folder2lmdb(dataset_path, write_frequency=5000):
    directory = os.path.expanduser(dataset_path)
    print("Loading dataset from %s" % directory)
    
    with open(dataset_path, 'r') as stream:
        data = yaml.load(stream)
        print(data)
        classes_name = data["classes"]["map"]
        classes_name.insert(0, 'background')
        ori_classes_name = data["classes"]["original"]
        trainval_dataset_path = data["trainval_dataset_path"]
        test_dataset_path = data["test_dataset_path"]
        ext_img = data["extention_names"]["image"]
        ext_anno = data["extention_names"]["annotation"]
        segmentation_enable = data["segmentation_enable"]
        if segmentation_enable:
            ext_seg = data["extention_names"]["segmentation"]
        
	
    #print(classes_name)
    if segmentation_enable:
        trainval_dataset =  \
            DatasetFromFile(trainval_dataset_path['imgs'],trainval_dataset_path['annos'],trainval_dataset_path['segs'],trainval_dataset_path['lists'],classes_name, \
            dataset_name=trainval_dataset_path['name'],phase = 'test',has_seg = segmentation_enable,difficultie=False,ext_img=ext_img,ext_anno=ext_anno,ext_seg=ext_seg,ori_classes_name=ori_classes_name)
            
        test_dataset =  \
            DatasetFromFile(test_dataset_path['imgs'],test_dataset_path['annos'],test_dataset_path['segs'],test_dataset_path['lists'],classes_name, \
            dataset_name=test_dataset_path['name'],phase = 'test',has_seg = segmentation_enable,difficultie=False,ext_img=ext_img,ext_anno=ext_anno,ext_seg=ext_seg,ori_classes_name=ori_classes_name)
    else :
        trainval_dataset =  \
            DatasetFromFile(trainval_dataset_path['imgs'],trainval_dataset_path['annos'],None,trainval_dataset_path['lists'],classes_name, \
            dataset_name=trainval_dataset_path['name'],phase = 'test',has_seg = segmentation_enable,difficultie=False,ext_img=ext_img,ext_anno=ext_anno,ori_classes_name=ori_classes_name)
            
        test_dataset =  \
            DatasetFromFile(test_dataset_path['imgs'],test_dataset_path['annos'],None,test_dataset_path['lists'],classes_name, \
            dataset_name=test_dataset_path['name'],phase = 'test',has_seg = segmentation_enable,difficultie=False,ext_img=ext_img,ext_anno=ext_anno,ori_classes_name=ori_classes_name)    
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
            if segmentation_enable:
                image,label,seg = data[0][0],data[0][1],data[0][2]
                txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label, seg)))
            else:
                image,label = data[0][0],data[0][1]
                txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps((image, label)))
            sum += len(label)
            
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
    parser.add_argument("-d", "--dataset", help="Path to original image dataset folder", default = 'data/voc_data.yaml')
    args = parser.parse_args()
    folder2lmdb(args.dataset)

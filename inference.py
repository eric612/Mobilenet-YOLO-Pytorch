import argparse
import os
import yaml
import torch
from models.voc.mbv2_yolo import yolo
import filetype
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='YOLO Inference')
parser.add_argument('-c', '--checkpoint', default='checkpoint/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to load checkpoint (default: checkpoint/checkpoint.pth.tar)')
parser.add_argument('-e', '--export', default='', type=str, metavar='PATH',
                    help='path to export model')                    
parser.add_argument('-y', '--yaml', default='models/voc/config.yaml', type=str, metavar='PATH',
                    help='path to load config')                     
parser.add_argument('-i', '--input', default='images/000166.jpg', type=str, metavar='PATH',
                    help='path to load input file') 
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                            '#d2f53c', '#fabebe', '#008080']            
CLASSES = (#'__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')                            
def main():
    args = parser.parse_args()
    assert os.path.isfile(args.yaml), 'Error: no config yaml file found!'
    with open(args.yaml, 'r') as f:
        config = yaml.load(f)      
    print(config)
    assert os.path.isfile(args.checkpoint), 'Error: no checkpoint found!'
    #checkpoint = torch.load(args.checkpoint)
    model = yolo(config=config)
    model = load_model(model, args.checkpoint)
    #model.load_state_dict(checkpoint['model'])    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.cuda()
    model = model.to(device)
    model.eval()
    model.yolo_losses[0].val_conf = 0.01 
    model.yolo_losses[1].val_conf = 0.01 
    kind = filetype.guess(args.input)
    if kind is None:
        print('Cannot guess file type!')
        return
    #print('File extension: %s' % kind.extension)
    #print('File MIME type: %s' % kind.mime)
    if kind.extension in ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'] :

        original_image = Image.open(args.input, mode='r')
        original_image = original_image.convert('RGB')
        annotated_image_ = cv2.cvtColor(np.asarray(original_image), cv2.COLOR_RGB2BGR)     
        height,width = annotated_image_.shape[0],annotated_image_.shape[1]
        #im_pil = Image.fromarray(annotated_image_)
        det_boxes = inference_image(model,original_image,device)
        #print(len(det_boxes))
        #print(det_boxes[0])
        
        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)     
        font = ImageFont.load_default().font
        # Suppress specific classes, if needed
        #box_location = [None]*4
        if det_boxes is not None :
            for bbox in det_boxes[0]:
               # print(bbox)
                
                box_location = bbox[:4].tolist()
                conf = bbox[4].item()
                cls_conf = bbox[5].item()
                cls_index = int(bbox[6].item())
                if conf*cls_conf>0.15:
                    box_location[0] = box_location[0]*width
                    box_location[1] = box_location[1]*height
                    box_location[2] = box_location[2]*width
                    box_location[3] = box_location[3]*height  
                    draw.rectangle(xy=box_location,outline=distinct_colors[0])
                     # Text
                    text_size = font.getsize(CLASSES[cls_index].upper())
                    text_location = [box_location[0] + 3., box_location[1] - text_size[1]]
                    textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                        box_location[1]]
                    draw.text(xy=text_location, text=CLASSES[cls_index].lower(), fill='white',
                              font=font)  
        cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 640, 480)
        cv2.imshow('frame', cv2.cvtColor(np.asarray(annotated_image), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0)                
        
def inference_image(model, original_image,device):
    # Transforms
    transform_test = transforms.Compose([
        transforms.Resize(size=(416,416), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Transform
    image = transform_test(original_image)

    # Move to default device
    image = image.to(device)
  
    detections = model(image.unsqueeze(0))  # (N, num_defaultBoxes, 4), (N, num_defaultBoxes, n_classes)

    return detections
def load_model(model, path_trained_weight):
    checkpoint_backbone = torch.load(path_trained_weight)
    
    pretrained_dict = checkpoint_backbone.state_dict()

    model_dict = model.state_dict()
    #for k, v in model_dict.items() :
        #if k[9:] in model_dict :
    #    print (k)    
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    if len(pretrained_dict.keys()) == 0:
        print('loading pretrain weight fail:{} '.format(path_trained_weight))
        input("Cont?")
    #print(pretrained_dict.keys())
    #print(model_dict.keys())
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("loaded the trained weights from {}".format(path_trained_weight))
    return model    
if __name__ == '__main__':
    main()
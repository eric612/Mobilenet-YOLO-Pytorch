import torch
import torch.nn as nn
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def wh_to_x2y2(bbox):
    bbox[...,0] = bbox[...,0] - bbox[...,2]/2
    bbox[...,1] = bbox[...,1] - bbox[...,3]/2
    bbox[...,2] = bbox[...,2] + bbox[...,0]
    bbox[...,3] = bbox[...,3] + bbox[...,1]
def nms(preds,num_classes) :
    nms_preds = list()
    assert len(preds) == 2 #only do two layers yolo 
    assert len(preds[0]) == len(preds[1])
    bs = len(preds[0])
    for b in range(bs):
        pred_per_img = torch.cat((preds[0][b],preds[1][b]),0)
        pred_boxes = torch.zeros(0,7, requires_grad=False).to(device)
        if pred_per_img.size(0):
            for i in range(num_classes) :                       
                mask = (pred_per_img[...,6] == i)                    
                pred_this_cls =  pred_per_img[mask]
                
                if pred_this_cls.size(0):
                    #print(pred_this_cls.shape,pred_per_img.shape)
                    boxes = pred_this_cls[...,:4]
                    scores = pred_this_cls[...,5]*pred_this_cls[...,4]
                    index = torchvision.ops.nms(boxes,scores,0.45)            
                    pred_boxes = torch.cat((pred_boxes,pred_this_cls[index]),0)
        nms_preds.append(pred_boxes)
    return nms_preds        
from collections import OrderedDict

import torch
import torch.nn as nn
from models.voc.yolo_config import *
from models.voc.mobilenetv2 import mobilenetv2
from models.voc.yolo_loss import *
from models.voc.yolo_detection import *
from torch.nn import init
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,depthwise=False):
        super(BasicConv, self).__init__()
        if depthwise == False :
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        else :
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False,groups = in_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)   
          
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        BasicConv(in_filters, in_filters, 3),
        BasicConv(in_filters, filters_list[0], 1),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
class yolo(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(yolo, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        #  backbone
        model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
        self.backbone = mobilenetv2(model_url)

        self.conv_for_P5 = BasicConv(1280,512,1)
        #print(num_anchors * (5 + num_classes))
        self.yolo_headP5 = yolo_head([1024, num_anchors * (5 + num_classes)],512)
        self.conv1_for_P4 = BasicConv(96,96,3)
        self.conv2_for_P4 = BasicConv(96,256,1)
        self.upsample = Upsample(512,256)
        self.yolo_headP4 = yolo_head([512, num_anchors * (5 + num_classes)],256)

        self.yolo_losses = []
        for i in range(2):
            self.yolo_losses.append(YOLOLoss(config["yolo"]["anchors"],config["yolo"]["mask"][i] \
                ,20,[config["img_w"],config["img_h"]],config["iou_thres"][i],config["noobj_scale"][i],config["nocls_scale"][i]).to(device))
        self.yolo_detection = []
        for i in range(2):
            self.yolo_detection.append(YOLO_Detection(config["yolo"]["anchors"],config["yolo"]["mask"][i] \
                ,20,[config["img_w"],config["img_h"]],config["iou_thres"][i]).to(device)) 
    def nms(self,preds) :
        nms_preds = list()
        assert len(preds) == 2 #only do two layers yolo 
        assert len(preds[0]) == len(preds[1])
        bs = len(preds[0])
        for b in range(bs):
            pred_per_img = torch.cat((preds[0][b],preds[1][b]),0)
            #print(preds[0][b].shape,preds[1][b].shape,pred_per_img.shape)
            #print(pred_per_img.shape)
            pred_boxes = torch.zeros(0,7, requires_grad=False).to(device)
            if pred_per_img.size(0):
                for i in range(self.num_classes) :                       
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
    def forward(self, x, targets=None):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #---------------------------------------------------#
        #print(x.shape)
        for i in range(2):
            self.yolo_losses[i].img_size = [x.size(2),x.size(3)]
        feat1, feat2 = self.backbone(x)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5) 

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        P4 = self.conv1_for_P4(feat1)
        P4 = self.conv2_for_P4(P4)
        P4 = torch.add(P4,P5_Upsample)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        output = self.yolo_losses[0](out0,targets),self.yolo_losses[1](out1,targets)
        if targets == None :
            output = self.nms(output)
        #if targets is not None :
        #    output = self.yolo_losses[0](out0,targets),self.yolo_losses[1](out1,targets)
        #else :
        #    output = self.yolo_detection[0](out0,targets),self.yolo_detection[1](out1,targets)
        #output1 = self.yolo_losses[0](out0,targets)
        #output2 = self.yolo_losses[0](out1,targets)
        
        return output
    
#def test():
#    net = yolo(3,20)
#    print(net)

#test()
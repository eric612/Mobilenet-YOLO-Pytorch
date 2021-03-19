import torch
import torch.nn as nn
import numpy as np
import math
from utils import AverageMeter
from utils.iou import *
from torch.autograd import Function
import gc
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision
class MySigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        sigmoid_eval = 1.0/(1.0 + torch.exp(-input))
        #input = sigmoid_eval
        return sigmoid_eval

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        #print(grad_output)
        grad_input = grad_output.clone()
        return grad_input
        
class YOLOLoss(nn.Module):
    def __init__(self, anchors, mask, num_classes, img_size,ignore_threshold,no_obj_scale=[0.2,0.1],no_cls_scale=[0.2,0.2],val_conf = 0.1):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.mask = mask;
        self.num_mask = len(mask)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.ignore_threshold = ignore_threshold
        self.lambda_xy = 2.5
        self.lambda_wh = 2.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.sm_loss = nn.SmoothL1Loss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss2 = nn.BCELoss()
        self.no_obj_scale = torch.tensor(no_obj_scale, requires_grad=False)
        self.no_cls_scale = torch.tensor(no_cls_scale, requires_grad=False)
        self.sigmoid = MySigmoid()
        self.nn_sigmoid = torch.nn.Sigmoid()
        self.val_conf = val_conf
 
    #def sigmoid(self, z):
    #    return 1/(1+np.exp(-z))     
      
    def weighted_l1_loss(self,input, target, weights):
        out = (input - target)      
        total = torch.sum(weights)
        out = out * weights 
        # expand_as because weights are prob not defined for mini-batch        
        loss = torch.sum(abs(out)/total) 
        #print(loss)
        return loss
    
    def weighted_mse_loss(self,input, target, weights):
        out = (input - target)**2      
        total = torch.sum(weights)
        out = out * weights / total
        # expand_as because weights are prob not defined for mini-batch        
        loss = torch.sum(out) 
        #print(loss)
        return loss
    '''
    def weighted_mse_loss(self,input, target, weights):
        out = (input - target)**2      
        #total = torch.sum(weights)
        out = out * weights 
        # expand_as because weights are prob not defined for mini-batch        
        #loss = torch.sum(out) 
        #print(loss)
        return out.mean()
    '''
    def get_pred_boxes(self,input, anchors, in_w, in_h):
        bs = input.size(0)
        #if bs > 0 :
        channel = self.bbox_attrs * self.num_mask;
        activation  = torch.zeros(bs, channel, in_h, in_w, requires_grad=True).to(device)
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(1, 1).to(device)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(1, 1).to(device)
        pred_boxes = torch.zeros(in_h, in_w,7, requires_grad=False).to(device)
        #pred_boxes = torch.zeros(in_h, in_w,4, requires_grad=False)
        outputs=list()
        for b in range(bs):
            preds = torch.zeros(0,7, requires_grad=False).to(device)
            for i in range(self.num_mask):
                idx = i * self.bbox_attrs                
                activation[b,idx:idx+2,...] = torch.sigmoid(input[b,idx:idx+2,...])
                activation[b,idx+2:idx+4,...] = torch.exp(input[b,idx+2:idx+4,...])
                start = idx+4
                end = idx + self.bbox_attrs
                activation[b,start:end,...] = torch.sigmoid(input[b,start:end,...])
                this_anchors = np.array(anchors)[self.mask]
                
                x = (activation[b,idx,...]+ grid_x)/in_w
                y = (activation[b,idx+1,...]+ grid_y)/in_h
                w = activation[b,idx+2,...]* this_anchors[i][0] 
                h = activation[b,idx+3,...]* this_anchors[i][1]                        

                #print(pred_boxes.shape,grid_x.shape,x.shape,((x + grid_x)/in_w).unsqueeze(2).shape)
                pred_boxes[..., 0] = x - w/2
                pred_boxes[..., 1] = y - h/2
                pred_boxes[..., 2] = x + w/2
                pred_boxes[..., 3] = y + h/2
                pred_boxes[..., 4] = activation[b,idx+4,...]
                #aa,bb = torch.max(output[b,idx+5:idx+self.bbox_attrs,...],dim=0)
                #print(activation[b,idx+5:idx+self.bbox_attrs,...].shape,aa.shape,bb.shape)
                pred_boxes[..., 5],pred_boxes[..., 6] = torch.max(activation[b,idx+5:idx+self.bbox_attrs,...],dim=0)  
                
                #pred_boxes = pred_boxes.reshape((in_w*in_h),7).contiguous
                mask = pred_boxes[...,4]>self.val_conf
                #print(pred_boxes[mask])
                #print(pred_boxes[mask])
                #pred_boxes2 = pred_boxes[mask]
                #outputs.cat(pred_boxes[mask])
                preds = torch.cat((preds,pred_boxes[mask]),0)
            outputs.append(preds)
        return outputs

    def forward(self, input, targets=None):
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w
        #print(self.img_size)
        #print(input.shape)
        scaled_anchors = [(a_w/self.img_size[0] , a_h/self.img_size[1] ) for a_w, a_h in self.anchors]
        
        #scaled_anchors2 = [(scaled_anchors[self.mask[i]][:] ) for i in range(self.num_mask)]
        #print(in_w,in_h,scaled_anchors)
        if targets is not None:
            #print(self.ignore_threshold)
            target,weights,output,recall,avg_iou,obj,no_obj,cls_score,count = self.get_target(targets,input, scaled_anchors,in_w, in_h,self.ignore_threshold)
            #target = target.to(device)
            #weights = weights.to(device)
            loss = self.weighted_mse_loss(output , target , weights)
            #loss = self.mse_loss(output[weights>=1] , target[weights>=1] )
          
            return loss, recall,avg_iou,obj,no_obj,cls_score,count

        else:
            preds = self.get_pred_boxes(input, scaled_anchors,in_w, in_h)
            #print(preds.shape)
            '''
            nms_preds = list()
            for b in range(bs):
                pred_per_img = preds[b]
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
                #print(pred_per_img.shape,pred_boxes.shape)
                nms_preds.append(pred_boxes)
            '''
            return preds

    def wh_to_x2y2(self,bbox):
        bbox[...,0] = bbox[...,0] - bbox[...,2]/2
        bbox[...,1] = bbox[...,1] - bbox[...,3]/2
        bbox[...,2] = bbox[...,2] + bbox[...,0]
        bbox[...,3] = bbox[...,3] + bbox[...,1] 
    # minimum convex box
    def box_c(self,box1,box2) :
        l = torch.min(box1[...,0],box2[...,0]).unsqueeze(0)
        t = torch.min(box1[...,1],box2[...,1]).unsqueeze(0)
        r = torch.max(box1[...,2],box2[...,2]).unsqueeze(0)
        b = torch.max(box1[...,3],box2[...,3]).unsqueeze(0)
        #print(t.shape)
        box_c = torch.cat((l,t,r,b),1)
        return box_c
    
    def box_giou(self,box1,box2):
        box_c = self.box_c(box1,box2)
        
        w = box_c[...,2] - box_c[...,0]
        h = box_c[...,3] - box_c[...,1]
        c = w*h;
        
        iou = find_jaccard_overlap(box1, box2)
        u = find_union(box1,box2)
        #giou_term = [iou if (k1 == 0)  else (k1 - k2)/k1 for k1,k2 in zip(c, u)]
        giou_term = (c-u)/c
        #print(iou,iou-giou_term)
        #print(c,u)
        return iou-giou_term;
 
    def IOU_Loss(self,gt_box,pred_box,grid_x,grid_y,anchors):
        w = gt_box[...,2] - gt_box[...,0]
        h = gt_box[...,3] - gt_box[...,1]
        tx = gt_box[...,0] - grid_x
        ty = gt_box[...,1] - grid_y        
        tw = torch.log(w/anchors[0])
        th = torch.log(h/anchors[1])
        #giou = self.box_giou(gt_box,pred_box)
        weight = 2.0 - (w*h)
        return tx,ty,tw,th,weight
    def class_loss(self,target_cls,target_weight,cls_idx):
        
        if target_weight[cls_idx]:
            target_cls[cls_idx] = 1
            target_weight[cls_idx] = 1                       
        else :
            target_cls[0:self.num_classes] = 0
            target_weight[0:self.num_classes] = 1
            target_cls[cls_idx] = 1
            #target_weight[cls_idx] = 1

    def get_target(self, target,input, anchors, in_w, in_h, ignore_threshold):
        bs = len(target)
        #if bs > 0 :
        channel = self.bbox_attrs * self.num_mask;
        #targets = torch.zeros(bs, channel, in_h, in_w, requires_grad=False).to(device)
        targets_weight = torch.zeros(bs, channel, in_h, in_w, requires_grad=False).to(device)
        #no_obj_mask = torch.ones(bs, self.num_mask, in_h, in_w, requires_grad=False).to(device)
        pred_boxes = torch.zeros(bs,self.num_mask,in_h, in_w,4, requires_grad=False).to(device)
        #obj_mask = torch.ones(bs, self.num_mask, in_h, in_w, requires_grad=False).to(device)
        #output = torch.zeros(bs, channel, in_h, in_w, requires_grad=False).to(device)
        #output = input.clone()
        pos_sum = 0
        FloatTensor = torch.cuda.FloatTensor 
        LongTensor = torch.cuda.LongTensor 
        # Calculate offsets for each grid
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(1, 1).to(device)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(1, 1).to(device)
        ious =  list()
        count = 0
        recall = 0
        no_obj = 0
        obj = list()
        cls_score = list()
        #output = self.sigmoid.apply(input)
        this_anchors = np.array(anchors)[self.mask]
        #print('\n',this_anchors)
        indice = torch.zeros(bs, channel, in_h, in_w, dtype=torch.bool, requires_grad=False).to(device)
        indice2 = torch.zeros(bs, channel, in_h, in_w, dtype=torch.bool, requires_grad=False).to(device)
        regression_mask = torch.zeros(bs, self.num_mask, in_h, in_w, dtype=torch.bool, requires_grad=False).to(device)
        nobj_num = self.num_mask*bs*in_w*in_h
        for i in range(self.num_mask):
            idx = i * self.bbox_attrs
            indice2[:,idx+4,...] = True
            indice[:,idx:idx+2,...] = True
        for b in range(bs):
            for t in range(len(target[b])):
                               
                gt = target[b][t].clone().detach()
                gx = gt[1] * in_w
                gy = gt[2] * in_h
                gw = gt[3] 
                gh = gt[4]
                gi = int(gx)
                gj = int(gy)
                
                #print(index)
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),
                                                  np.array(anchors)), 1)) 
                gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                anch_ious = find_jaccard_overlap(gt_box, anchor_shapes).squeeze(0)
                best_n = torch.argmax(anch_ious)
                anch_ious_this = anch_ious[self.mask] 
                iou_thresh_list = (anch_ious_this>0.213).tolist()
                bn = self.num_anchors + 1 
                
                if best_n in self.mask :
                    bn = self.mask.index(best_n) 
                
                    count+= 1
                    index = [bn*self.bbox_attrs+y+5 for y in range(self.num_classes)] #conf,cls
                    indice[b,index,gj,gi] = True            
                    regression_mask[b,bn,gj,gi] = True
                '''
                for m in range(self.num_mask):
                    if m == bn or m in iou_thresh_list:                    
                        #print(this_anchors)
                        bn = m
                        count+= 1
                        regression_mask[b,m,gj,gi] = True
                        index = [bn*self.bbox_attrs+y+5 for y in range(self.num_classes)] #conf,cls
                        indice[b,index,gj,gi] = True
                '''
        indice = torch.logical_or(indice,indice2)
        #print(indice)
        output = self.sigmoid.apply(input)*indice + input*(~indice)
        no_obj = torch.sum(output[indice2]).item()
        
        #get preds bbox
        for i in range(self.num_mask):
            idx = i * self.bbox_attrs                
            xy = (output[:,idx:idx+2,...].detach())
            wh = torch.exp(output[:,idx+2:idx+4,...].detach())

            pred_boxes[:,i,..., 0] = (xy[:,0,...]+ grid_x)/in_w
            pred_boxes[:,i,..., 1] = (xy[:,1,...]+ grid_y)/in_h
            pred_boxes[:,i,..., 2] = (wh[:,0,...]* this_anchors[i][0])
            pred_boxes[:,i,..., 3] = (wh[:,1,...]* this_anchors[i][1])
               

        targets = output.clone().detach()
        self.wh_to_x2y2(pred_boxes)
        for b in range(bs):
                        
            gt_boxes = target[b][...,1:].clone().detach().to(device)
            self.wh_to_x2y2(gt_boxes)
            #print(gt_boxes.shape)
            
            
            pred_boxes2 = pred_boxes[b].view((in_w*in_h*self.num_mask, 4))
            pred_iou = find_jaccard_overlap(gt_boxes,pred_boxes2)
            #print(pred_iou.shape)
            pred_iou,_ = torch.max(pred_iou,0)
            #print(pred_iou.shape)
            pred_iou = pred_iou.view((self.num_mask,in_h,in_w))
            for i in range(self.num_mask):
                idx = i * self.bbox_attrs
                #print('be',targets_weight[b,idx+4][5])
                #print(pred_iou[i]>0.5)
                #print(pred_iou[i])
                m = pred_iou[i]<ignore_threshold
                targets_weight[b,idx+4,m] = 1 
                targets[b,idx+4,m] = 0 
                #no_obj_mask[b,idx+4,pred_iou[i]<ignore_threshold] = 0
           
            for t in range(len(target[b])):
                gt = target[b][t].clone().detach()
                gx = gt[1] * in_w
                gy = gt[2] * in_h
                gw = gt[3] 
                gh = gt[4] 
                gx2 = gt[1] 
                gy2 = gt[2] 
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)    
                # Get shape of gt box
                #gt_box = torch.FloatTensor([0, 0, gw, gh]).unsqueeze(0)
                gt_box_xy = torch.FloatTensor([gx2-gw/2, gy2-gh/2, gx2+gw/2, gy2+gh/2]).unsqueeze(0).to(device)

                for m in range(self.num_mask):
                    if regression_mask[b,m,gj,gi]:                    
                        #print(this_anchors)
                        bn = m
                        #count+= 1                                          
                        index = bn*self.bbox_attrs                
                        pred = pred_boxes[b,bn,gj,gi,...].unsqueeze(0)
                        iou = find_jaccard_overlap(gt_box_xy, pred)
                        if iou>0:
                            ious.append(iou.item())
                        else :
                            ious.append(0)
                        if iou>ignore_threshold :
                            recall = recall + 1
                        tx,ty,tw,th,weight = self.IOU_Loss(gt_box_xy,pred,gi,gj,this_anchors[bn])
                        #print(giou,iou)
                        #print(gt_box,pred,iou.item())
                        # Coordinates
                        targets[b, index, gj, gi] = gx-gi
                        targets[b, index+1, gj, gi] = gy-gj
                        targets_weight[b, index:(index+4), gj, gi] = weight
                        targets[b, index+2, gj, gi] = tw
                        targets[b, index+3, gj, gi] = th
                        # object
                        targets[b, index+4, gj, gi] = 1
                        targets_weight[b, index+4, gj, gi] = 1
                        no_obj = no_obj - output[b,index+4,gj,gi].item()
                        obj.append(output[b,index+4,gj,gi].item())
                        
                        #no_obj_mask[b, bn, gj, gi] = 0
                        cls_index = int(target[b][t][0].item())-1
                        cls_tensor = targets[b, index+5:index+self.bbox_attrs, gj, gi]
                        cls_weight = targets_weight[b, index+5:index+self.bbox_attrs, gj, gi]
                        self.class_loss(cls_tensor,cls_weight,cls_index)
                        cls_score.append(output[b, cls_index+5+index, gj, gi].item())
                        '''
                        # One-hot encoding of label      
                        cls_index = index + 5 + int(target[b][t][0].item())-1
                        if targets_weight[b, cls_index, gj, gi]:
                            targets[b, cls_index, gj, gi] = 1
                            targets_weight[b, cls_index, gj, gi] = 1                       
                        else :
                            targets[b, index+5:index+self.bbox_attrs, gj, gi] = 0
                            #targets_weight[b, index+5:index+self.bbox_attrs, gj, gi] = self.no_cls_scale
                            targets_weight[b, index+5:index+self.bbox_attrs, gj, gi] = 1
                            #output[b,index+5:index+self.bbox_attrs, gj, gi] = self.sigmoid.apply(input[b,index+5:index+self.bbox_attrs, gj, gi])
                            targets[b, cls_index, gj, gi] = 1
                            targets_weight[b, cls_index, gj, gi] = 1
                            cls_score.append(output[b, cls_index, gj, gi].item())
                        '''
                    
                    
                    #targets_weight[b, index+4:index+self.bbox_attrs, gj, gi] = 1
        nobj_num = nobj_num - count
        no_obj = no_obj/nobj_num

        if count>0:
            avg_iou = sum(ious)/count;
            recall = recall/count
            obj_avg = sum(obj) / len(obj) 
            cls_avg = sum(cls_score) / len(cls_score)           
        else:
            avg_iou = recall = 0.5
            obj_avg = cls_avg = 0.5
            

        return targets,targets_weight,output,recall,avg_iou,obj_avg,no_obj,cls_avg,count/bs


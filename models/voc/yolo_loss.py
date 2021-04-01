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
class MSigmoid(Function):
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
        # Maximum likelihood and gradient descent demonstration
        # https://blog.csdn.net/yanzi6969/article/details/80505421
        # https://xmfbit.github.io/2018/03/21/cs229-supervised-learning/
        # https://zlatankr.github.io/posts/2017/03/06/mle-gradient-descent
        grad_input = grad_output.clone()
        return grad_input
        
class YOLOLoss(nn.Module):
    def __init__(self, anchors, mask, num_classes, img_size,ignore_threshold,val_conf = 0.1):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.mask = mask;
        self.num_mask = len(mask)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size
        self.ignore_threshold = ignore_threshold
        self.sigmoid = MSigmoid()
        self.nn_sigmoid = torch.nn.Sigmoid()
        self.val_conf = val_conf
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.label_smooth_eps = 0.1
    
    def weighted_mse_loss(self,input, target, weights):
        out = (input - target)**2      
        total = torch.sum(weights)
        out = out * weights / total
        # expand_as because weights are prob not defined for mini-batch        
        loss = torch.sum(out) 
        #print(loss)
        return loss

    def pre_maps(self,bs,is_cuda,anchors, in_w, in_h):
    
        FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor
        this_anchors = np.array(anchors)[self.mask]
        anchor_w = FloatTensor(this_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(this_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(bs,self.num_mask,in_h,in_w,1).to(device)   
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(bs,self.num_mask,in_h,in_w,1).to(device)        
        grid_x = torch.linspace(0, in_w-1, in_w).repeat(in_w, 1).repeat(bs * self.num_mask, 1, 1).view(bs,self.num_mask,in_h,in_w,1).type(FloatTensor)
        grid_y = torch.linspace(0, in_h-1, in_h).repeat(in_h, 1).t().repeat(bs * self.num_mask, 1, 1).view(bs,self.num_mask,in_h,in_w,1).type(FloatTensor)
        grid_xy = torch.cat((grid_x,grid_y),4)
        anchor_wh = torch.cat((anchor_w,anchor_h),4)
        return grid_xy,anchor_wh
        
    def get_target(self, target,input, anchors, in_w, in_h, ignore_threshold):
    
        bs = input.size(0)
        this_anchors = np.array(anchors)[self.mask]
        FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
        targets_weight = torch.zeros(bs, self.num_mask, in_h, in_w,self.num_classes+1, requires_grad=False).to(device)
        pred_boxes = torch.zeros(bs,self.num_mask,in_h, in_w,0, requires_grad=False).to(device)
        prediction = input.view(bs,  self.num_mask,self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous() 
        xy = self.sigmoid.apply(prediction[..., 0:2]) 
        wh = torch.exp(prediction[..., 2:4]) 
        output = self.sigmoid.apply(prediction[..., 4:])

        grid_xy,anchor_wh = self.pre_maps(bs,input.is_cuda,anchors, in_w, in_h)
        pred_boxes = torch.cat((pred_boxes,(xy + grid_xy)/FloatTensor([in_w,in_h])),4)
        pred_boxes = torch.cat((pred_boxes,wh * anchor_wh),4)
        self.wh_to_x2y2(pred_boxes)
        

        count = recall = ious = obj = cls_score = 0
        #output = torch.cat((xy,prediction[..., 2:4],conf_cls),4).to(device)
        targets = output.clone().to(device)
        no_obj = torch.sum(output[...,0])
        no_cnt = output[...,0].numel()
        targets_weight_parts = targets_weight[...,0]  
        targets_parts = targets[...,0] 
        anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 2)),np.array(anchors)), 1)) 
        iou_loss = torch.FloatTensor(0).to(device)
        iou_weight = torch.FloatTensor(0).to(device)
        in_dim = torch.Tensor([in_w,in_h])
        #print(need_grad_tensor.view(,self.num_classes+1).shape)
        for b in range(bs):
            if len(target[b]) == 0 :
                targets_weight_parts[b] = 1
                targets_parts[b] = 0
                continue
            gt_boxes = target[b][...,1:].clone().detach().to(device)
            self.wh_to_x2y2(gt_boxes)   
            
            pred_boxes2 = pred_boxes[b].view((in_w*in_h*self.num_mask, 4)).to(device)
            pred_iou = find_jaccard_overlap(gt_boxes,pred_boxes2).to(device)
            #print(pred_iou.shape)
            pred_iou,_ = torch.max(pred_iou,0)

            pred_iou = pred_iou.view((self.num_mask,in_h,in_w))
            
            #for i in range(self.num_mask):
            m = pred_iou<ignore_threshold
            targets_weight_parts[b,...,m] = 1
            targets_parts[b,...,m] = 0

            gt = target[b].clone().detach()
            gxgy = gt[...,1:3] * in_dim 
            gt[...,1:3] = 0
            gt_box = gt[...,1:]
            gt[...,0] = gt[...,0] - 1
            anch_ious = find_jaccard_overlap(gt_box, anchor_shapes)
            best_n = torch.argmax(anch_ious,1)
            #mask = best_n[:] in self.mask
            for t in range(len(target[b])):                                               
                gi = int(gxgy[t,0])
                gj = int(gxgy[t,1])                
                #anch_ious_this = anch_ious[self.mask] 
                #iou_thresh_list = (anch_ious_this>0.213).tolist()
                #bn = self.num_anchors + 1 
                if best_n[t] in self.mask :
                    bn = self.mask.index(best_n[t])  
                    k = bn 
                #for k in range(self.num_mask):
                    #if k == bn or iou_thresh_list[k] :
                    count+= 1                
                    cls_index = int(gt[t,0])
                    
                    targets_parts[b,k,gj,gi] = 1 
                    targets_weight_parts[b,k,gj,gi] = 1 
                    conf = output[b,k,gj,gi,0].item()
                    obj = obj + conf
                    no_obj = no_obj - conf
                    gt_box_xy = gt_boxes[t].unsqueeze(0)
                    pred = pred_boxes[b, k, gj, gi].unsqueeze(0)

                    giou,iou = self.box_ciou(gt_box_xy,pred)

                    iou_loss = torch.cat((iou_loss,(1. - giou).to(device)))
                    area = 2.0 - self.get_area(gt_box_xy)
                    
                    iou_weight = torch.cat((iou_weight,(area).to(device)))
                    if iou>ignore_threshold :
                        recall = recall + 1                         
                    ious = ious + iou.item()
                    cls_tensor = targets[b, k, gj, gi,1:]
                    cls_weight = targets_weight[b, k, gj, gi,1:]
                    self.class_loss(cls_tensor,cls_weight,cls_index)
                    cls_score = cls_score + output[b,k,gj,gi,1+cls_index].item()
        if count > 0:                
            obj_avg =  obj/count 
            cls_avg =  cls_score/count
            no_obj = no_obj/(no_cnt-count)
            avg_iou = ious/count
            recall = recall/count
        else :
            recall = obj_avg = cls_avg = no_obj = avg_iou = 0
        return targets,targets_weight,output,recall,avg_iou,obj_avg,no_obj,cls_avg,count/bs,iou_loss,iou_weight
        
    def get_pred_boxes(self,input, anchors, in_w, in_h):
    
        bs = input.size(0)
        pred_boxes = torch.zeros(bs,self.num_mask,in_h, in_w,0, requires_grad=False).to(device)
        #pred_boxes = torch.zeros(in_h, in_w,4, requires_grad=False)
        outputs=list()
        prediction = input.view(bs,  self.num_mask,self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous() 
        xy = torch.sigmoid(prediction[..., 0:2]) 
        wh = torch.exp(prediction[..., 2:4]) 
        conf_cls = torch.sigmoid(prediction[..., 4:])       # Conf
        
        FloatTensor = torch.cuda.FloatTensor if input.is_cuda else torch.FloatTensor
        grid_xy,anchor_wh = self.pre_maps(bs,input.is_cuda,anchors, in_w, in_h)
        
        pred_boxes = torch.cat((pred_boxes,(xy + grid_xy)/FloatTensor([in_w,in_h])),4)
        pred_boxes = torch.cat((pred_boxes,wh * anchor_wh),4)
        self.wh_to_x2y2(pred_boxes)        
        pred_boxes = torch.cat((pred_boxes,conf_cls[...,0].unsqueeze(4)),4)
        score,cls_idx = torch.max(conf_cls[...,1:self.bbox_attrs],dim=4)
        pred_boxes = torch.cat((pred_boxes,score.unsqueeze(4),cls_idx.float().unsqueeze(4)),4)
        pred_boxes = pred_boxes.to(device)
        mask = pred_boxes[...,4]>self.val_conf
        for b in range(bs):
            outputs.append(pred_boxes[b,mask[b]])
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

        if targets is not None:
            #print(self.ignore_threshold)
            target,weights,output,recall,avg_iou,obj,no_obj,cls_score,count,iou_losses,iou_weights = self.get_target(targets,input, scaled_anchors,in_w, in_h,self.ignore_threshold)
            loss = self.weighted_mse_loss(output , target , weights)
            iou_target = torch.zeros_like(iou_losses)
            #iou_loss= torch.sum(iou_target-iou_losses)
            iou_loss = self.weighted_mse_loss(iou_losses,iou_target,iou_weights)/iou_losses.numel()
            #iou_loss = self.mse_loss(iou_losses,iou_target)/iou_losses.numel()
            #print(iou_loss)
            #iou_loss = torch.Tensor(iou_loss)
            #print(loss,iou_loss)
            #loss = torch.cat((loss.unsqueeze(0) ,iou_loss.unsqueeze(0)))
            loss = loss + iou_loss*0.01
            return loss, recall,avg_iou,obj,no_obj,cls_score,count 

        else:
            preds = self.get_pred_boxes(input, scaled_anchors,in_w, in_h)

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
    def box_ciou(self,box1,box2):
        box_c = self.box_c(box1,box2)
        c = self.get_area(box_c)       
        iou = find_jaccard_overlap(box1, box2)

        w1,h1 = box1[...,2] - box1[...,0],box1[...,3] - box1[...,1]
        w2,h2 = box2[...,2] - box2[...,0],box2[...,3] - box2[...,1]
        x1,y1 = (box1[...,2] + box1[...,0])/2,(box1[...,1] + box1[...,3])/2
        x2,y2 = (box2[...,2] + box2[...,0])/2,(box2[...,1] + box2[...,3])/2
        u = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
        if c==0 :
            ciou_term = iou
        else :
            d = u/c
            ar_gt  = w2/h2
            ar_pred  = w1/h1
            ar_loss = 4 / (math.pi * math.pi) * (torch.atan(ar_gt) - torch.atan(ar_pred)) * (torch.atan(ar_gt) - torch.atan(ar_pred));
            alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
            ciou_term = d + alpha * ar_loss;
        #print(iou,iou-giou_term)
        #print(c,u)
        return iou-ciou_term,iou
   
    def box_giou(self,box1,box2):
        box_c = self.box_c(box1,box2)
        c = self.get_area(box_c)
        
        #iou = find_jaccard_overlap(box1, box2)
        u = find_union(box1,box2)
        i = find_intersection(box1,box2)
        iou = i/u
        #giou_term = [iou if (k1 == 0)  else (k1 - k2)/k1 for k1,k2 in zip(c, u)]
        if c==0 :
            giou_term = iou
        else :
            giou_term = (c-u)/c
        #print(iou,iou-giou_term)
        #print(c,u)
        return iou-giou_term,iou
    def get_area(self,box):
        return (box[...,2] - box[...,0]) * (box[...,3] - box[...,1])   
    def get_aspect_ratio(self,box):
        return (box[...,2] - box[...,0]) / (box[...,3] - box[...,1])           
    def IOU_Loss(self,gt_box,pred_box,input,output,accumulate):
 
        X = self.get_area(pred_box)
        Xhat = self.get_area(gt_box)
        
        pred_l,pred_t,pred_r,pred_b = pred_box[...,0],pred_box[...,1],pred_box[...,2],pred_box[...,3]
        gt_l,gt_t,gt_r,gt_b = gt_box[...,0],gt_box[...,1],gt_box[...,2],gt_box[...,3]
        
        Ih = torch.min(pred_b, gt_b) - torch.max(pred_t, gt_t)
        Iw = torch.min(pred_r, gt_r) - torch.max(pred_l, gt_l)
        I = Iw*Ih # intersection area
        #print(Iw,Ih,I)
        
        #m = I > 0
        #if m == False:
        #    print(Iw,Ih,I)
        U = X + Xhat - I; # Union area
        Cw = torch.max(pred_r, gt_r) - torch.min(pred_l, gt_l);
        Ch = torch.max(pred_b, gt_b) - torch.min(pred_t, gt_t);
        C = Cw * Ch;
        #iou = find_jaccard_overlap(gt_box, pred_box)
        #print(pred_box,gt_box)
        #if I<0 :
        #    I = 0
        #print((I/U)==iou)
            
        dX_wrt_t = -1 * (pred_r - pred_l);
        dX_wrt_b = -dX_wrt_t;
        dX_wrt_l = -1 * (pred_b - pred_t);
        dX_wrt_r = -dX_wrt_l;
        
        dI_wrt_t = (pred_t > gt_t)*(-Iw)
        dI_wrt_b = (pred_b > gt_b)*(Iw)
        dI_wrt_l = (pred_l > gt_l)*(-Ih)
        dI_wrt_r = (pred_r > gt_r)*(Ih)

        # derivative of U with regard to x
        dU_wrt_t = dX_wrt_t - dI_wrt_t
        dU_wrt_b = dX_wrt_b - dI_wrt_b
        dU_wrt_l = dX_wrt_l - dI_wrt_l
        dU_wrt_r = dX_wrt_r - dI_wrt_r  
        
        dC_wrt_t = (pred_t < gt_t)*(-1 * Cw)
        dC_wrt_b = (pred_b > gt_b)*Cw
        dC_wrt_l = (pred_l < gt_l)*(-1 * Ch) 
        dC_wrt_r = (pred_r > gt_r)*Ch 

        p_dt = p_db = p_dl = p_dr = 0
        if U > 0 :
            p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U)
            p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U)
            p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U)
            p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U)
            #p_dt = ((U+I) * dI_wrt_t)/ (U*I ) - (dX_wrt_t) / U 
            #p_db = ((U+I) * dI_wrt_b)/ (U*I ) - (dX_wrt_t) / U 
            #p_dl = ((U+I) * dI_wrt_l)/ (U*I ) - (dX_wrt_t) / U 
            #p_dr = ((U+I) * dI_wrt_r)/ (U*I ) - (dX_wrt_t) / U 
        if C > 0 :
            # apply "C" term from gIOU
            p_dt += ((C * dU_wrt_t) - (U * dC_wrt_t)) / (C * C);
            p_db += ((C * dU_wrt_b) - (U * dC_wrt_b)) / (C * C);
            p_dl += ((C * dU_wrt_l) - (U * dC_wrt_l)) / (C * C);
            p_dr += ((C * dU_wrt_r) - (U * dC_wrt_r)) / (C * C);
           
        delta_x = ((p_dl + p_dr))
        delta_y = ((p_dt + p_db))
        delta_w = ((-0.5 * p_dl) + (0.5 * p_dr))
        delta_h = ((-0.5 * p_dt) + (0.5 * p_db))
        #tx,ty,tw,th,_ = self.DenseBoxLoss(gt_box,pred_box,grid_x,grid_y,anchors,in_w,in_h)
        #print(output[...,0]-tx,delta_x)
        if accumulate:
            tx = (output[...,0] + delta_x*0.5).item()
            ty = (output[...,1] + delta_y*0.5).item()
            tw = (output[...,2] + (delta_w*torch.exp(input[...,2]))*0.5).item()
            th = (output[...,3] + (delta_h*torch.exp(input[...,3]))*0.5).item()        
        else :
            tx = (input[...,0] + delta_x*0.5).item()
            ty = (input[...,1] + delta_y*0.5).item()
            tw = (input[...,2] + (delta_w*torch.exp(input[...,2]))*0.5).item()
            th = (input[...,3] + (delta_h*torch.exp(input[...,3]))*0.5).item()
        #print(tw,th)
        #delta_w = delta_w*torch.exp(delta_w);
        #delta_h = delta_h*torch.exp(delta_h);    
        #print(p_dt,p_db,p_dl,p_dr)
        #else :
        #    tx,ty,tw,th,_ = self.DenseBoxLoss(gt_box,pred_box,grid_x,grid_y,anchors,in_w,in_h)
        target = torch.Tensor([tx,ty,tw,th]).to(device)
        return target,(2.0-Xhat),I/U
        
    def DenseBoxLoss(self,gt_box,pred_box,grid_x,grid_y,anchors,in_w,in_h):
        w = gt_box[...,2] - gt_box[...,0]
        h = gt_box[...,3] - gt_box[...,1]
        x = gt_box[...,0] + w / 2
        y = gt_box[...,1] + h / 2
        tx = x * in_w - grid_x
        ty = y * in_h - grid_y        
        tw = torch.log(w/anchors[0])
        th = torch.log(h/anchors[1])
        #giou = self.box_giou(gt_box,pred_box)
        weight = 2.0 - (w*h)
        target = torch.Tensor([tx,ty,tw,th]).to(device)
        iou = find_jaccard_overlap(gt_box, pred_box)
        return target,weight,iou
    def class_loss(self,target_cls,target_weight,cls_idx):
        y_true = (1 - self.label_smooth_eps) + 0.5*self.label_smooth_eps;
        y_false = 0.5*self.label_smooth_eps;
        if target_weight[cls_idx]:
            target_cls[cls_idx] = y_true
            target_weight[cls_idx] = 1                       
        else :
            target_cls[0:self.num_classes] = y_false
            target_weight[0:self.num_classes] = 1
            target_cls[cls_idx] = y_true
            #target_weight[cls_idx] = 1




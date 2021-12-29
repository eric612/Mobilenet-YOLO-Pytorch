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
import cv2
        
class SegLoss(nn.Module):
    class sigmoid(Function):
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
    def __init__(self,num_classes):
        super(SegLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = nn.Threshold(0.5, 0.)
        return

    
    def weighted_mse_loss(self,input, target, weights):
        out = (input - target)**2      
        total = torch.sum(weights)
        out = out * weights / total
        # expand_as because weights are prob not defined for mini-batch        
        loss = torch.sum(out) 
        #print(loss)
        return loss



    def forward(self, input, targets=None):
        if targets is not None:
            truth = targets.clone().to(device)
            truth = truth.permute(0,3,1,2)
            #print(input.shape,truth.shape)
            #.to(device)
            #print(truth)
            #print(truth>0.1)
            output = self.sigmoid.apply(input)
            #result = output[0,0,...]
            #print(result.shape)
            #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('frame', 640, 480)        
            #cv2.imshow('frame', result.cpu().detach().numpy())
            #key = cv2.waitKey(1) 
            obj = torch.masked_select(output, truth>=0.5)
            no_obj = torch.masked_select(output, truth<0.5)
            #mask_truth = torch.masked_select(truth, truth>=0.3)
            #threshold = torch.tensor([0.3]).to(device)
            #results = (truth>threshold).float()*1
            #results = obj + no_obj*truth
            #print(results)
            #print(torch.mean(output))
            weights = torch.ones_like(input).to(device)
            loss = self.weighted_mse_loss(output , truth , weights)
            #print(loss)
            return loss*0.05,torch.mean(obj).item(),torch.mean(no_obj).item()
        else:
            output = self.sigmoid.apply(input)
            result = output[0,...].cpu().detach().numpy()      
            return result






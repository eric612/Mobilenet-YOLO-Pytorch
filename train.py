import argparse
import os
import random
import shutil
import time
import warnings
from progress.bar import (Bar, IncrementalBar)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import folder2lmdb
import cv2
from models.voc.mbv2_yolo import yolo
from models.voc.yolo_loss import *
from utils import Bar, Logger, AverageMeter
from utils.eval_mAP import *
from pprint import PrettyPrinter
import yaml
pp = PrettyPrinter()
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate') 
parser.add_argument('--warm-up', '--warmup',  default=[], type=float,
                    metavar='warmup', help='warm up learning rate')                    
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--schedule', type=int, nargs='+', default=[125,200,250],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#parser.add_argument('-e', '--evaluate', type=bool, default=False, help='Evaluate mAP? default=False')   
random.seed(10992)
                              
def main():

    with open('models/voc/config.yaml', 'r') as f:
        config = yaml.load(f) 
    with open('data/voc_data.yaml', 'r') as f:
        dataset_path = yaml.load(f)         
    print(config)
    best_acc = 0  # best test accuracy
    args = parser.parse_args()
    start_epoch = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_folder = folder2lmdb.ImageFolderLMDB                                 
    train_dataset = image_folder(
        db_path=dataset_path["trainval_dataset_path"]["lmdb"],
        transform_size=config["train_img_size"],
        phase='train'
    )       
    test_dataset = image_folder(
        db_path=dataset_path["test_dataset_path"]["lmdb"],
        transform_size=[[config["img_w"],config["img_h"]]],
        phase='test'
    )    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, config["batch_size"], shuffle=False,
        num_workers=4, pin_memory=True,collate_fn=test_dataset.collate_fn) 
    model = yolo(config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.cuda()
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()

    params = model.parameters()
    optimizer = optim.AdamW(params=params,lr = args.lr)   
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)    
    title = 'voc-training-process'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        print(args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.yolo_losses[0].val_conf = checkpoint['conf'] 
        model.yolo_losses[1].val_conf = checkpoint['conf'] 
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = args.lr
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch    ', 'Loss     ', 'Precision ', 'Time      ', 'IOU      ', 'Learning Rate'])
    test_acc = 0 
    if args.evaluate:
        for epoch in range(1):
            test_acc = test(test_loader, model, optimizer, epoch , config)
        return
        
    #ls = len(args.warm_up)
    for epoch in range(start_epoch, args.epochs):
        if epoch in args.warm_up:
            adjust_learning_rate(optimizer, 0.1)
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch   
        if epoch in args.warm_up: 
            adjust_learning_rate(optimizer, 10)
        if epoch in args.schedule:
            #load_best_checkpoint(model=model, save_path=args.save_path)
           
            save_checkpoint({
                    'epoch': epoch ,
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'conf' : model.yolo_losses[0].val_conf,
                }, False,model, checkpoint=args.checkpoint,filename='epoch%d_checkpoint.pth.tar'%epoch) 
            adjust_learning_rate(optimizer, 0.5)
            print('adjusted to current lr: '
                  '{}'.format([param_group['lr'] for param_group in optimizer.param_groups]))  
            
        log = False
        if epoch%2 == 0 :
            log = True
            st = time.time()
            print('\nEpoch: [%3d | %3d] LR: %f        | loss   | cnt | iou   | obj   | no_obj | class | recall | cnt2 | iou2  | obj2  | no_obj2 | class2 | recall2 |' \
                    % (epoch, args.epochs, optimizer.param_groups[0]['lr']))
        
        train_loss,iou = train(train_loader, model, optimizer, epoch)
        
        if not log :
            test_acc = test(test_loader, model, optimizer, epoch , config)  
            logger.append([epoch + 1, train_loss , test_acc, time.time()-st,iou, optimizer.param_groups[0]['lr']])
            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc) 
            save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                    'conf' : model.yolo_losses[0].val_conf,
                }, is_best,model, checkpoint=args.checkpoint)
            
        
def train(train_loader, model, optimizer,epoch):
    model.train()
    bar = IncrementalBar('Training', max=len(train_loader),width=12)
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    recall = [AverageMeter(),AverageMeter()]
    iou = [AverageMeter(),AverageMeter()]
    obj = [AverageMeter(),AverageMeter()]
    no_obj = [AverageMeter(),AverageMeter()]
    conf_loss = [AverageMeter(),AverageMeter()]
    cls_loss = [AverageMeter(),AverageMeter()]
    cls_score = [AverageMeter(),AverageMeter()]
    count = [AverageMeter(),AverageMeter()]
    #end = time.time()
    for batch_idx, (images,targets) in enumerate(train_loader):
        #print(targets)
        #data_time.update(time.time() - end)
        bs = images.size(0)
        #print(images.shape)
        #print(i,targets[0])
        optimizer.zero_grad()
        images = images.to(device)  # (batch_size (N), 3, H, W)
        outputs = model(images,targets)
        #losses0 = yolo_losses[0](outputs[0],targets)
        #losses1 = yolo_losses[1](outputs[1],targets) 
        t_loss = list()
        for i,l in enumerate(outputs):
            #print(l[0])
            t_loss.append(l[0])  
            recall[i].update(l[1])
            iou[i].update(l[2])
            obj[i].update(l[3])
            no_obj[i].update(l[4])
            cls_score[i].update(l[5])
            count[i].update(l[6])
            #conf_loss.update(l[5])
            #cls_loss.update(l[6])
        loss = sum(t_loss)
        losses.update(loss.item(),bs)
        loss.backward()
        optimizer.step()
        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()     
        bar.suffix  = \
            '%(percent)3d%% | {total:} | {loss:.4f} | {cnt1:2.1f} | {iou1:.3f} | {obj1:.3f} | {no_obj1:.4f} | {cls1:.3f} | {rec1:.3f}  | {cnt2:2.1f}  | {iou2:.3f} | {obj2:.3f} | {no_obj2:.4f}  | {cls2:.3f}  | {rec2:.3f}   |'\
            .format(
            #batch=batch_idx + 1,
            #size=len(train_loader),
            #data=data_time.avg,
            #bt=batch_time.avg,
            total=bar.elapsed_td,
            loss=losses.avg,
            #loss1=losses[0].avg,
            #loss2=losses[1].avg,
            cnt1=(count[0].avg),
            cnt2=(count[1].avg),
            #recall=recall.avg,
            iou1=iou[0].avg,
            iou2=iou[1].avg,
            obj1=obj[0].avg,
            no_obj1=no_obj[0].avg,
            cls1=cls_score[0].avg,
            obj2=obj[1].avg,
            no_obj2=no_obj[1].avg,
            cls2=cls_score[1].avg,
            rec1=recall[0].avg,
            rec2=recall[1].avg,
            #cls=cls_loss.avg,
            )                

        bar.next()
    bar.finish()
    return losses.avg,(iou[0].avg+iou[1].avg)/2
    
def test(test_loader, model, optimizer,epoch , config):
    
    # switch to evaluate mode
    model.eval()
    n_classes = config['yolo']['classes'];
    
    end = time.time()
    #bar = Bar('Validating', max=len(test_loader))
    bar = IncrementalBar('Validating', max=len(test_loader),width=32)
    #for batch_idx, (inputs, targets) in enumerate(testloader):
    n_gt = [0]*n_classes
    correct = [0]*n_classes
    n_pred = [0]*n_classes
    n_iou = [0]*n_classes
    n_images = 0
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list() 
    gt_box = 0
    pred_box = 0

    for batch_idx, (images,targets) in enumerate(test_loader):
        images = images.to(device)  # (batch_size (N), 3, H, W)      
        labels = [torch.Tensor(l).to(device) for l in targets] 
        bs = len(labels)
        # compute output
        with torch.no_grad():
            detections = model(images)  # (N, num_defaultBoxes, 4), (N, num_defaultBoxes, n_classes)
            for sample_i in range(bs):
                
                # Get labels for sample where width is not zero (dummies)
                # print(len(labels[0]),labels[sample_i])
                target_sample = labels[sample_i]
                gt_box = gt_box + len(target_sample)
                tx1, tx2 = torch.unsqueeze((target_sample[...,1] - target_sample[...,3] / 2),1), torch.unsqueeze((target_sample[...,1] + target_sample[...,3] / 2),1)
                ty1, ty2 = torch.unsqueeze((target_sample[...,2] - target_sample[...,4] / 2),1), torch.unsqueeze((target_sample[...,2] + target_sample[...,4] / 2),1)
                box = torch.cat((tx1,ty1,tx2,ty2),1)
                size = target_sample.size(0)
 
                true_boxes.append(box)
                true_labels.append(target_sample[...,0])
                true_difficulties.append(torch.zeros(size, requires_grad=False))
                #print(detections[0][sample_i].shape,detections[1][sample_i].shape)
                preds = detections[sample_i]
                pred_box = pred_box + len(preds)
                if preds is not None:                                
                    det_boxes.append(preds[...,:4])
                    det_labels.append((preds[...,6]+1).to(device))
                    conf = (preds[...,4] * preds[...,5]).to(device)
                    det_scores.append(conf)
                else :
                    empty = torch.empty(0).to(device)
                    det_boxes.append(empty)
                    det_labels.append(empty)
                    det_scores.append(empty)
                
                n_images = n_images + 1  
            

        # measure elapsed time
        sum_gt = sum(n_gt)
        sum_n_pred= sum(n_pred)
        # plot progress
        bar.suffix  = '({batch}/{size}) | Total: {total:} | ETA: {eta:}| n_img: {n_img:} | gt_box: {gt_box:} | pred_box: {pred_box:}'.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),

                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    n_img=n_images,
                    gt_box=gt_box,
                    pred_box=pred_box
                    )
        bar.next()
    bar.finish()
    print("\nVal conf. is %f\n" % (model.yolo_losses[0].val_conf))
    model.yolo_losses[0].val_conf = adjust_confidence(gt_box,pred_box,model.yolo_losses[0].val_conf)
    model.yolo_losses[1].val_conf = adjust_confidence(gt_box,pred_box,model.yolo_losses[1].val_conf)
    
    # Calculate mAP
    APs, mAP, TP, FP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, n_classes=21)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    return mAP
def save_checkpoint(state, is_best,model, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    #save_onnx(filepath,model)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
def adjust_confidence(gt_box_num,pred_box_num,conf):
    if pred_box_num>gt_box_num*3 :
        conf = conf + 0.01
    elif pred_box_num<gt_box_num*2 and conf>0.01:
        conf = conf - 0.01
    
    return conf
def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("Change learning rate.\n The new LR is %f\n" % (optimizer.param_groups[0]['lr']))        
if __name__ == '__main__':
    main()

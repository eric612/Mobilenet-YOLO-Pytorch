# Mobilenet-YOLO-Pytorch

Like my previous project [Mobilenet-YOLO](https://github.com/eric612/MobileNet-YOLO) , the loss function was very simular to original implemention of darknet

![training process](/images/show.gif)

## Model

A pytorch implementation of MobileNet-YOLO detection network , train on 07+12 , test on VOC2007 (imagenet pretrained , not coco)

Network|mAP|Resolution|download|
:---:|:---:|:---:|:---:|
MobileNetV2|72.1|352|[checkpoint](https://drive.google.com/drive/folders/11iNLZA5sOZP2tiTQB6pz6TAA2u5xyYCa?usp=sharing)|


## Training steps

1. Download dataset VOCdevkit/ , if already have , please skip this step
```
sh scripts/VOC2007.sh
sh scripts/VOC2012.sh
``` 
2. Create lmdb
```
 sh scripts/create.sh 
``` 
3. Start training
```
sh scripts/train.sh 
```  
## yolov3 training 

see [branch](https://github.com/eric612/Mobilenet-YOLO-Pytorch/tree/yolov3)

## Hyper parameter optimization 

```
nnictl create --config config.yml
```

## Demo

Download  checkpoint, and save at $Mobilenet-YOLO-Pytorch/checkpoint/

```
sh scripts/inference.sh 
``` 

## Under construction

- [ ] A new detector
- [ ] yolov4,yolov5 ...
- [ ] Multi-Task 
- [ ] Hyper Parameter Tuning
- [ ] Prning 
- [ ] Porting (KL520 , ncnn , caffe , ...)

## Acknowledgements

[AlexeyAB](https://github.com/AlexeyAB/darknet)

[diggerdu](https://github.com/diggerdu/Generalized-Intersection-over-Union)

[BobLiu20](https://github.com/BobLiu20/YOLOv3_PyTorch)

[bubbliiiing](https://github.com/bubbliiiing/yolov4-tiny-pytorch)

[aleju](https://github.com/aleju/imgaug)

[rmccorm4](https://github.com/rmccorm4/PyTorch-LMDB)

[hysts](https://github.com/hysts/pytorch_image_classification)

[utkuozbulak](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)

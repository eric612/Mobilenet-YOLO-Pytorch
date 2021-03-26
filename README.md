# Mobilenet-YOLO-Pytorch

Include mobilenet series (v1,v2,v3...) and yolo series (yolov3,yolov4,...)

The first version will be a pure yolov3 model like my previous project [Mobilenet-YOLO](https://github.com/eric612/MobileNet-YOLO) 

## Model

A pytorch implementation of MobileNet-YOLO detection network , train on 07+12 , test on VOC2007 (imagenet pretrained , not coco)

Network|mAP|Resolution|yolov3|yolov4|
:---:|:---:|:---:|:---:|:---:|
MobileNetV2|71.2|352|✓| |
MobileNetV2| |352| |✓|
MobileNetV3| | | | |
MobileNetV3| | | | |

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
## Demo

Download [checkpoint](https://drive.google.com/file/d/1PPfmv5aHz014jBiKiH2hL-YAQDOrm2hx/view?usp=sharing) , and save at $Mobilenet-YOLO-Pytorch/checkpoint/

```
sh scripts/inference.sh 
``` 

## Under construction

- [ ] A new detector
- [ ] yolov4,yolov5 ...
- [ ] Multi-Task 
- [ ] Hyper Parameter Tuning
- [ ] Prning 
- [ ] Porting (KL520 , ncnn , ...)

## Acknowledgements

[AlexeyAB](https://github.com/AlexeyAB/darknet)

[diggerdu](https://github.com/diggerdu/Generalized-Intersection-over-Union)

[BobLiu20](https://github.com/BobLiu20/YOLOv3_PyTorch)

[bubbliiiing](https://github.com/bubbliiiing/yolov4-tiny-pytorch)

[aleju](https://github.com/aleju/imgaug)

[rmccorm4](https://github.com/rmccorm4/PyTorch-LMDB)

[hysts](https://github.com/hysts/pytorch_image_classification)

[utkuozbulak](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)
# Mobilenet-YOLO-Pytorch

Include mobilenet series (v1,v2,v3...) and yolo series (yolov3,yolov4,...)

The first version will be a pure yolov3 model like my previous project [Mobilenet-YOLO](https://github.com/eric612/MobileNet-YOLO) 

## Model

A caffe implementation of MobileNet-YOLO detection network , train on 07+12 , test on VOC2007 (imagenet pretrained , not coco)

Network|mAP|Resolution|yolov3|yolov4|
:---:|:---:|:---:|:---:|:---:|
MobileNetV2|70.x|352|✓| |
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

## Under construction

- [ ] A new detector
- [ ] yolov4,yolov5 ...
- [ ] Multi-Task 
- [ ] Hyper Parameter Tuning
- [ ] Prning 
- [ ] Porting (KL520 , ncnn , ...)

## Acknowledgements

[YOLOv3_PyTorch](https://github.com/BobLiu20/YOLOv3_PyTorch)

[yolov4-tiny-pytorch](https://github.com/bubbliiiing/yolov4-tiny-pytorch)

[imgaug](https://github.com/aleju/imgaug)

[PyTorch-LMDB](https://github.com/rmccorm4/PyTorch-LMDB)

[pytorch_image_classification](https://github.com/hysts/pytorch_image_classification)

[pytorch-custom-dataset-examples](https://github.com/utkuozbulak/pytorch-custom-dataset-examples)

# Towards-Realtime-MOT
**NEWS:** 
- **[2019.10.11]** Training and evaluation data uploaded! Please see [DATASET_ZOO.md](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) for details.
- **[2019.10.01]** Demo code and pre-trained model released!

## Introduction
This repo is the a codebase of the Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. Techical details are described in our [arXiv preprint paper](https://arxiv.org/pdf/1909.12605v1.pdf). By using this repo, you can simply achieve **MOTA 64%+** on the "private" protocol of [MOT-16 challenge](https://motchallenge.net/tracker/JDE), and with a near real-time speed at **18~24 FPS** (Note this speed is for the entire system, including the detection step! ) .

We hope this repo will help researches/engineers to develop more practical MOT systems. For algorithm development, we provide training data, baseline models and evaluation methods to make a level playground. For application usage, we also provide a small video demo that takes raw videos as input without any bells and whistles.

## Requirements
* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.2.0 
* python-opencv
* [py-motmetrics](https://github.com/cheind/py-motmetrics) (`pip install motmetrics`)
* cython-bbox (`pip install cython_bbox`)
* (Optional) ffmpeg (used in the video demo)
* (Optional) [syncbn](https://github.com/ytoon/Synchronized-BatchNorm-PyTorch) (compile and place it under utils/syncbn, or simply replace with nn.BatchNorm [here](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/models.py#L12))
* ~~[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) (Their GPU NMS is used in this project)~~


## Video Demo
<img src="assets/MOT16-03.gif" width="400"/>   <img src="assets/MOT16-14.gif" width="400"/>
<img src="assets/IMG_0055.gif" width="400"/>   <img src="assets/000011-00001.gif" width="400"/>

Usage:
```
python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root
```

## docker demo example
```bash
docker build -t towards-realtime-mot docker/

docker run --rm --gpus all -v $(pwd)/:/Towards-Realtime-MOT -ti towards-realtime-mot /bin/bash
cd /Towards-Realtime-MOT;
python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root

```

## Dataset zoo
Please see [DATASET_ZOO.md](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) for detailed description of the training/evaluation datasets.
## Pretrained model and baseline models
Darknet-53 ImageNet pretrained: [[DarkNet Official]](https://pjreddie.com/media/files/darknet53.conv.74)

JDE-1088x608-uncertainty: [[Google Drive]](https://drive.google.com/open?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA) [[Baidu NetDisk]](https://pan.baidu.com/s/1Ifgn0Y_JZE65_qSrQM2l-Q)
## Test on MOT-16 Challenge

## Training instruction
- Download the training datasets.  
- Edit `cfg/ccmcpe.json`, config the training/validation combinations. A dataset is represented by an image list, please see `data/*.train` for example. 
- Run the training script:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py 
```

We use 8x Nvidia Titan Xp to train the model, with a batch size of 32. You can adjust the batch size (and the learning rate together) according to how many GPUs your have. You can also train with smaller image size, which will bring faster inference time. But note the image size had better to be multiples of 32 (the down-sampling rate).

### Train with custom datasets
Adding custom datsets is quite simple, all you need to do is to organize your annotation files in the same format as in our training sets. Please refer to [DATASET_ZOO.md](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) for the dataset format. 


## Acknowledgement
A large portion of code is borrowed from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) and [longcw/MOTDT](https://github.com/longcw/MOTDT), many thanks to their wonderful work!

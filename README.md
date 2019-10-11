# Towards-Realtime-MOT
**NEWS:** 
- **[2019.10.11]** Training and evaluation data uploaded! Please see [DATASET_ZOO.MD](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.MD) for details.
- **[2019.10.01]** Demo code and pre-trained model released!

## Introduction
This repo is the a codebase of the Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. Techical details are described in our [arXiv preprint paper](https://arxiv.org/pdf/1909.12605v1.pdf). By using this repo, you can simply achieve **MOTA 64%+** on the "private" protocol of [MOT-16 challenge](https://motchallenge.net/tracker/JDE), and with a near real-time speed at **18~24 FPS** (Note this speed is for the entire system, including the detection step! ) .

We hope this repo will help researches/engineers to develop more practical MOT systems. For algorithm development, we provide training data, baseline models and evaluation methods to make a level playground. For application usage, we also provide a small video demo that takes raw videos as input without any bells and whistles.

## Requirements
* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.0.1
* [syncbn](https://github.com/ytoon/Synchronized-BatchNorm-PyTorch) (Optional, compile and place it under utils/syncbn, or simply replace with nn.BatchNorm [here](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/models.py#L12))
* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) (Their GPU NMS is used in this project)
* python-opencv
* ffmpeg (Optional, used in the video demo)
* [py-motmetrics](https://github.com/cheind/py-motmetrics) (Simply `pip install motmetrics`)

## Video Demo
<img src="assets/MOT16-03.gif" width="400"/>   <img src="assets/MOT16-14.gif" width="400"/>
<img src="assets/IMG_0055.gif" width="400"/>   <img src="assets/000011-00001.gif" width="400"/>

Usage:
```
python demo.py --input-video path/to/your/input/video --weights path/to/model/weights
               --output-format video --output-root path/to/output/root
```
## Dataset zoo
Please see [DATASET_ZOO.MD](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.MD) for detailed description of the training/evaluation datasets.
## Pretrained model and baseline models
Darknet-53 ImageNet pretrained: [[DarkNet Official]](https://pjreddie.com/media/files/darknet53.conv.74)

JDE-1088x608-uncertainty: [[Google Drive]](https://drive.google.com/open?id=1nlnuYfGNuHWZztQHXwVZSL_FvfE551pA) [[Baidu NetDisk]](https://pan.baidu.com/s/1Ifgn0Y_JZE65_qSrQM2l-Q)
## Test on MOT-16 Challenge

## Training instruction

### Train with custom datasets



## Acknowledgement
A large portion of code is borrowed from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) and [longcw/MOTDT](https://github.com/longcw/MOTDT), many thanks to their wonderful work!

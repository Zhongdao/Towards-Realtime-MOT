"""Demo file for running the JDE tracker on custom video sequences for pedestrian tracking. 

This file is the entry point to running the tracker on custom video sequences. It loads images from the provided video sequence, uses the JDE tracker for inference and outputs the video with bounding boxes indicating pedestrians. The bounding boxes also have associated ids (shown in different colours) to keep track of the movement of each individual. 

Examples:
        $ python demo.py --input-video path/to/your/input/video --weights path/to/model/weights --output-root path/to/output/root


Attributes:
    input-video (str): Path to the input video for tracking.
    output-root (str): Output root path. default='results'
    weights (str): Path from which to load the model weights. default='weights/latest.pt'
    cfg (str): Path to the cfg file describing the model. default='cfg/yolov3.cfg'
    iou-thres (float): IOU threshold for object to be classified as detected. default=0.5
    conf-thres (float): Confidence threshold for detection to be classified as object. default=0.5
    nms-thres (float): IOU threshold for performing non-max supression. default=0.4
    min-box-area (float): Filter out boxes smaller than this area from detections. default=200
    track-buffer (int): Size of the tracking buffer. default=30
    output-format (str): Expected output format, can be video, or text. default='video'
    

Todo:
    * Add compatibility for non-GPU machines (would run slow)
    * More documentation
"""

import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.utils import *
from utils.io import read_results
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
from utils.parse_config import parse_model_cfg
import utils.datasets as datasets
import torch
from track import eval_seq


logger.setLevel(logging.INFO)

def track(opt):    
    result_root = opt.output_root if opt.output_root!='' else '.'
    mkdir_if_missing(result_root)

    cfg_dict = parse_model_cfg(opt.cfg)
    opt.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]

    # run tracking
    timer = Timer()
    accs = []
    n_frame = 0

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate 

    frame_dir = None if opt.output_format=='text' else osp.join(result_root, 'frame')
    try:
        eval_seq(opt, dataloader, 'mot', result_filename,
                 save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
    except Exception as e:
        logger.info(e)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'result.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='demo.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    track(opt)


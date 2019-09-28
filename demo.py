import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm

from tracker.multitracker import JDETracker
from utils import visualization as vis
from utils.utils import *
from utils.log import logger
from utils.timer import Timer
from utils.evaluation import Evaluator
import utils.datasets as datasets
import torch
from track import eval_seq


def track(opt):
    logger.setLevel(logging.INFO)
    result_root = opt.output_root if opt.output_root!='' else '.'
    mkdir_if_missing(result_root)

    # run tracking
    timer = Timer()
    accs = []
    n_frame = 0

    logger.info('start tracking...')
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
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--img-size', type=int, default=(1088, 608), help='size of each image dimension')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', help='expected output format, can be video, tracklet or text')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    track(opt)


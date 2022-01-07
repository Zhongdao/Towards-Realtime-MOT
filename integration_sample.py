from tracker.multitracker import JDETracker
import cv2
import matplotlib.pyplot as plt
import torch
from utils.parse_config import parse_model_cfg
from utils import visualization as vis
import numpy as np
import argparse
import time
import collections
import itertools

parser = argparse.ArgumentParser()
# parser.add_argument('arggg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument('--cfg', type=str, default='cfg/yolov3_576x320.cfg', help='cfg file path')
parser.add_argument('--weights', type=str, default='jde_576x320_uncertainty.pt', help='path to weights file')
parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
parser.add_argument('--test-mot16', action='store_true', help='tracking buffer')
parser.add_argument('--save-images', action='store_true', help='save tracking results (image)')
parser.add_argument('--save-videos', action='store_true', help='save tracking results (video)')
parser.add_argument('-f', action='store_true', help='Internal jupyter stuff')
opt = parser.parse_args()
opt

cfg_dict = parse_model_cfg(opt.cfg)
opt.img_size = (int(cfg_dict[0]['width']), int(cfg_dict[0]['height']))

tracker = JDETracker(opt)



class FPS:
    def __init__(self,avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)
    def __call__(self):
        self.frametimestamps.append(time.time())
        if(len(self.frametimestamps) > 1):
            return len(self.frametimestamps)/(self.frametimestamps[-1]-self.frametimestamps[0])
        else:
            return 0.0




def frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        cv2.waitKey(1)
        if success:
            yield frame
        else:
            break

fps = FPS()
for frame in frames("../videos/11.mp4"):
    frame = cv2.resize(frame,opt.img_size)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = torch.from_numpy(img).cuda()
    blob = blob.permute(2, 0, 1)[None]/255

    online_targets = tracker.update(blob, frame)

    online_tlwhs = [t.tlwh for t in online_targets]
    online_ids = [t.track_id for t in online_targets]
    online_im = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=2,
                                  fps=fps())
    cv2.imshow("window", online_im)

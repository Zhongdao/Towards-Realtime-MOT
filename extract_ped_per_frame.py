import argparse
import json
import time
from pathlib import Path

from sklearn import metrics
from scipy import interpolate
import torch.nn.functional as F
from models import *
from utils.utils import *
from torchvision.transforms import transforms as T
from utils.datasets import LoadImages, JointDataset, collate_fn

def extract_ped_per_frame(
        cfg,
        input_root,
        output_root,
        weights,
        batch_size=16,
        img_size=416,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
        nID=14455,
):
    mkdir_if_missing(output_root)
    
    # Initialize model
    model = Darknet(cfg, img_size, nID)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'], strict=False)
    else:  # darknet format
        load_darknet_weights(model, weights)

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    vlist = os.listdir(input_root)
    vlist = [osp.join(input_root, v, 'img1') for v in vlist]

    for vpath in vlist:
        vroot = osp.join('/',*vpath.split('/')[:-1])
        out_vroot = vroot.replace(input_root, output_root)
        mkdir_if_missing(out_vroot)
        dataloader = LoadImages(vpath, img_size)
        for frame_id, (frame_path, frame, frame_ori) in enumerate(dataloader):
            frame_ground_id = frame_path.split('/')[-1].split('.')[0]
            if frame_id % 20 == 0:
                print('Processing frame {} of video {}'.format(frame_id, frame_path))
            blob = torch.from_numpy(frame).cuda().unsqueeze(0)
            pred = model(blob)
            pred = pred[pred[:,:,4] > conf_thres]
            if len(pred) > 0:
                dets = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0].cpu()
                scale_coords(img_size, dets[:, :4], frame_ori.shape).round()
                frame_dir = osp.join(out_vroot, frame_ground_id)
                mkdir_if_missing(frame_dir)
                dets = dets[:, :5]
            
                for ped_id, det in enumerate(dets):
                    box = det[:4].int()
                    conf = det[4]
                    ped = frame_ori[box[1]:box[3], box[0]:box[2]]
                    ped_path = osp.join(frame_dir, ('{:04d}_'+ '{:d}_'*4 + '{:.2f}.jpg').format(ped_id, *box, conf))
                    cv2.imwrite(ped_path, ped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=40, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/mot_64/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--img-size', type=int, default=(1088, 608), help='size of each image dimension')
    parser.add_argument('--print-interval', type=int, default=10, help='size of each image dimension')
    parser.add_argument('--input-root', type=str, default='/home/wangzd/datasets/youtube/data/0004/frame', help='path to input frames')
    parser.add_argument('--output-root', type=str, default='/home/wangzd/datasets/youtube/data/0004/ped_per_frame', help='path to output frames')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        extract_ped_per_frame(
            opt.cfg,
            opt.input_root,
            opt.output_root,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.iou_thres,
            opt.conf_thres,
            opt.nms_thres,
            opt.print_interval,
        )


import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch

from lib.utils.log import logger
from lib.tracker import matching
from lib.utils.kalman_filter import KalmanFilter
from lib.model.faster_rcnn.resnet import resnet_deploy
from lib.model.utils.config import cfg
from lib.model.rpn.bbox_transform import clip_boxes, bbox_transform_inv
from lib.model.nms.nms_wrapper import nms

from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):

    def __init__(self, tlwh, score, temp_feat):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.temp_feat = temp_feat

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.time_since_update = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self._tlwh = new_track.tlwh
        self.temp_feat = new_track.temp_feat
        self.time_since_update = 0
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.time_since_update = 0
        self.tracklet_len += 1

        self._tlwh = new_track.tlwh
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.temp_feat = new_track.temp_feat

    @property
    @jit
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        return self._tlwh.copy()

    @property
    @jit
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    @jit
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    @jit
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDTracker(object):
    def __init__(self, checksession=3, checkepoch=24, checkpoint=663, det_thresh=0.92, frame_rate=30):
        self.classes = np.asarray(['__background__', 'pedestrian'])

        self.fasterRCNN = resnet_deploy(self.classes, 101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()

        input_dir = osp.join('models', 'res101', 'mot17det')
        if not os.path.exists(input_dir):
            raise Exception('There is no input directory for loading network from ' + input_dir)
        load_name = os.path.join(input_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(checksession, checkepoch, checkpoint))
        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
        print('load model successfully!')
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()

        self.frame_id = 0
        self.det_thresh = det_thresh
        self.buffer_size = int(frame_rate / 30.0 * cfg.TRACKING_BUFFER_SIZE)
        self.max_time_lost = self.buffer_size
        #self.fmap_buffer = deque([], maxlen=self.buffer_size)

    def update(self, im_blob):
        self.frame_id += 1

        '''Forward'''
        im_blob = im_blob.cuda()
        im_info = torch.Tensor([[im_blob.shape[1], im_blob.shape[2], 1, ],]).float().cuda()
        self.im_info = im_info
        boxes, temp_feat, base_feat = self.predict(im_blob, im_info)

        '''Detections'''
        detections = [STrack(STrack.tlbr_to_tlwh((t, l, b, r)), s, f) for (t, l, b, r, s), f in zip(boxes, temp_feat)]


        return detections

    def predict(self, im_blob, im_info):
        im_blob = im_blob.permute(0,3,1,2)
        # Trivial input
        gt_boxes = torch.zeros(1, 1, 6).to(im_blob)
        num_boxes = gt_boxes[:, :, 0].squeeze()
        with torch.no_grad():
            rois, cls_prob, bbox_pred, base_feat = self.fasterRCNN(im_blob, im_info, gt_boxes, num_boxes)
        scores = cls_prob.data
        inds_first = torch.nonzero(scores[0, :, 1] > self.det_thresh).view(-1)
        if inds_first.numel() > 0:
            rois = rois[:, inds_first]
            scores = scores[:,inds_first]
            bbox_pred = bbox_pred[:, inds_first]

            refined_rois = self.fasterRCNN.bbox_refine(rois, bbox_pred, im_info)
            template_feat = self.fasterRCNN.roi_pool(base_feat, refined_rois)
            pred_boxes = refined_rois.data[:, :, 1:5]

            cls_scores = scores[0, :, 1]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[0]
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            temp_feat = template_feat[order]
            keep_first = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS).view(-1).long()
            cls_dets = cls_dets[keep_first]
            temp_feat = temp_feat[keep_first]
            output_box = cls_dets.cpu().numpy()
        else:
            output_box = []
            temp_feat = []
        return output_box, temp_feat, base_feat


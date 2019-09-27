import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch

from utils.utils import *
from utils.log import logger
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)

    def update_features(self, feat):
        print(1)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = 0.9 *self.smooth_feat + 0.1 * feat
        self.features.append(temp_feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)  


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
        #self.features.append(new_track.curr_feat)
        self.update_features(new_track.curr_feat)
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
            #self.features.append( new_track.curr_feat)
            self.update_features(new_track.curr_feat)

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

class IOUTracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt 
        self.model = Darknet(opt.cfg, opt.img_size, nID=14455)
        #load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model']) 
        self.model.cuda().eval()
        
        self.tracked_stracks = []   # type: list[STrack]
        self.lost_stracks = []      # type: list[STrack]
        self.removed_stracks = []   # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        #self.fmap_buffer = deque([], maxlen=self.buffer_size)

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        t1 = time.time()
        '''Forward'''
        with torch.no_grad():
            pred = self.model(im_blob)
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0]
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh((t, l, b, r)), s, None) for (t, l, b, r, s) in dets[:, :5]]
        else:
            detections = []

        t2 = time.time()
        #print('Forward: {} s'.format(t2-t1))


        '''matching for tracked targets'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        #dists = self.track_matching(strack_pool, detections, base_feat)
        dists = matching.iou_distance(strack_pool, detections)
        #dists[np.where(iou_dists>0.4)] = 1.0
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        t3 = time.time()
        #print('First match {} s'.format(t3-t2))

        #'''Remained det/track, use IOU between dets and tracks to associate directly'''
        #detections = [detections[i] for i in u_detection]
        #r_tracked_stracks = [strack_pool[i] for i in u_track ]
        #dists = matching.iou_distance(r_tracked_stracks, detections)
        #matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        #for itracked, idet in matches:
        #    r_tracked_stracks[itracked].update(detections[idet], self.frame_id)
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """step 4: init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """step 6: update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        t4 = time.time()
        #print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks) 
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        #self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        t5 = time.time()
        #print('Final {} s'.format(t5-t4))
        return output_stracks


class AETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.model = Darknet(opt.cfg, opt.img_size, nID=14455)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'])
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        t1 = time.time()
        '''Forward'''
        with torch.no_grad():
            pred = self.model(im_blob)
        pred = pred[pred[:, :, 4] > self.opt.conf_thres]
        if len(pred) > 0:
            dets = non_max_suppression(pred.unsqueeze(0), self.opt.conf_thres, self.opt.nms_thres)[0].cpu()
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30) for
                          (tlbrs, f) in zip(dets[:, :5], dets[:, -self.model.emb_dim:])]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        '''matching for tracked targets'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)


        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        #strack_pool = tracked_stracks
        dists = matching.embedding_distance(strack_pool, detections)
        iou_dists = matching.iou_distance(strack_pool, detections)
        dists[np.where(iou_dists>0.99)] = 1.0
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)


        # detections = [detections[i] for i in u_detection]
        # dists = matching.embedding_distance(self.lost_stracks, detections)
        # iou_dists = matching.iou_distance(self.lost_stracks, detections)
        # dists[np.where(iou_dists>0.7)] = 1.0
        #
        # matches, u_track_lost, u_detection = matching.linear_assignment(dists, thresh=0.7)
        #
        # for itracked, idet in matches:
        #     track = self.lost_stracks[itracked]
        #     det = detections[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(detections[idet], self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)



        '''Remained det/track, use IOU between dets and tracks to associate directly'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked ]
        r_lost_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state!=TrackState.Tracked ]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # '''Remained det/track, use IOU between dets and tracks to associate directly'''
        # detections = [detections[i] for i in u_detection]
        # dists = matching.iou_distance(r_lost_stracks, detections)
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.25)
        #
        # for itracked, idet in matches:
        #     track = r_lost_stracks[itracked]
        #     det = detections[idet]
        #     if track.state == TrackState.Tracked:
        #         track.update(det, self.frame_id)
        #         activated_starcks.append(track)
        #     else:
        #         track.re_activate(det, self.frame_id, new_id=False)
        #         refind_stracks.append(track)
        #
        # for it in u_track:
        #     track = r_lost_stracks[it]
        #     if not track.state == TrackState.Lost:
        #         track.mark_lost()
        #         lost_stracks.append(track)



        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """step 4: init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """step 6: update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        t4 = time.time()
        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        t5 = time.time()
        # print('Final {} s'.format(t5-t4))
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            


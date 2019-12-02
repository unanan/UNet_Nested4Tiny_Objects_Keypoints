# * heatmap.py:
#
# *
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
import logging
from collections import Counter
import numpy as np
import numpy.matlib
import cv2
import torch

# TODO: Need to write in GPU Mode
class HeatmapPattern():
    '''
    Base Class of Heatmap
    '''
    def __init__(self, pattern, w, h, radius = 3, match_method = 'match_distmin'):
        '''
        :param pattern: Heatmap pattern, must be list with 'list' elements.
        e.g. [[0,1,3],[2,4],[5]] - means create 3 heatmaps which there draws keypoints index-0,1,3 on 1st heatmap,
        keypoints index-2,4 on 2nd, keypoint index-5 on 3rd.
        :param w: target width of output heatmap
        :param h: target height of output heatmap
        :param radius: the dot(s)' radius on the heatmap(s)
        '''
        if not isinstance(pattern, list):
            raise TypeError("'pattern' must be list.")
        if len(pattern)>4:
            logging.warning("Elements of 'pattern' is suggested to be less than 4.")
        for ele in pattern:
            if not isinstance(ele, list):
                raise TypeError("'pattern' must be list with elements as type 'list'. "
                                "e.g. [[0,1,3],[2,4],[5]] - means create 3 heatmaps which there draws "
                                "keypoints index-0,1,3 on 1st heatmap, keypoints index-2,4 on 2nd heatmap, "
                                "keypoint index-5 on 3rd heatmap.")
        flatten_pattern = sum(pattern,[])
        if len(set(flatten_pattern))!=len(flatten_pattern):
            raise ValueError("Number in 'pattern' must not repeat.")

        if not isinstance(w, int) or not isinstance(h, int):
            raise TypeError("'w'& 'h' must be int")

        if not isinstance(match_method, str):
            raise TypeError("'match_method' must be str")

        self.pattern = pattern
        self.w = w
        self.h = h
        self.radius = radius
        self.matchmethod = getattr(self,match_method)


    def match_distmin(self, predpoints, targets, index_list):
        '''
        - Create the sequence of predicted points by calculating the minimum distance
        :param predpoints: prediced points of a heatmap
        :param targets: all the keypoints of specific sequence of a image
        :param index_list: keypoints index list on this heatmap
        :return: output_points: re-arranged sequence of the predicted points
        '''
        target_points = [targets[idx] for idx in index_list]
        points = []  #e.g.[[(0, [100, 120]), (1, [200, 110]), (2, [60,30])],...]
        output_points = []  #TODO

        for tarpt in target_points:
            points.append(sorted(
                enumerate(predpoints),
                key=lambda x: (x[1][0]-tarpt[0])**2+(x[1][1]-tarpt[1])**2,
                reverse=False))

        #TODO:Need to finish the method
        for index, count in Counter([row[0][0] for row in points]).items():
            if count>1:
                pass
        return output_points


    def create_heatmap(self, targets):
        '''
        - Create heatmap
        :param targets: with shape [N, C, 2], '2' means [x,y]
        :return: heatmaps with shape [N, C, H, W]
        '''
        pass


    def transfer_points(self, preds, targets):
        pass



class Heatmap(HeatmapPattern):
    '''
    Heatmap Class
    '''
    def region_segment_(self, pred):
        mask = cv2.medianBlur(pred, 3).copy()

        mask[mask > 0] = 255.0

        mask = np.uint8(mask)
        opening = mask.copy()
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        sure_bg = cv2.dilate(opening, np.ones((3, 3), np.uint8), iterations=2)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)

        ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        markers = cv2.watershed(mask[:, :, np.newaxis].repeat([3], axis=2), markers)

        out_markers_dict = []  # element are tuples
        for marker in np.unique(markers):
            if marker < 2:  # 从2开始才是单独的marker
                continue

            # markerpos = np.dstack(np.where(markers == marker))
            # if len(markerpos.shape) == 3:
            #     markerpos = markerpos[0, :, :]

            # markerpos = markerpos[markerpos[:, 0].argsort()]  # ltsort=0按y高度排序，ltsort=1按x高度排序

            # (lefttop(x,y)，开关状态，marker矩阵)
            out_markers_dict.append(markers == marker)

        return out_markers_dict


    # TODO: 改成计算最大能量区域的方法（把threshold去掉）
    def extract_points_(self, pred, num, threshold=0.5):
        assert len(pred.shape)==2, "Heatmap assertion failed. It should be [H, W]"

        # Notice that the target points coordinate values are resized with the resize of the picture in dataloader
        origin_height = self.h
        origin_width = self.w

        # cut off the value under the threshold
        heatm = pred.copy()
        heatm[heatm < threshold] = 0

        markers = self.region_segment_(heatm)

        heatm_list = []
        if len(markers) != 0:
            for marker in markers:
                heatm_ = heatm.copy()
                heatm_[~marker] = 0
                heatm_list.append([heatm_, np.max(heatm_)])
            heatm_list = sorted(heatm_list, key=lambda k: k[1], reverse=True)  # Sort as lightest order(high confidence)

            output_points = []
            for idx, item in enumerate(heatm_list[:num]):
                heatm_ = item[0]
                heatm_max_ = item[1]
                yx = np.where(heatm_ == heatm_max_)
                if len(yx[0]) != 0:
                    y = int(origin_height * int(yx[0][0]) / self.h)
                    x = int(origin_width * int(yx[1][0]) / self.w)

                    output_points.append([x, y])

            return output_points
        else:
            logging.warning("Heatmap: No markers after Region Segmentation. Retrying on lower threshold..")
            heatm = pred.copy()
            heatm[heatm < threshold * 0.9] = 0
            markers = self.region_segment_(heatm)
            heatm_list = []
            if len(markers) != 0:
                for marker in markers:
                    heatm_ = heatm.copy()
                    heatm_[~marker] = 0
                    heatm_list.append([heatm_, np.max(heatm_)])
                heatm_list = sorted(heatm_list, key=lambda k: k[1], reverse=True)

                output_points = []
                for idx, item in enumerate(heatm_list[:num]):
                    heatm_ = item[0]
                    heatm_max_ = item[1]
                    yx = np.where(heatm_ == heatm_max_)
                    if len(yx[0]) != 0:
                        y = int(origin_height * int(yx[0][0]) / self.h)
                        x = int(origin_width * int(yx[1][0]) / self.w)

                        output_points.append([x, y])

                return output_points
            else:
                logging.warning("OilGauge: Retry failed. No markers after Region Segmentation.")
            return []


    def create_heatmap(self,targets):
        '''
        - Create heatmap
        :param targets: with shape [N, C, 2], '2' means [x,y]
        :return: heatmaps with shape [N, C, H, W]
        '''

        # number of data in 'targets'
        N = targets.shape[0]

        # output target heatmap matrix is of shape [N, len(self.pattern), self.h, self.w] with element value 0.0~1.0
        target_matrix = np.zeros((N, len(self.pattern), self.h, self.w)).astype(np.float32)

        for n in range(N):
            for hmap_idx, hmap in enumerate(self.pattern):
                for pt_idx in hmap:
                    center_x = targets[n, pt_idx, 0]
                    center_y = targets[n, pt_idx, 1]

                    Gauss_map = np.sqrt((np.matlib.repmat(np.arange(self.w), self.h, 1) -
                                         np.matlib.repmat(center_x, self.h, self.w)) ** 2 +
                                        (np.transpose(np.matlib.repmat(np.arange(self.h), self.w, 1)) -
                                         np.matlib.repmat(center_y, self.h, self.w)) ** 2)
                    Gauss_map = np.exp(-0.5 * Gauss_map / self.radius)
                    target_matrix[n, hmap_idx] += Gauss_map
                target_matrix[n, hmap_idx] = target_matrix[n, hmap_idx] / np.max(target_matrix[n, hmap_idx])

        return target_matrix


    def transfer_points(self, preds, targets):
        '''
        - Tranfer the preds(heatmaps) to points(coordinates)
        :param preds: with shape [N, C, H, W]
        :param targets: with shape [N, S, 2], '2' means [x,y]
        :return: Tensor of predicted points
        '''
        assert len(preds.shape) == 4,               "preds shape should be [N, C, H, W]"
        assert len(targets.shape) == 3,             "targets shape should be [N, C, 2]"
        assert preds.shape[0] == targets.shape[0],  "1st dimension of preds and targets must be equal"
        assert targets.shape[2] == 2,               "3rd dimension of targets must be 2"

        # number of data
        N = preds.shape[0]

        output_points_tensor = []
        for n in range(N):
            output_points_tensor.append([])
            for hmap_idx, hmap in enumerate(self.pattern):
                points = self.extract_points_(preds[n, hmap_idx], len(hmap))
                points = self.matchmethod(points, targets[n], self.pattern[hmap_idx])
                output_points_tensor[-1].append(points)
        return torch.FloatTensor(output_points_tensor)


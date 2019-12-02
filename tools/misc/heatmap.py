# * heatmap.py:
#
# *
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
# from visdom import Visdom
# viz = Visdom(env='heatmap pattern')
import logging
import numpy as np
import numpy.matlib

# TODO: Need to write in GPU Mode
class HeatmapPattern():
    '''
    Base Class of Heatmap
    '''

    def __init__(self, pattern, w, h, radius = 3):
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

        self.pattern = pattern
        self.w = w
        self.h = h
        self.radius = radius


    def create_heatmap(self, targets):
        pass



class Heatmap(HeatmapPattern):

    def create_heatmap(self,targets):
        # target: N C 2
        # 2 means [x,y]
        N, C, _ = targets.shape

        target_matrix = np.zeros((N, 4, self.h, self.w)).astype(np.float32)  # 0.0~1.0

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




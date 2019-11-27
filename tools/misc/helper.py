# * Helper:
#   concluds some helper tool functions.
#
# * Test Status: Not tested
#

# -*- coding: utf-8 -*
import os
import numpy as np
import torch


class Save_Handle(object):
    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


def cal_cos(fea, gallery):
    fea = fea[np.newaxis, :]
    scores = np.sum(fea * gallery, axis=1)
    scores = scores / np.linalg.norm(gallery, axis=1) / np.linalg.norm(fea)
    return scores

# def cal_triplet():

def cal_result(query_feas, query_labels, gal_feas, gal_labels, level):
    acc_count = 0
    all_pure = 0.0
    for idx in range(len(query_feas)):
        fea = query_feas[idx]
        label = query_labels[idx]
        scores = cal_cos(fea, gal_feas)
        top_k_idx = np.argsort(scores)[::-1][:level]
        top_k_label = gal_labels[top_k_idx]
        if label == top_k_label[0]:
            acc_count += 1
        pure = 1.0 * np.sum(np.equal(label, top_k_label)) / level
        all_pure += pure

    acc = 1.0 * acc_count / len(query_feas)
    avg_pure = all_pure / len(query_feas)
    return acc, avg_pure



def create_heatmap(target, image_height, image_width):
    # target: N C 2  (C=5)
    N, C, _ = target.shape
    R_corner = 3
    R_pointer = 3
    target_matrix = np.zeros((N, 4, image_height, image_width)).astype(np.float32)  # 0.0~1.0

    for n in range(N):

        #region GearKnob
        #channel0,point0
        pt =0;ch=0
        center_x = target[n, pt, 0]
        center_y = target[n, pt, 1]

        Gauss_map = np.sqrt((np.matlib.repmat(np.arange(image_width), image_height, 1) -
                             np.matlib.repmat(center_x, image_height, image_width)) ** 2 +
                            (np.transpose(np.matlib.repmat(np.arange(image_height), image_width, 1)) -
                             np.matlib.repmat(center_y, image_height, image_width)) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        target_matrix[n, ch] = Gauss_map


        #channel1,point1~3
        ch = 1
        for pt in range(1,4): # 角点后三个点
            center_x = target[n, pt, 0]
            center_y = target[n, pt, 1]

            #
            Gauss_map = np.sqrt((np.matlib.repmat(np.arange(image_width), image_height, 1) -
                                 np.matlib.repmat(center_x, image_height, image_width)) ** 2 +
                                (np.transpose(np.matlib.repmat(np.arange(image_height), image_width, 1)) -
                                 np.matlib.repmat(center_y, image_height, image_width)) ** 2)
            Gauss_map = np.exp(-0.5 * Gauss_map / R_pointer)
            target_matrix[n, ch] += Gauss_map
        target_matrix[n, ch] = target_matrix[n, ch]/np.max(target_matrix[n, ch])


        # channel2,point4
        pt = 4;ch = 2 #指针头
        center_x = target[n, pt, 0]
        center_y = target[n, pt, 1]

        Gauss_map = np.sqrt((np.matlib.repmat(np.arange(image_width), image_height, 1) -
                             np.matlib.repmat(center_x, image_height, image_width)) ** 2 +
                            (np.transpose(np.matlib.repmat(np.arange(image_height), image_width, 1)) -
                             np.matlib.repmat(center_y, image_height, image_width)) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        target_matrix[n, ch] = Gauss_map


        # channel3,point5~6
        ch=3
        for pt in range(5,C): # 后2个点在一个channel
            center_x = target[n, pt, 0]
            center_y = target[n, pt, 1]

            #
            Gauss_map = np.sqrt((np.matlib.repmat(np.arange(image_width), image_height, 1) -
                                 np.matlib.repmat(center_x, image_height, image_width)) ** 2 +
                                (np.transpose(np.matlib.repmat(np.arange(image_height), image_width, 1)) -
                                 np.matlib.repmat(center_y, image_height, image_width)) ** 2)
            Gauss_map = np.exp(-0.5 * Gauss_map / R_pointer)
            target_matrix[n, ch] += Gauss_map
        target_matrix[n, ch] = target_matrix[n, ch]/np.max(target_matrix[n, ch])
        #endregion

        #region Busbar
        # for idx in range(4):
        #     pt =idx; ch=idx
        #     center_x = target[n, pt, 0]
        #     center_y = target[n, pt, 1]
        #
        #     Gauss_map = np.sqrt((np.matlib.repmat(np.arange(image_width), image_height, 1) -
        #                          np.matlib.repmat(center_x, image_height, image_width)) ** 2 +
        #                         (np.transpose(np.matlib.repmat(np.arange(image_height), image_width, 1)) -
        #                          np.matlib.repmat(center_y, image_height, image_width)) ** 2)
        #
        #     Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        #     target_matrix[n, ch] = Gauss_map
        #endregion

    return target_matrix

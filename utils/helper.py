import os
import numpy as np

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




# * Loss: BCE Loss
#   Referenced the 3rd-party codes.
#   Loss for heatmap (pixel-by-pixel)
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
import torch
import torch.nn as nn



def BCE_loss(results, labels, topk=10):
    # if len(results.shape) == 1:
    #     results = results.view(1, -1)
    # batch_size, class_num = results.shape
    # labels = labels.view(-1, 1)
    # # one_hot_target = torch.zeros(batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004 * 2]
    # lovasz_loss = lovasz_hinge(results,labels )#one_hot_target

    error = torch.abs(labels - torch.sigmoid(results))#one_hot_target
    error = error.topk(topk, 1, True, True)[0].contiguous()
    target_error = torch.zeros_like(error).float().cuda()
    error_loss = nn.BCELoss(reduce=True)(error, target_error)

    # labels = labels.view(-1)
    # indexs_new = (labels != 5004 * 2).nonzero().view(-1)
    # if len(indexs_new) == 0:
    #     return error_loss
    # results_nonew = results[torch.arange(0, len(results))[indexs_new], labels[indexs_new]].contiguous()
    # target_nonew = torch.ones_like(results_nonew).float().cuda()
    # nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)

    return error_loss # nonew_loss + error_loss + lovasz_loss * 0.5


class BCELogitsLossWithMask(nn.Module):

    def __init__(self, size_average=True):
        super(BCELogitsLossWithMask, self).__init__()
        self.size_average = size_average



    def forward(self, input, target, mask=None):
        '''
        :param input: Variable of shape (N, C, H, W)  logits 0~1
        :param target:  Variable of shape (N, C, H, W)  0~1 float
        :param mask: Variable of shape (N, C)  0. or 1.  float
        :return:
        '''


        # print(target[0,0,1]) # target: N C 2  input: N C H W
        _,C,_ = target.shape

        if not (C == input.shape[1]):
            raise ValueError("Target channel ({}) must be the same as input channel ({})".format(C, input.shape[1]))

        N, C, H, W = input.shape
        # print(input.shape)
        # target_matrix = np.zeros((N, C, H, W)).astype(np.float32)  #0.0~1.0
        # for n in range(N):
        #     for c in range(C):
        #         target_matrix[n,c,int(target[n,c,1]/TARGET_SIZE),int(target[n,c,0]/TARGET_SIZE)] = 1.0

        target_matrix = self.produceTargetHeatmap(target,H, W)
        # print(input, np.max(target_matrix))
        target_matrix = torch.from_numpy(target_matrix)


        # BCELogitsLossWithMask
        max_val = (-input).clamp(min=0)
        loss =  - input * target_matrix + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() #TODO:Need to check #input
        if self.size_average:
            # w, h = input.shape
            return loss.sum() / (H*W)
        else:
            return loss.sum()


        # neg_abs = - input.abs()
        # loss = input.clamp(min=0) - input * target_matrix + (1 + neg_abs.exp()).log()
        # return loss.mean()
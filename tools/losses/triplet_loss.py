# * Loss: Triplet Loss
#   Referenced the 3rd-party codes.
#   Loss for Embedding Heatmap
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses if size_average else losses.sum() #.mean()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    #TODO:UNAGUO New added scenes
    def forward(self, embeddings, target):

        # print(embeddings.size())
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


#Embedding loss: Global & local
def getLoss(classcriterion, globalcriterion,localcriterion, global_feat, local_feat, results,labels):
    # gl=global_loss(TripletLoss(margin=1.5), global_feat, labels)[0]
    if type(global_feat) not in (tuple, list):
        global_feat = (global_feat,)
    if type(local_feat) not in (tuple, list):
        M, m, d = local_feat.size()
        # print(local_feat.size())
        local_feat = local_feat.contiguous().view(M , d* m)
        local_feat = (local_feat,)
    global_feat += (labels,)
    local_feat += (labels,)
    gl = globalcriterion(*global_feat)
    gl = gl[0] if type(gl) in (tuple, list) else gl
    # ll=local_loss(TripletLoss(margin=2.0), local_feat, labels)[0]
    ll = localcriterion(*local_feat)
    ll = ll[0] if type(ll) in (tuple, list) else ll
    # triple_loss = gl + ll

    # loss_ = sigmoid_loss(results, labels, topk=30)
    #criterion = nn.CrossEntropyLoss()

    labels = labels.long()
    # print(results.size())
    # print(labels.size())
    # print(torch.max(labels))
    loss_ = classcriterion(results,labels)/8#/2#
    #
    # losssum = gl+ll+loss_

    # loss = triple_loss + loss_
    # print("Loss: gl:{:.2f}, ll:{:.2f}, rl:{:.2f}".format(gl,ll,loss_))
    return gl,ll,loss_

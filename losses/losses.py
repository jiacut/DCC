"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            #raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
            return torch.tensor([0.0]).cuda()
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim = 1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing
        self.lamda = 2.0

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim = 1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts = True)
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight = weight, reduction='mean')

        return loss


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss

class AGCLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(AGCLoss, self).__init__()
        self.xentropy = nn.CrossEntropyLoss().cuda()
        self.lamda = entropy_weight
        self.softmax = nn.Softmax(dim=1)
        self.temperature = 1.0

    def forward(self, ologits, plogits):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        b, c = ologits.shape[0] // 2, ologits.shape[1]
        assert ologits.shape == plogits.shape, ('Inputs are required to have same shape')

        ologits = self.softmax(ologits)
        plogits = self.softmax(plogits)
        ologits_ = torch.split(ologits, b, dim=0)[1]
        ologits = torch.cat(torch.split(ologits, b, dim=0), dim=1)
        plogits = torch.cat(torch.split(plogits, b, dim=0), dim=1)
        
        # one-hot
        similarity = torch.mm(F.normalize(ologits.t(), p=2, dim=1), F.normalize(plogits, p=2, dim=0))
        similarity = torch.cat(torch.split(similarity, c, dim=1), dim=0)
        loss_ce = self.xentropy(similarity, torch.arange(c).repeat(1,4).squeeze().cuda())
        # similarity = torch.split(similarity, c, dim=0)
        # loss_ce = self.xentropy(similarity[0], torch.arange(c).cuda()) # weak - weak
        # loss_ce = self.xentropy(torch.cat([similarity[1], similarity[2]], dim=0), torch.arange(c).repeat(1,2).cuda()) # weak - strong
        
        # balance regularisation
        o = ologits_.sum(0).view(-1)
        o /= o.sum()
        loss_ne = math.log(o.size(0)) + (o * o.log()).sum()

        loss = loss_ce + self.lamda * loss_ne

        return loss, loss_ce, loss_ne
'''
class RGCLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(RGCLoss, self).__init__()
        self.temperature = temperature
        self.lamda = 0.1 # infoNCE损失中 分母应该全是负样本 考虑到正样本中存在假正样本情况 特添加此参数 同时该参数的值应该设置较小

    def forward(self, features, neighbors_features, conf_pred_mask, label_mask, neighbor_num):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.size()
        n_ = n // 2

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = torch.cat([features[:,0], features[:,2]], dim=0)
#         anchor = contrast_features

        if neighbors_features is not None:
            contrast_features = torch.split(contrast_features, n_*b, dim=0)
            neighbors_features = torch.cat(torch.unbind(neighbors_features, dim=1), dim=0)
            neighbors_features = torch.split(neighbors_features, neighbor_num*b, dim=0)
            contrast_features = torch.cat([contrast_features[0],neighbors_features[0],contrast_features[1], neighbors_features[1]], dim=0).cuda()
            assert(10*b == contrast_features.shape[0])

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        dot_product = torch.cat(torch.split(dot_product, (n_ + neighbor_num)*b, dim=1), dim=0)

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        union_mask = ((conf_pred_mask + label_mask) > 0).float()
        pseudo_mask = torch.scatter(union_mask.repeat(1, n_ + neighbor_num), 1, torch.arange(b).unsqueeze(1).cuda(), 0).cuda()
        pseudo_mask = torch.cat([pseudo_mask, union_mask.repeat(2, n_ + neighbor_num), pseudo_mask], dim=0)
        
        posit_mask = torch.scatter(conf_pred_mask.repeat(1, n_ + neighbor_num), 1, torch.arange(b).unsqueeze(1).cuda(), 0).cuda()
        posit_mask = torch.cat([posit_mask, conf_pred_mask.repeat(2, n_ + neighbor_num), posit_mask], dim=0)
        
        assert pseudo_mask.shape == exp_logits.shape
        assert posit_mask.shape == exp_logits.shape
 
        ones_matrix = torch.ones_like(pseudo_mask)
        prob = exp_logits / (((ones_matrix - pseudo_mask).cuda() * exp_logits).sum(1, keepdim=True) + posit_mask * exp_logits)
        
        # Mean log-likelihood for positive
        loss = -((torch.log(prob) * posit_mask).sum(1) / posit_mask.sum(1)).mean()
        # loss = -((torch.log(prob) * posit_mask).sum(1) / posit_mask.sum(1))
        # loss = torch.split(loss, b, 0)
        # loss = loss[0].mean() # weak - weak
        # loss = torch.cat([loss[1], loss[2]], dim=0).mean() # weak - strong

        return loss
'''

class RGCLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(RGCLoss, self).__init__()
        self.temperature = temperature
        self.lamda = 0.1 # infoNCE损失中 分母应该全是负样本 考虑到正样本中存在假正样本情况 特添加此参数 同时该参数的值应该设置较小

    def forward(self, features, neighbors_features, neighbor_num, tau_plus, debiased=True):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.size()
        n_ = n // 2

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = torch.cat([features[:,0], features[:,2]], dim=0)

        if neighbors_features is not None:
            contrast_features = torch.split(contrast_features, n_*b, dim=0)
            neighbors_features = torch.cat(torch.unbind(neighbors_features, dim=1), dim=0)
            neighbors_features = torch.split(neighbors_features, neighbor_num*b, dim=0)
            contrast_features = torch.cat([contrast_features[0],neighbors_features[0],contrast_features[1], neighbors_features[1]], dim=0).cuda()
            assert(10*b == contrast_features.shape[0])

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        dot_product = torch.cat(torch.split(dot_product, (n_ + neighbor_num)*b, dim=1), dim=0)

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()
        exp_logits = torch.exp(logits)
        
        mask = torch.eye(b, dtype=bool).repeat(n, n_ + neighbor_num).cuda()
        
        neg = exp_logits.masked_select(~mask).view(n * b, -1)
        pos = exp_logits.masked_select(mask).view(n * b, -1)
        assert((n_ + neighbor_num)*(b-1) == neg.shape[1])
        assert((n_ + neighbor_num) == pos.shape[1])
        
        if debiased:
            N = b * (n_ + neighbor_num) - (n_ + neighbor_num)
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1, keepdim=True)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / self.temperature))
        else:
            Ng = neg.sum(dim=-1)

        loss = (- torch.log(pos / (pos + Ng) )).mean()
        
        return loss

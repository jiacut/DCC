"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import math
import faiss
import numpy as np
import torch.nn.functional as F
from utils.utils import AverageMeter, ProgressMeter
from utils.common_config import adjust_beta
from utils.evaluate_utils import get_accuracy

def gcc_train(train_loader, model, criterion1, criterion2, optimizer, grad_scaler, epoch, p, aug_feat_memory, log_output_file, only_train_pretext=True):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    constrastive_losses = AverageMeter('Constrast Loss', ':.4e')
    cluster_losses = AverageMeter('Cluster Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),[losses, constrastive_losses, cluster_losses], prefix="Epoch: [{}]".format(epoch), output_file=log_output_file)
    
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        weak_images = batch['image'].cuda(non_blocking=True)
        strong_images = batch['strong_image'].cuda(non_blocking=True)
        weak_augmented = batch['weak_augmented'].cuda(non_blocking=True)
        strong_augmented = batch['strong_augmented'].cuda(non_blocking=True)

#         tau_plus = 1.0 / p['num_classes']
        tau_plus = p["tau_plus"]

        with torch.cuda.amp.autocast():
            b, c, h, w = weak_images.size()
            input_ = torch.cat([weak_images.unsqueeze(1), weak_augmented.unsqueeze(1), strong_images.unsqueeze(1), strong_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            targets = batch['target'].cuda(non_blocking=True)
            constrastive_features, cluster_outs = model(input_)
            constrastive_features = constrastive_features.view(b, 4, -1)
            cluster_outs = [out_.view(b, 4, -1) for out_ in cluster_outs]

            if not only_train_pretext:
                strong_neighbors = batch['strong_neighbors'].cuda(non_blocking=True)
                weak_neighbors = batch['weak_neighbors'].cuda(non_blocking=True)
                neighbor_num = weak_neighbors.shape[1]
                neighbor_input_ = torch.cat([weak_neighbors, strong_neighbors], dim=1)
                assert neighbor_input_.shape == (b, 2*neighbor_num, c, h, w)

                neighbor_input_ = neighbor_input_.view(-1, c, h, w)
                neighbor_input_ = neighbor_input_.cuda(non_blocking=True)
                neighbor_features = model(neighbor_input_)[0]
                neighbor_features = neighbor_features.view(b, 2*neighbor_num, -1)

                constrastive_loss = criterion1(constrastive_features, neighbor_features, neighbor_num, tau_plus)

            else:
                # the first stage
                constrastive_loss = criterion1(constrastive_features, None, 0, tau_plus)

            aug_feat_memory.push(constrastive_features.clone().detach()[:, 1], batch['meta']['index'])

            if not only_train_pretext:
                weak_neighbor = batch['weak_neighbor'].cuda(non_blocking=True)
                strong_neighbor = batch['strong_neighbor'].cuda(non_blocking=True)
                strong_neighbor_output = model(strong_neighbor)[1]
                weak_neighbor_output = model(weak_neighbor)[1]

                # Loss for every head
                total_loss, consistency_loss, entropy_loss = [], [], []
                for image_and_aug_output_subhead, weak_neighbor_cluster, strong_neighbor_cluster in zip(cluster_outs, weak_neighbor_output, strong_neighbor_output):
                    image_output_subhead = torch.cat([image_and_aug_output_subhead[:,1], image_and_aug_output_subhead[:,3]], dim=0)
                    neightbor_output_subhead = torch.cat([weak_neighbor_cluster, strong_neighbor_cluster], dim=0) # 邻居样本的弱和强
                    total_loss_, consistency_loss_, entropy_loss_ = criterion2(image_output_subhead, neightbor_output_subhead)

                    total_loss.append(total_loss_)
                    consistency_loss.append(consistency_loss_)
                    entropy_loss.append(entropy_loss_)

                cluster_loss = torch.sum(torch.stack(total_loss, dim=0))
            else:
                cluster_loss = torch.tensor([0.0]).cuda()

            loss = p['alpha'] * constrastive_loss + cluster_loss

        losses.update(loss.item())
        constrastive_losses.update(constrastive_loss.item())
        cluster_losses.update(cluster_loss.item())

        grad_scaler(loss, optimizer, parameters=model.parameters(), create_graph=False, update_grad=True)
        optimizer.zero_grad() # 参数梯度归零 防止与上一次迭代的梯度关联

        if i % 25 == 0:
            progress.display(i)

def sl_search(logits, threshold, targets, num_classes):
    anchor_logits = F.softmax(logits, dim=1)
    anchor_prob, anchor_pred = torch.max(anchor_logits, dim=1)
    conf_anchor_prob = anchor_prob > threshold
    conf_anchor_pred = torch.linspace(-logits.size(0), -1, logits.size(0)).cuda()
    conf_anchor_pred = torch.masked_scatter(conf_anchor_pred, conf_anchor_prob, anchor_pred.masked_select(conf_anchor_prob).float())
    conf_pred_mask = torch.eq(conf_anchor_pred.unsqueeze(1), conf_anchor_pred.unsqueeze(1).T).float().cuda()
    #conf_pred_mask = torch.scatter(conf_pred_mask, 1, torch.arange(conf_pred_mask.size(0)).unsqueeze(1).cuda(), 0).cuda()

    accuracy = 0.0
    count = 0.0
    if conf_anchor_prob.sum() > 0:
        pseudo = torch.masked_select(anchor_pred, conf_anchor_prob.squeeze())
        true = torch.masked_select(targets, conf_anchor_prob.squeeze())
        accuracy = get_accuracy(pseudo, true, num_classes)
        count = conf_anchor_prob.sum()
    
    return (conf_pred_mask, accuracy, count)

def selflabel_train(train_loader, model, criterion1, criterion2, optimizer, grad_scaler, epoch, p, ema=None, output_file=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    feature_losses = AverageMeter('Constrast Loss', ':.4e')
    selflabel_losses = AverageMeter('Selflabel Loss', ':.4e')
    label_count = AverageMeter('Count', ':.2f')
    label_ratio = AverageMeter('Acc', ':.2f')
    progress = ProgressMeter(len(train_loader), [losses, feature_losses, selflabel_losses, label_count, label_ratio], prefix="Epoch: [{}]".format(epoch), output_file=output_file)
    
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        images = batch['val_image'].cuda(non_blocking=True)
        weak_images = batch['image'].cuda(non_blocking=True)
        strong_images = batch['strong_image'].cuda(non_blocking=True)
        weak_augmented = batch['weak_augmented'].cuda(non_blocking=True)
        strong_augmented = batch['strong_augmented'].cuda(non_blocking=True)
        
        tau_plus = 1.0 / p['num_classes']

        with torch.cuda.amp.autocast():
            b, c, h, w = weak_images.size()
            input_ = torch.cat([weak_images.unsqueeze(1), weak_augmented.unsqueeze(1), strong_images.unsqueeze(1), strong_augmented.unsqueeze(1)], dim=1)
            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            targets = batch['target'].cuda(non_blocking=True)
            constrastive_features, cluster_outs = model(input_)
            constrastive_features = constrastive_features.view(b, 4, -1)
            cluster_outs = [out_.view(b, 4, -1) for out_ in cluster_outs]

            with torch.no_grad():
                outs = model(images)[1]
                _, accuracy, count = sl_search(outs[0], p['confidence_threshold'], targets, p['num_classes'])
                label_ratio.update(accuracy)
                label_count.update(count)

            feature_loss = criterion1(constrastive_features, None, 0, tau_plus)
            selflabel_loss = criterion2(outs[0], cluster_outs[0][:, 2])

            loss = feature_loss + p['beta'] * selflabel_loss

        feature_losses.update(feature_loss.item())
        selflabel_losses.update(selflabel_loss.item())
        losses.update(loss.item())

        grad_scaler(loss, optimizer, parameters=model.parameters(), create_graph=False, update_grad=True)
        optimizer.zero_grad()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i % 25 == 0:
            progress.display(i)
"""
Authors: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np
import sys

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate, get_predictions, hungarian_evaluate
from utils.memory import MemoryBank
from utils.train_utils import gcc_train
from utils.utils import fill_memory_bank, fill_memory_bank_mean
from utils.grad_scaler import NativeScalerWithGradNormCount as NativeScaler
from termcolor import colored
from utils.aug_feat import AugFeat
from data import ConcatDataset

import warnings
warnings.filterwarnings('ignore')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

# Parser
parser = argparse.ArgumentParser(description='Graph Contrastive Clustering')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
 
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,c=palette[colors.astype(int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
 
    # We add the labels for each digit.
    txts = []
    for i in range(10):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=36)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
 
    return f, ax, sc, txts

def tsne(features, labels, epoch):
    features = features.numpy()
    labels = labels.numpy()
    labels_num = np.unique(labels)

    sub_features = None
    sub_labels = np.array([])
    nums = [500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

    for i in labels_num:
        label_i = np.equal(labels, i)
        feature_i = features[label_i][:1000]
        feature_num = feature_i.shape[0]
        feature = feature_i[np.random.choice(feature_num, nums[i])]
        sub_labels = np.concatenate((sub_labels, np.ones(nums[i]) * i), axis=0)
        if sub_features is None:
            sub_features = feature
        else:
            sub_features = np.concatenate((sub_features, feature), axis=0)
    
    # print(sub_features.shape)

    td_features = TSNE(n_components=2, learning_rate='auto',init='random').fit_transform(sub_features)
    scatter(td_features, sub_labels)
    plt.savefig(f'tsne_{epoch}.png', dpi=120)

def main():
    org_feat_memory = AugFeat('./org_feat_memory', 4)
    aug_feat_memory = AugFeat('./aug_feat_memory', 4)

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))
    with open (p['log_output_file'], 'a+') as fw:
        fw.write(str(p) + "\n")

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    # print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p) # 强数据增强 + 弱数据增强
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p) # 验证集上的数据增强
    print('Validation transforms:', val_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True,split='train+unlabeled') # Split is for stl-10
    train_dataloader = get_train_dataloader(p, train_dataset)

    val_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True,split='train') # Dataset w/o augs for knn eval
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, to_end2end_dataset=True, split='train') # Dataset for performance test
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                p['model_kwargs']['features_dim'],
                                p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion1, criterion2 = get_criterion(p)
    print('Criterion is {}'.format(criterion1.__class__.__name__))
    print('Criterion is {}'.format(criterion2.__class__.__name__))
    criterion1 = criterion1.cuda()
    criterion2 = criterion2.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Gradient Scaler
    print(colored('Retrieve grad_scaler', 'blue'))
    grad_scaler = NativeScaler()
    print(grad_scaler)

    # Checkpoint
    if os.path.exists(p['end2end_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['end2end_checkpoint']), 'blue'))
        checkpoint = torch.load(p['end2end_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scaler.load_state_dict(checkpoint['grad_scaler'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']# 10000 for evaluate directly
        # if start_epoch >= p['init_epochs']:
        #     train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True, split='train') # Split is for stl-10
        #     train_dataloader = get_train_dataloader(p, train_dataset)
    else:
        print(colored('No checkpoint file at {}'.format(p['end2end_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    best_acc = 0.0
    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        if epoch <= p['init_epochs']:
            print('Train pretext...')
            gcc_train(train_dataloader, model, criterion1, criterion2, optimizer, grad_scaler,
                    epoch, p, aug_feat_memory, p['log_output_file'], True)
        else:
            print('Train pretext and clustering...')
            gcc_train(train_dataloader, model, criterion1, criterion2, optimizer, grad_scaler,
                    epoch, p, aug_feat_memory, p['log_output_file'], False)

        # Evaluate
        if epoch > 0 and epoch % 5 == 0:
            print ("Start to evaluate...")
            # predictions = get_predictions(p, base_dataloader, model)
            predictions, features, targets = get_predictions(p, base_dataloader, model, return_features=True)
            lowest_loss_head = 0
            clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=None)
            print(clustering_stats, len(base_dataloader.dataset))
            with open (p['log_output_file'], 'a+') as fw:
                fw.write(str(clustering_stats) + "\n")

            if clustering_stats['ACC'] > best_acc:
                best_acc = clustering_stats['ACC']
                print ('Best acc: ', best_acc)
                # Checkpoint
                print('Checkpoint ...')
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 'grad_scaler': grad_scaler.state_dict(),
                        'epoch': epoch + 1}, p['end2end_checkpoint'])
                
                # tsne(features, targets, epoch)

        # Update memory bank
        if epoch >= p['init_epochs'] and epoch % 5 == 0:
            if epoch == p['init_epochs']:
                train_dataset = get_train_dataset(p, train_transforms, to_end2end_dataset=True, split='train') # Split is for stl-10
                train_dataloader = get_train_dataloader(p, train_dataset)

            # Fill memory bank
            topk = 5 # 超参 需要微调(软标签缘故)
            fill_memory_bank_mean(val_dataloader, aug_feat_memory, org_feat_memory, memory_bank_val)
            indices, acc, detail_acc = memory_bank_val.mine_nearest_neighbors(topk, model) # K近邻 + 软近邻
            print('Accuracy of top-[1,3,5] nearest neighbors on val set is %.2f, %.2f, %.2f' %(detail_acc[0]*100, detail_acc[1]*100, detail_acc[2]*100))
            with open (p['log_output_file'], 'a+') as fw:
                for acc in detail_acc:
                    fw.write(str(acc))
            np.save(p['topk_neighbors_val_path'], indices)
            train_dataset.update_neighbors(indices)

            torch.cuda.empty_cache() # 释放显存中不再引用的变量

    predictions, features, targets = get_predictions(p, base_dataloader, model, return_features=True)
    lowest_loss_head = 0
    clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
    print(clustering_stats)

    with open (p['features'], 'wb') as f:
        np.save(f, features)
    with open (p['features'] + "_label", 'wb') as f:
        np.save(f, targets)

    # Save final model
    torch.save(model.state_dict(), p['end2end_model'])

def plot_case(config_env, config_exp):
    import random
    # p = create_config(args.config_env, args.config_exp)
    p = create_config(config_env, config_exp)
    train_dataset = get_train_dataset(p, None, to_end2end_dataset=True, split='train+unlabeled')
    train_dataloader = get_train_dataloader(p, train_dataset)

    imageset = {}
    '''cifar10'''
    # for image in train_dataset:
    #     if len(imageset) >= 10:
    #         break
    #     elif not imageset.get(image['class_name']) and len(image['error_neighbor']) > 0 and len(image['true_neighbor']) > 0:
    #         imageset[image['class_name']] = [image['image'], random.choice(image['true_neighbor']), random.choice(image['error_neighbor'])]

    '''imagenet10'''
    for image in train_dataset:
        if len(imageset) >= 10:
            break
        elif not imageset.get(image['target']) and len(image['error_neighbor']) > 0 and len(image['true_neighbor']) > 0:
            imageset[image['target']] = [image['image'], random.choice(image['true_neighbor']), random.choice(image['error_neighbor'])]

    return imageset
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # fig = plt.figure()

    # print(class_names[0])
    # print(len(imageset[class_names[0]]))

    # assert 1 == 0

    # for i in range(3):
    #     for j in range(1, 11):
    #         plt.subplot(3, 10, 3*i+j)
    #         name = class_names[j]
    #         img = imageset[name]
    #         img = img.numpy()
    #         # img = np.transpose(img, (1, 2, 0))
    #         plt.imshow(img)
    #         plt.axis('off')
    
    # plt.savefig('cifar_cases.png', bbox_inches='tight')

if __name__ == '__main__':
    main()

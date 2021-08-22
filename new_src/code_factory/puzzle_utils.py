##https://github.com/OFRIN/PuzzleCAM/blob/master/train_classification_with_puzzle.py

import math

import torch
import torch.nn.functional as F

def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    h_per_patch = h // num_pieces_per_line
    w_per_patch = w // num_pieces_per_line
    
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)

def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))
    num_pieces_per_line = int(math.sqrt(num_pieces))
    
    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):

        ext_w_list = []
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])
            index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))

    features = torch.cat(ext_h_list, dim=2)
    return features

def puzzle_module(x, func_list, num_pieces):
    tiled_x = tile_features(x, num_pieces)

    for func in func_list:
        tiled_x = func(tiled_x)
        
    merged_x = merge_features(tiled_x, num_pieces, x.size()[0])
    return merged_x


"""
from code_factory.puzzle_utils import tile_features,merge_features


        for i, (images, labels,indexes) in tk0:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            onehot_label = torch.eye(CFG.model.n_classes)[labels].to(device)
            ### mix系のaugumentation=========

            ### mix系のaugumentation おわり=========

            tiled_images = tile_features(images, 4)
            #print(tiled_images.shape)
            BATCH_SIZE = images.shape[0]


            if CFG.train.amp:
                with autocast():
                    y_preds,features = model(images.float(),with_cam=True)
                    tiled_logits, tiled_features = model(tiled_images, with_cam=True)
                    re_features = merge_features(tiled_features, 4,BATCH_SIZE)
                    #print(tiled_logits.shape,re_features.shape)
                    class_mask = onehot_label.unsqueeze(2).unsqueeze(3)
                    #print(features.shape,re_features.shape)
                    re_loss = re_loss_fn(features, re_features) * class_mask
                    re_loss = re_loss.mean()
                    #args.re_loss_option == 'selection'
                    
                    if CFG.augmentation.mix_p>rand:
                        loss_ = mixup_criterion(criterion, y_preds, y_a, y_b, lam)
                        p_class_loss = mixup_criterion(criterion,gap_fn(re_features).squeeze().squeeze(),y_a, y_b, lam)
                    elif CFG.augmentation.mix_p<=rand and CFG.augmentation.mix_p>0:
                        loss_ = criterion(y_preds,onehot_label)
                        p_class_loss = criterion(gap_fn(re_features).squeeze().squeeze(), onehot_label)
                    else:
                        loss_ = criterion(y_preds, labels)
                        p_class_loss = criterion(gap_fn(re_features).squeeze().squeeze(), labels)
                        #p_class_loss = torch.zeros(1).cuda()

                    alpha = CFG.puzzle_cam.alpha
                    loss = loss_ + p_class_loss + alpha * re_loss*CFG.puzzle_cam.beta




"""

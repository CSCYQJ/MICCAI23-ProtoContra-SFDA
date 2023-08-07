import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import surface_distance as surfdist


def connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

def cal_dice_score(input,target):
    
    input = input.numpy()
    # input = connectivity_region_analysis(input)
    input_flat = input.flatten()
    target_flat = target.numpy().flatten()
    inter = np.sum(input_flat * target_flat)
    union = input_flat.sum() + target_flat.sum()
    # compute the dice score
    score = np.zeros_like(inter, dtype="float32")
    score[union>0] = 2*inter[union>0]/union[union>0]
    score[union==0] = np.nan
    return score

def cal_iou_score(input,target):
    
    input = input.numpy()
    # input = connectivity_region_analysis(input)
    input_flat = input.flatten()
    target_flat = target.numpy().flatten()
    inter = np.sum(input_flat * target_flat)
    union = input_flat.sum() + target_flat.sum() - inter
    # compute the dice score
    score = np.zeros_like(inter, dtype="float32")
    score[union>0] = inter[union>0]/union[union>0]
    score[union==0] = np.nan
    return score

def MultiDiceScore(preds,target,num_classes,include_bg=False):

    dice_score_list = []
    target = F.one_hot(target,num_classes).float()
    if isinstance(preds, dict):
        seg = preds['seg']
    else:
        seg = preds
    seg = F.one_hot(seg.argmax(dim=0),num_classes).float()

    if include_bg:
        for i in range(num_classes):
            dice_score = cal_dice_score(seg[...,i], target[...,i])
            dice_score_list.append(dice_score)
    else:
        for i in range(1,num_classes):
            dice_score = cal_dice_score(seg[...,i], target[...,i])
            dice_score_list.append(dice_score)
    return dice_score_list

def MultiIoUScore(preds,target,num_classes,include_bg=False):

    iou_score_list = []
    target = F.one_hot(target,num_classes).float()
    if isinstance(preds, dict):
        seg = preds['seg']
    else:
        seg = preds
    seg = F.one_hot(seg.argmax(dim=0),num_classes).float()

    if include_bg:
        for i in range(num_classes):
            iou_score = cal_iou_score(seg[...,i], target[...,i])
            iou_score_list.append(iou_score)
       
    else:
        for i in range(1,num_classes):
            iou_score = cal_iou_score(seg[...,i], target[...,i])
            iou_score_list.append(iou_score)
    return iou_score_list

def cal_average_surface_distance(input,target):
    input = input.cpu().numpy().astype(np.bool8)
    target = target.cpu().numpy().astype(np.bool8)
    surface_distances = surfdist.compute_surface_distances(input, target, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
    return (avg_surf_dist[0]+avg_surf_dist[1])/2

def MultiASD(preds,target,num_classes,include_bg=False):

    asd_list = []
    target = F.one_hot(target,num_classes)
    if isinstance(preds, dict):
        seg = preds['seg']
    else:
        seg = preds
    seg = F.one_hot(seg.argmax(dim=0),num_classes)
    # print(seg.shape,target.shape)
    if include_bg:
        for i in range(num_classes):
            asd = cal_average_surface_distance(seg[...,i], target[...,i])
            asd_list.append(asd)
    else:
        for i in range(1,num_classes):
            asd = cal_average_surface_distance(seg[...,i], target[...,i])
            asd_list.append(asd)
    return asd_list

def mean_dice(results, gt_seg_maps,num_classes,organ_list):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_dice_mat = []
    dice_metric = {}
    for i in range(num_imgs):
        dice_coef = MultiDiceScore(results[i],gt_seg_maps[i],num_classes)
        total_dice_mat.append(dice_coef)
    total_dice_mat = np.array(total_dice_mat)
    for j,organ in enumerate(organ_list):
        dice_metric['{:}_dice'.format(organ)] = total_dice_mat[:,j].mean()
    dice_metric['dice_avg'] = total_dice_mat.mean()
    return dice_metric
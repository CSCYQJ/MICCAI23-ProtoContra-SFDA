import SimpleITK as sitk
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import skimage.morphology as mor
import skimage.measure as measure

avg_pool = nn.AvgPool3d(3, 1, 1)


def torch_binary_erosion_3d(input_tensor, se_size=3, times=1):
    assert len(input_tensor.size()) == 5
    for i in range(times):
        input_tensor = avg_pool(input=input_tensor.float())
        input_tensor = input_tensor == 1

    return input_tensor

def torch_binary_dilate_3d(input_tensor, se_size=3, times=1):
    assert len(input_tensor.size()) == 5
    for i in range(times):
        input_tensor = avg_pool(input=input_tensor.float())
        input_tensor = input_tensor > 0

    return input_tensor


def torch_binary_opening_3d(input_tensor, se_size=3, times=1):
    res = torch_binary_erosion_3d(input_tensor, se_size=3, times=times)
    res = torch_binary_dilate_3d(res, se_size=3, times=times)
    return res


def torch_binary_closintg_3d(input_tensor, se_size=3, times=1):
    res = torch_binary_dilate_3d(input_tensor, se_size=3, times=times)
    res = torch_binary_erosion_3d(res, se_size=3, times=times)
    return res

def delete_bed_torch(volume, fg_th=-500, device=0):
    
    
    assert len(volume.shape) == 3
    volume = volume[None][None]
    img_mask = np.array(volume) >= fg_th
    with torch.no_grad():
        volume_tensor = torch.from_numpy(img_mask.astype(np.uint8)).cuda(device)
        # volume_tensor = volume_tensor.unsqueeze(0)
        closed = torch_binary_closintg_3d(torch_binary_opening_3d(volume_tensor, 3, 2), 3, 2)
        img_mask = closed[0][0].cpu().numpy().astype(np.bool)
    fg_mask = mor.label(img_mask)
    fg_prop = measure.regionprops(fg_mask)

    fg_prop.sort(key=lambda x: x.area, reverse=True)
    fg_mask = (fg_mask == fg_prop[0].label)

    bg_mask = 1 - fg_mask
    bg_mask = mor.label(bg_mask)
    bg_prop = measure.regionprops(bg_mask)
    bg_prop.sort(key=lambda x: x.area, reverse=True)
    bg_mask = (bg_mask == bg_prop[0].label)

    inner_mask = 1 - bg_mask

    prop = measure.regionprops(inner_mask.astype(np.uint8))
    bbox = prop[0].bbox
    fg_vol = volume[:, :, 0:volume.shape[2], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    # bed_removed_voxel = ((0,bbox[1],bbox[2]), (volume.shape[2], bbox[4], bbox[5]))
    bbox = [[0,volume.shape[2]],[bbox[1],bbox[4]],[bbox[2],bbox[5]]]
    

    return fg_vol[0][0], bbox

from tensorboardX import SummaryWriter
import numpy as np
import torch
import os
import time
import sys
import cv2

class Visualizer():
    def __init__(self, opt):
        self.opt = opt

        # tf summary writer
        self.summary_writer = SummaryWriter(os.path.join(opt['checkpoint_dir'], 'tf_logs'))

        # create a logging file to store training losses
        self.log_name = os.path.join(opt['checkpoint_dir'], 'console_logs', 'loss_log.txt')

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False
    
    def add_mask(self,img,mask):
        
        label2color = {0:np.array([0,0,0]),1:np.array([251,111,111]),2:np.array([240,130,40]),3:np.array([206,242,151]),4:np.array([238,172,255]),5:np.array([0,255,255])}
        
        pred_mask = np.zeros_like(img).astype(np.uint8)
        for l,color in label2color.items():
            pred_mask[mask==l,:] = color
        img_pred = cv2.addWeighted(img,0.5,pred_mask,0.5,0,0)
        return img_pred
    
    def add_heatmap(self,img,entropy_map):

        entropy_map = (entropy_map - entropy_map.min())/(entropy_map.max()-entropy_map.min())
        entropy_map = (entropy_map * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(entropy_map,cv2.COLORMAP_JET)
        img_uncertainty = cv2.addWeighted(img,0.7,heatmap[:,:,::-1],0.3,0,0)
        return img_uncertainty

    
    def display_current_results(self, epoch, visuals,is_resize=False):
        
        images,preds,gt_segs = visuals['images'],visuals['preds'],visuals['gt_segs']
        concat_results = []
        for i in range(min(images.shape[0],8)):
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img,img,img],axis=-1)
            elif img.ndim == 3:
                img = img.transpose((1,2,0))
            img = (img-img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)
            pred = preds[i].astype(np.uint8)
            gt = gt_segs[i].astype(np.uint8)
            img_pred = self.add_mask(img,pred)
            img_gt = self.add_mask(img,gt)
            concat_results.append(np.concatenate([img,img_pred,img_gt],axis=1))
        concat_results = np.concatenate(concat_results,axis=0)
        if is_resize:
            concat_results = cv2.resize(concat_results,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.opt['checkpoint_dir'], 'visuals',  'img_gt_pred_' + str(epoch) + '.png'),concat_results[:,:,::-1])
        
    def display_current_PFA(self, epoch, visuals,is_resize=False):
        
        images,entropy_maps, preds,gt_segs = visuals['images'],visuals['entropy_maps'],visuals['preds'],visuals['gt_segs']
        concat_results = []
        for i in range(min(images.shape[0],8)):
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img,img,img],axis=-1)
            elif img.ndim == 3:
                img = img.transpose((1,2,0))
            img = (img-img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)
            entropy_map = entropy_maps[i]
            pred = preds[i].astype(np.uint8)
            gt = gt_segs[i].astype(np.uint8)
            img_uncertain = self.add_heatmap(img,entropy_map)
            img_pred = self.add_mask(img,pred)
            img_gt = self.add_mask(img,gt)
            concat_results.append(np.concatenate([img,img_uncertain,img_pred,img_gt],axis=1))
        concat_results = np.concatenate(concat_results,axis=0)
        if is_resize:
            concat_results = cv2.resize(concat_results,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.opt['checkpoint_dir'], 'visuals',  'img_ent_pred_gt' + str(epoch) + '.png'),concat_results[:,:,::-1])
        
        
    def display_current_CL(self, epoch, visuals,is_resize=False):
        
        images,entropy_maps_teacher,preds_teacher,entropy_maps, preds,gt_segs = visuals['images'],visuals['entropy_maps_teacher'],visuals['preds_teacher'],visuals['entropy_maps'],visuals['preds'],visuals['gt_segs']
        concat_results = []
        for i in range(min(images.shape[0],8)):
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img,img,img],axis=-1)
            elif img.ndim == 3:
                img = img.transpose((1,2,0))
            img = (img-img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)
            entropy_map_teacher = entropy_maps_teacher[i]
            pred_teacher = preds_teacher[i].astype(np.uint8)
            entropy_map = entropy_maps[i]
            pred = preds[i].astype(np.uint8)
            gt = gt_segs[i].astype(np.uint8)
            img_uncertain_teacher = self.add_heatmap(img,entropy_map_teacher)
            img_pred_teacher = self.add_mask(img,pred_teacher)
            img_uncertain = self.add_heatmap(img,entropy_map)
            img_pred = self.add_mask(img,pred)
            img_gt = self.add_mask(img,gt)
            concat_results.append(np.concatenate([img,img_uncertain_teacher,img_pred_teacher,img_uncertain,img_pred,img_gt],axis=1))
        concat_results = np.concatenate(concat_results,axis=0)
        if is_resize:
            concat_results = cv2.resize(concat_results,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.opt['checkpoint_dir'], 'visuals',  'img_entt_predt_ent_pred_gt' + str(epoch) + '.png'),concat_results[:,:,::-1])
        
    def display_current_Pseudo(self, epoch, visuals,is_resize=False):
        
        images,preds_s,preds_w,gt_segs = visuals['images'],visuals['pred_s'],visuals['pred_w'],visuals['gt_segs']
        concat_results = []
        for i in range(min(images.shape[0],8)):
            img = images[i]
            if img.ndim == 2:
                img = np.stack([img,img,img],axis=-1)
            elif img.ndim == 3:
                img = img.transpose((1,2,0))
            img = (img-img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)
            pred_s = preds_s[i].astype(np.uint8)
            pred_w = preds_w[i].astype(np.uint8)
            gt = gt_segs[i].astype(np.uint8)
            img_pred_s = self.add_mask(img,pred_s)
            img_pred_w = self.add_mask(img,pred_w)
            img_gt = self.add_mask(img,gt)
            concat_results.append(np.concatenate([img,img_pred_s,img_pred_w,img_gt],axis=1))
        concat_results = np.concatenate(concat_results,axis=0)
        if is_resize:
            concat_results = cv2.resize(concat_results,dsize=None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self.opt['checkpoint_dir'], 'visuals',  'img_pred_pseudo_gt' + str(epoch) + '.png'),concat_results[:,:,::-1])
        

    def plot_current_losses(self, epoch, losses):
        self.summary_writer.add_scalars('Losses', losses, epoch)

    def plot_current_metrics(self, epoch, metrics, name='Metrics'):
        self.summary_writer.add_scalars(name, metrics, epoch)

    def plot_current_histogram(self, epoch, data):
        for k, v in data.items():
            self.summary_writer.add_histogram('Histogram/' + k, v, epoch)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, iters, times, losses):
        """
        print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(iters: %d' % (iters)
        for k, v in times.items():
            message += ", %s time: %.3fs" % (k, v)
        message += ") "
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v.mean())

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

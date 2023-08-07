import torch
import torch.nn as nn
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset,PatientDataset,MyBatchSampler
from torch.utils.data import DataLoader

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice, SoftMatchWeighting,FixedThresholding, DistAlignEMA

from tqdm import tqdm
import pdb


class PseudoLabel_Trainer():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        
        ### initialize dataloaders
        if self.opt['patient_level_dataloader']:
            train_dataset = PatientDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True, weak_strong_aug=True)
            patient_sampler = MyBatchSampler(train_dataset,self.opt['batch_size'])
            self.train_dataloader = DataLoader(train_dataset,batch_sampler=patient_sampler,num_workers=self.opt['num_workers'])
        else:
            self.train_dataloader = DataLoader(
                MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True, weak_strong_aug=True),
                batch_size=self.opt['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.opt['num_workers']
            )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='val', split_train=False),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models
        self.use_ema = self.opt['use_ema']
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')

        self.model = get_model(self.opt)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.opt['gpu_id'])
        
        if self.use_ema:
            self.ema_model = get_model(self.opt)
            self.ema_model.load_state_dict(checkpoint)
            self.ema_model = self.ema_model.to(self.opt['gpu_id'])
            self.ema_model.eval()
        
        self.total_epochs = self.opt['total_epochs']
        self.total_steps = self.total_epochs * len(self.train_dataloader)
       
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        ## pseudo label setting
        self.match_type = self.opt['match_type']
        
        
        if self.match_type == 'naive':
            self.masking = None
        elif self.match_type == 'fixmatch':
            self.masking = FixedThresholding(self.opt['p_cutoff'])
        elif self.match_type == 'softmatch':
            self.masking = SoftMatchWeighting(self.opt['num_classes'],per_class=self.opt['per_class'])
            self.use_dist_align = self.opt['use_dist_align']
            if self.use_dist_align:
                self.dist_align = DistAlignEMA(self.opt['num_classes'])
        ## losses
        self.criterion_pseudo = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1,2,2,1]).to(self.opt['gpu_id']),reduction='none')

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.set_seed(self.opt['random_seed'])
        self.model_resume()
        
    def set_seed(self,seed):     
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, step, dice):
        if step != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))

    
    def save_best_models(self, step, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        for file in os.listdir(os.path.join(checkpoint_dir, 'saved_models')):
            if 'best_model' in file:
                os.remove(os.path.join(checkpoint_dir, 'saved_models', file))
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state,os.path.join(checkpoint_dir, 'saved_models','best_model_step_{}_dice_{:.4f}.pth'.format(step,dice)))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params,lr=self.opt['lr'],betas=(0.9, 0.999), weight_decay=0.0005)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        return optimizer, scheduler
    
    def model_resume(self):
        if self.opt['continue_train']:
            if os.path.isfile(self.opt['resume']):
                print("=> Loading checkpoint '{}'".format(self.opt['resume']))
            state = torch.load(self.opt['resume'])
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.start_epoch = state['epoch']
        else:
            self.start_epoch = 0
            print("=> No checkpoint, train from scratch !")
    
    def ema_update(self):
        # Use the true average until the exponential average is more correct
        global_step = self.iter_counter.steps_so_far
        ema_decay = self.opt['ema_decay']
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data*(1 - ema_decay))

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        self.model.train()
        imgs_w, imgs_s, gt = data[0],data[1],data[2]
        b, c, h, w = imgs_w.shape
        _, prob_s = self.model(imgs_s)
        pred_s = torch.argmax(prob_s, dim=1)
        prob_s = rearrange(prob_s, 'b c h w -> (b h w) c')
        
        with torch.no_grad():
            if self.use_ema:
                    _, prob_w = self.ema_model(imgs_w)
            else:
                self.model.eval()
                _, prob_w = self.model(imgs_w)
                self.model.train()
                
        prob_w = rearrange(prob_w, 'b c h w -> (b h w) c')
        prob_w = F.softmax(prob_w,dim=1)
        pseudo_label = torch.argmax(prob_w, dim=1)
        if self.match_type == 'naive':
            mask = torch.ones_like(pseudo_label).float()
        elif self.match_type == 'fixmatch':
            mask = self.masking.masking(prob_w)
        elif self.match_type == 'softmatch':
            if self.use_dist_align:
                prob_w = self.dist_align.dist_align(prob_w)
            mask = self.masking.masking(prob_w)
        
        mask = mask.to(prob_s.device)
        pseudo_ce_loss = self.criterion_pseudo(prob_s, pseudo_label)
        pseudo_ce_loss = pseudo_ce_loss * mask
        loss = pseudo_ce_loss.mean()
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        gt = gt.flatten()
        fg_idxs = gt > 0
        fg_quantity = mask[fg_idxs].mean()
        tp_region = (pseudo_label[fg_idxs] == gt[fg_idxs]) * (mask[fg_idxs] / torch.sum(mask[fg_idxs]))
        fg_quality  = torch.sum(tp_region)
        trade_off = {}
        adapt_losses = {}
        adapt_losses['pseudo_ce_loss'] = loss.detach()
        trade_off['quality'] = fg_quality.detach()
        trade_off['quantity'] = fg_quantity.detach()
        pseudo_label[mask == 0] = 5
        pseudo_label = rearrange(pseudo_label,'(b h w) -> b h w',b=b,h=h,w=w)
        return adapt_losses,trade_off,pred_s,pseudo_label
    
    @torch.no_grad()
    def validate_one_step(self, data):
        
        self.model.eval()
        imgs = data[0]
        h,w = imgs.shape[2:]
        _,pred = self.model(imgs)
        pred = F.softmax(pred, dim=1)
        return pred

    
    def launch(self):
        self.initialize()
        self.train()

    def train(self):
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            train_losses = {}
            for it, (img_s,img_w,segs,_) in enumerate(train_iterator):
                # pdb.set_trace()
                img_s = img_s.to(self.opt['gpu_id'])
                img_w = img_w.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                with self.iter_counter.time_measurement("train"):
                    losses,trade_off,pred_s,pred_w = self.train_one_step([img_s,img_w,segs])
                    if self.use_ema:
                        self.ema_update()
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0) 
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    train_iterator.set_postfix(ce_loss = train_losses['pseudo_ce_loss'].item()/(it+1), quality = trade_off['quality'].item(), quantity = trade_off['quantity'].item())
                    
                with self.iter_counter.time_measurement("maintenance"):
                    if self.iter_counter.needs_displaying():
                        visuals = {'images':img_s[:,1].detach().cpu().numpy(),  'pred_s':pred_s.detach().cpu().numpy(),
                        'pred_w':pred_w.detach().cpu().numpy(),
                        'gt_segs':segs.detach().cpu().numpy()}
                        self.visualizer.display_current_Pseudo(self.iter_counter.steps_so_far,visuals)
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, losses)
                    self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far,trade_off,'Quantity_vs_Quality')
                    if self.iter_counter.needs_evaluation_steps():
                        val_metrics = {}
                        sample_dict = {}
                        val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))
                        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                            val_imgs = val_imgs.to(self.opt['gpu_id'])
                            val_segs = val_segs.to(self.opt['gpu_id'])

                            predict = self.validate_one_step([val_imgs, val_segs])
                            for i,name in enumerate(val_names):

                                sample_name,index = name.split('_')[0],int(name.split('_')[1])
                                sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(predict[i].detach().cpu(),val_segs[i].detach().cpu(),index)]
                            
                        pred_results_list = []
                        gt_segs_list = []
                        
                        for k in sample_dict.keys():

                            sample_dict[k].sort(key=lambda ele: ele[2])
                            preds = []
                            targets = []
                            for pred,target,_ in sample_dict[k]:
                                if target.sum()==0:
                                    continue
                                preds.append(pred)
                                targets.append(target)
                            pred_results_list.append(torch.stack(preds,dim=-1))
                            gt_segs_list.append(torch.stack(targets,dim=-1))
                                
                        # pdb.set_trace()
                        val_metrics['dice'] = mean_dice(pred_results_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
                            

                        if val_metrics['dice']['dice_avg'] > self.best_avg_dice:
                            self.best_avg_dice = val_metrics['dice']['dice_avg']
                            self.save_best_models(self.iter_counter.steps_so_far,val_metrics['dice']['dice_avg'])
                        else:
                            if self.iter_counter.needs_saving_steps():
                                self.save_models(self.iter_counter.steps_so_far,val_metrics['dice']['dice_avg'])
                        self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far, val_metrics['dice'],'Dice_metrics')
                        self.schedular.step()
                self.iter_counter.record_one_iteration()
            self.iter_counter.record_one_epoch()
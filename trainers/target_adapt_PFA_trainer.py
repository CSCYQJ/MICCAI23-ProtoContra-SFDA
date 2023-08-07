import torch
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset,PatientDataset,MyBatchSampler
from torch.utils.data import DataLoader
from losses import ProtoLoss

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice

from tqdm import tqdm
import pdb


class PFA_Trainer():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):

        ### initialize dataloaders
        if self.opt['patient_level_dataloader']:
            train_dataset = PatientDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True)
            patient_sampler = MyBatchSampler(train_dataset,self.opt['batch_size'])
            self.train_dataloader = DataLoader(train_dataset,batch_sampler=patient_sampler,num_workers=self.opt['num_workers'])
        else:
            self.train_dataloader = DataLoader(
                MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True),
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

        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.opt['gpu_id'])
        self.source_prototypes = self.model.outc.conv.weight.view((self.opt['num_classes']*self.opt['num_prototypes'],self.opt['output_dim']))
        self.model.outc.requires_grad = False
        self.total_epochs = self.opt['total_epochs']

        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)


        ## losses
        self.criterion_proto = ProtoLoss(self.opt['nav_t'],self.opt['beta'],self.opt['num_classes'],self.opt['gpu_id'],self.opt['s_par'])

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

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, min_lr=1e-7) # maximize dice score
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
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

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        imgs = data[0]

        target_f,predict = self.model(imgs,only_feature=True)

        target_f = rearrange(target_f, 'b c h w -> (b h w) c')
        t2p_loss, p2t_loss = self.criterion_proto(self.source_prototypes,target_f)
        loss = t2p_loss + p2t_loss
        
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


        adapt_losses = {}
        adapt_losses['t2p_loss'] = t2p_loss.detach()
        adapt_losses['p2t_loss'] = p2t_loss.detach()
        adapt_losses['total_loss'] = loss.detach()

        return predict, adapt_losses
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        _,predict = self.model(imgs)

        self.model.train()

        return predict

    def launch(self):
        self.initialize()
        self.train()
        
    def train(self):
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            train_losses = {}
            for it, (images,segs,_) in enumerate(train_iterator):
                # pdb.set_trace()
                images = images.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                with self.iter_counter.time_measurement("train"):
                    predicts, losses = self.train_one_step([images, segs])
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0) 
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    train_iterator.set_postfix(t2p_loss = train_losses['t2p_loss'].item()/(it+1), p2t_loss = train_losses['p2t_loss'].item()/(it+1), total_loss = train_losses['total_loss'].item()/(it+1))
                
                with self.iter_counter.time_measurement("maintenance"):
                    
                    if self.iter_counter.needs_displaying():
                        if isinstance(predicts, dict):
                            predicts = predicts['seg']
                        probs = F.softmax(predicts,dim=1)
                        entropy_maps = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                        visuals = {'images':images[:,1].detach().cpu().numpy(),'entropy_maps':entropy_maps.detach().cpu().numpy(),'preds':torch.argmax(predicts,dim=1).detach().cpu().numpy(),
                                   'gt_segs':segs.detach().cpu().numpy()}
                        self.visualizer.display_current_PFA(self.iter_counter.steps_so_far,visuals)
                        self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, losses)
                        
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
            
                
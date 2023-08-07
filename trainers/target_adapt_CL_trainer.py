import torch
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset,PatientDataset,MyBatchSampler
from torch.utils.data import DataLoader
from losses import PseudoLabel_Loss

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice

from tqdm import tqdm
import pdb


class CL_Trainer():
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
        self.teacher_model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.teacher_model.load_state_dict(checkpoint)
        self.model = self.model.to(self.opt['gpu_id'])
        self.teacher_model = self.teacher_model.to(self.opt['gpu_id'])
        self.source_prototypes = self.model.outc.conv.weight.view((self.opt['num_classes']*self.opt['num_prototypes'],self.opt['output_dim']))
        
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
        self.model.outc.requires_grad = False
        self.total_epochs = self.opt['total_epochs']
        self.total_steps = self.total_epochs * len(self.train_dataloader)

        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        # build class-wise memory bank
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        for i in range(self.opt['num_classes']):
            self.memobank.append([torch.zeros(0, self.opt['output_dim'])])
            self.queue_size.append(30000)
            self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        self.queue_size[0] = 50000
        
        # build query prototype
        if self.opt['momentum_prototype']:
            self.momentum_prototype = torch.zeros(
                (
                    self.opt['num_classes'],
                    self.opt['num_queries'],
                    1,
                    self.opt['output_dim'],
                )).to(self.opt['gpu_id'])
        else:
            self.momentum_prototype = None

        ## losses
        self.criterion_pseudo = PseudoLabel_Loss()

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
    
    def label_onehot(self, inputs):
        one_hot_inputs = F.one_hot(inputs,self.opt['num_classes'])
        return one_hot_inputs.permute(0, 3, 1, 2)
    
    def compute_contra_memobank_loss(
        self,
        target_f,
        pseudo_label_teacher,
        prob_teacher,
        low_mask,
        high_mask,
        target_f_teacher,
    ):
        # current_class_threshold: delta_p (0.3)
        current_class_threshold = self.opt['current_class_threshold']
        low_rank, high_rank = self.opt['low_rank'], self.opt['high_rank']
        temp = 0.1
        num_queries = self.opt['num_queries']
        num_negatives = self.opt['num_negatives']

        num_feat = target_f.shape[1]
        low_valid_pixel = pseudo_label_teacher * low_mask
        # high_valid_pixel = pseudo_label_teacher * high_mask

        target_f = target_f.permute(0, 2, 3, 1)
        target_f_teacher = target_f_teacher.permute(0, 2, 3, 1)

        seg_feat_all_list = []
        seg_feat_low_entropy_list = []  # candidate anchor pixels
        seg_num_list = []  # the number of low_valid pixels in each class
        seg_proto_list = []  # the center of each class

        _, prob_indices_teacher = torch.sort(prob_teacher, 1, True)
        prob_indices_teacher = prob_indices_teacher.permute(
            0, 2, 3, 1
        )  # (num_unlabeled, h, w, num_cls)

        valid_classes = []
        positive_masks = []
        negative_masks = []
        new_keys = []

        for i in range(self.opt['num_classes']):
            low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
            prob_seg = prob_teacher[:, i, :, :]
            target_f_mask_low_entropy = (
                prob_seg > current_class_threshold
            ) * low_valid_pixel_seg.bool()

            seg_feat_all_list.append(target_f[low_valid_pixel_seg.bool()])
            seg_feat_low_entropy_list.append(target_f[target_f_mask_low_entropy])

            # positive sample: center of the class
            seg_proto_list.append(
                torch.mean(
                    target_f_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
                )
            )

            # generate class mask for unlabeled data
            # prob_i_classes = prob_indices_teacher[target_f_mask_high_entropy[num_labeled :]]
            class_mask = torch.sum(
                prob_indices_teacher[:, :, :, low_rank:high_rank].eq(i), dim=3
            ).bool()


            negative_mask = high_mask[:,0].bool() * class_mask

            keys = target_f_teacher[negative_mask].detach()

            new_keys.append(
                self.dequeue_and_enqueue(
                    keys=keys,
                    class_idx=i
                )
            )

            if low_valid_pixel_seg.sum() > 0:
                seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
                valid_classes.append(i)
                positive_masks.append(target_f_mask_low_entropy)
                negative_masks.append(negative_mask)

        if (
            len(seg_num_list) <= 1
        ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
            
            return new_keys, torch.tensor(0.0) * target_f.sum(), valid_classes, positive_masks, negative_masks

        else:
            reco_loss = torch.tensor(0.0).to(self.opt['gpu_id'])
            seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
            valid_seg = len(seg_num_list)  # number of valid classes


            prototype = torch.zeros(
                (prob_indices_teacher.shape[-1], num_queries, 1, num_feat)
            ).to(self.opt['gpu_id'])

            for i in range(valid_seg):
                if (
                    len(seg_feat_low_entropy_list[i]) > 0
                    and self.memobank[valid_classes[i]][0].shape[0] > 0
                ):
                    # select anchor pixel
                    seg_low_entropy_idx = torch.randint(
                        len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                    )
                    anchor_feat = (
                        seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().to(self.opt['gpu_id'])
                    )
                else:
                    # in some rare cases, all queries in the current query class are easy
                    reco_loss = reco_loss + 0 * target_f.sum()
                    continue

                # apply negative key sampling from memory bank (with no gradients)
                with torch.no_grad():
                    negative_feat = self.memobank[valid_classes[i]][0].clone().to(self.opt['gpu_id'])

                    high_entropy_idx = torch.randint(
                        len(negative_feat), size=(num_queries * num_negatives,)
                    )
                    negative_feat = negative_feat[high_entropy_idx]
                    negative_feat = negative_feat.reshape(
                        num_queries, num_negatives, num_feat
                    )
                    if self.opt['use_source_prototypes']:
                        positive_feat = (
                            self.source_prototypes[i]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(num_queries, 1, 1)
                            .to(self.opt['gpu_id'])
                        )  # (num_queries, 1, num_feat)
                        
                    else:
                        positive_feat = (
                            seg_proto[i]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(num_queries, 1, 1)
                            .to(self.opt['gpu_id'])
                        )  # (num_queries, 1, num_feat)
                        if self.momentum_prototype is not None:
                            if not (self.momentum_prototype == 0).all():
                                ema_decay = min(1 - 1 / (self.iter_counter.steps_so_far / self.opt['batch_size']), 0.999)
                                positive_feat = (
                                    1 - ema_decay
                                ) * positive_feat + ema_decay * self.momentum_prototype[
                                    valid_classes[i]
                                ]
                                
                            prototype[valid_classes[i]] = positive_feat.clone()

                    all_feat = torch.cat(
                        (positive_feat, negative_feat), dim=1
                    )  # (num_queries, 1 + num_negative, num_feat)

                seg_logits = torch.cosine_similarity(
                    anchor_feat.unsqueeze(1), all_feat, dim=2
                )
                # pdb.set_trace()

                reco_loss = reco_loss + F.cross_entropy(
                    seg_logits / temp, torch.zeros(num_queries).long().to(self.opt['gpu_id'])
                )

            if self.momentum_prototype is None:
                return new_keys, reco_loss / valid_seg, valid_classes, positive_masks, negative_masks

            else:
                self.momentum_prototype = prototype
                return new_keys, reco_loss / valid_seg, valid_classes, positive_masks, negative_masks

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        alpha_t = self.opt['low_entropy_threshold'] * (
            1 - self.iter_counter.steps_so_far / self.total_steps
        )
        # get losses
        imgs = data[0]
        target_f_teacher, pred_teacher = self.teacher_model(imgs,only_feature=True)
        prob_teacher = F.softmax(pred_teacher, dim=1)
        pseudo_label_teacher = torch.argmax(pred_teacher, dim=1)

        # unsupervised loss
        drop_percent = self.opt['drop_percent']
        percent_unreliable = (100 - drop_percent) * (1 - self.iter_counter.steps_so_far / self.total_steps)
        drop_percent = 100 - percent_unreliable
        
        target_f,pred = self.model(imgs,only_feature=True)
        total_loss = 0
        adapt_losses = {}
        if self.opt['use_pseudo']:
            pesudo_unsup_loss = self.criterion_pseudo(pred,pseudo_label_teacher.clone(),drop_percent,prob_teacher.detach())
            total_loss += pesudo_unsup_loss
            adapt_losses['pseudo_loss'] = pesudo_unsup_loss.detach()
        if self.opt['use_contra']:
            with torch.no_grad():
                entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + 1e-10), dim=1)
                low_entropy_mask = 0
                high_entropy_mask = 0
                for i in range(pred.shape[1]):
                    if torch.sum(entropy[pseudo_label_teacher == i]) > 5:
                        if i == 0:
                            low_thresh,high_thresh = np.percentile(
                            entropy[pseudo_label_teacher == i].detach().cpu().numpy().flatten(), [90 - alpha_t/4, 100 - alpha_t/4]
                            )
                        else:
                            low_thresh,high_thresh = np.percentile(
                            entropy[pseudo_label_teacher == i].detach().cpu().numpy().flatten(), [90 - alpha_t, 100 - alpha_t]
                            )
                        high_entropy_mask = high_entropy_mask + entropy.ge(high_thresh).bool() * (pseudo_label_teacher == i).bool()
                        low_entropy_mask = low_entropy_mask + entropy.le(low_thresh).bool() * (pseudo_label_teacher == i).bool()
                

            new_keys, contra_loss, valid_classes,pos_masks,neg_masks = self.compute_contra_memobank_loss(
                                    target_f,
                                    self.label_onehot(pseudo_label_teacher),
                                    prob_teacher,
                                    low_entropy_mask.unsqueeze(1),
                                    high_entropy_mask.unsqueeze(1),
                                    target_f_teacher)
            total_loss += contra_loss
            adapt_losses['contra_loss'] = contra_loss.detach()
        
        self.grad_scaler.scale(total_loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


        adapt_losses = {}

        adapt_losses['total_loss'] = total_loss.detach()

        return pred_teacher,pred,adapt_losses
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, class_idx):
        
        queue=self.memobank[class_idx]
        queue_ptr=self.queue_ptrlis[class_idx]
        queue_size=self.queue_size[class_idx]
        keys = keys.clone().cpu()

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)

        queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
        if queue[0].shape[0] >= queue_size:
            queue[0] = queue[0][-queue_size:, :]
            ptr = queue_size
        else:
            ptr = (ptr + batch_size) % queue_size  # move pointer

        queue_ptr[0] = ptr

        return batch_size
    
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
                    predicts_teacher,predicts, losses = self.train_one_step([images, segs])
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0) 
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    train_iterator.set_postfix(total_loss = train_losses['total_loss'].item()/(it+1))
                
                with self.iter_counter.time_measurement("maintenance"):
                    
                    if self.iter_counter.needs_displaying():
                        probs_teacher = F.softmax(predicts_teacher,dim=1)
                        probs = F.softmax(predicts,dim=1)
                        entropy_maps_teacher = -torch.sum(probs_teacher * torch.log(probs_teacher + 1e-10), dim=1)
                        entropy_maps = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                        visuals = {'images':images[:,1].detach().cpu().numpy(),     'entropy_maps_teacher':entropy_maps_teacher.detach().cpu().numpy(),'preds_teacher':torch.argmax(probs_teacher,dim=1).detach().cpu().numpy(),'entropy_maps':entropy_maps.detach().cpu().numpy(),
                        'preds':torch.argmax(probs,dim=1).detach().cpu().numpy(),
                        'gt_segs':segs.detach().cpu().numpy()}
                        self.visualizer.display_current_CL(self.iter_counter.steps_so_far,visuals)
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
            
                
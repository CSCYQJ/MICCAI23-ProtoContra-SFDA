import torch


class Masking():
    """
    Base MaskingHook, used for computing the mask of unlabeled (consistency) loss
    define MaskingHook in each algorithm when needed, and call hook inside each train_step
    easy support for other settings
    """
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def update(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def masking(self, logits_x_ulb=None, idx_ulb=None, *args, **kwargs):
        """
        generate mask for unlabeled loss

        Args:
            logits_x_ulb: unlabeled batch logits (or probs, need to set softmax_x_ulb to False)
            idx_ulb: unlabeled batch index
        """
        raise NotImplementedError


class FixedThresholding(Masking):
    """
    Common Fixed Threshold used in fixmatch, uda, pseudo label, et. al.
    """
    def __init__(self, p_cutoff, *args, **kwargs):
        self.p_cutoff = p_cutoff
        
    @torch.no_grad()
    def masking(self, logits_x_ulb, *args, **kwargs):

        # logits is already probs
        probs_x_ulb = logits_x_ulb.detach()
        max_probs, _ = torch.max(probs_x_ulb, dim=1)
        mask = max_probs.ge(self.p_cutoff).to(max_probs.dtype)
        return mask
    
# class FlexMatchThresholdingHook(MaskingHook):
#     """
#     Adaptive Thresholding in FlexMatch
#     """
#     def __init__(self, p_cutoff, ulb_dest_len, num_classes, thresh_warmup=True, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.p_cutoff = p_cutoff
#         self.ulb_dest_len = ulb_dest_len
#         self.num_classes = num_classes
#         self.thresh_warmup = thresh_warmup
#         self.selected_label = torch.ones((self.ulb_dest_len,), dtype=torch.long, ) * -1
#         self.classwise_acc = torch.zeros((self.num_classes,))

#     @torch.no_grad()
#     def update(self, *args, **kwargs):
#         pseudo_counter = Counter(self.selected_label.tolist())
#         if max(pseudo_counter.values()) < self.ulb_dest_len:  # not all(5w) -1
#             if self.thresh_warmup:
#                 for i in range(self.num_classes):
#                     self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
#             else:
#                 wo_negative_one = deepcopy(pseudo_counter)
#                 if -1 in wo_negative_one.keys():
#                     wo_negative_one.pop(-1)
#                 for i in range(self.num_classes):
#                     self.classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())

#     @torch.no_grad()
#     def masking(self, logits_x_ulb, idx_ulb, *args, **kwargs):
#         if not self.selected_label.is_cuda:
#             self.selected_label = self.selected_label.to(logits_x_ulb.device)
#         if not self.classwise_acc.is_cuda:
#             self.classwise_acc = self.classwise_acc.to(logits_x_ulb.device)

#         # logits is already probs
#         probs_x_ulb = logits_x_ulb.detach()
#         max_probs, max_idx = torch.max(probs_x_ulb, dim=1)
#         # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
#         # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
#         mask = max_probs.ge(self.p_cutoff * (self.classwise_acc[max_idx] / (2. - self.classwise_acc[max_idx])))  # convex
#         # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
#         select = max_probs.ge(self.p_cutoff)
#         mask = mask.to(max_probs.dtype)

#         # update
#         if idx_ulb[select == 1].nelement() != 0:
#             self.selected_label[idx_ulb[select == 1]] = max_idx[select == 1]
#         self.update()

#         return mask

class SoftMatchWeighting(Masking):
    """
    SoftMatch learnable truncated Gaussian weighting
    """
    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum
        self.prob_max_mu_t = None
        self.prob_max_var_t = None
        
        # initialize Gaussian mean and variance
        # if not self.per_class:
        #     self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
        #     self.prob_max_var_t = torch.tensor(1.0)
            
        # else:
        #     self.prob_max_mu_t = torch.ones((self.num_classes)) / self.num_classes
        #     self.prob_max_var_t =  torch.ones((self.num_classes))
            

    @torch.no_grad()
    def update(self, probs_x_ulb):
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs) # torch.quantile(max_probs, 0.5)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            if self.prob_max_mu_t is not None:
                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t.item()
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t.item()
            else:
                self.prob_max_mu_t = prob_max_mu_t
                self.prob_max_var_t = prob_max_var_t
        else:
            prob_max_mu_t = torch.zeros((self.num_classes)).to(probs_x_ulb.device)
            prob_max_var_t = torch.ones((self.num_classes)).to(probs_x_ulb.device)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            if self.prob_max_mu_t is not None:
                self.prob_max_mu_t = self.m * self.prob_max_mu_t + (1 - self.m) * prob_max_mu_t
                self.prob_max_var_t = self.m * self.prob_max_var_t + (1 - self.m) * prob_max_var_t
            else:
                self.prob_max_mu_t = prob_max_mu_t
                self.prob_max_var_t = prob_max_var_t
        return max_probs, max_idx
    
    @torch.no_grad()
    def masking(self, logits_x_ulb, *args, **kwargs):
        # logits is already probs
        probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)
        if not self.prob_max_mu_t.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(logits_x_ulb.device)
        if not self.prob_max_var_t.is_cuda:
            self.prob_max_var_t = self.prob_max_var_t.to(logits_x_ulb.device)
        max_probs, max_idx = probs_x_ulb.max(dim=1)
        # compute weight
        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]
        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask
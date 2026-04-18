"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, device, temperature=1.0):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature

    def forward(self, text_features, image_features, t_label, i_targets): 
        batch_size = text_features.shape[0] 
        batch_size_N = image_features.shape[0] 
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device) 

        logits = torch.div(torch.matmul(text_features, image_features.T), self.temperature)
        #print('SupConLoss: logits max= ', logits.max(1)[0].cpu().detach())
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach() 
        exp_logits = torch.exp(logits) 
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
        loss = - mean_log_prob_pos.mean()

        return loss



def CrossModalConLoss(img_features, text_features, img_labels, text_labels, text_other_feats=None, text_other_labels=None, temp=0.07): 
    
    if text_other_feats is not None:
        text_all_feats = torch.cat((text_features, text_other_feats), dim=0)
        text_all_labels = torch.cat((text_labels, text_other_labels))
    else:
        text_all_feats = text_features
        text_all_labels = text_labels

    batch_size = img_features.shape[0] 
    batch_size_N = text_all_feats.shape[0] 

    mask = torch.eq(img_labels.unsqueeze(1).expand(batch_size, batch_size_N), \
        text_all_labels.unsqueeze(0).expand(batch_size, batch_size_N)).float().cuda()

    logits = torch.div(torch.matmul(img_features, text_all_feats.T), temp)
    # for numerical stability
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach() 
    exp_logits = torch.exp(logits) 
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1) 
    loss = - mean_log_prob_pos.mean()

    return loss




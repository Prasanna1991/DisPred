import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.autograd import grad


base_temperature=0.07

def getConstrastiveLoss(features, labels, device, temperature, normalize=False):
    if normalize:
        features = F.normalize(features, dim=1)
    features = features.repeat(1,2).view(features.size(0), 2, -1)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, torch.t(labels)).float().to(device)
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, torch.t(contrast_feature)),
        temperature)
    
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)

    # mask-out self-contrast cases
    logits_mask = torch.ones_like(mask).scatter_(1,
        torch.arange(features.size(0) * anchor_count).view(-1, 1).to(device),
        0
    ).to(device)

    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, features.size(0)).mean()

    return loss
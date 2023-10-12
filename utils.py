
import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from fastai.vision.all import *
import subprocess


#### Grad scaler
# Fix fastai bug to enable fp16 training with dictionaries


def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self):
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision





def adjusted_loss(pred, target):
    # Extracting predictions corresponding to the mask
    p = pred[target['mask'][:,:pred.shape[1]]]

    # Extracting true values and clipping them
    y = target['react'][target['mask']].clip(0, .83)

    # Calculating the unweighted loss
    individual_losses = F.l1_loss(p, y, reduction='none')

    # Removing NaN values from individual_losses
    mask_non_nan = ~torch.isnan(individual_losses)
    individual_losses = individual_losses[mask_non_nan]

    # Extracting and clipping the signal-to-noise ratios and ensuring they have the same shape as the loss
    sn = target['sn'].clip(0, 1)

    # Expanding sn to match the shape of predictions and applying the mask
    sn = sn.unsqueeze(1).expand(-1, pred.shape[1], -1)
    sn = sn[target['mask'][:,:pred.shape[1]]]

    # Filtering sn with the same condition used for individual_losses
    sn = sn[mask_non_nan]

    # Weighting the loss by the clipped signal-to-noise ratio and calculating the mean
    final_loss = (individual_losses * sn).mean()

    # Clearing temporary tensors to free up memory
    del p, y, individual_losses, mask_non_nan, sn
    torch.cuda.empty_cache()

    return final_loss




"""
def loss(pred, target):
    # Extracting predictions corresponding to the mask
    p = pred[target['mask'][:,:pred.shape[1]]]

    # Extracting true values and clipping them
    y = target['react'][target['mask']].clip(0, .83)

    # Calculating the unweighted loss
    individual_losses = F.l1_loss(p, y, reduction='none')

    # Removing NaN values from individual_losses
    mask_non_nan = ~torch.isnan(individual_losses)
    individual_losses = individual_losses[mask_non_nan]

    # Filtering sn with the same condition used for individual_losses
    sn = target['sn'][target['mask'][:,:pred.shape[1]]]
    sn = sn[mask_non_nan]

    # Weighting the loss by the clipped signal-to-noise ratio and calculating the mean
    final_loss = (individual_losses * sn).mean()

    return final_loss"""



def loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0, .83) #.clip(.67,1.5) .clip(0,1)

    not_unk = ~torch.isnan(y)
    p = p[not_unk]
    y = y[not_unk]

    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()

    return loss



class MAE(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x,self.y = [],[]

    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,.83) #.clip(0,.83)  .clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss




def push_to_github(file_path, commit_message):
    try:
        # Add the file to the local repository
        subprocess.run(["git", "add", file_path], check=True)
        
        # Commit the change
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # Push the commit to the remote repository
        subprocess.run(["git", "push"], check=True)
        print(f"Successfully pushed {file_path} to GitHub.")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to push {file_path} to GitHub.")
        print(str(e))

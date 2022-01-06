# -*- coding: utf-8 -*-
import numpy as np
import torch
from model import *

#======================================================================================================================
def calc_stats_data(array):
  mean = torch.mean(array,dim=0)
  diffs = array - mean
  var = torch.mean(torch.pow(diffs, 2.0),dim=0)
  std = torch.sqrt(var)
  zscores = diffs / std
  skews = torch.mean(torch.pow(zscores, 3.0),dim=0)
  kurtoses = torch.mean(torch.pow(zscores, 4.0),dim=0) - 3.0 
  # print(mean, std, skews, kurtoses)
  return torch.cat((mean, std, skews, kurtoses)).reshape(1,-1)

def calc_stats_data_bin(array):
  mean = torch.mean(array,dim=0)
  diffs = array - mean
  var = torch.mean(torch.pow(diffs, 2.0),dim=0)
  std = torch.sqrt(var)
  return torch.cat((mean, std)).reshape(1,-1)

def stats_data(Yobs, X, W, n_sources, train_size, bin_feats_4moments=True, cont_feats=[], bin_feats=[]):
  lst = []
  for s in range(n_sources):
    idx = range(s*train_size, (s+1)*train_size)
    Yobs_g = Yobs[idx,:]
    X_g = X[idx,:]
    W_g = W[idx,:].reshape(-1,1)
    Yobs0_g = Yobs_g[W_g==0].reshape(-1,1)
    Yobs1_g = Yobs_g[W_g==1].reshape(-1,1)

    if bin_feats_4moments==True:
      lst.append(torch.cat((calc_stats_data(Yobs0_g), calc_stats_data(Yobs1_g),
                            calc_stats_data(W_g), calc_stats_data(X_g)),dim=1))
    else:

      lst.append(torch.cat((calc_stats_data(Yobs0_g), calc_stats_data(Yobs1_g),
                            calc_stats_data_bin(W_g), calc_stats_data(X_g[:,cont_feats]),
                            calc_stats_data_bin(X_g[:,bin_feats])),dim=1))
  stats_ = torch.cat(lst, dim=0)
  return stats_

def detach_to_numpy(p):
  lst = []
  for x in p:
    lst.append(x.detach().cpu().numpy())
  return lst
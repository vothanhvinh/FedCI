# -*- coding: utf-8 -*-
import numpy as np
import torch
from model import *
from model_utils import *
from evaluation import Evaluation

#======================================================================================================================
def train_model(Yobs, X, W, Yobste, Xte, Wte, y, y_cf, x, w, yte, y_cfte, xte, wte,
                n_sources, train_size, test_size, dim_h=200, n_iterations=1000, learning_rate=1e-3,
                display_per_iter=50, inter_dependency=True, bin_feats_4moments=True, cont_feats=[], bin_feats=[]):
  # Collect statistic of each group
  stats_feats = stats_data(Yobs, X, W, n_sources, train_size,
                           bin_feats_4moments=bin_feats_4moments, cont_feats=cont_feats, bin_feats=bin_feats)
  model_server = FederatedCausalNet(Yobs=Yobs[:1,:], X=X[:1,:], W=W[:1,:], stats_feats=stats_feats,
                                    dim_h=dim_h, n_sources=n_sources, idx_source=0, inter_dependency=inter_dependency,
                                    bin_feats_4moments=bin_feats_4moments, cont_feats=cont_feats).to(device)
  model_sources = []
  for s in range(n_sources):
    idx = range(s*train_size, (s+1)*train_size)
    model_s = FederatedCausalNet(Yobs=Yobs[idx,:], X=X[idx,:], W=W[idx,:], stats_feats=stats_feats, 
                                 dim_h=dim_h, n_sources=n_sources, idx_source=s, inter_dependency=inter_dependency,
                                 bin_feats_4moments=bin_feats_4moments, cont_feats=cont_feats).to(device)
    model_sources.append(model_s)

  optimizer_server = torch.optim.Adagrad(model_server.parameters(), lr=learning_rate)
  optimizer_sources = [torch.optim.Adagrad(model_sources[s].parameters(), lr=learning_rate) for s in range(n_sources)]

  val_neg_elbo_best = np.Inf
  early_stopping_count = 0
  early_stopping_max = 100

  loss_store = []
  for t in range(n_iterations):
    # Compute gradient in each source
    for s in range(len(model_sources)):
      # Compute gradient on Source s
      idx = range(s*train_size, (s+1)*train_size)
      loss_source = model_sources[s](s)
      
      optimizer_sources[s].zero_grad()
      loss_source.backward()
      grad_dict_source = {key:param.grad for key, param in model_sources[s].named_parameters()} # store gradients to grad_dict_source

      # Update gradient on Server
      loss_server = model_server(0)
      loss_server.backward() # The purpuse of this line is to allocate memory for gradient of each parameter, i.e param.grad as below
      optimizer_server.zero_grad()
      for key, param in model_server.named_parameters():
        if (param.grad is not None) and param.requires_grad:
          param.grad += grad_dict_source[key]
      optimizer_server.step()
    
    # Collect all parameters on Server
    # state_server = model_server.state_dict()
    param_dict_server = {key:param.data for key, param in model_server.named_parameters()}
    # server_params = []
    # for group in optimizer_server.param_groups:
    #   for p in group['params']:
    #     if p.requires_grad:
    #       server_params.append(p.data)


    # Send updated parameters to Sources. In a real application, we need to make a tcp connection to sources machines to send parameters.
    val_neg_elbo = 0
    for s in range(len(model_sources)):
      # model_sources[s].load_state_dict(state_server)
      for key, param in model_sources[s].named_parameters():
        if param.requires_grad and param_dict_server[key] is not None:
          # param.data -= param.data
          param.data = param_dict_server[key] + 0 # + 0 will copy content of param_dict_server[key] to param.data; otherwise, it only assign reference

      # # Compute loss on validation data
      # idxva = range(s*val_size, (s+1)*val_size)
      # for j in range(10):
      #   neg_elbo = model_sources[s](Yobsva[idxva,:], Xva[idxva,:], Wva[idxva,:], mean_Yobs0, mean_Yobs1, mean_X, mean_W, s)
      #   val_neg_elbo += neg_elbo.item()

    # # Save the current best model (similar to CEVAE)
    # if val_neg_elbo_best > val_neg_elbo:
    #   print('Improved ELBO: old = {}, new = {}'.format(val_neg_elbo_best, val_neg_elbo))
    #   val_neg_elbo_best = val_neg_elbo
    #   torch.save(model_server.state_dict(), 'mytraining-data3.pt')
    #   early_stopping_count = 0
    # else:
    #   early_stopping_count += 1
    
    if t%display_per_iter == 0:
      print('Iter: {}'.format(t))

      Ymis_pred_tr = []
      err_ate_source_tr = []
      Ymis_pred_te = []
      err_ate_source_te = []
      for s in range(len(model_sources)):
        idx = range(s*train_size, (s+1)*train_size)
        idxte = range(s*test_size, (s+1)*test_size)
        meanATE_tr, varATE_tr, meanYmis_tr, covYmis_tr = detach_to_numpy(model_sources[s].predictATE(Yobs[idx,:], X[idx,:],
                                                                                                 W[idx,:], idx_source=s))
        meanATE_te, varATE_te, meanYmis_te, covYmis_te = detach_to_numpy(model_sources[s].predictATE(Yobste[idxte,:], Xte[idxte,:],
                                                                                                 Wte[idxte,:], idx_source=s))
        Ymis_pred_tr.append(meanYmis_tr.reshape(-1))
        Ymis_pred_te.append(meanYmis_te.reshape(-1))

        y0 = (1-w[idx])*y[idx] + w[idx]*y_cf[idx]
        y1 = w[idx]*y[idx] + (1-w[idx])*y_cf[idx]
        y0pred = (1-w[idx])*y[idx] + w[idx]*meanYmis_tr.reshape(-1)
        y1pred = w[idx]*y[idx] + (1-w[idx])*meanYmis_tr.reshape(-1)
        evaluator_tr = Evaluation(m0=y0, m1=y1)
        err_ate_source_tr.append(evaluator_tr.absolute_err_ate(y0pred=y0pred,y1pred=y1pred))

        y0 = (1-wte[idxte])*yte[idxte] + wte[idxte]*y_cfte[idxte]
        y1 = wte[idxte]*yte[idxte] + (1-wte[idxte])*y_cfte[idxte]
        y0pred = (1-wte[idxte])*yte[idxte] + wte[idxte]*meanYmis_te.reshape(-1)
        y1pred = wte[idxte]*yte[idxte] + (1-wte[idxte])*meanYmis_te.reshape(-1)
        evaluator_te = Evaluation(m0=y0, m1=y1)
        err_ate_source_te.append(evaluator_te.absolute_err_ate(y0pred=y0pred,y1pred=y1pred))

      Ymis_pred_tr = np.concatenate(Ymis_pred_tr)
      Ymis_pred_te = np.concatenate(Ymis_pred_te)

      y0 = (1-w)*y + w*y_cf
      y1 = w*y + (1-w)*y_cf
      y0pred = (1-w)*y + w*Ymis_pred_tr.reshape(-1)
      y1pred = w*y + (1-w)*Ymis_pred_tr.reshape(-1)
      evaluator_tr = Evaluation(m0=y0, m1=y1)
      print('Train: pehe {}, err_ate {}, err_ate(s) {}'.format(evaluator_tr.pehe(y0pred=y0pred,y1pred=y1pred),
                                                               evaluator_tr.absolute_err_ate(y0pred=y0pred,y1pred=y1pred),
                                                               np.mean(err_ate_source_tr)))

      y0 = (1-wte)*yte + wte*y_cfte
      y1 = wte*yte + (1-wte)*y_cfte
      y0pred = (1-wte)*yte + wte*Ymis_pred_te.reshape(-1)
      y1pred = wte*yte + (1-wte)*Ymis_pred_te.reshape(-1)
      evaluator_te = Evaluation(m0=y0, m1=y1)
      print('Test: pehe {}, err_ate {}, err_ate(s) {}'.format(evaluator_te.pehe(y0pred=y0pred,y1pred=y1pred),
                                                             evaluator_te.absolute_err_ate(y0pred=y0pred,y1pred=y1pred),
                                                             np.mean(err_ate_source_te)))
      print('=================================================================')
    # if early_stopping_count == early_stopping_max:
    #   print('Early Stopping')
    #   break
    # loss_store.append(loss.item())
  return model_server, model_sources
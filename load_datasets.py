# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class DATA1(object):
  def __init__(self,use_one_hot=False):
    data = np.load('data/DATA-1/data-synthetic.npz', allow_pickle=True)
    self.data_lst_Delta = data['data_lst_Delta']
    self.n_sources = data['n_sources'].item()
    self.source_size = data['source_size'].item()
    self.Delta_lst = data['Delta_lst']
    self.n_replicates = data['n_replicates'].item()
    self.train_size = 50
    self.test_size = 550
    self.val_size = 400
    self.use_one_hot = use_one_hot

    # which features are binary
    self.binfeats = []
    # which features are continuous
    self.contfeats = list(range(0,20))

  def get_train_valid_test_combined(self, m_sources):
    for i in range(self.n_replicates):
      data = self.data_lst_Delta[0][i]

      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      if self.use_one_hot==True and m_sources > 0:
        one_hot = pd.get_dummies(np.concatenate([[i]*self.source_size for i in range(self.n_sources)])).values
        one_hot = one_hot[:,:m_sources]
        x = np.concatenate((x,one_hot),axis=1)
        print('Use one-hot encoding, d_x = {}'.format(x.shape[1]))
      else:
        print('Do not use one-hot encoding, d_x = {}'.format(x.shape[1]))

      itr = np.concatenate([range(i,i+self.train_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      ite = np.concatenate([range(i+self.train_size,i+self.train_size+self.test_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      iva = np.concatenate([range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
      valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
      test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
      yield train, valid, test, self.contfeats, self.binfeats

class DATA2(object):
  def __init__(self,use_one_hot=False):
    data = np.load('data/DATA-2/data-synthetic-largescale.npz', allow_pickle=True)
    self.data_lst_Delta = data['data_lst_Delta']
    self.n_sources = data['n_sources'].item()
    self.source_size = data['source_size'].item()
    self.Delta_lst = data['Delta_lst']
    self.n_replicates = data['n_replicates'].item()
    self.train_size = 50
    self.test_size = 550
    self.val_size = 400
    self.use_one_hot = use_one_hot

    # which features are binary
    self.binfeats = []
    # which features are continuous
    self.contfeats = list(range(0,20))

  def get_train_valid_test_combined(self, m_sources):
    for i in range(self.n_replicates):
      data = self.data_lst_Delta[0][i]

      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      if self.use_one_hot==True and m_sources > 0:
        one_hot = pd.get_dummies(np.concatenate([[i]*self.source_size for i in range(self.n_sources)])).values
        one_hot = one_hot[:,:m_sources]
        x = np.concatenate((x,one_hot),axis=1)
        print('Use one-hot encoding, d_x = {}'.format(x.shape[1]))
      else:
        print('Do not use one-hot encoding, d_x = {}'.format(x.shape[1]))

      itr = np.concatenate([range(i,i+self.train_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      ite = np.concatenate([range(i+self.train_size,i+self.train_size+self.test_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      iva = np.concatenate([range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
      valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
      test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
      yield train, valid, test, self.contfeats, self.binfeats


# IHDP dataset
class IHDP(object):
  def __init__(self,use_one_hot=False):
    self.n_sources = 3
    self.source_size = 249
    self.n_replicates = 10
    self.train_size = 83
    self.test_size = 83
    self.val_size = 83
    self.use_one_hot = use_one_hot

    # which features are binary
    self.binfeats = list(range(6,25))
    # which features are continuous
    self.contfeats = list(range(0,6))

  def get_train_valid_test_combined(self, m_sources):
    for i in range(self.n_replicates):
      data = pd.read_csv('data/IHDP/csv/ihdp_npci_{}.csv'.format(i+1),header=None).values

      t, y, y_cf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
      mu_0, mu_1, x = data[:, 3][:, np.newaxis], data[:, 4][:, np.newaxis], data[:, 5:]

      if self.use_one_hot==True and m_sources > 0:
        one_hot = pd.get_dummies(np.concatenate([[i]*self.source_size for i in range(self.n_sources)])).values
        one_hot = one_hot[:,:m_sources]
        x = np.concatenate((x,one_hot),axis=1)
        print('Use one-hot encoding, d_x = {}'.format(x.shape[1]))
      else:
        print('Do not use one-hot encoding, d_x = {}'.format(x.shape[1]))

      itr = np.concatenate([range(i,i+self.train_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      ite = np.concatenate([range(i+self.train_size,i+self.train_size+self.test_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      iva = np.concatenate([range(i+self.train_size+self.test_size,i+self.train_size+self.test_size+self.val_size)
                                    for i in range(0, m_sources*self.source_size, self.source_size)])
      train = (x[itr], t[itr], y[itr]), (y_cf[itr], mu_0[itr], mu_1[itr])
      valid = (x[iva], t[iva], y[iva]), (y_cf[iva], mu_0[iva], mu_1[iva])
      test = (x[ite], t[ite], y[ite]), (y_cf[ite], mu_0[ite], mu_1[ite])
      yield train, valid, test, self.contfeats, self.binfeats
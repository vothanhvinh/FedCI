# -*- coding: utf-8 -*-

import numpy as np

class Evaluation():
  def __init__(self, m0, m1):
    self.m0 = m0
    self.m1 = m1

  def pehe(self, y0pred, y1pred):
    return np.sqrt(np.mean(((y1pred - y0pred) - (self.m1 - self.m0))**2))

  def absolute_err_ate(self, y0pred, y1pred):
    return np.abs(np.mean(y1pred - y0pred) - np.mean(self.m1 - self.m0))

  def calc_stats(self, y0pred, y1pred):
    ae = self.absolute_err_ate(y0pred, y1pred)
    pehe = self.pehe(y0pred, y1pred)
    return pehe, ae

  def pehe2(self, itepred):
    return np.sqrt(np.mean((itepred - (self.m1 - self.m0))**2))

  def absolute_err_ate2(self, itepred):
    return np.abs(np.mean(itepred) - np.mean(self.m1 - self.m0))

  def calc_stats2(self, itepred):
    ae = self.absolute_err_ate2(itepred)
    pehe = self.pehe2(itepred)
    return pehe, ae




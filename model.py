# -*- coding: utf-8 -*-
import torch
import gpytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#=============================================================================================================================================
class FederatedCausalNet(torch.nn.Module):
  def __init__(self, Yobs, X, W, stats_feats, dim_h, n_sources, idx_source, bin_feats_4moments=True, cont_feats=[], inter_dependency=True):
    super().__init__()
    self.inter_dependency = inter_dependency
    self.Yobs = Yobs
    self.X = X
    self.W = W
    self.device = Yobs.device
    self.stats_Yobs0 = stats_feats[:,0:4]
    self.stats_Yobs1 = stats_feats[:,4:8]
    self.stats_W = stats_feats[:,8:12]
    self.stats_X = stats_feats[:,12:]
    self.dim_stats = 4
    # Define constants
    self.dim_x = X.shape[1]
    self.dim_y = 1
    self.dim_w = 1
    self.dim_h = dim_h
    self.n_sources = n_sources
    self.idx_source = idx_source

    self.d0 = self.dq = 100
    self.V0 = torch.tensor([[1.0,0.5],
                            [0.5,1.0]])
    self.n0 = self.nq = 100
    self.S0 = torch.tensor([[1.0,0.5],
                            [0.5,1.0]])
    self.ONE = torch.tensor([[1.0, 0.0],
                             [0.0,0.0]])
    self.TWO = torch.tensor([[0.0, 0.0],
                             [0.0,1.0]])
    self.THREE = torch.tensor([[0.0, 0.0],
                               [1.0,0.0]])

    self.mtkernelYobs = gpytorch.kernels.MaternKernel(nu=2.5)
    # self.mtkernelYobs = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=(self.dim_w+self.dim_y+self.dim_x)).to(self.device)
    # self.mtkernelYobs = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_y, active_dims=torch.tensor([0])) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_x, active_dims=torch.tensor(list(range(1,self.dim_y+self.dim_x)))) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_w, active_dims=torch.tensor([self.dim_w+self.dim_y+self.dim_x-1]))

    self.mtkernelQ = gpytorch.kernels.MaternKernel(nu=2.5)
    # self.mtkernelQ = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=(self.dim_x+self.dim_y+self.dim_w)).to(self.device)
    # self.mtkernelQ = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_y, active_dims=torch.tensor([0])) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_x, active_dims=torch.tensor(list(range(1,self.dim_y+self.dim_x)))) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_w, active_dims=torch.tensor([self.dim_w+self.dim_y+self.dim_x-1]))

    # self.mtkernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_x)
    self.mtkernel = gpytorch.kernels.MaternKernel(nu=2.5)


    self.muPg = torch.nn.Parameter(torch.zeros((n_sources, 2)), requires_grad=True)

    self.mtkernelPg = gpytorch.kernels.MaternKernel(nu=2.5)
    # self.mtkernelPg = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.stats_X.shape[1])
    
    self.mtkernelQg = gpytorch.kernels.MaternKernel(nu=2.5)
    # self.mtkernelQg = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.dim_y*2+self.dim_x+self.dim_w)
    # self.mtkernelQg = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.stats_Yobs0.shape[1],
    #                                                 active_dims=torch.tensor([0,1,2,3])) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.stats_Yobs1.shape[1],
    #                                           active_dims=torch.tensor([4,5,6,7])) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.stats_X.shape[1],
    #                                           active_dims=torch.tensor(list(range(8,8+self.stats_X.shape[1])))) \
    #           + gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.stats_W.shape[1],
    #                                           active_dims=torch.tensor(list(range(8+self.stats_X.shape[1],8+self.stats_X.shape[1]+self.stats_W.shape[1]))))

    # Parameters for matrix V
    self.nu_logit = torch.nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)
    self.rho_logit = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

    # Parameters for matrix S
    self.delta_logit = torch.nn.Parameter(torch.tensor([0.0,0.0], requires_grad=True))
    self.eta_logit = torch.nn.Parameter(torch.tensor([0.0], requires_grad=True))

    # Model for P(Yobs|.)
    self.modelYshare = torch.nn.Sequential(
        torch.nn.Linear(self.dim_x, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
    )

    self.modelY0 = torch.nn.Sequential(
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_y)
    )

    self.modelY1 = torch.nn.Sequential(
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_y)
    )

    # # Model for Q(Ymis|.)
    # self.model_meanQ = torch.nn.Sequential(
    #     torch.nn.Linear(self.dim_x + self.dim_y, self.dim_h),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(self.dim_h, self.dim_h),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(self.dim_h, self.dim_h),
    #     torch.nn.ReLU(),
    # )

    # self.model_meanQ0 = torch.nn.Sequential(
    #     torch.nn.Linear(self.dim_h, self.dim_h),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(self.dim_h, self.dim_y)
    # )

    # self.model_meanQ1 = torch.nn.Sequential(
    #     torch.nn.Linear(self.dim_h, self.dim_h),
    #     torch.nn.ReLU(),
    #     torch.nn.Linear(self.dim_h, self.dim_y)
    # )

    # Model for Q(g)
    if bin_feats_4moments==True:
      dim_in_Qg = (self.dim_x + self.dim_y + self.dim_y + self.dim_w)*self.dim_stats
    else:
      dim_in_Qg = (self.dim_y + self.dim_y)*self.dim_stats + self.dim_w*2 + (self.dim_x-len(cont_feats))*2 + len(cont_feats)*self.dim_stats 
    self.model_meanQg = torch.nn.Sequential(
        torch.nn.Linear(dim_in_Qg, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, self.dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(self.dim_h, 2),
    )
  
  def rand_wishartI(self, p, d, device='cpu'):
    eps = torch.randn((d, p), device=self.device)
    return torch.sum(torch.bmm(eps.reshape(-1, p,1), eps.reshape(-1,1,p)),dim=0)

  def fKobs(self, W, K, Phi, Sig):
    K1 = ((1-W)*(1-W).t()*Phi[0,0] + W*W.t()*Phi[1,1] + (1-W)*W.t()*Phi[0,1] + W*(1-W).t()*Phi[1,0])*K
    K2 = torch.diag(((1-W)*(1-W)*Sig[0,0] + W*W*Sig[1,1]).flatten())
    return K1 + K2

  def fKmis(self, W, K, Phi, Sig):
    K1 = (W*W.t()*Phi[0,0] + (1-W)*(1-W).t()*Phi[1,1] + W*(1-W).t()*Phi[0,1] + (1-W)*W.t()*Phi[1,0])*K
    K2 = torch.diag((W*W*Sig[0,0] + (1-W)*(1-W)*Sig[1,1]).flatten())
    return K1 + K2

  def fKmo(self, W, K, Phi, Sig):
    K1 = ((1-W)*(1-W).t()*Phi[1,0] + W*W.t()*Phi[0,1] + (1-W)*W.t()*Phi[1,1] + W*(1-W).t()*Phi[0,0])*K
    K2 = torch.diag(((1-W)*Sig[1,0] + W*Sig[0,1]).flatten())
    return K1 + K2

  def modelY(self, X, W, Phi_half, g=None):
    mu0 = self.modelY0(self.modelYshare(X))
    mu1 = self.modelY1(self.modelYshare(X))

    if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True: 
      muObs = (1-W)*Phi_half[0,0]*(mu0 + g[0]) + W*(Phi_half[1,0]*(mu0 + g[0]) + Phi_half[1,1]*(mu1 + g[1]))
      muMis = W*Phi_half[0,0]*(mu0 + g[0]) + (1-W)*(Phi_half[1,0]*(mu0 + g[0]) + Phi_half[1,1]*(mu1 + g[1]))
    else:#elif self.n_sources==1 or self.inter_dependency==False:
      muObs = (1-W)*Phi_half[0,0]*mu0 + W*(Phi_half[1,0]*mu0 + Phi_half[1,1]*mu1)
      muMis = W*Phi_half[0,0]*mu0 + (1-W)*(Phi_half[1,0]*mu0 + Phi_half[1,1]*mu1)
    return muObs, muMis
    # return torch.zeros((X.shape[0],1), device=device), torch.zeros((X.shape[0],1), device=self.device),
    # torch.zeros((X.shape[0],1), device=device), torch.zeros((X.shape[0],1), device=self.device)
    # if self.n_sources > 1 and self.inter_dependency==True: 
    #   muObs = (1-W)*Phi_half[0,0]*g[0] + W*(Phi_half[1,0]*g[0] + Phi_half[1,1]*g[1])
    #   muMis = W*Phi_half[0,0]*g[0] + (1-W)*(Phi_half[1,0]*g[0] + Phi_half[1,1]*g[1])
    # elif self.n_sources==1 or self.inter_dependency==False:
    #   muObs = torch.zeros((X.shape[0],1), device=device)
    #   muMis = torch.zeros((X.shape[0],1), device=device)
    # return muObs, muMis

  def covYobs(self, Ymis, X, W):
    in_ = torch.cat((Ymis, X, W), dim=1)
    return self.mtkernelYobs(in_)

  def meanQ(self, Yobs, X, W):
    in_ = torch.cat((Yobs, X), dim=1)
    return W*self.model_meanQ1(self.model_meanQ(in_)) + (1-W)*self.model_meanQ0(self.model_meanQ(in_))

  def covQ(self, Yobs, X, W):
    in_ = torch.cat((Yobs, X, W), dim=1)
    return self.mtkernelQ(in_)

  def meanPg(self):
    return self.muPg

  def covPg(self, X):
    return self.mtkernelPg(X)

  def meanQg(self, stats_Yobs0, stats_Yobs1, stats_X, stats_W):
    in_ = torch.cat((stats_Yobs0, stats_Yobs1, stats_X, stats_W), dim=1)
    return self.model_meanQg(in_)

  def covQg(self, stats_Yobs0, stats_Yobs1, stats_X, stats_W):
    in_ = torch.cat((stats_Yobs0, stats_Yobs1, stats_X, stats_W), dim=1)
    return self.mtkernelQg(in_)

  def predictYmis(self, Yobs, X, W, idx_source):
    X_all = torch.cat((self.X, X),dim=0)
    Yobs_all = torch.cat((self.Yobs, Yobs),dim=0)
    W_all = torch.cat((self.W, W),dim=0)

    m_mo_est = torch.zeros((Yobs_all.shape[0],1),device=self.device)
    S_mo_est = torch.zeros((Yobs_all.shape[0],Yobs_all.shape[0]),device=self.device)
    m_mo_lst = []
    for i in range(100):
      #################
      # Draw Phi, Sig #
      #################
      # Compute rho, eta
      self.rho = torch.sigmoid(self.rho_logit)
      self.eta = torch.sigmoid(self.eta_logit)
      self.nu = torch.exp(self.nu_logit)
      self.delta = torch.exp(self.delta_logit)

      # Compute matrices V and S, see ONE, TWO, THREE in __init__
      V_half = self.nu[0]*self.ONE \
              + torch.sqrt(1-self.rho**2)*self.nu[1]*self.TWO \
              + self.rho*self.nu[1]*self.THREE
      # V = V_half.matmul(V_half.t())
      S_half = self.delta[0]*self.ONE \
              + torch.sqrt(1-self.eta**2)*self.delta[1]*self.TWO \
              + self.rho*self.delta[1]*self.THREE
      # S = S_half.matmul(S_half.t())

      # Sample Phi, Sig using reparameterization trick
      eps_Phi = self.rand_wishartI(2, self.dq, device=self.device)
      Phi = V_half.mm(eps_Phi).mm(V_half.t())

      eps_Sig = self.rand_wishartI(2, self.nq, device=self.device)
      Sig = S_half.mm(eps_Sig).mm(S_half.t())

      #################
      # Draw g        #
      #################
      if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
        mean_Qg = self.meanQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W)
        cov_Qg = self.covQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W).evaluate()
        eps_g = torch.randn((cov_Qg.shape[0],2), device=self.device)
        sample_g = mean_Qg + torch.cholesky(cov_Qg).mm(eps_g)

      # Compute kernel matrices
      K = self.mtkernel(X_all).evaluate()
      Kobs = self.fKobs(W_all, K, Phi, Sig)
      Kmis = self.fKmis(W_all, K, Phi, Sig)
      Kmo = self.fKmo(W_all, K, Phi, Sig)
      Kom = Kmo.t()
      KobsInv = torch.inverse(Kobs)

      Phi_half = torch.cholesky(Phi)
      if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
        muObs, muMis = self.modelY(X_all, W_all, Phi_half, sample_g[idx_source,:])
      else:#elif self.n_sources==1 or self.inter_dependency==False:
        muObs, muMis = self.modelY(X_all, W_all, Phi_half)

      m_mo = muMis + Kmo.mm(KobsInv).mm(Yobs_all-muObs)
      S_mo = Kmis - Kmo.mm(KobsInv).mm(Kom)

      m_mo_est += m_mo.detach().clone()/100
      S_mo_est += S_mo.detach().clone()/100

      m_mo_lst.append(m_mo.detach().clone())

    mean = m_mo_est[-X.shape[0]:,:]

    C = torch.zeros((X.shape[0],X.shape[0]),device=self.device)
    for i in range(len(m_mo_lst)):
      C += (m_mo_lst[i][-X.shape[0]:,:]).mm((m_mo_lst[i][-X.shape[0]:,:]).t())/100
    C = C - mean.mm(mean.t())
    Cov = S_mo_est[-X.shape[0]:,-X.shape[0]:] + C

    return mean, Cov

  def predictATE(self, Yobs, X, W, idx_source):
    meanYmis, covYmis = self.predictYmis(Yobs=Yobs, X=X, W=W, idx_source=idx_source)
    mean = torch.sum((2*W-1).t().mm(Yobs) + (1-2*W).t().mm(meanYmis))/Yobs.shape[0]
    var = torch.sum((1-2*W).t().mm(covYmis).mm(1-2*W))/(Yobs.shape[0]**2)
    return mean, var, meanYmis, covYmis


  def sampleYmis(self, Yobs, X, W, idx_source, n_samples=100):
    X_all = torch.cat((self.X, X),dim=0)
    Yobs_all = torch.cat((self.Yobs, Yobs),dim=0)
    W_all = torch.cat((self.W, W),dim=0)

    m_mo_est = torch.zeros((Yobs_all.shape[0],1),device=self.device)
    S_mo_est = torch.zeros((Yobs_all.shape[0],Yobs_all.shape[0]),device=self.device)
    m_mo_lst = []
    Ymis_samples = []
    for i in range(n_samples):
      #################
      # Draw Phi, Sig #
      #################
      # Compute rho, eta
      self.rho = torch.sigmoid(self.rho_logit)
      self.eta = torch.sigmoid(self.eta_logit)
      self.nu = torch.exp(self.nu_logit)
      self.delta = torch.exp(self.delta_logit)

      # Compute matrices V and S, see ONE, TWO, THREE in __init__
      V_half = self.nu[0]*self.ONE \
              + torch.sqrt(1-self.rho**2)*self.nu[1]*self.TWO \
              + self.rho*self.nu[1]*self.THREE
      # V = V_half.matmul(V_half.t())
      S_half = self.delta[0]*self.ONE \
              + torch.sqrt(1-self.eta**2)*self.delta[1]*self.TWO \
              + self.rho*self.delta[1]*self.THREE
      # S = S_half.matmul(S_half.t())

      # Sample Phi, Sig using reparameterization trick
      eps_Phi = self.rand_wishartI(2, self.dq, device=self.device)
      Phi = V_half.mm(eps_Phi).mm(V_half.t())

      eps_Sig = self.rand_wishartI(2, self.nq, device=self.device)
      Sig = S_half.mm(eps_Sig).mm(S_half.t())

      #################
      # Draw g        #
      #################
      if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
        mean_Qg = self.meanQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W)
        cov_Qg = self.covQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W).evaluate()
        eps_g = torch.randn((cov_Qg.shape[0],2), device=self.device)
        sample_g = mean_Qg + torch.cholesky(cov_Qg).mm(eps_g)

      # Compute kernel matrices
      K = self.mtkernel(X_all).evaluate()
      Kobs = self.fKobs(W_all, K, Phi, Sig)
      Kmis = self.fKmis(W_all, K, Phi, Sig)
      Kmo = self.fKmo(W_all, K, Phi, Sig)
      Kom = Kmo.t()
      KobsInv = torch.inverse(Kobs)

      Phi_half = torch.cholesky(Phi)
      if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
        muObs, muMis = self.modelY(X_all, W_all, Phi_half, sample_g[idx_source,:])
      else:#elif self.n_sources==1 or self.inter_dependency==False:
        muObs, muMis = self.modelY(X_all, W_all, Phi_half)

      m_mo = muMis + Kmo.mm(KobsInv).mm(Yobs_all-muObs)
      S_mo = Kmis - Kmo.mm(KobsInv).mm(Kom)


      eps = torch.randn((m_mo.shape[0],1), device=self.device)
      Ymis_sample = m_mo + torch.cholesky(S_mo).mm(eps)
      Ymis_samples.append(Ymis_sample[-X.shape[0]:,:])

    return Ymis_samples

  def sampleATE(self, Yobs, X, W, idx_source, n_samples=100):
    Ymis_samples = self.sampleYmis(Yobs=Yobs, X=X, W=W, idx_source=idx_source, n_samples=n_samples)

    ate_samples = torch.zeros((n_samples,1),device=self.device)
    for i,Ymis_sample in enumerate(Ymis_samples):
      ate_sample = torch.sum((2*W-1).t().mm(Yobs) + (1-2*W).t().mm(Ymis_sample))/Yobs.shape[0]
      ate_samples[i,0] = ate_sample
    return ate_samples

  '''Override method to(device), default is cpu'''
  def to(self, device='cpu'):
    super().to(device)
    self.V0 = self.V0.to(device)
    self.S0 = self.S0.to(device)
    self.ONE = self.ONE.to(device)
    self.TWO = self.TWO.to(device)
    self.THREE = self.THREE.to(device)
    return self

  def forward(self, idx_source):
    # Compute rho, eta
    self.rho = torch.sigmoid(self.rho_logit)
    self.eta = torch.sigmoid(self.eta_logit)
    self.nu = torch.exp(self.nu_logit)
    self.delta = torch.exp(self.delta_logit)

    # Compute matrices V and S, see ONE, TWO, THREE in __init__
    V_half = self.nu[0]*self.ONE \
            + torch.sqrt(1-self.rho**2)*self.nu[1]*self.TWO \
            + self.rho*self.nu[1]*self.THREE
    V = V_half.matmul(V_half.t())

    S_half = self.delta[0]*self.ONE \
            + torch.sqrt(1-self.eta**2)*self.delta[1]*self.TWO \
            + self.rho*self.delta[1]*self.THREE
    S = S_half.matmul(S_half.t())
    
    # Sample Phi, Sig using reparameterization trick
    eps_Phi = self.rand_wishartI(2, self.dq, device=self.device)
    Phi = V_half.mm(eps_Phi).mm(V_half.t())

    eps_Sig = self.rand_wishartI(2, self.nq, device=self.device)
    Sig = S_half.mm(eps_Sig).mm(S_half.t())

    # Sample g using reparameterization trick
    if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
      mean_Qg = self.meanQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W)
      cov_Qg = self.covQg(self.stats_Yobs0, self.stats_Yobs1, self.stats_X, self.stats_W).evaluate()
      eps_g = torch.randn((cov_Qg.shape[0],2), device=self.device)
      sample_g = mean_Qg + torch.cholesky(cov_Qg).mm(eps_g)
    
    # Compute kernel matrices
    K = self.mtkernel(self.X).evaluate()
    Kobs = self.fKobs(self.W, K, Phi, Sig)
    # Kmis = self.fKmis(W, K, Phi, Sig)
    # Kmo = self.fKmo(W, K, Phi, Sig)
    # Kom = Kmo.t()
    # KmisInv = torch.inverse(Kmis)

    # Compute KL divergence
    if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
      mug = self.meanPg()
      Kg = self.covPg(self.stats_X).evaluate()
      KgInv = torch.inverse(Kg)
      sgn_Qg, logabsdet_Qg = torch.slogdet(cov_Qg)
      sgn_Kg, logabsdet_Kg = torch.slogdet(Kg)
      klg0 = 0.5*(torch.trace(KgInv.mm(cov_Qg)) + (mean_Qg[:,0:1] - mug[:,0:1]).t().mm(KgInv).mm(mean_Qg[:,0:1] - mug[:,0:1])\
                  + sgn_Kg*logabsdet_Kg - sgn_Qg*logabsdet_Qg)
      klg1 = 0.5*(torch.trace(KgInv.mm(cov_Qg)) + (mean_Qg[:,1:] - mug[:,1:]).t().mm(KgInv).mm(mean_Qg[:,1:] - mug[:,1:])\
                  + sgn_Kg*logabsdet_Kg - sgn_Qg*logabsdet_Qg)

    sgn_Phi, logabsdet_Phi = torch.slogdet(torch.inverse(self.V0).mm(V))
    klPhi = -0.5*self.d0*sgn_Phi*logabsdet_Phi + 0.5*self.dq*torch.trace(torch.inverse(self.V0).mm(V))

    sgn_Sig, logabsdet_Sig = torch.slogdet(torch.inverse(self.S0).mm(S))
    klSig = -0.5*self.n0*sgn_Sig*logabsdet_Sig + 0.5*self.nq*torch.trace(torch.inverse(self.S0).mm(S))

    # Sample Ymis using reparameterization trick
    # eps_Ymis = torch.randn((cov_Ymis.shape[0],1), device=self.device)
    # Ymis = mean_Ymis + torch.cholesky(cov_Ymis).mm(eps_Ymis)

    # Compute negative log conditional likelihood (reconstruction loss)
    # m_om = self.modelYobs(Ymis, X, W, Phi, Sig) #Kom.mm(KmisInv).mm(Ymis)
    # S_om = self.covYobs(Ymis, X, W).evaluate() #Kobs - Kom.mm(KmisInv).mm(Kmo)
    # m_om = muObs
    # S_om = Kobs #covYobs(Ymis, X, W).evaluate()
    Phi_half = torch.cholesky(Phi)
    if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
      muObs, muMis = self.modelY(self.X, self.W, Phi_half, sample_g[idx_source,:])
    else:#elif self.n_sources==1 or self.inter_dependency==False:
      muObs, muMis = self.modelY(self.X, self.W, Phi_half)
    sgn, logabsdet = torch.slogdet(Kobs)
    loss_recon = 0.5*(sgn*logabsdet) + 0.5*(self.Yobs - muObs).t().mm(torch.inverse(Kobs)).mm(self.Yobs - muObs)

    # Mean loss
    if self.inter_dependency==True:#self.n_sources > 1 and self.inter_dependency==True:
      loss = (loss_recon + klg0 + klg1 + klPhi + klSig)/self.Yobs.shape[0]
    else:#elif self.n_sources==1 or self.inter_dependency==False:
      loss = (loss_recon + klPhi + klSig)/self.Yobs.shape[0]
    return loss

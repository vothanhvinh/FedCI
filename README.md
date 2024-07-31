## Bayesian Federated Estimation of Causal Effects from Observational Data

Vo, T. V., Lee, Y., Hoang, T. N., & Leong, T. Y. (2022). Bayesian Federated Estimation of Causal Effects from Observational Data. In The 38th Conference on Uncertainty in Artificial Intelligence. <br />[<a href="https://proceedings.mlr.press/v180/vo22a/vo22a.pdf" target="_blank">Download PDF</a>] [<a href="https://openreview.net/forum?id=BEl3vP8sqlc" target="_blank">OpenReview</a>] [<a href="https://proceedings.mlr.press/v180/vo22a/vo22a-supp.pdf" target="_blank">Supplementary PDF</a>]

Please cite: 

```
@inproceedings{vo2022bayesian,
  title={Bayesian Federated Estimation of Causal Effects from Observational Data},
  author={Vo, Thanh Vinh and Lee, Young and Hoang, Trong Nghia and Leong, Tze-Yun},
  booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
  year={2022}
}
```
and
```
@article{vo2024federated,
  title={Federated Causal Inference from Observational Data},
  author={Vo, Thanh Vinh and Lee, Young and Leong, Tze-Yun},
  journal={arXiv preprint arXiv:2308.13047},
  year={2024}
}
```

## Table of Contents
- [Requirements](#requirements)
- [Import packages](#import-packages)
- [Prepare device: GPU or GPU](#prepare-device--gpu-or-gpu)
- [Load data](#load-data)
- [Convert numpy arrays to tensors](#convert-numpy-arrays-to-tensors)
- [Configure hyperparameters](#configure-hyperparameters)
- [Train the model](#train-the-model)
- [Some examples](#some-examples)
- [Some experimental results](#some-experimental-results)


## Requirements
This code has been tested on:
```
gpytorch==1.3.0
pytorch==1.7.0+cu101
```

## Import packages

```python
import numpy as np
import torch
import pandas as pd
import gpytorch
from scipy import stats
from model_train import train_model
from model_utils import *
from evaluation import Evaluation
```

## Prepare device: GPU or GPU

```python
if torch.cuda.is_available():
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  print('Use ***GPU***')
  print(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024,'GB')
else:
  print('Use CPU')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
If you have GPU, the above code would use it by default. Otherwise, it would use CPU.

## Load data
There are 3 datasets: ```DATA-1, DATA-2, IHDP```. 
Use the following codes to load ```IHDP``` dataset:
```python
from load_datasets import IHDP
dataset = IHDP()
source_size = dataset.source_size
train_size = dataset.train_size
test_size = dataset.test_size
val_size = dataset.val_size
```

To load a new dataset, create a new class for that dataset in ```load_datasets.py```. The class should be similar to that of ```IHDP```.

## Convert numpy arrays to tensors
* Let ```y, y_cf, x, w``` be ```numpy``` arrays of the training data.
* Let ```yte, y_cfte, xte, wte``` be ```numpy``` arrays of the training data.
First we need to transform these numpy arrays into tensors:
```python
# Tensors of training data
N = y.shape[0]
X = torch.from_numpy(x.reshape(N,-1)).float().to(device)
Yobs = torch.from_numpy(y.reshape(-1,1)).float().to(device)
W = torch.from_numpy(w.reshape(-1,1)).float().to(device)

# Tensors of testing data
Nte = yte.shape[0]
Xte = torch.from_numpy(xte.reshape(Nte,-1)).float().to(device)
Yobste = torch.from_numpy(yte.reshape(-1,1)).float().to(device)
Wte = torch.from_numpy(wte.reshape(-1,1)).float().to(device)
```

## Configure hyperparameters
```python
n_iterations = 2000
learning_rate = 1e-3
display_per_iter = 200
```

## Train the model

```python
model_server, model_sources = train_model(Yobs=Yobs, X=X, W=W, # for training
                                          Yobste=Yobste, Xte=Xte, Wte=Wte, # for testing
                                          y=y, y_cf=y_cf, x=x, w=w, # To print the errors
                                                                    # of ATE & ITE on training set
                                          yte=yte, y_cfte=y_cfte, xte=xte, wte=wte, # To print the errors
                                                                                    # of ATE & ITE on testing set
                                          n_sources=n_sources, train_size=train_size, test_size=test_size,
                                          n_iterations=n_iterations, learning_rate=learning_rate,
                                          display_per_iter=display_per_iter)
```
The above code would print out the errors of ATE and ITE on training and testing data for every ```display_per_iter``` iterations.

## Some examples

We recommend to start with these examples:
* example_DATA-1.ipynb
* example_DATA-1.ipynb
* example_ihdp.ipynb

## Some experimental results

***

**Federated causal inference analysis**

<img src="https://github.com/vothanhvinh/FedCI/blob/main/pics/FedCI-analysis.jpg" width="480">

***

**The impact of inter-dependency**

<img src="https://github.com/vothanhvinh/FedCI/blob/main/pics/FedCI-inter-dependency.jpg" width="480">

***

**Contrasting with baselines**

<img src="https://github.com/vothanhvinh/FedCI/blob/main/pics/Table1.jpg" width="480">

***

**The  estimated  distribution  of  ATE**

_**Synthetic dataset**_

<img src="https://github.com/vothanhvinh/FedCI/blob/main/pics/FedCI-uncertain-synthetic.jpg" width="480">

<br />

_**IHDP dataset**_

<img src="https://github.com/vothanhvinh/FedCI/blob/main/pics/FedCI-uncercain-ihdp.jpg" width="480">

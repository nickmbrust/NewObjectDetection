import numpy as np
import torch
import LearningTools as LT
import rave as r


PATHMAD = 'MADELON/'
pathdatatrain = PATHMAD + 'madelon_train.data'
pathlabelstrain = PATHMAD + 'madelon_train.labels'

pathtestdata = PATHMAD + 'madelon_valid.data'

pathtestlabels = PATHMAD + 'madelon_valid.labels'

x = np.loadtxt(pathdatatrain)
y = np.loadtxt(pathlabelstrain)
xt = np.loadtxt(pathtestdata)
yt = np.loadtxt(pathtestlabels)
device = 'cuda:0'
x = torch.tensor(x).float().to(device)
y = torch.tensor(y).float().to(device)
xt = torch.tensor(xt).float().to(device)
yt = torch.tensor(yt).float().to(device)
trainingrave = r.RAVE()
trainingrave.add(x, y)

XXN, XYN, pi = trainingrave.standardize()
xt = xt - trainingrave.mx
xt = xt/pi

betas, indicies = LT.OFSA(XXN, trainingrave.mxy.view(1, -1), 2048, 10)

print(LT.get_loss(xt, yt, betas.t()))
print(LT.test(xt, yt, betas.t()))
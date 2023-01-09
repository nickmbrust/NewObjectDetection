from ast import LShift
from hashlib import algorithms_available
from operator import index
from re import X
from statistics import mode
from tokenize import Double
import torch
import rave as r
import sklearn as sk
import LearningTools as LT
from dataloader import loadimgs, loadRave
import featureExtractor as ext
import features as f
import numpy as np
import gc
from torch.utils.data import TensorDataset, DataLoader, Dataset
from getdata import *
PATH = 'dataset/'
pathtrain = PATH+ 'training/'
pathtest = PATH+ 'test/'
swsltransformer = f.swsl_transform(128)
cliptransformer = f.Clip_transform(128)

alg = 'FSAU'
exttype = 'SWSL'
if exttype == 'SWSL':
    eta = 0.1
elif exttype == 'clip':
    eta = 5

#imgs, labelstest = loadimgs(pathtest, swsltransformer, 'swsl')

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")


# Xtest = ext.swslextract(imgs)
# print(len(Xtest))
# if exttype == 'clip':
#     averages, postiverave, negativerave = getCliprave()
# else:
#     averages, postiverave, negativerave = getSWSLrave()

positiverave, negativerave = loadRave("SWSLclassfeats/", "n02772753.tar")
averages = r.RAVE()
averages.add_rave(positiverave)
averages.add_rave(negativerave)
XXn, XYn, pi = averages.standardize()


if alg == "LS":
    print("Calculating Least Squares")
    betas = LT.OLS(averages.mxx.to(device), averages.mxy.to(device))
elif alg == "FSA":
    print("Calulating using OFSA")
    betas, indicies = LT.OFSA(XXn.to(device), averages.mxy.t().to(device),2048, 200)
elif alg == "FSAU":
    betas= LT.FSAunbalanced(positiverave, negativerave, 100, 50, eta)
print(betas)

if exttype == 'clip':
    imgs, labelstest = loadimgs(pathtest, cliptransformer, 'clip')
else:
    imgs, labelstest = loadimgs(pathtest, swsltransformer, 'swsl')
ytest = []
if exttype == "clip":
    for h in range(73):
        if labelstest[h] == "backpack":
            ytest.append(1.0)
        else:
            ytest.append(-1.0)
else:
    for h in range(len(labelstest)):
        if labelstest[h] == "backpack":
            ytest.append(1.0)
        else:
            ytest.append(-1.0)
ytest = torch.tensor(ytest)
if exttype == 'clip':
    Xtest = ext.clipextract(imgs, labelstest)
else:
    Xtest = ext.swslextract(imgs)
if alg == 'FSA':
    Xtest = Xtest - averages.mx
    Xtest = Xtest * pi
    #Xtest = Xtest[:, indicies]

#LT.test(Xtest.to(device), ytest.to(device), betas.t())
LT.testunlabled(Xtest.to(device), betas.t(), imgs)
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
from dataloader import loadimgs
import featureExtractor as ext
import features as f
import numpy as np
import gc
from torch.utils.data import TensorDataset, DataLoader, Dataset
PATH = 'dataset/'
pathtrain = PATH+ 'training/'
pathtest = PATH+ 'test/'
swsltransformer = f.swsl_transform(128)
cliptransformer = f.Clip_transform(128)

alg = 'FSA'


imgs, labelstest = loadimgs(pathtest, swsltransformer, 'swsl')
ytest = []

for h in range(len(labelstest)):
    if labelstest[h] == "racket":
        ytest.append(1.0)
    else:
        ytest.append(-1.0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")


Xtest = ext.swslextract(imgs)
print(len(Xtest))
positive = torch.load('SWSLclassfeats/n02772753.tar')
postivefeats = positive['features']
positivey = torch.ones((postivefeats.shape[0],1), dtype=torch.float)
positivey = torch.tensor(positivey).to(device)
postiverave = r.RAVE()
negatives = torch.ones((1092, 1), dtype = int)


with torch.no_grad():
    postiverave.add(postivefeats.to(device), positivey)
    print(postiverave.mxx.shape)

negativerave = r.RAVE()   
negativeclass = torch.load('SWSLclassfeats/n04045857.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))

del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n02853991.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03266479.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03276921.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))

del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03443167.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
negativerave.add(negativefeats.to(device), -negativeey.to(device))

del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03802912.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))

del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04076546.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))

del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04190372.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)

negativerave.add(negativefeats.to(device), -negativeey.to(device))
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04513584.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
del negativeclass, negativefeats, negativeey


averages = r.RAVE()
averages.add_rave(postiverave)
averages.add_rave(negativerave)


XXn, XYn, pi = averages.standardize()


if alg == "LS":
    print("Calculating Least Squares")
    betas = LT.OLS(averages.mxx.to(device), averages.mxy.to(device))
elif alg == "FSA":
    print("Calulating using OFSA")
    betas, indicies = LT.OFSA(XXn.to(device), averages.mxy.t().to(device),2048, 200)
elif alg == "FSAU":
    betas = LT.FSAunbalanced(postiverave, negativerave, 100, 50)
print(betas)


imgs, labelstest = loadimgs(pathtest, swsltransformer, 'swsl')
ytest = []

for h in range(len(labelstest)):
    if labelstest[h] == "backpack":
        ytest.append(1.0)
    else:
        ytest.append(-1.0)
ytest = torch.tensor(ytest)
Xtest = ext.swslextract(imgs)
if alg == 'FSA' or 'FSAU':
    Xtest = Xtest - averages.mx
    Xtest = Xtest * pi
    Xtest = Xtest[:, indicies]
print(LT.err(Xtest.to(device), ytest.to(device), betas.t()))

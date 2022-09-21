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
Xtrain, labels = loadimgs(pathtrain, swsltransformer, 'swsl')
ytrain = []
for k in range(len(labels)):
    if labels[k] == "n04045857":
       ytrain.append(1.0)
    else:
       ytrain.append(-1.0)
alg = 'FSA'


#imgs, labelstest = loadimgs(pathtest, swsltransformer, 'swsl')
# ytest = []
# truepositive=0
# for h in range(len(labelstest)):
#     if labelstest[h] == "racket":
#         ytest.append(1.0)
#     else:
#         ytest.append(-1.0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")


#Xtest = ext.swslextract(imgs)
#print(len(Xtest))
positive = torch.load('SWSLclassfeats/n04045857.tar')
postivefeats = positive['features']
print(postivefeats.shape)
positivey = torch.ones((1, postivefeats.shape[1]), dtype=torch.float)
positivey = torch.tensor(positivey).to(device)
postiverave = r.RAVE()
print(positivey.t())
print(postivefeats)
with torch.no_grad():
    postiverave.add(postivefeats.to(device).t(), positivey.t())
    print(postiverave.mxx.shape)
    


averages = r.RAVE()
averages.add_rave(postiverave)
#averages.add_rave(negativerave)




if alg == "LS":
    print("Calculating Least Squares")
    betas = LT.OLS(averages.mxx.to(device), averages.mxy.to(device))
elif alg == "FSA":
    print("Calulating using OFSA")
    print(averages.mxx.shape)
    print(averages.mxy)
    betas = LT.OFSA(averages.mxx, averages.mxy, 100, 10)
print(betas)
# classification = []
# with torch.no_grad():
#     for x in range(len(Xtest)):
#         yest = 1/(1+np.exp(Xtest[x].cpu().numpy()@ betas))
#         classification = np.round(yest)

#         tp = 0
#         fp = 0
#         tn = 0
#         fn = 0
      
      
       
#         if classification == 1.0:
#             if classification == ytest[x]:
#                 tp = tp +1
#             elif classification != ytest[x]:
#                 fp = fp+1
#         elif classification == 0.0:
#             if classification== ytest[x]:
#                 tn = tn +1
#             elif classification != ytest[x]:
#                 fn = fn+1

#     print(fn)
#     print(tp)
#     print(fp)
#     print(tn)
#     precision = tp/(tp+fp)
#     recall = tp/(tp+fn)
# print('Precision: ' + str(precision))
# print('Recall: ' + str(recall))
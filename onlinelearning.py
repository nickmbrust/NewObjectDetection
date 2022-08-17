import torch
import rave as r
import sklearn as sk
import LearningTools as LT
from dataloader import loadimgs
PATH = 'dataset/'
pathtrain = PATH+ 'training/'
Xtrain, labels = loadimgs(pathtrain)

ytrain = []
for k in range(len(labels)):
    if labels[k] == "backpack":
       ytrain.append(1.0)
    else:
       ytrain.append(0.0)
#Xtest, ytest = loaddata
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")

positive = torch.load('SWSLclassfeats/n02772753.tar')
postivefeats = positive['features']

postivefeats = postivefeats.to(device)
postiverave = r.RAVE()
print(type(postiverave))
postiverave.add(postivefeats)

print(postiverave)

negativeclasses = []
negativeclasses.append(torch.load('SWSLclassfeats/n02853991.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n03266479.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n03276921.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n03443167.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n03802912.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n04045857.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n04076546.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n04190372.tar'))
negativeclasses.append(torch.load('SWSLclassfeats/n04513584.tar'))

negativefeats = []
for i in range(0, 8):
    negativefeats.append(negativeclasses[i]['features'])
    print(type(negativefeats[i]))
negativerave = r.RAVE()

for j in range(len(negativefeats)):
    negativerave.add(negativefeats[j].to(device))


betas = LT.OLS(postiverave.mxx, ytrain, 0.99)
print(betas)
print(len(betas))
#yest = Xtest @ betas
#print(yest)
#error = yest - ytest
#print(error)

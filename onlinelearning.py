import torch
import rave as r
import sklearn as sk
import LearningTools as LT
from dataloader import loadimgs
import featureExtractor as ext
import features as f
import numpy as np
PATH = 'dataset/'
pathtrain = PATH+ 'training/'
pathtest = PATH+ 'test/'
swsltransformer = f.swsl_transform(128)

Xtrain, labels = loadimgs(pathtrain, swsltransformer)
ytrain = []
for k in range(len(labels)):
    if labels[k] == "n04045857":
       ytrain.append(1.0)
    else:
       ytrain.append(0.0)



imgs, labelstest = loadimgs(pathtest, swsltransformer)
ytest = []
truepositive=0
for h in range(len(labelstest)):
    if labelstest[h] == "racket":
        ytest.append(1.0)
    else:
        ytest.append(0.0)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")

Xtest = ext.swslextract(imgs)
print(len(Xtest))
positive = torch.load('SWSLclassfeats/n04045857.tar')
postivefeats = positive['features']

postivefeats = postivefeats.to(device)
postiverave = r.RAVE()
postiverave.add(postivefeats)

print(postiverave)

# negativeclasses = []
# negativeclasses.append(torch.load('SWSLclassfeats/n02853991.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n03266479.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n03276921.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n03443167.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n03802912.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n04076546.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n04190372.tar'))
# negativeclasses.append(torch.load('SWSLclassfeats/n04513584.tar'))

# negativefeats = []
# for i in range(0, 8):
#     negativefeats.append(negativeclasses[i]['features'])
#     print(type(negativefeats[i]))
# negativerave = r.RAVE()

# for j in range(len(negativefeats)):
#     negativerave.add(negativefeats[j].to(device))


betas = LT.OLS(postiverave.mxx, ytrain, 0.99)
print(betas)
with torch.no_grad():
    
    yest = Xtest.cpu().numpy() @ betas
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(yest)):
        dist = np.absolute(ytest[i]-yest[i])
        if dist <= 1:
            yest[i] = 1.0
            if yest[i] == ytest[i]:
                tp = tp+1
            elif yest[i] != ytest[i]:
                fp = fp+1
        else:
            yest[i] = 0.0
            if yest[i] == ytest[i]:
                tn = tn+1
            elif yest[i] != ytest[i]:
                fn = fn+1
        print(yest[i])
print(fn)
print(tp)
print(fp)
print(tn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print('Precision: ' + str(precision))
print('Recall: ' + str(recall))
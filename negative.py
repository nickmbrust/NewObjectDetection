import rave as r
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")

negativerave = r.RAVE()

negativeclass = torch.load('SWSLclassfeats/n02772753.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n02853991.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03266479.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03276921.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03443167.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n03802912.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04076546.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04190372.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey
negativeclass = torch.load('SWSLclassfeats/n04513584.tar')
negativefeats = negativeclass['features']
negativeey = torch.ones((negativefeats.shape[0], 1), dtype=torch.float)
print(negativeey.shape)
print(negativeey)
negativerave.add(negativefeats.to(device), -negativeey.to(device))
print(negativerave.mxy.shape)
del negativeclass, negativefeats, negativeey

print(negativerave.mxx.shape)
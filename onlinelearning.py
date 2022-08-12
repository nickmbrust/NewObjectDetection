import torch
import rave as r

positive = torch.load('SWSLclassfeats/n02772753.tar')
postivefeats = positive['features']

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
negativerave = r.RAVE()

for j in len(negativefeats):
    negativerave.add(negativefeats[j])



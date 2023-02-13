import dataloader as dl
import featureExtractor as ext
import torch
from features import swsl_transform, Clip_transform

PATH = 'dataset/training/n0'

pathload = PATH + '4513584'

# swslpreprocess = swsl_transform(128)
# imgs, labels = dl.loadClass(pathload, swslpreprocess)
# swslfeatures = ext.swslextract(imgs)
# print(swslfeatures.shape)
# torch.save({"features": swslfeatures}, 'SWSLclassfeats/background.tar')
# del imgs, labels, swslfeatures


clippreprocess = Clip_transform(288)
imgs, labels = dl.loadClass(pathload, clippreprocess)
clipfeatures = ext.newclip(imgs)
print(clipfeatures)
torch.save({"features": clipfeatures}, 'Clipclassfeatures/n04513584.tar')

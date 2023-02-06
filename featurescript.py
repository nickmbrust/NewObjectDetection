import dataloader as dl
import featureExtractor as ext
import torch
from features import swsl_transform, Clip_transform

PATH = 'backgrounddata/images'

pathload = PATH

# swslpreprocess = swsl_transform(128)
# imgs, labels = dl.loadClass(pathload, swslpreprocess)
# swslfeatures = ext.swslextract(imgs)
# print(swslfeatures.shape)
# torch.save({"features": swslfeatures}, 'SWSLclassfeats/background.tar')
# del imgs, labels, swslfeatures


clippreprocess = Clip_transform(128)
imgs, labels = dl.loadClass(pathload, clippreprocess)
clipfeatures = ext.newclip(imgs)
print(clipfeatures)
torch.save({"features": clipfeatures}, 'Clipclassfeatures/background.tar')

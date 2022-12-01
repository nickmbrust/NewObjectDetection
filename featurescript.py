import dataloader as dl
import featureExtractor as ext
import torch
from features import swsl_transform

PATH = 'dataset/training/n0'

pathload = PATH+ '2772753'

# swslpreprocess = swsl_transform(128)
# imgs, labels = dl.loadClass(pathload, swslpreprocess)
# swslfeatures = ext.swslextract(imgs)
# print(swslfeatures.shape)
# torch.save({"features": swslfeatures}, 'SWSLclassfeats/n02772753.tar')
# del imgs, labels, swslfeatures



imgs, labels = dl.loadClip(pathload)
clipfeatures = ext.clipextract(imgs, labels)
print(clipfeatures)
print(clipfeatures.shape)
torch.save({"features": clipfeatures}, 'Clipclassfeatures/n02772753.tar')

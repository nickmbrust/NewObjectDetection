import dataloader as dl
import featureExtractor as ext
import torch
from features import swsl_transform

PATH = 'dataset/training/n0'

pathload = PATH+ '4513584'

swslpreprocess = swsl_transform(128)
imgs, labels = dl.loadClass(pathload, swslpreprocess)
swslfeatures = ext.swslextract(imgs)
print(swslfeatures.shape)
torch.save({"features": swslfeatures}, 'SWSLclassfeats/n04513584.tar')
del imgs, labels, swslfeatures



# imgs, labels = dl.loadClip(pathload)
# clipfeatures = ext.clipextract(imgs, labels)
# torch.save({"features": clipfeatures}, 'Clipclassfeatures/n04513584.tar')

#from torchsummary import summary
from re import X
from torch.utils.data import TensorDataset, DataLoader
import torch
from dataloader import loadimgs, loadClass, loadClip
import features as f
from PIL import Image
from matplotlib import cm
import clip
import torch
import gc
import numpy as np
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

PATH = 'backgroundata/'

pathtrain = PATH + 'images'
pathtest = PATH + 'test'
def swslextract(x):
    #print(type(x[0]))
    #SWSLset = TensorDataset(x)
    dataloaderSWSL = DataLoader(x, batch_size = 3)
    print(dataloaderSWSL)
    torch.hub.list('facebookresearch/semi-supervised-ImageNet1K-models')
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    model = model.to(device)
    model.eval()
    batchfeatures = []
    for imgBatch in dataloaderSWSL:
        batchfeatures.append(f.swslfeatures(model, imgBatch.to(device)))
        del imgBatch
    swslfeats = torch.tensor(batchfeatures[0])
    with torch.no_grad():
        torch.cat(batchfeatures, out = swslfeats)
    return swslfeats

def clipextract(x, y):
# # ClIP
    modelclip, preprocess = clip.load('RN50x4', device)
    images = []
    text = clip.tokenize(y).to(device)
    features = []
    for i in range(73):
        img = preprocess(x[i]).unsqueeze(0)
        featuresclass= modelclip.encode_image(img.to(device))
        features.append(featuresclass)
    with torch.no_grad():
        featuresclip = torch.cat(features)
    return featuresclip
# torch.save({"features": clipfeats}, 'Clipclassfeatures/n03628765.tar')
#torch.save({"features": swslfeats}, 'backswsl/backgroundfeats.tar')

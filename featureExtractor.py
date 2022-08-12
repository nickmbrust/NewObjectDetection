from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import torch
from dataloader import loadimgs, loadClass, loadClip
import features as f
from PIL import Image
from matplotlib import cm
import clip
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

PATH = 'dataset/'

pathtrain = PATH + 'training/n02853991/'
pathtest = PATH + 'test'

swsltransformer = f.swsl_transform(128)

XtrainSWSL, ytrain = loadClass(pathtrain, swsltransformer)
# XtrainClip, _ = loadClip(pathtrain)
# ClIP
modelclip, preprocess = clip.load('RN50x4', device)
# print(Xtrain[1].ndarray.dtype)
# text = clip.tokenize(ytrain).to(device)
images = []
imagefeature = []
# for i in range(len(Xtrain)):
# images = preprocess(Image.open(pathtrain+'/ukulele0.png')).unsqueeze(0).to(device)

# with torch.no_grad():
# imagefeature = model.encode_image(images)
# textfeats = model.encode_text(text)
# Clipset = TensorDataset(torch.tensor(XtrainClip), text)
SWSLset = TensorDataset(torch.tensor(XtrainSWSL))

# dataloaderClip=DataLoader(Clipset, batch_size=64, drop_last=True)
dataloaderSWSL = DataLoader(SWSLset, batch_size=64, drop_last=True)
clipfeats = []
# summary(modelclip, (3,128,128))
torch.hub.list('facebookresearch/semi-supervised-ImageNet1K-models')
model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
model = model.to(device)
model.eval()

# for i in range(len(XtrainClip)):
# img= preprocess(XtrainClip[i]).unsqueeze(0).to(device)
# clipfeats.append(modelclip.encode_image(img))

for imgBatch in dataloaderSWSL:
    swslfeats = f.swslfeatures(model, imgBatch[0].to(device))
# torch.save({"features": clipfeats}, 'Clipclassfeatures/n03628765.tar')
torch.save({"features": swslfeats}, 'SWSLclassfeats/n02853991.tar')

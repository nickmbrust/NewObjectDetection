#from torchsummary import summary
from torch.utils.data import TensorDataset, DataLoader
import torch
from dataloader import loadimgs, loadClass, loadClip
import features as f
from PIL import Image
from matplotlib import cm
#import clip
import torch

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
    dataloaderSWSL = DataLoader(x, batch_size=64, drop_last=True)
    torch.hub.list('facebookresearch/semi-supervised-ImageNet1K-models')
    model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    model = model.to(device)
    model.eval()
    for imgBatch in dataloaderSWSL:
        swslfeats = f.swslfeatures(model, imgBatch.to(device))
    return swslfeats


# def clipextract(x):
# # ClIP
#     modelclip, preprocess = clip.load('RN50x4', device)
# # print(Xtrain[1].ndarray.dtype)
# # text = clip.tokenize(ytrain).to(device)
# images = []
# imagefeature = []
# # for i in range(len(Xtrain)):
# # images = preprocess(Image.open(pathtrain+'/ukulele0.png')).unsqueeze(0).to(device)

# # with torch.no_grad():
# # imagefeature = model.encode_image(images)
# # textfeats = model.encode_text(text)
# # Clipset = TensorDataset(torch.tensor(XtrainClip), text)
   

# # dataloaderClip=DataLoader(Clipset, batch_size=64, drop_last=True)
    
# clipfeats = []
# summary(modelclip, (3,128,128))
# for i in range(len(XtrainClip)):
# img= preprocess(XtrainClip[i]).unsqueeze(0).to(device)
# clipfeats.append(modelclip.encode_image(img)) 
# torch.save({"features": clipfeats}, 'Clipclassfeatures/n03628765.tar')
#torch.save({"features": swslfeats}, 'backswsl/backgroundfeats.tar')

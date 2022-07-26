import torch
from dataloader import loadimgs
import features as f
from PIL import Image
import clip

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
PATH = 'dataset/'

pathtrain = PATH+ 'training'
pathtest = PATH+ 'test'

Xtrain, ytrain = loadimgs(pathtrain)

model, preprocess = clip.load('RN50x4', device)

text = clip.tokenize(ytrain).to(device)
images = []
imagefeature = []

for i in range(len(Xtrain)):
    images[i] = preprocess(Xtrain[i]).unsqueeze(0).to(device)

with torch.nograd():
    for j in range(len(images)):
         imagefeature[j] = model.encode_image(images[j])
         textfeats = model.encode_text(text)

torch.save({"imagefeatures": imagefeature, "textfeatures": textfeats}, 'features.tar')



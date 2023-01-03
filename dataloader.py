import torch
import numpy as np
import os
import sklearn.utils
import cv2
import PIL.Image as Image
import rave as R
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    

def loadimgs(mainDir, transformer, type):
    mainDir = mainDir
    cat = os.listdir(mainDir)
    X = []
    y = []
    for i in cat:
        folderPATH = os.path.join(mainDir, i)
        imgList = os.listdir(folderPATH)
        cimgList = imgList.copy()
        for j in cimgList:
            if j.endswith('.png') != True:
                imgList.remove(j)
        for img in imgList:
            imgPath = os.path.join(folderPATH, img)
            loadImg = Image.open(imgPath).convert('RGB')
            if type == 'swsl':
                loadImg = transformer(loadImg)
            X.append(loadImg)
            y.append(i)
    if type == 'swsl':
        X = np.array(X)
        
    X, y = sklearn.utils.shuffle(X, y)
    return X, y
  

def loadClass(classFolder, transformer):
    classDir = classFolder
    cat = os.listdir(classDir)
    X = []
    y = []
    for i in range(0, len(cat)):
        imgPath = os.path.join(classDir, cat[i])
        loadImg = Image.open(imgPath).convert('RGB')
        loadImg = transformer(loadImg)
        X.append(loadImg.numpy())
        y.append(cat[i])
    X = np.array(X)
    return X, y


def loadClip(classFolder):
    classDir = classFolder
    cat = os.listdir(classDir)
    X = []
    y = []
    for i in range(0, len(cat)):
        imgPath = os.path.join(classDir, cat[i])
        loadImg = Image.open(imgPath)
        # loadImg = transformer(loadImg)
        X.append(loadImg)
        y.append(cat[i])

    return X, y

def loadRave(Path, positive):
    positivepath = os.path.join(Path, positive)
    positiveclass = torch.load(positivepath, map_location=device)
    positivefeatures = positiveclass['features']
    positiverave = R.RAVE()
    positivey = torch.ones((positivefeatures.shape[0],1), dtype=torch.float)
    positiverave.add(positivefeatures, positivey.to(device))
    negativerave = R.RAVE()
    li = os.listdir(Path)
    for i in li:
        negpath = os.path.join(Path, i)
        if negpath == positivepath:
            continue
        negclass = torch.load(negpath, map_location=device)
        negfeats = negclass['features']
        negativeey = torch.ones((negfeats.shape[0], 1), dtype=torch.float)
        negativerave.add(negfeats, -negativeey.to(device))
        print(negpath, " done")
    print()
    return positiverave, negativerave
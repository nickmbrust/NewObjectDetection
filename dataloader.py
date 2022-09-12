import torch
import numpy as np
import os
import sklearn.utils
import cv2
import PIL.Image as Image


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


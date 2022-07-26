import torch
import numpy as np
import os
import sklearn.utils
import cv2

def loadimgs(mainDir):
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
            loadImg = cv2.imread(imgPath, 0)
            X.append(loadImg)
            y.append(i)
    X = np.array(X, dtype=np.float)
    X, y = sklearn.utils.shuffle(X, y)
    return X,y

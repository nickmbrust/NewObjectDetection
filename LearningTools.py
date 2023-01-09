from asyncio import base_tasks
import torch
import rave
import numpy as np
from sklearn.metrics import *
import torchvision.transforms as T
from PIL import Image
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on cuda")
else:
    device = torch.device("cpu")
    print("running on cpu")

def OLS(SXX, SXY):
    
    beta = torch.linalg.solve(SXX, SXY)
    

    return beta

def standard(X, y):
    mean_x = torch.mean(X)
    mean_y = torch.mean(y)
    std_x = torch.std(X)
    std_y = torch.std(y)

    X = (X - mean_x)/std_x
    y = (y - mean_y)/std_y
    return X

def OFSA(SXX, SXY, k, T):
 
    #print(SXY.shape)
    beta = torch.zeros((1, SXY.shape[1])).to(device)
    eta = 0.0001
    p = len(SXX)
    mu = 0.1
    global_indices = torch.arange(0, SXX.shape[0])
    for t in range(T):
      
       
        
        delta =  eta*(beta@SXX-SXY)
        beta = beta - delta
        print(beta)
        
        max = np.max((0, (T-t)/(t*mu+T)))
        Mt = k + (p-k)*np.max((0, (T-t)/(t*mu+T)))
        Mt = int(Mt)
    
        absbetas = torch.abs(beta)
   
        indices = torch.argsort(-absbetas)
        #print(indices[0][:Mt])
        beta = beta[:, indices[0][:Mt]]
        
    
        SXX = SXX[:, indices[0][:Mt]]
        SXX = SXX[indices[0][:Mt], :]
        SXY = SXY[:, indices[0][:Mt]]
     

        print(get_loss(SXX, SXY, beta))
        print('Completed ' + str(t+1) + ' times')
    global_indices = global_indices[indices[0][0:Mt]]
    return beta, global_indices

def FSAunbalanced(Xpos, Xneg, k, T, eta):
     
    #print(SXY.shape)
    SXXp = Xpos.mxx
    SXXn = Xneg.mxx
    mup = Xpos.mx
    mun = Xneg.mx
    wp = 1/(SXXp.shape[0])
    wn = 1/(SXXn.shape[0])
    beta = torch.zeros((1, SXXp.shape[0])).to(device)
    beta0 = torch.zeros((1, SXXp.shape[0])).to(device)
    p = len(SXXp)
    mu = 0.1
    global_indices = torch.arange(0, SXXn.shape[0])
    for t in range(T):

        deltab0 = eta*(beta@(wp*mup + wn*mun)+(wp + wn)*beta0 + wn - wp )
        beta0 = beta-deltab0
      
        deltab =  eta*(beta@(wp*SXXp + wn*SXXn)+(wp*mup + wn*mun)*beta0 + wn*mun - wp*mup )
        beta = beta - deltab
       
        print(beta0.shape)
        print(beta.shape)
        # max = np.max((0, (T-t)/(t*mu+T)))
        # Mt = k + (p-k)*np.max((0, (T-t)/(t*mu+T)))
        # Mt = int(Mt)
    
        # absbetas = torch.abs(beta.t())
   
        # indices = torch.argsort(-absbetas)
        # print(indices[0][:Mt])
        # beta = beta[indices[0][:Mt], :]
        
    
        # SXX = SXX[:, indices[0][:Mt]]
        # SXX = SXX[indices[0][:Mt], :]
        # SXY = SXY[indices[0][:Mt], :]
    return beta

def test(Xtest, Ytest, betas):
    yhat = Xtest.float() @ betas.view(-1,1)
    yhat[yhat<0] = -1
    yhat[yhat>=0] = 1
    Ytest = Ytest.detach().cpu().numpy()
    yhat = yhat.t().detach().cpu().numpy()
    precision = average_precision_score(Ytest, yhat[0])
    recall = recall_score(Ytest, yhat[0])
    F1 = f1_score(Ytest, yhat[0])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)

def testunlabled(Xtest, betas, pics):
    transform = T.ToPILImage()
    yhat = Xtest.float() @ betas.view(-1,1)
    yhat[yhat<0] = -1
    yhat[yhat>=0] = 1
    for i in range(len(yhat)):
        if yhat[i] == 1:
            img = pics[i]/255
            save = transform(img)
            save.save('returnimages/'+str(i)+'.jpg')

def get_loss(x, y, beta): 
    xbeta = x@beta.view(-1, 1)
    yxbeta = y.view(-1,1)*xbeta
    loss = torch.log(1 + torch.exp(-yxbeta))
    ll_s = torch.mean(loss)
    return ll_s 

def err(x,y,beta): 
    xbeta = x@beta.view(-1,1)
    xbeta[xbeta<0] = -1
    xbeta[xbeta>=0] = 1
    
    return torch.mean((xbeta.view(-1)!=y).float())
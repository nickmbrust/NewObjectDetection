from asyncio import base_tasks
import torch
import rave
import numpy as np
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

def FSAunbalanced(Xpos, Xneg, k, T):
     
    #print(SXY.shape)
    SXXp = Xpos.mxx
    SXXn = Xneg.mxx
    mup = Xpos.mx
    mun = Xneg.mx
    wp = 1/(SXXp.shape[0])
    wn = 1/(SXXn.shape[0])
    beta = torch.zeros((SXXp.shape[0], 1)).to(device)
    eta = 1
    p = len(SXXp)
    mu = 0.1
    global_indices = torch.arange(0, SXXn.shape[0])
    for t in range(T):
      
       
        
        delta =  eta*((wp*SXXp + wn*SXXn)*beta+(wp*mup + wn*mun)*beta[0] + wn*mun - wp*mup )
        print(delta)
        beta = beta - delta
        
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
    yhat = Xtest @ betas
   
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(yhat.shape[0]):
        item = yhat[i].item()
        if item < 0:
            guess= -1
        else:
            guess = 1
        if guess == Ytest[i]:
            if guess == 1:
                tp+=1
            if guess == -1:
                tn+=1
        if guess != Ytest[i]:
            if guess == 1:
                fp+=1
            if guess == -1:
                fn+=1
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(precision, recall)

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
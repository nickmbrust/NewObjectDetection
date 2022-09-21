import torch
import rave
import numpy as np

def OLS(SXX, SXY):
    
    beta = torch.linalg.solve(SXX, SXY)

    return beta


def OFSA(SXX, SXY, k, T):
    SXX = np.array(SXX.cpu())
    SXY = np.array(SXY.cpu())
    #print(SXY.shape)
    beta = np.zeros((SXY.shape[0], 1))
    eta = 0.01
    p = len(SXX)
    mu = 1
    global_indices = torch.arange(0, SXX.shape[0])
    for t in range(T):
        delta =  eta*(SXX @ beta-SXY)
        print(delta.shape)
        beta = beta - delta
       # print(beta.shape)
        max = np.max((0, (T-t)/(t*mu+T)))
        # Mt = k + (p-k)*np.max((0, (T-t)/(t*mu+T)))
        # Mt = int(Mt)
        # print(Mt)
        # absbetas = np.abs(beta)
        # indices = np.argsort(-absbetas)
        # print(beta.shape)
        # beta = beta.T
        # beta = beta[indices[0:Mt]].T
        # global_indices = global_indices[indices[0:Mt]]    
    return beta
import torch
import rave
import numpy as np
def OLS(SXX, lmda):
    n = len(SXX)
    beta = np.zeros(SXX.shape[1],1)
    errorplot  = np.zeros((len(SXX),1))
    rn = np.eye(SXX.shape[1])

    for i in range(n):
        X = np.reshape(SXX[i],(SXX[i].shape[0], 1))
        top = ((1/lmda)*rn@X)
        bot = (1+(1/lmda)*X.T@rn@X)
        kn = top/bot
        loss = ytrue[i]-beta.t@X
        beta = beta - kn @ loss
        rn = (1/lmda)*rn-(1/lmda)*kn*X.T*rn
        errorplot[i] = np.linalg.norm(X@beta-ytrue)

    return beta

def OFSA(SXX, SXY, k, T):
    beta = 0
    loss = 1
    p = len(SXX)
    mu = np.average(SXX)
    M = []
    for t in range(T):
        beta = beta - loss*(SXX*beta-SXY)
        M[t] = k + (p-k)*np.max(0, (T-t)/(t*mu+T))

    return beta
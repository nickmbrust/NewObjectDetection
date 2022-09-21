import torch

class RAVE:
    def __init__(self):
        self.n = 0

    def add(self, X, y):
        n, p = X.shape
        Sx = torch.sum(X, dim=0)
        #Sy = torch.sum(y)
        Sxx = X.t() @ X
        print(Sxx.shape)
        Sxy = X.t() @ y
        #Syy = y.t() @ y
        if self.n == 0:
            self.n = n
            self.mx = Sx / n
            #self.my = Sy / n
            self.mxx = Sxx / n
            self.mxy = Sxy / n
            #self.myy = Syy / n
        else:
            self.mx = self.mx * (self.n / (self.n + n)) + Sx / (self.n + n)
            #self.my = self.my * (self.n / (self.n + n)) + Sy / (self.n + n)
            self.mxx = self.mxx * (self.n / (self.n + n)) + Sxx / (self.n + n)
            self.mxy = self.mxy * (self.n / (self.n + n)) + Sxy / (self.n + n)
            #self.myy = self.myy * (self.n / (self.n + n)) + Syy / (self.n + n)
            self.n = self.n + n

    def add_rave(self, rave):
        n = rave.n
        if self.n == 0:
            self.n = rave.n
            self.mx = rave.mx.clone()
            #self.my = rave.my.clone()
            self.mxx = rave.mxx.clone()
            self.mxy = rave.mxy.clone()
            #self.myy = rave.myy.clone()
        else:
            n0 = self.n / (self.n + n)
            n1 = n / (self.n + n)
            self.mx = self.mx * n0 + rave.mx * n1
           #self.my = self.my * n0 + rave.my * n1
            self.mxx = self.mxx * n0 + rave.mxx * n1
            self.mxy = self.mxy * n0 + rave.mxy * n1
            #self.myy = self.myy * n0 + rave.myy * n1
            self.n = self.n + n

    def subtract_mu(self,mu):
        # xx=(x-1*mu)^T(x-1*mu)/n=x^Tx/n-mu^Tmx-mx^Tmu+mu^Tmu
        mpn=mu.view(-1,1)@self.mx.view(1,-1)
        xx=self.mxx-mpn-mpn.t()+mu.view(-1,1)@mu.view(1,-1)
        return xx,self.mx-mu
    
    def multiply(self,xx,mx,si):
        xx=xx*si.view(-1,1)
        xx=xx*si.view(1,-1)
        return xx,mx*si

    def standardize_x(self):
        # standardize the raves for x
        var_x = torch.diag(self.mxx) - self.mx ** 2
        std_x = torch.sqrt(var_x)
        Pi = 1 / std_x

        XXn = self.mxx - self.mx.view(-1, 1) @ self.mx.view(1, -1)
        XXn *= Pi.view(1, -1)
        XXn *= Pi.view(-1, 1)
        self.sxi = Pi.clone()
        return (XXn, Pi)

    def standardize(self):
        # standardize the raves
        XXn, Pi = self.standardize_x()

        Temp1 = Pi * self.mxy
        Temp2 = self.my * Pi * self.mx
        XYn = Temp1 - Temp2

        return (XXn, XYn, Pi)

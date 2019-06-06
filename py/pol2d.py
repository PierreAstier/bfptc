import numpy as np
import itertools

class pol2d :
    def __init__(self, x,y,z,order, w=None):
        self.orderx = min(order,x.shape[0]-1)
        self.ordery = min(order,x.shape[1]-1)
        G = self.monomials(x.ravel(), y.ravel())
        if w is None:
            self.coeff,_,rank,_ = np.linalg.lstsq(G,z.ravel())
        else :
            self.coeff,_,rank,_ = np.linalg.lstsq((w.ravel()*G.T).T,z.ravel()*w.ravel())

    def monomials(self, x, y) :
        ncols = (self.orderx+1)*(self.ordery+1)
        G = np.zeros(x.shape + (ncols,))
        ij = itertools.product(range(self.orderx+1), range(self.ordery+1))
        for k, (i,j) in enumerate(ij):
            G[...,k] = x**i * y**j
        return G
            
    def eval(self, x, y) :
        G = self.monomials(x,y)
        return np.dot(G, self.coeff)


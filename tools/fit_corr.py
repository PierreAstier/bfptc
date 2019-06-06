#!/usr/bin/env python                                                                                                                        

import scipy.optimize as opt
from bfstuff.integ_et import *
import pyfits as pf
import sys


class Corr() :
    def __init__(self,corr_fits_name, sigma_corr_fits_name) :
        self.meas_corr = pf.getdata(corr_fits_name).T
        sigma = pf.getdata(sigma_corr_fits_name).T
        index_pos = sigma != 0
        self.weight=np.zeros(sigma.shape)
        self.weight[index_pos] = 1/sigma[index_pos]**2
        self.weight[0,0] = 0 # not measured actually.
        self.fit_range=15

    def model(self, params) :
        c = Config(params[0], params[1], 
                   params[2], params[3], params[4])
        fr = self.fit_range
        m = np.array([[c.EvalCorr(i,j) for j in range(fr)] for i in range(fr)])
        m[0,0] = 0 
        # fit normalization and offset to data:
        w= self.weight[0:fr,0:fr]
        y = self.meas_corr[0:fr,0:fr]
        sxx=(w*m*m).sum()
        sx=(w*m).sum()
        s1=(w).sum()
        sxy=(w*m*y).sum()
        sy=(w*y).sum()
        d= sxx*s1-sx*sx
        self.norm =  (s1*sxy-sx*sy)/d
        self.offset = (-sx*sxy+sxx*sy)/d        
        return self.norm*m + self.offset


    def write_model(self,params,filename, maxrange=10) :
        c = Config(params[0], params[1], 
                   params[2], params[3], params[4])
        self.model(params) # compute normalization
        c.WriteModel(filename, self.norm, maxrange)


    def chi2(self,params) :
        fr = self.fit_range
        m = self.model(params)
        w= self.weight[0:fr,0:fr]
        y = self.meas_corr[0:fr,0:fr]
        chi2 = (w*(m-y)**2).sum()
        print params, chi2
        return chi2

    def weighted_res(self,params) :
        fr = self.fit_range
        m = self.model(params)
        w= self.weight[0:fr,0:fr]
        y = self.meas_corr[0:fr,0:fr]
        wres = w*(m-y)
        return wres.flatten()
    

if __name__ == "__main__" :

    try :
        tofit = Corr(sys.argv[1],sys.argv[2])
    except :
        print "usage : %s <corr_slope.fits> <sig_corr_slope.fits>"%sys.argv[0]
        
    if len(sys.argv) < 3:
        raise ValueError("Wrong number of arguments")
    output_name = "output_model.dict"
    if len(sys.argv)==4 : output_name = sys.argv[3]

    params = np.array([5.10544252,3.7806627,   4.38392108,  0.65042887,  4.89542185])
    tofit.weight[0,0] = 0
    # switch from minimize to leastsq, but not tested yet
    #result = opt.minimize(tofit.chi2, params,method='BFGS',options=dict({'maxiter':3000}))
    params, cov_params, _, mesg, ierr = opt.leastsq(tofit.weighted_res, params, full_output = True)
    if ierr not in [1,2,3,4] : raise RuntimeError(mesg)
    print params
    print cov_params
    print "chi2", tofit.chi2(params)
    print "norm=%f, offset=%f"%(tofit.norm, tofit.offset)
    print "Writing model...%s"%output_name
    
    tofit.write_model(params, output_name)





        





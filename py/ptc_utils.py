from __future__ import print_function
import numpy as np
import matplotlib.pyplot as pl
import math

import scipy.interpolate as interp


import ptcfit

class load_params :
    """
    Prepare covariances for the PTC fit:
    - eliminate data beyond saturation
    - eliminate data beyond r (ignored in the fit
    - optionnaly (subtract_distant_value) subtract the extrapolation from distant covariances to closer ones, separately for each pair.
    - start: beyond which the modl is fitted
    - offset_degree: polynomila degree for the subtraction model
    """
    def __init__(self):
        self.r = 8
        self.maxmu = 2e5
        self.maxmu_el = 1e5
        self.subtract_distant_value = True
        self.start=12
        self.offset_degree = 1
        


def load_data(tuple_name,params) :
    """
    Returns a list of cov_fits, indexed by amp number.
    tuple_name can be an actual tuple (rec array), rather than a file name containing a tuple.

    params drives what happens....  the class load_params provides default values
    params.r : max lag considered
    params.maxmu : maxmu in ADU's

    params.subtract_distant_value: boolean that says if one wants to subtract a background to the measured covariances (mandatory for HSC flat pairs).
    Then there are two more needed parameters: start, offset_degree

    """
    if (tuple_name.__class__ == str) :
        nt = np.load(tuple_name) 
    else :
        nt = tuple_name
    exts = np.array(np.unique(nt['ext']), dtype = int)
    cov_fit_list = {}
    for ext in exts :
        print('extension=', ext)
        ntext = nt[nt['ext'] == ext]
        if params.subtract_distant_value :
            c = ptcfit.cov_fit(ntext,r=None)
            c.subtract_distant_offset(params.r, params.start, params.offset_degree)
        else :
            c = ptcfit.cov_fit(ntext, params.r)
        this_maxmu = params.maxmu            
        # tune the maxmu_el cut
        for iter in range(3) : 
            cc = c.copy()
            cc.set_maxmu(this_maxmu)
            cc.init_fit()# allows to get a crude gain.
            gain = cc.get_gain()
            if (this_maxmu*gain < params.maxmu_el) :
                this_maxmu = params.maxmu_el/gain
                continue
            cc.set_maxmu_electrons(params.maxmu_el)
            break
        cov_fit_list[ext] = cc
    return cov_fit_list



def fit_data(tuple_name, maxmu = 1.4e5, maxmu_el = 1e5, r=8) :
    """
    The first argument can be a tuple, instead of the name of a tuple file.
    returns 2 dictionnaries, one of full fits, and one with b=0

    The behavior of this routine should be controlled by other means.
    """
    lparams = load_params()
    lparams.subtract_distant_value = False
    lparams.maxmu = maxmu
    lparams.maxmu = maxmu_el = maxmu_el
    lparams.r = r
    cov_fit_list = load_data(tuple_name, lparams)
    # exts = [i for i in range(len(cov_fit_list)) if cov_fit_list[i] is not None]
    alist = []
    blist = []
    cov_fit_nob_list = {} # [None]*(exts[-1]+1)
    for ext,c in cov_fit_list.iteritems() :
        print('fitting channel %d'%ext)
        c.fit()
        cov_fit_nob_list[ext] = c.copy()
        c.params['c'].release()
        c.fit()
        a = c.get_a()
        alist.append(a)
        print(a[0:3, 0:3])
        b = c.get_b()
        blist.append(b)
        print(b[0:3, 0:3])
    a = np.asarray(alist)
    b = np.asarray(blist)
    for i in range(2):
        for j in range(2) :
            print(i,j,'a = %g +/- %g'%(a[:,i,j].mean(), a[:,i,j].std()),
                  'b = %g +/- %g'%(b[:,i,j].mean(), b[:,i,j].std()))
    return cov_fit_list, cov_fit_nob_list


# subtract the "long distance" offset from measured covariances



def CHI2(res,wy):
    wres = res*wy
    return (wres*wres).sum()
    

    
# pass fixed arguments using curve_fit:    
# https://stackoverflow.com/questions/10250461/passing-additional-arguments-using-scipy-optimize-curve-fit



def select_from_tuple(t, i, j, ext):
    cut = (t['i'] == i) & (t['j'] == j) & (t['ext'] == ext)
    return t[cut]


def eval_nonlin(tuple, knots = 20, verbose = False, fullOutput=False):
    """
    it will be faster if the tuple only contains the variances
    return value: a dictionnary of correction spline functions (one per amp)
    """
    amps = np.unique(tuple['ext'].astype(int))
    res={}
    if fullOutput:
        x = {}
        y = {}
    for i in amps :
        t = tuple[tuple['ext'] == i]
        clap = t['c1']
        mu = t['mu1']
        if fullOutput :
            res[i], x[i], y[i] = fit_nonlin_corr(mu,clap, knots=knots, verbose=verbose, fullOutput=fullOutput)
        else :
            res[i] = fit_nonlin_corr(mu,clap, knots=knots, verbose=verbose, fullOutput=fullOutput)
    if fullOutput:
        return res,x,y
    else :
        return res

    
# I don't know who wrote it...
def mad(data, axis=0, scale=1.4826):
    """
    Median of absolute deviation along a given axis.  

    Normalized to match the definition of the sigma for a gaussian
    distribution.
    """
    if data.ndim == 1:
        med = np.ma.median(data)
        ret = np.ma.median(np.abs(data-med))
    else:
        med = np.ma.median(data, axis=axis)
        if axis>0:
            sw = np.ma.swapaxes(data, 0, axis)
        else:
            sw = data
        ret = np.ma.median(np.abs(sw-med), axis=0)
    return scale * ret



#    mcc = interp.splev(x, s) # model values
#    dd = interp.splder(s)  # model derivative
#    der = interp.splev(x,dd) # model derivative values

def fit_nonlin_corr(xin, yclapin, knots = 20, loop = 20, verbose = False, fullOutput=False):
    """
    xin : the data to be "linearized"
    yclapin : the (hopefully) linear reference
    returns a  spline that can be used for correction uisng "scikit.splev"
    if full_output==True, returns spline,x,y  (presumably to plot)
    """
    # do we need outlier rejection ?
    # the xin has to be sorted, although the doc does not say it....
    index = xin.argsort()
    x = xin[index]
    yclap = yclapin[index]
    chi2_mask = np.isfinite(yclap) # yclap = nan kills the whole thing
    xx = x
    yyclap = yclap
    for i in range(loop):
        xx = xx[chi2_mask]
        # first fit the scaled difference between the two channels we are comparing
        yyclap = yyclap[chi2_mask]
        t = np.linspace(xx[0], xx[-1], knots)
        s = interp.splrep(xx, yyclap, task=-1, t=t[1:-1])
        model = interp.splev(xx, s)     # model values
        res = model - yyclap
        sig = mad(res)
        res = np.abs(res)
        if (res> (5 * sig)).sum()>0 : # remove one at a time
            chi2_mask = np.ones(len(xx)).astype(bool)
            chi2_mask[np.argmax(res)] = False
            continue
        else : break
    # normalize so that the gain does not change too much
    yyclap_norm = yyclap * xx.mean()/yyclap.mean()
    s = interp.splrep(xx, yyclap_norm, task=-1, t=t[1:-2])
    model = interp.splev(xx, s)     # model values
    # compute gain residuals
    print("nonlin gain residuals : %g"%(model/yyclap_norm-1).std())
    if verbose :     
        print("fit_nonlin loops=%d sig=%f res.max = %f"%(i,sig, res.max()))
    if fullOutput :
        return s, xx, yyclap_norm
    return s

def correct_tuple_for_nonlin(tuple, nonlin_corr=None, verbose=False, draw = False):
    """
    Compute the non-linearity correction  using the
    'c1' field, and applies it.
    Return value: corrected tuple
    """
    t00 = tuple[(tuple['i'] == 0)  & (tuple['j'] == 0)]
    if nonlin_corr == None :
        if draw :
            nonlin_corr = eval_nonlin_draw(t00, verbose=verbose)
        else :
            nonlin_corr = eval_nonlin(t00, verbose=verbose)
    amps = np.unique(t00['ext']).astype(int)
    # sort the tuple by amp, i.e. extension
    index = tuple['ext'].argsort()
    stuple=tuple[index]
    tuple = stuple # release memory ? not sure
    # find out where each amp starts and ends in the tuple
    ext = stuple['ext']
    diff = ext[1:] - ext[:-1]
    boundaries = [0]+ [ i+1 for i in range(len(diff)) if diff[i]!=0]+[len(ext)]
    amps = np.unique(ext).astype(int)
    start = [None]*int(amps.max()+1)
    end = [None]*int(amps.max()+1)
    for k in range(len(boundaries)-1):
        b = boundaries[k]
        amp = int(ext[b])
        start[amp]= b
        end[amp] = boundaries[k+1]
        # print amp,start[amp], end[amp], ext[start[amp]],ext[end[amp]-1]
    # now applyr the nonlinearity correction        
    for amp in amps:
        tamp = stuple[stuple['ext'] == amp]
        x = 0.5*(tamp['mu1'] + tamp['mu2'])
        iamp = int(amp)
        s = nonlin_corr[iamp]
        mu_corr = interp.splev(x, s) # model values
        dd = interp.splder(s)  # model derivative
        der = interp.splev(x,dd) # model derivative values
        stuple['mu1'][start[iamp]:end[iamp]] = mu_corr
        stuple['mu2'][start[iamp]:end[iamp]] = mu_corr
        stuple['var'][start[iamp]:end[iamp]] *= (der**2)
        stuple['cov'][start[iamp]:end[iamp]] *= (der**2)
    return stuple

#    return mcc, cvc, pix

def apply_quality_cuts(nt0, satu_adu=1.35e5, sig_ped=3):
    """
    dispersion of the pedestal and saturation
    """
    cut = (nt0['sp1']<sig_ped)  & (nt0['sp2']<sig_ped) & (nt0['mu1']<satu_adu)
    return nt0[cut]


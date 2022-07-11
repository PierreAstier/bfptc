from __future__ import print_function
import numpy as np
from bfptc import pol2d as pol2d

"""
This code implements the model (and the fit thereof) described in
https://arxiv.org/pdf/1905.08677.pdf
For the time beeing it uses as input a  numpy recarray which contains
one row per covariance and per pair: see routine make_cov_array
"""

from bfptc.ptc_utils import mad as mad


def compute_old_fashion_a(fit, mu_el) :
    """
    Compute the a coefficients the old way (slope of cov/var at a given flux mu-el)

    Returns the a array, computed this way, to be compare to the actual a_array from the model (fit.get_a())
    """
    gain = fit.get_gain()
    mu_adu = np.array([mu_el/gain])
    model = fit.eval_cov_model(mu_adu)
    var = model[0,0,0]
    # model is in ADU**2, so is var, mu is in adu.
    # So for a result in electrons^-1, we have to convert mu to electrons
    return model[0,:,:]/(var*mu_el)
    
        
        

from scipy.signal import fftconvolve


def make_cov_array(nt, r=8) :
    """from a tuple that contains rows with (at least):
    mu1, mu2, cov ,var, i, j, npix
    
    With one entry per lag, and image pair. 
    Different lags (i.e. different i and j) from the same
    image pair have the same values of mu1 and mu2. When i==j==0, cov
    = var.

    r is the maximum range to select from tuple. 

     If the input tuple contains several video channels, one should
     select the data of a given channel *before* entering this
     routine, as well as apply (e.g.) saturation cuts.

    returns cov[k_mu, j, i] , vcov[(same indices)], and mu[k]
    where the first index of cov matches the one in mu.

    Notes: this routine implements the loss of variance due to 
    clipping cuts when measuring variances and covariance. This is the
    *wrong* place to doit: it should happen inside the measurement code, 
    where the cuts are readily available.

    """
    if r is not None :
        cut = (nt['i']<r) & (nt['j'] < r)
        ntc = nt[cut]
    else : ntc = nt
    # increasing mu order, so that we can group measurements with the same mu 
    mu_tmp = (ntc['mu1'] + ntc['mu2'])*0.5
    ind = np.argsort(mu_tmp)
    ntc = ntc[ind]
    # should group measurements on the same image pairs(same average)
    mu = 0.5*(ntc['mu1']+ntc['mu2'])
    xx = np.hstack(([mu[0]], mu))
    delta = xx[1:] - xx[:-1]
    steps, = np.where(delta>0)
    ind = np.zeros_like(mu, dtype = int)
    ind[steps] = 1
    ind = np.cumsum(ind) # this acts as an image pair index.
    # now fill the 3-d cov array (and variance)
    mu_val = np.array(np.unique(mu)) # have to convert because type is 'NTuple'
    i = ntc['i'].astype(int)
    j = ntc['j'].astype(int)
    c = 0.5*ntc['cov']
    n = ntc['npix']
    v = 0.5*ntc['var']
    # book and fill
    cov = np.ndarray((len(mu_val), np.max(i)+1, np.max(j)+1))
    vcov = np.zeros_like(cov)
    cov[ind, i, j] = c
    vcov[ind, i, j] = v**2/n
    vcov[:, 0,0] *= 2 # var(v) = 2*v**2/N
    # compensate for loss of variance and covariance due to outlier elimination
    # when computing variances (cut to 4 sigma): 1 per mill for variances and twice as
    # much for covariances:
    fact = 1.00107
    cov *= fact*fact
    cov[:, 0,0] /= fact
    # done
    return cov, vcov, mu_val


def symmetrize(a):
    # copy over 4 quadrants prior to convolution.
    target_shape = list(a.shape)
    r1,r2 = a.shape[-1],a.shape[-2]
    target_shape[-1] = 2*r1-1
    target_shape[-2] = 2*r2-1
    asym = np.ndarray(tuple(target_shape))
    asym[...,r2-1:   ,r1-1:   ]= a
    asym[...,r2-1:   ,r1-1::-1]= a
    asym[...,r2-1::-1,r1-1::-1]= a
    asym[...,r2-1::-1,r1-1:   ]= a
    return asym


# a few utilities to transform a covariance tuple into a ptcfit
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
                if this_maxmu<0 :
                    print(" the initialization went crazy hope the full fit, gets better")
                    break
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



# The actual PTC/covariance curves fit 
from bfptc.fitparameters import FitParameters
import copy

from scipy.optimize import leastsq


class cov_fit :
    def __init__(self, nt, r=8) :
        self.cov, self.vcov, self.mu = make_cov_array(nt, r)
        self.sqrt_w  = 1./np.sqrt(self.vcov)
        self.r = self.cov.shape[1]
        

    def subtract_distant_offset(self,r=8, start=15, degree=1) :
        assert(start < self.r)
        for k in range(len(self.mu)) :
            # I want a copy because it is going to be altered
            w = self.sqrt_w[k,...] + 0.
            sh = w.shape
            i,j =  np.meshgrid(range(sh[0]), range(sh[1]), indexing='ij')
            # kill the core for the fit
            w[:start,:start] = 0
            poly = pol2d.pol2d(i,j,self.cov[k,...], degree+1, w=w)
            back = poly.eval(i,j)
            self.cov[k,...] -= back
        self.r = r
        self.cov = self.cov[:,:r,:r]
        self.vcov = self.vcov[:,:r,:r]
        self.sqrt_w = self.sqrt_w[:,:r,:r]
        
    def set_maxmu(self, maxmu) :
        # mus are sorted at construction
        index = self.mu<maxmu
        k = index.sum()
        self.mu = self.mu[:k]
        self.cov = self.cov[:k,...]
        self.vcov = self.vcov[:k,...]
        self.sqrt_w = self.sqrt_w[:k,...]

    def set_maxmu_electrons(self,maxmu_el) :
        g = self.get_gain()
        kill = (self.mu*g > maxmu_el)
        self.sqrt_w[kill,:,:] = 0
                        

        
    def copy(self) :
        cop = copy.deepcopy(self)
        # deepcopy does not work for FitParameters, for now(06/18).
        if hasattr(self, 'params'):
            cop.params = self.params.copy()
        return cop
        
    def init_fit(self) :
        """
        performs a crude parabolic fit of the data in order to start 
        the full fit close to the solution
        """
        # number of parameters for 'a'
        len_a = self.r*self.r
        # define parameters : c corresponds to a*b in the paper. 
        self.params=FitParameters([('a', len_a), ('c', len_a), ('noise', len_a), ('gain', 1)])
        self.params['gain'] = 1.
        # obvious : c=0 in a first go.
        self.params['c'].fix(val = 0.)
        # plumbing: extract stuff from the parameter structure
        # I think those are references
        a = self.params['a'].full.reshape(self.r, self.r)
        noise = self.params['noise'].full.reshape(self.r,self.r)
        gain = self.params['gain'].full[0]
        old_chi2 = 1e30
        for iter in range(5):   # iterate the fit to account for higher orders
            # the chi2 does not necessarily go down, so one could
            # stop when it increases
            model = self.eval_cov_model() # this computes the full model.
            # loop on lags
            for i in range(self.r) :
                for j in range(self.r) :
                    # fit a given lag with a parabola
                    p = np.polyfit(self.mu, self.cov[:,i,j] - model[:,i,j],
                                   2, w = self.sqrt_w[:,i,j])
                    # model equation in the paper:
                    a[i,j] += p[0]
                    noise[i,j] += p[2]*gain*gain
                    if (i+j==0) :
                        gain  = 1./(1/gain+p[1])
                        self.params['gain'].full[0] = gain
                    #if (i+j==0) : print(p, gain, a[0,0])
            chi2 = self.chi2()
            print('iter,chi2 a00 gain = ', iter, chi2, a[0,0], gain)
            if chi2 > old_chi2 : break
            old_chi2 = chi2

                
    def get_param_values(self):
        """
        return an array of free parameter values (it is a copy)
        """
        return self.params.free + 0.

    def set_param_values(self, p):
        self.params.free = p

    def eval_cov_model(self, mu=None) :
        """ 
        by default, computes the cov_model for the mus stored (self.mu)
        returns cov[Nmu, self.r, self.r]. The PTC is cov[:, 0, 0].
        mu and cov are in ADUs and ADUs squared. to use electrons for both,
        the gain should be set to 1.
        This routine implements the model in 1905.08677
        """
        sa = (self.r, self.r)
        a = self.params['a'].full.reshape(sa)
        c = self.params['c'].full.reshape(sa)
        gain = self.params['gain'].full[0]
        noise = self.params['noise'].full.reshape(sa)
        # pad a with zeros and symmetrize
        a_enlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        a_enlarged[0:sa[0], 0:sa[1]] = a
        asym = symmetrize(a_enlarged)
        # pad c with zeros and symmetrize
        c_enlarged = np.zeros((int(sa[0]*1.5)+1, int(sa[1]*1.5)+1))
        c_enlarged[0:sa[0], 0:sa[1]] = c
        csym = symmetrize(c_enlarged)
        a2 = fftconvolve(asym, asym, mode = 'same')
        a3 = fftconvolve(a2, asym, mode = 'same')
        ac = fftconvolve(asym, csym, mode = 'same')
        (xc,yc) = np.unravel_index(np.abs(asym).argmax(), a2.shape)
        range = self.r
        a1 = a[np.newaxis, :,:]
        a2 = a2[np.newaxis, xc:xc+range, yc:yc+range]
        a3 = a3[np.newaxis, xc:xc+range, yc:yc+range]
        ac = ac[np.newaxis, xc:xc+range, yc:yc+range]
        c1 = c[np.newaxis, : : ]
        if mu is None : mu = self.mu
        # assumes that mu is 1d
        bigmu = mu[:, np.newaxis, np.newaxis]*gain
        # c (=a*b in the paper) also has a contribution to the last term, that is absent for now.
        model = bigmu/(gain*gain)*(a1*bigmu+2./3.*(bigmu*bigmu)*(a2+c1)+(1./3.*a3+5./6.*ac)*(bigmu*bigmu*bigmu)) + noise[np.newaxis,:,:]/gain**2
        # add the Poisson term, and the read out noise (variance, obviously) 
        model[:,0,0] += mu/gain 
        return model

    def get_a(self) :
        return self.params['a'].full.reshape(self.r, self.r)

    def get_b(self) :
        return self.params['c'].full.reshape(self.r, self.r)/self.get_a()

    def get_c(self) :
        return self.params['c'].full.reshape(self.r, self.r)
    
    def _get_cov_params(self,what):
        indices = self.params[what].indexof()
        #indicesp = [i for i in range(len(indices)) if indices[i]>=0 ]
        i1 = indices[:,np.newaxis]
        i2 = indices[np.newaxis, :]
        covp = self.cov_params[i1,i2]
        return covp

    def get_a_cov(self) :
        cova = self._get_cov_params('a')
        return cova.reshape((self.r, self.r, self.r,self.r))

    def get_a_sig(self) :
        cova = self._get_cov_params('a')
        return np.sqrt(cova.diagonal()).reshape((self.r, self.r))

    def get_b_cov(self) :
        # b = c/a
        covb = self._get_cov_params('c')
        aval = self.get_a().flatten()
        factor = np.outer(aval,aval)
        covb /= factor
        return covb.reshape((self.r, self.r, self.r,self.r))

    def get_c_cov(self) :
        cova = self._get_cov_params('c')
        return cova.reshape((self.r, self.r, self.r,self.r))

    def get_gain(self) :
        return self.params['gain'].full[0]

    def get_ron(self) :
        return self.params['noise'].full[0]

    def get_noise(self) :
        return self.params['noise'].full.reshape(self.r, self.r)
    
    def set_a_b(self, a,b) :
        self.params['a'].full = a.flatten()
        self.params['c'].full = a.flatten()*b.flatten()
    
    def chi2(self) :
        return (self.weighted_res()**2).sum()

    def wres(self, params = None) :
        if params is not None:
            self.set_param_values(params)
        model = self.eval_cov_model()
        return ((model-self.cov)*self.sqrt_w)
    
    def weighted_res(self, params = None):
        """
        to be used via:
        c = cov_fit(nt)
        c.init_fit()
        coeffs, cov, _, mesg, ierr = leastsq(c.weighted_res, c.get_param_values(), full_output=True )
        works nicely indeed.
        """
        return self.wres(params).flatten()
    

    def fit(self, p0 = None, nsig = 5) :
        """
        Carries out a fit using scipy.optimize.leastsq.
        May raise RuntimeError
        """
        if p0 is None:
            p0 = self.get_param_values()
        n_outliers = 1
        while (n_outliers != 0) : 
            for iter in range(2):
                try :
                    coeffs, cov_params, _, mesg, ierr = leastsq(self.weighted_res, p0, full_output=True, maxfev=40000)
                except RuntimeError :
                    if mesg.find('Number of calls to function has reached') == 0:
                        print ('having hard time to fit, trying to insist')
                    else:
                        raise RuntimeError(msg)
                except TypeError as e : #happens when there is less data than parameters
                    # re throw it as a RunTimeError that 
                    # should be caught in the calling routine:
                    raise RuntimeError(str(e))
            wres = self.weighted_res(coeffs)
            # do not count the outliers as significant : 
            sig = mad(wres[wres != 0]) 
            mask = (np.abs(wres)>(nsig*sig))
            self.sqrt_w.flat[mask] = 0 #flatten makes a copy
            n_outliers = mask.sum()
            print(" dropped %d outliers (/%d)"%(n_outliers, len(mask)))

        if ierr not in [1,2,3,4] :
            print("minimisation failed ", mesg)
            raise RuntimeError(mesg)
        if cov_params is None:
            print(' the fit did not deliver the covariance matrix')
        self.cov_params = cov_params
        return coeffs
    
    def ndof(self):
        mask = self.sqrt_w != 0
        return mask.sum() - len(self.params.free)
            
    def get_normalized_fit_data(self, i, j, divide_by_mu=True) :
        """
        from a cov_fit, selects from (i,j) and 
        returns mu*gain, cov[i,j]*gain**2 model*gain**2 and sqrt_w/gain**2
        """
        gain = self.get_gain()
        x = self.mu*gain
        y = self.cov[:,i,j]*(gain**2)
        model = self.eval_cov_model()[:,i,j]*(gain**2)
        w = self.sqrt_w[:,i,j]/(gain**2)
        # select data used for the fit:
        mask = w != 0
        w = w[mask]
        model = model[mask]
        x = x[mask]
        y = y[mask]
        if (divide_by_mu) :
            y /= x
            model /= x
            w *= x
        return x,y,model,w
    

    

    def __call__(self, params) :
        self.set_param_values(params)
        chi2 = self.chi2()
        print('chi2 = ',chi2)
        return chi2


import scipy.optimize as opt


def fft_cov_fun(x, a) :
    return 1/(2*a)*(np.exp(2*a*x)-1)


class cov_fit_fft :
    def __init__(self, nt, r=8) :
        self.r = r
        self.cov, self.vcov, self.mu = make_cov_array(nt, self.r)
        sc = self.cov.shape
        cov_enlarged = np.zeros((sc[0], int(sc[1]*1.5)+1, int(sc[2]*1.5)+1))
        cov_enlarged[:, 0:sc[1], 0:sc[2]] = self.cov
        cov_sym = symmetrize(cov_enlarged)
        #cov_sym = symmetrize(self.cov)
        self.cov_sym = cov_sym
        self.p_sym_ = np.fft.fft2(cov_sym) # fft over the two last indices.
        self.p_sym_ = np.fft.fftshift(self.p_sym_, axes=(-2,-1))
        l1 = self.p_sym_.shape[1]
        hl1 = l1/2
        argi = (np.linspace(0,l1-1,l1)-hl1)*(hl1)/float(l1)
        shift_i = np.exp(1j*2.*np.pi*argi)
        l2 = self.p_sym_.shape[2]
        hl2 = l2/2
        argj = (np.linspace(0,l2-1,l2)-hl2)*(hl2)/float(l2)
        shift_j = np.exp(1j*2.*np.pi*argj)
        shifts = np.outer(shift_i, shift_j)
        self.p_sym = (self.p_sym_*np.expand_dims(shifts, 0)).real
        # cook up for the gain:
        param = np.polyfit(self.mu, self.cov[:,0,0], 3, w = 1/np.sqrt(self.vcov[:, 0, 0]))
        self.g = 1/param[-2]
        print('found a gain = ', self.g)
        

    def fit(self) :
        cs = self.p_sym.shape
        self.fa = np.zeros_like(self.p_sym[0,...])
        for ik in range(cs[2]) :
            for jk in range(cs[1]) :
                y = self.g**2*self.p_sym[:, ik, jk]
                wy = self.g**2*np.sqrt(self.vcov[:, 0, 0])
                x = self.g**2*self.mu
                # guess some initial value
                p_poly = np.polyfit(x, y, 2, w = wy)
                p0 = np.ndarray((1))                
                p0[0] = p_poly[0]
                coeffs, cov, _, mesg, ierr = leastsq(WeightedRes(fft_cov_fun ,x,y, sigma = 1/wy), p0, full_output=True)
                if ierr not in [1,2,3,4] : raise RuntimeError(mesg)
                self.fa[ik,jk] = coeffs[0]
        tmp = np.fft.ifftshift(self.fa)
        self.a_ = np.fft.fftshift(np.fft.ifft2(tmp))
        #it turns out that a_ is real up to rounding errors
        self.a = self.a_.real
        
        
class WeightedRes:
    def __init__(self, model, x, y, sigma=None):
        self.x = x
        self.y = y
        self.model = model
        self.sigma= sigma

# could probably be more efficient (i.e. have two different functions)
    def __call__(self, params) :
        if self.sigma is None:
            res = self.y-self.model(self.x,*params)
        else : 
            res = (self.y-self.model(self.x,*params))/self.sigma
        return res

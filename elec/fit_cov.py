#!/usr/bin/env python
from integ_et import *
import astropy.io.fits  as pf

from bfptc.fitparameters import FitParameters
import copy

class electro_fit() :
    """
    class to handle the electrostatic fit of area coefficients
    The actual electrostatic calculations are done in integ_et.py
    """
    def __init__(self,meas_a_fits_name, sig_a_fits_name) :
        """
        """
        self.meas_a = pf.getdata(meas_a_fits_name)
        # beware : meas_a is the "a" array as measured, the "a" parameter is the (half) size of the charge cloud in the ccd.
        siga = pf.getdata(sig_a_fits_name)
        self.sqrt_w = 1/siga
        self.fit_range=min(self.meas_a.shape[0], 8)
        self.fitting_offset = False
        self.params = FitParameters([('z_q',1), ('zsh',1),('zsv',1),('a',1),('b',1),('thickness',1), ('pixsize',1)])

    def set_params(self, dict) :
        for name,val in dict.items() :
            self.params[name] = val

    def get_params(self) :
        """
        return a  copy of the free params vector
        """
        return self.params.free + 0.
            
    def get_a(self):
        fr = self.fit_range
        return self.meas_a[0:fr,0:fr]

    def model(self, free_params = None) :
        m = self.raw_model(free_params)
        a,b = self.normalize_model(m)
        return a*m+b

    def raw_model(self,free_params = None) :
        if free_params is not None :
            # assign what the minimizer is asking for:  
            self.params.free = free_params
        # need all the parameters as a dictionnary:
        dict = { key: self.params[key].full[0]+0 for key in list(self.params._pars.struct.slices.keys())}
        # push them into the electrostatic calculator
        c = CcdGeom(**dict)
        fr = self.fit_range
        # compute the observables
        # m = np.array([[c.EvalAreaChange(i,j) for j in range(fr+1)] for i in range(fr+1)])
        # We use analytic integrals, hence the "Fast" routine.
        if hasattr(self,'npair') :
            m = c.EvalAreaChangeSidesFast(fr,npair=self.npair)
        else:
            m = c.EvalAreaChangeSidesFast(fr)
        
        # I am almost sure it is useless to compute (just above)
        # more than we use (just below).
        m = m [:fr,:fr]
        return m
    
    def normalize_model(self, m) :
        """
        The overall normalization is a linear parameter. 
        We just hide from the minimizer by computing the optimal value given
        the other parameters.
        """
        # fit normalization and possibly a large distance offset to data :
        fr = m.shape[0]
        sqrtw= self.sqrt_w[ :fr, :fr]
        w = sqrtw**2
        y = self.meas_a[ :fr, :fr]
        if (self.fitting_offset) :
            sxx=(w*m*m).sum()
            sx=(w*m).sum()
            s1=(w).sum()
            sxy=(w*m*y).sum()
            sy=(w*y).sum()
            d= sxx*s1-sx*sx
            a =  (s1*sxy-sx*sy)/d
            b = (-sx*sxy+sxx*sy)/d        
            return a*m+b
        else :
            # just scale
            a = (w*y*m).sum()/(w*m*m).sum()
            b = 0
        return a,b

    def wres_array(self,params) :
        fr = self.fit_range
        m = self.model(params)
        w= self.sqrt_w[0:fr,0:fr]
        y = self.meas_a[0:fr,0:fr]
        # these are two 2-d arrays to be mutiplied term to term:
        start = (w*(m-y))
        # and the result has the same size as both of them.
        return start

    def write_results_txt(self,filename, npy_file_name) :
        npy = np.load(npy_file_name)
        f = open(filename,'w')
        f.write('@RANGE %d\n'%self.fit_range)
        #f.write('@Z_END %f\n'%z_end)        
        f.write('#i:\n#j:\n#aN:\n#aE:\n#aS:\n#aW:\n#ath:\n#ameas:\n#sig_ameas:\n#end\n')
        npy = npy.view(np.recarray)
        for row in npy:
            f.write('%d %d %g %g %g %g %g %g %g\n'%(row.i,row.j,
                                                    row.aN,row.aE,row.aS,row.aW,row.ath, row.ameas, row.sig_ameas))
        f.close()


    class boundary_shifts :

        def __init__(self, el_fit, z_end):
            assert z_end>0 
            dict = { key: el_fit.params[key].full[0]+0 for key in list(el_fit.params._pars.struct.slices.keys())}
            c = CcdGeom(**dict)
            ii,jj = np.meshgrid( list(range(el_fit.fit_range)), list(range(el_fit.fit_range)))
            ii = ii.flatten()
            jj = jj.flatten()
            self.aN = np.ndarray((el_fit.fit_range,el_fit.fit_range))
            self.aS = np.zeros_like(self.aN)
            self.aE = np.zeros_like(self.aN)
            self.aW = np.zeros_like(self.aN)
            self.ath = np.zeros_like(self.aN)
            a,b = el_fit.normalize_model(el_fit.raw_model())
            for (i,j) in zip(ii,jj) :
                self.aN[i,j] = -(a*c.Integ_Ey_fast(i,j,1, z_end = z_end)+0.25*b)
                self.aS[i,j] = -(a*c.Integ_Ey_fast(i,j,-1, z_end = z_end)+0.25*b)
                self.aW[i,j] = -(a*c.Integ_Ex_fast(i,j,-1, z_end = z_end)+0.25*b)
                self.aE[i,j] = -(a*c.Integ_Ex_fast(i,j,1, z_end = z_end)+0.25*b)
            self.ath = a*c.EvalAreaChangeSidesFast(el_fit.fit_range, z_end=z_end)+b

        import copy
            
        def __rmul__(self, factor) :
            """
            """
            res = copy.deepcopy(self)
            res.aN *= factor
            res.aS *= factor
            res.aE *= factor
            res.aW *= factor
            res.ath *= factor
            return res

        def __add__(self, other) :
            """
            """
            res = copy.deepcopy(self)
            res.aN += other.aN
            res.aS += other.aS
            res.aE += other.aE
            res.aW += other.aW
            res.ath += other.ath
            return res

                
    def write_results_np(self,filename, conversion_weights=None) :
        """
        If provided, conversion_weights is expected to be a list of pairs of (depth, probability)
        the routine computes the model corresponding to this probablity distribution.
        If conversion_depth is not provided , then [(0, 1.)] is used as the distribution.
        """
        z_end = self.params["thickness"].full[0]
        if conversion_weights is None:
            conversion_weights = (np.array([0.]),np.array([1.]))
        # need the parameters as a dictionnary
        #dict = { key: self.params[key].full[0]+0 for key in list(self.params._pars.struct.slices.keys())}
        #c = CcdGeom(**dict)
        #ii,jj = np.meshgrid( list(range(self.fit_range)), list(range(self.fit_range)))
        #ii = ii.flatten()
        #jj = jj.flatten()
        res = None
        (d,p) = conversion_weights
        p /= p.sum() # normalize to 1.
        for (depth, prob) in zip(d,p) :
            if res is None:
                res = prob*electro_fit.boundary_shifts(self, z_end-depth)
            else :
                res = res + prob*electro_fit.boundary_shifts(self, z_end-depth)
        
        # have to compute the normalization and offset. I am not sure that the offset is really meaningful.
        # it is set to zero in practice here
        #a,b = self.normalize_model(self.raw_model())
        #res.aN = a*res.aN+0.25*b
        #res.aS = a*res.aS+0.25*b
        #res.aE = a*res.aE+0.25*b
        #res.aW = a*res.aW+0.25*b
        #res.ath = a*res.ath+b
        tuple = np.recarray(res.aN.size, dtype=[('i',np.int), ('j',np.int),
                                          ('aN', np.float),
                                          ('aW', np.float),
                                          ('aS', np.float),
                                          ('aE', np.float),
                                          ('ath', np.float),
                                          ('ameas',np.float),
                                          ('sig_ameas',np.float)])

        for k,((i,j),v) in  enumerate(np.ndenumerate(res.aN)):
            row=tuple[k]
            row.i = i
            row.j = j
            row.aN = res.aN[i,j]
            row.aS = res.aS[i,j]
            row.aE = res.aE[i,j]
            row.aW = res.aW[i,j]
            row.ath = res.ath[i,j]            
            row.ameas = self.meas_a[i,j]
            sig = 1./self.sqrt_w[i,j] if self.sqrt_w[i,j]>0 else -1
            row.sig_ameas = sig
        np.save(open(filename,'wb'),tuple)                

        
    def wres(self,params) :
        """
        This is the routine for leastsq. 
        returns a 1-d array of weighted residuals.
        implements constraints as residuals that increase rapidly 
        when constraints are violated. 
        This technique allows us to use leastsq which is much 
        better than anything els I tried. 
        """
        wres = self.wres_array(params).flatten()
        # constraints :
        n_constraints = 5
        nt = wres.size
        ret = np.ndarray((nt+n_constraints))
        ret[:nt] = wres
        # z_q >0
        z_q = self.params['z_q'].full[0]
        ret[nt] = np.exp(-(z_q-0.1)*300)
        nt += 1
        # zsh > z_q
        zsh = self.params['zsh'].full[0]
        ret[nt] =  np.exp((z_q-zsh)*300)
        nt +=1
        # zsv > z_q
        zsv = self.params['zsv'].full[0]
        ret[nt] =  np.exp((z_q-zsv)*300)
        nt +=1
        # 0.35 pixsize > a, same for b
        a = self.params['a'].full[0]
        b = self.params['b'].full[0]
        pixsize = self.params['pixsize'].full[0]
        ret[nt] =  np.exp((a-0.35*pixsize)*300)
        nt +=1
        ret[nt] =  np.exp((b-0.35*pixsize)*300)
        nt +=1

        print('chi2 %g'%(ret**2).sum(), ' params ', params)
        return ret
    
    def __call__(self,params) :
        chi2 = ((self.wres(params))**2).sum()
        print(params, chi2)
        return chi2
    
from scipy.optimize import leastsq

def test_cont() :
    """
    this routine was meant to test the continuity of the electric field of the 
    model ( in CcdModel) at distances across a change of calculation technique.
    Not sure it is still meaningful.
    """
    params = np.array([ 1.,  1.5, 2. ,  2. , 5.] )
    t = 30.
    c = Config(ConfigParams(*params), CcdGeom(t, 18.))
    rho = np.linspace(0,200,num=200)
    X = np.zeros((len(rho),3))
    X[...,1] = rho
    X[:,2] = t/2.
    return X, c.Exyz(X)

from bfptc.ptcfit import symmetrize
import sys



def test_n():
    """
    test routine to see how results depend on how far teh series is summed.
    """
    tofit = electro_fit("a.fits","siga.fits")
    tofit.fit_range = 10
    tofit.set_params({'thickness': 200., 'pixsize' : 15.,
                      'z_q': 5.58829214, 'zsv': 5.59954644,
                      'zsh': 7.66267472,
                      'a' : 5.23366811, 'b': 2.85267152})
    tofit.params.fix('pixsize')
    tofit.params.fix('thickness')
    tofit.write_results_np('toto.npy')
    tofit.write_results_txt('bfshifts_O.list','toto.npy')              
    tofit.npair=11
    tofit.write_results_np('toto.npy')
    tofit.write_results_txt('bfshifts_11.list','toto.npy')
    tofit.npair=12
    tofit.write_results_np('toto.npy')
    tofit.write_results_txt('bfshifts_12.list','toto.npy')              
    tofit.npair=13
    tofit.write_results_np('toto.npy')
    tofit.write_results_txt('bfshifts_13.list','toto.npy')              

import sys
if __name__ == "__main__" :
    """
    nothing serious has been designed to interface this fit to the 
    outside world. the data is provided as a.fits and siga.fits.
    Then you have to define the initial parameters and the CCD thicknes 
    and pixels size. 
    """
    # thickness = 200, pixsize = 15
    # (CCD Hamamatsu for HSC, see the abstract of doi: 10.1093/pasj/psx063
    tofit = electro_fit("a.fits","siga.fits")
    tofit.fit_range = 10
    tofit.set_params({'thickness': 200., 'pixsize' : 15., 'z_q' : 1, 'zsh':3., 'zsv' : 3., 'a':2., 'b':6.})
    tofit.params.fix('pixsize')
    tofit.params.fix('thickness')
    #tofit.params.fix('thickness')
    #tofit.params.fix('a')
    #tofit.params.fix('b')
    # drop beginning of serial direction
    #print("Warning : masking the 3 first serial data points")
    #tofit.sqrt_w[0,0] = 0
    #tofit.sqrt_w[1,0] = 0
    #tofit.sqrt_w[2,0] = 0
    #tag = '_noserial'
    tag = ''
    
    values = {'z_q': 3.98788573, 'zsv': 6.64889138, 'zsh': 4.76154228,
              'b': 5.23784427, 'a' :2.56314585 }

    tofit.set_params(values)
    params = tofit.get_params()
    print('params',params)

    # params = np.array([4.25842783, 6.57151066, 4.73493689, 5.23812866, 3.15376615])
    if True :
        coeffs, cov_params, _, mesg, ierr = leastsq(tofit.wres, params, full_output=True)
        print(coeffs, cov_params, mesg, ierr)
    else :
        coeffs = params
    print('chi2 %g'%tofit(coeffs))
    print('model : \n',tofit.model(coeffs))
    print('data : \n',tofit.get_a())
    print('chi2 :\n',tofit.wres_array(coeffs)**2)
    print('params', tofit.params)
    print('sums : data %g model %g'%
          (symmetrize(tofit.get_a()).sum(), symmetrize(tofit.model(coeffs)).sum()))
    tofit.write_results_np('avalues%s.npy'%tag)
    tofit.write_results_txt('bfshifts%s.list'%tag,'avalues%s.npy'%tag)
    (z,p) = np.load('zprob.npy')
    print('integrating z band')
    tofit.write_results_np('avalues_z.npy', (z,p))
    tofit.write_results_txt('bfshifts_z.list', 'avalues_z.npy')
    conversion_depth = np.load('yprob.npy')
    print('integrating y band')
    tofit.write_results_np('avalues_y.npy', conversion_depth)
    tofit.write_results_txt('bfshifts_y.list', 'avalues_y.npy')
    
    m,o = tofit.normalize_model(tofit.raw_model())
    for r in range(tofit.get_a().shape[0],28,5) :
        tofit.fit_range = r
        model = m * tofit.raw_model(coeffs) + o
        s = symmetrize(model).sum()
        print('range %d sum %g'%(r,s))


        
    
    
    



#!/usr/bin/env python 

from __future__ import print_function

try :
    import astropy.io.fits as pf
except :
    import pyfits as pf

    
import numpy as np
import sys



# the same routine could be written using (numpy) MaskedArrays.
# They do not seem that efficient, and we'll use a floating point mask
# to compute its Fourier transform
def find_mask(im, nsig, w=None) :
    if w is None : w = np.ones(im.shape)
    #  mu is used for sake of numerical precision in the sigma
    # computation, and is needed (at full precision) to identify outliers. 
    count = w.sum()
    # different from (w*im).mean()
    mu = (w*im).sum()/count
    # same comment for the variance. 
    sigma = np.sqrt((((im-mu)*w)**2).sum()/count)
    for iter in range(3) :
        outliers = np.where(np.abs((im-mu)*w)>nsig*sigma)
        w[outliers] = 0
        #  does not work :
        # count = im.size-ouliers[0].size
        count = w.sum()
        mu = (w*im).sum()/count
        newsig = np.sqrt((((im-mu)*w)**2).sum()/count)
        if (np.abs(sigma-newsig)<0.02*sigma) :
            sigma = newsig
            break
        sigma = newsig
    return w

def write_fits(filename, d) :
    l = pf.HDUList()
    l.append(pf.ImageHDU(data=d))
    l.writeto(filename, clobber=True)
    l.close()

# a simplified utilty to compute the sky background.

def fit_back(im, stepx, stepy=None) :
    """
    fits a polynomial sky to the image. Misses a mask argument
    """
    if stepy is None : stepy = stepx
    nx = im.shape[0]/stepx
    if nx*stepx<im.shape[0] : nx += 1
    ny = im.shape[1]/stepy
    if ny*stepy<im.shape[1] : ny += 1
    medians = np.ndarray((nx,ny))
    xpos = np.ndarray(nx)
    ypos = np.ndarray(ny)
    for ix in range(nx) :
        maxx = min(stepx*(ix+1), im.shape[0])
        xpos[ix] = 0.5*(maxx-1+stepx*ix)
        for iy in range(ny):
            maxy = min(stepy*(iy+1), im.shape[1])
            ypos[iy] = 0.5*(maxy-1+stepy*iy)
            medians[ix,iy] = np.median(im[ix*stepx:maxx, iy*stepy:maxy])
    nx,ny = im.shape
    # some sort of reduced coordinates for numerical stability.
    # It does improve things, by several orders of magnitude.
    if True :
        ax=1./float(nx)
        bx = -0.5
        ay = 1./float(ny)
        by = -0.5
    else :
        ax=1.
        bx=0.
        ay=1.
        by=0.
    x,y = np.meshgrid(ax*xpos+bx,ay*ypos+by,indexing='ij')
    p = pol2d(x,y,medians,2)
    # now evaluate at all positions of the image
    x,y = np.meshgrid(np.linspace(bx,ax*(nx-1)+bx, num=nx),
                      np.linspace(by,ay*(ny-1)+by, num=ny)+by,indexing='ij')
    return p.eval(x,y)

# This class is mostly justified by writing the "cov" function with
# only 2 arguments.
class cov_fft :
    def __init__(self, diff,w, parameters) :
        """
        This class computed (via FFT), the nearby pixel correlation function.
        The range is controlled by "parameters", as well as 
        the actual FFT shape.
        """
        self.parameters = parameters
        maxrange = parameters.maxrange
        
        # check that the zero padding implied by "fft_shape"
        # is large enough for the required correlation range
        fft_shape = parameters.fft_shape
        assert(fft_shape[0]>diff.shape[0]+maxrange+1)
        assert(fft_shape[1]>diff.shape[1]+maxrange+1)
        tim = np.fft.rfft2(diff*w, parameters.fft_shape)
        tmask = np.fft.rfft2(w, parameters.fft_shape)
        # sum of  "squares"
        self.pcov = np.fft.irfft2(tim*tim.conjugate())
        # sum of values (depends on the offets indeed)
        self.pmean= np.fft.irfft2(tim*tmask.conjugate())
        # number of w!=0 pixels. 
        self.pcount= np.fft.irfft2(tmask*tmask.conjugate())
    
    def cov(self, dx,dy) :
        """
        covariance for dx,dy averaged with dx,-dy if both non zero.
        """
        # compensate rounding errors
        npix1 = int(round(self.pcount[dy,dx]))
        cov1 = self.pcov[dy,dx]/npix1-self.pmean[dy,dx]*self.pmean[-dy,-dx]/(npix1*npix1)
        if (dx == 0 or dy == 0): return cov1, npix1
        npix2 = int(round(self.pcount[-dy,dx]))
        cov2 = self.pcov[-dy,dx]/npix2-self.pmean[-dy,dx]*self.pmean[dy,-dx]/(npix2*npix2)
        return 0.5*(cov1+cov2), npix1+npix2

    def report_cov_fft(self) :
        maxrange = self.parameters.maxrange
        l = []
        # (dy,dx) = (0,0) has to be first 
        for dy in range(maxrange+1) :
            for dx in range (0,maxrange+1) :
                cov,npix = self.cov(dx,dy)
                if (dx==0 and dy == 0) : var = cov
                l.append((dx,dy,var,cov,npix))

        return l

def compute_cov_fft(diff, w, parameters) :
    c = cov_fft(diff, w, parameters)
    return c.report_cov_fft()

def cov_direct_value(diff, w, dx,dy):
    """
    Computes in direct space the covariance of image diff at lag (i,j)
    for the pixels in w. Assumes that w contains 0's and 1's.
    returns the covariance and the number of pixel pairs used to evaluate it.
    """
    (ncols,nrows) = diff.shape
    # switching both signs does not change anything:
    # it just swaps im1 and im2 below
    if (dx<0) : (dx,dy) = (-dx,-dy)
    # now, we have dx >0. We have to distinguish two cases
    # depending on dy sign
    if dy>=0 :
        im1 = diff[dy:, dx:]
        w1 = w[dy:, dx:]
        im2 = diff[:ncols-dy, :nrows-dx]
        w2=w[:ncols-dy, :nrows-dx]
    else:
        im1 = diff[:ncols+dy, dx:]
        w1 = w[:ncols+dy, dx:]
        im2 = diff[-dy:, :nrows-dx]
        w2 = w[-dy:, :nrows-dx]
    # use the same mask for all 3 calculations
    w_all = w1*w2
    # do not use mean() because w=0 pixels would then count. 
    npix = w_all.sum()
    im1_times_w = im1*w_all
    s1 = im1_times_w.sum()/npix
    s2 = (im2*w_all).sum()/npix
    p = (im1_times_w*im2).sum()/npix
    cov = p-s1*s2
    return cov,npix
    
    


def compute_cov_direct(diff, w, parameters) :
    """
    Compute covariances the old way. For more than ~25 lags
    (parameters.maxrange), it is slower than the FFT way.
    """
    maxrange = parameters.maxrange
    l=[]
    var = 0
    # (dy,dx) = (0,0) has to be first 
    for dy in range(maxrange+1):
        for dx in range (0,maxrange+1) :
            if (dx*dy>0):
                cov1,npix1 = cov_direct_value(diff, w, dx, dy)
                cov2,npix2 = cov_direct_value(diff, w, dx,-dy)
                cov = 0.5*(cov1+cov2)
                npix = npix1+npix2
            else:
                cov,npix = cov_direct_value(diff, w, dx, dy)
            if (dx==0 and dy == 0) : var = cov
            l.append("%d %d %f %f %d"%(dx,dy,var,cov,npix))
    return l


# assumes mask only contains 0's and 1's
def masked_mean(im, mask) :
    return (im*mask).sum()/mask.sum()


def fft_size(s) :
    # maybe there exists something more clever....
    x = int(np.log(s)/np.log(2.))
    return int(2**(x+1))
    


class ComputeCov :
    def __init__(self, im1,im2, params) :
        self.im1 = im1
        self.im2 = im2
        self.params = params
    
    def compute_cov(self):
        if (im1.check_images(im2) == False) :
            return []
        tuple_rows = []
        rescale_before_subtraction = im1.rescale_before_subtraction()
        v1,t1 = self.im1.other_sensors()
        v2,t2 = self.im2.other_sensors()
        if v1 != '' :
            other_values =(float(v1),float(v2))
            other_tags = [t1+'1',t2+'2']
        else :
            other_values = ()
            other_tags = ()
        extensions = self.im1.segment_ids()
        
        for ext in extensions :
            ext1 = self.im1.get_segment(ext)
            sim1_full = self.im1.prepare_segment(ext1)
            sim1 = self.im1.clip_frame(sim1_full)
            ped1, sig_ped1 = self.im1.ped_stat(ext1)
            ext2 = self.im2.get_segment(ext)
            sim2_full = self.im2.prepare_segment(ext2)
            sim2 = self.im2.clip_frame(sim2_full)
            ped2, sig_ped2 = self.im2.ped_stat(ext2)
            #
            channel = im1.channel_index(ext1)
            assert channel==im2.channel_index(ext2) , 'Different channel indices in a pair'
            nsig2 = 4
            w1 = find_mask(sim1, self.params.nsig1)
            w2 = find_mask(sim2, self.params.nsig1)
            # check is there are masks provided by the sensor itself (e.g. vignetting)
            ws = self.im1.sensor_mask(ext)
            if ws is not None : 
                ws = self.im1.clip_frame(ws)
                w1 *= (1-ws)
                w2 *= (1-ws)

            if w1.sum() ==0 or w2.sum() == 0 : continue

            #      print("masked fractions %f %f"%(float(m1.mask.sum())/sim1.size, float(m2.mask.sum())/sim2.size))
            mu1 = masked_mean(sim1, w1)
            mu2 = masked_mean(sim2, w2)
            fact =  mu1/mu2 if (rescale_before_subtraction)  else 1
            print("file1,file2 = %s %s mu1,mu2 = %f %f fact=%f ext=%d"%(self.im1.filename, self.im2.filename, mu1, mu2, fact, ext))

            if abs(fact-1)>0.1 :
                print("%s and %s have too different averages in ext %d ! .... ignoring them"%(self.im1.filename, self.im2.filename, ext))
                continue
            # Compensate the flux difference, so that if the images are 
            # proportional to each other, any spatial structure vanishes.
            diff = (sim1*mu2-sim2*mu1)/(0.5*(mu1+mu2))
            w12 = w1*w2
            if self.params.subtract_sky :
                diff -= fit_back(diff,250)
            wdiff = find_mask(diff, self.params.nsig2, w12)
            w = w12*wdiff
            sh = diff.shape
            self.params.fft_shape = (fft_size(sh[0]+self.params.maxrange), fft_size(sh[1]+self.params.maxrange))
            covs = compute_cov_fft(diff, w, self.params)
            #covs = compute_corr_direct(diff, w, params)
            # checked that the output tuples are identical
            # covs is a list of ascii lines, each containing i j var cov npix
            tuple_rows += [(mu1, mu2) + cov + (channel, ped1, sig_ped1, ped2, sig_ped2)+ other_values for cov in covs]
        tags = ['mu1','mu2','i','j','var','cov','npix','ext','s1','sp1','s2','sp2']+other_tags
        return tuple_rows, tags

    
    
import pickle
import scipy.interpolate as interp

from bfstuff.filehandlers import *


class parameters :

    class regions :
        def __init__(self):
            self.xover = slice(525,540)
            # E2V "bad" lines : 1983 and 1982
            # in segment coordinates

    class margin :
        pass

    def __init__(self):
        self.regions= self.regions()
        self.margin = self.margin()
        self.margin.bottom = self.margin.top = 20
        self.margin.right = self.margin.left = 20
        self.debug = 0
        self.maxrange = 28 # can be modified on the command line


import argparse

if __name__ == "__main__" :   
    params = parameters()

    # should provide a way to alter that from the command line
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in file_handlers.iteritems()])
    
    usage=" to compute covariances of differences of pairs of flats"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("pairs_file", help= " list of pairs (2 filenames per line)")
    
    parser.add_argument( "-t", "--tuple_name",
                       dest = "tuple_name", 
                       type = str,
                       help = "output tuple name (default : %(default)s)", 
                       default = "tuple.list"  )
    parser.add_argument( "-m", "--maxrange",
                       type=int,
                       dest = "maxrange", 
                       help = "how far we evaluate correlations", 
                       default = 28  )
    parser.add_argument( "-j", "--jmax",
                       dest = "jmax", 
                       type = int,
                       help = "max j value considered in each extension", 
                       default = "1990"  )
    parser.add_argument( "-d", "--deferred-charge", 
                       action="store_true",  # default is False
                       dest = "correct_deferred_charge", 
                       help = "correct for deferred charge (using ./cte.pkl)")
    parser.add_argument( "-n", "--nonlin-correction", 
                       action="store_true",  # default is False
                       dest = "correct_nonlinearity", 
                       help = "correct non linearity (using ./nonlin.pkl)")
    parser.add_argument( "-s", "--subtract_sky", 
                         action="store_true",  # default is False
                         dest = "subtract_sky", 
                         help = "subtract a smooth background from the difference")
    parser.add_argument( "-f", "--file-handler", 
                         dest = "file_handler_tag",
                         help = help_fh_tags)

    options = parser.parse_args()
#    if (len(args) == 0) : 
#        parser.print_help()
#        sys.exit(1)

    params.maxrange= options.maxrange
    params.subtract_sky= options.subtract_sky

    try :
        file_handler = file_handlers[options.file_handler_tag]
    except KeyError:
        print('valid values for -f :\n%s',help_fh_tags)
        sys.exit(0)
    
    params.correct_deferred_charge = options.correct_deferred_charge
    if options.correct_deferred_charge:
        try :
            f = open('cte.pkl','rb')
        except IOError:
            print('cannot find ./cte.pkl for deferred charge correction')
            raise
        print('INFO: loading deferred charge correction') 
        params.dc_corr = pickle.load(f)
        f.close()

    params.correct_nonlinearity = options.correct_nonlinearity
    params.nsig1 = 5 # number of sigams on each input images
    params.nsig2 = 4 # on the difference
    if options.correct_nonlinearity :
        try :
            f = open('nonlin.pkl','rb')
        except IOError:
            print('cannot find ./nonlin.pkl for deferred charge correction')
            raise
        print('INFO: loading nonlinearity correction') 
        params.nonlin_corr = pickle.load(f)
        f.close()

    
    
    #tuple = open(options.tuple_name,"w")
    #tuple.write('@OPTIONS %s\n'%' '.join(sys.argv))
    print('line commmand options : %s\n'%' '.join(sys.argv))
    np.seterr(invalid='raise')
    tuple_records = []
    
    f = open(options.pairs_file)
    for i,l in enumerate(f.readlines()) :
        try :
            f1 = l.split()[0]
            f2 = l.split()[1]
        except :
            print("ignore line in pair file : %s"%l)
            continue

        im1 = file_handler(f1,params)
        im2 = file_handler(f2,params)
        cc = ComputeCov(im1,im2, params)
        t1 = im1.time_stamp()
        t2 = im2.time_stamp()
        tuple_entries, tags = cc.compute_cov()
        alltags = tags + [' t1','t2']

        #if i==0 : # write tuple header (column names)
            #for tag in tags.split() : tuple.write('#%s:\n'%tag)
            #tuple.write("#end\n")
        #for line in tuple_entries :
            #tuple.write("%s %s %s\n"%(line, t1, t2))
        tuple_records += tuple_entries
    print('alltags', alltags)
    print(' Converting a list of tuples into a numpy recarray')
    nt = np.rec.fromrecords(tuple_records, names=alltags)
    print('writing tuple %s to disk'%options.tuple_name)
    np.save(options.tuple_name, nt)


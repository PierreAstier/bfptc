#!/usr/bin/env python



try :
    import astropy.io.fits as pf
except ImportError :
    import pyfits as pf

    
import numpy as np
import sys


def write_fits(filename, d) :
    l = pf.HDUList()
    l.append(pf.ImageHDU(data=d))
    l.writeto(filename, clobber=True)
    l.close()



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
            l.append((dx,dy,var,cov,npix))
    return l

def cross_cov_direct_value(diff1, w1, diff2, w2, dx,dy):
    """
    Computes in direct space the covariance of image diff at lag (i,j)
    for the pixels in w. Assumes that w contains 0's and 1's.
    returns the covariance and the number of pixel pairs used to evaluate it.
    """
    (ncols,nrows) = diff1.shape
    # switching both signs does not change anything:
    # it just swaps im1 and im2 below
    # not sure... for the cross correlation, do it anyway.
    if (dx<0) : (dx,dy) = (-dx,-dy)
    # now, we have dx >0. We have to distinguish two cases
    # depending on dy sign
    if dy>=0 :
        im1 = diff1[dy:, dx:]
        w1 = w1[dy:, dx:]
        im2 = diff2[:ncols-dy, :nrows-dx]
        w2=w2[:ncols-dy, :nrows-dx]
    else:
        im1 = diff1[:ncols+dy, dx:]
        w1 = w1[:ncols+dy, dx:]
        im2 = diff2[-dy:, :nrows-dx]
        w2 = w2[-dy:, :nrows-dx]
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

    
def compute_cross_cov_direct(diff1, w1, diff2, w2, parameters) :
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
                cov1,npix1 = cross_cov_direct_value(diff1, w1, diff2, w2, dx, dy)
                cov2,npix2 = cross_cov_direct_value(diff1, w1, diff2, w2, dx,-dy)
                cov = 0.5*(cov1+cov2)
                npix = npix1+npix2
            else:
                cov,npix = cross_cov_direct_value(diff1, w1, diff2, w2,dx, dy)
            if (dx==0 and dy == 0) : var = cov
            l.append((dx,dy,var,cov,npix))
    return l



# assumes mask only contains 0's and 1's
def masked_mean(im, mask) :
    return (im*mask).sum()/mask.sum()


def fft_size(s) :
    # maybe there exists something more clever....
    x = int(np.log(s)/np.log(2.))
    return int(2**(x+1))
    

from bfptc.cov_utils import find_mask, fit_back


class ComputeCov :
    def __init__(self, im1,im2, params) :
        self.im1 = im1
        self.im2 = im2
        self.params = params
        self.rescale_before_subtraction = im1.rescale_before_subtraction()
        self.v1, self.t1 = self.im1.other_sensors()
        self.v2, self.t2 = self.im2.other_sensors()
        if self.v1 != '' :
            self.other_values = tuple(np.atleast_1d(self.v1))+tuple(np.atleast_1d(self.v2))
            self.other_tags = [t+'1' for t in np.atleast_1d(self.t1)]+[t+'2' for t in np.atleast_1d(self.t2)]
        else :
            self.other_values = ()
            self.other_tags = []
        self.extensions = self.im1.segment_ids()

    def diff_image(self, ext) :
        """
        evaluates the difference image for extension ext
        (listed in self.extensions)
        returns mu1,mu2, w, diff
        (mu1 and mu2 are averages, w is the weight (0. and 1.), diff is the images. 
        mu1 = None means that somthing went wrond
        """
        sim1_full = self.im1.prepare_segment(ext)
        sim1 = self.im1.clip_frame(sim1_full)
        sim2_full = self.im2.prepare_segment(ext)
        sim2 = self.im2.clip_frame(sim2_full)
        #
        channel = im1.channel_index(ext)
        assert channel==im2.channel_index(ext) , 'Different channel indices in a pair'
        print(' means (full then clipped): ',sim1_full.mean(), sim2_full.mean(), sim1.mean(), sim2.mean())
        # check fs there are masks provided by the sensor itself (e.g. vignetting)
        ws = self.im1.sensor_mask(ext)
        if ws is not None : 
            ws = self.im1.clip_frame(ws)
            w1 = (1-ws)
            w2 = (1-ws)
            if w1.sum() ==0 : # no need to go further
                return None,0,w1, sim1-sim2
        else :
            w1 = None
            w2 = None

        if self.params.subtract_sky_before_clipping :
            w1 = find_mask(sim1 - fit_back(sim1,50), self.params.nsig_image, w1)
            w2 = find_mask(sim2 - fit_back(sim2,50), self.params.nsig_image, w2)
        else:
            w1 = find_mask(sim1, self.params.nsig_image, w1)
            w2 = find_mask(sim2, self.params.nsig_image, w2)

        if w1.sum() ==0 or w2.sum() == 0 :
            return None,0,0,0
        mu1 = masked_mean(sim1, w1)
        mu2 = masked_mean(sim2, w2)
        fact =  mu1/mu2 if (mu2 != 0 and self.rescale_before_subtraction)  else 1
        print(("file1,file2 = %s %s mu1,mu2 = %f %f fact=%f ext=%d"%(self.im1.filename, self.im2.filename, mu1, mu2, fact, channel)))

        if abs(fact-1)>0.1 :
            print(("%s and %s have too different averages in ext %d ! .... ignoring them"%(self.im1.filename, self.im2.filename, channel)))
            return None,0,0,0

        # Compensate the flux difference, so that if the images are 
        # proportional to each other, any spatial structure vanishes.
        diff = (sim1*mu2-sim2*mu1)/(0.5*(mu1+mu2))
        w12 = w1*w2
        w = find_mask(diff, self.params.nsig_diff, w12)
        return mu1,mu2,w,diff
        
        
    def compute_cov(self):
        if (im1.check_images(im2) == False) :
            return [], []
        tuple_rows = []
        for ext in self.extensions :
            mu1,mu2,w,diff = self.diff_image(ext)
            if mu1 is None : # something went wrong
                continue
            ped1, sig_ped1 = self.im1.ped_stat(ext)
            ped2, sig_ped2 = self.im2.ped_stat(ext)
            channel = im1.channel_index(ext)
            sh = diff.shape
            if self.params.maxrange > 3 : # Not sure about the exact value that optimizes the whole thing
                self.params.fft_shape = (fft_size(sh[0]+self.params.maxrange), fft_size(sh[1]+self.params.maxrange))
                covs = compute_cov_fft(diff, w, self.params)
            else :
                covs = compute_cov_direct(diff, w, params)
            # checked that the output tuples are identical
            # covs is a list of ascii lines, each containing i j var cov npix
            tuple_rows += [(mu1, mu2) + cov + (channel, ped1, sig_ped1, ped2, sig_ped2)+ self.other_values for cov in covs]
        tags = ['mu1','mu2','i','j','var','cov','npix','ext','s1','sp1','s2','sp2']+ self.other_tags
        return tuple_rows, tags


    def compute_cross_cov(self):
        if (im1.check_images(im2) == False) :
            return [], []
        tuple_rows = []
        # store all data in order to read/prepare only once.
        mus = {}
        diffs = {}
        ws = {}
        for ext in self.extensions :
            mu1,mu2, w, diff = self.diff_image(ext)
            if mu1 is None : # something went wrong
                continue
            mus[ext] = 0.5*(mu1+mu2)
            diffs[ext] = diff+ 0. # store a copy
            ws[ext] = w + 0.
        # double loop on extensions : 
        for ext1,mu1 in mus.items():
            diff1 = diffs[ext1]
            w1 = ws[ext1]
            #ped1, sig_ped1 = self.im1.ped_stat(ext)
            #ped2, sig_ped2 = self.im2.ped_stat(ext)
            channel1 = im1.channel_index(ext1)
            # Not sure about the exact value that optimizes the whole thing
            for ext2,mu2 in mus.items(): 
                if ext2<=ext1 : continue
                diff2 = diffs[ext2]
                w2 = ws[ext2]
                channel2 = im1.channel_index(ext2)
                #if self.params.maxrange > 3 : 
                #    sh = diff.shape
                #    self.params.fft_shape = (fft_size(sh[0]+self.params.maxrange),
                #                         fft_size(sh[1]+self.params.maxrange))
                #    covs = compute_cov_fft(diff, w, self.params)
                #else :
                covs = compute_cross_cov_direct(diff1, w1, diff2, w2, params)
                tuple_rows += [(mu1, mu2) + cov + (channel1, channel2) + self.other_values for cov in covs]
        tags = ['mu1','mu2','i','j','var','cov','npix','ext1','ext2']+ self.other_tags
        return tuple_rows, tags

    
    
    
import pickle
import scipy.interpolate as interp

from bfptc.filehandlers import *
import bfptc.envparams as envparams

import argparse

if __name__ == "__main__" :   
    params = envparams.EnvParams()
    # the default there can be overriden in a file provided
    # through the BFPARAMS environment variable

    # should provide a way to alter that from the command line
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in list(file_handlers.items())])
    
    usage=" to compute covariances of differences of pairs of flats"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("pairs_file", help= " list of pairs (2 filenames per line)")
    parser.add_argument( "-f", "--file-handler", 
                         dest = "file_handler_tag",
                         required = True,                         
                         help = help_fh_tags)
    
    parser.add_argument( "-t", "--tuple_name",
                       dest = "tuple_name", 
                       type = str,
                       help = "output tuple name (default : %(default)s)", 
                       default = "tuple"  )
    parser.add_argument( "-m", "--maxrange",
                       type=int,
                       dest = "maxrange", 
                       help = "how far we evaluate correlations", 
                       default = None  )
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

    parser.add_argument( "-c", "--cross-covariances", 
                       action="store_true",  # default is False
                       dest = "cross_cov", 
                       help = "compute cross-amp covariances ")

    options = parser.parse_args()
    #    if (len(args) == 0) : 
    #        parser.print_help()
    #        sys.exit(1)

    if options.maxrange is not None :
        params.maxrange= options.maxrange

    if options.cross_cov and options.tuple_name == "tuple" :
        print("renaming tuple to cross-tuple")
        options.tuple_name = "cross-tuple"
        
        
    try :
        file_handler = file_handlers[options.file_handler_tag]
    except KeyError:
        print(('valid values for -f :\n%s',help_fh_tags))
        sys.exit(0)

    params.nsig_image = 5 # number of sigams on each input images
    params.nsig_diff = 4 # on the difference
    params.correct_deferred_charge = options.correct_deferred_charge
    params.correct_nonlinearity = options.correct_nonlinearity

    print('command line : %s\n'%' '.join(sys.argv))
    print('control parameters\n:',params)
            
    np.seterr(invalid='raise')
    tuple_records = []
    
    f = open(options.pairs_file)
    alltags=[]
    for i,l in enumerate(f.readlines()) :
        try :
            f1 = l.split()[0]
            f2 = l.split()[1]
        except :
            print(("ignore line in pair file : %s"%l))
            continue

        im1 = file_handler(f1,params)
        im2 = file_handler(f2,params)
        cc = ComputeCov(im1,im2, params)
        t1 = im1.time_stamp()
        t2 = im2.time_stamp()
        if options.cross_cov :
            entries, tags = cc.compute_cross_cov()
        else :
            entries, tags = cc.compute_cov()
        if len(entries) == 0: continue
        # add t1, t2 at the end (tuples are immutable, hence new list) 
        tuple_entries = []
        for row in entries :
            tuple_entries.append(row+(t1,t2))
        alltags = tags + ['t1','t2']

        #if i==0 : # write tuple header (column names)
            #for tag in tags.split() : tuple.write('#%s:\n'%tag)
            #tuple.write("#end\n")
        #for line in tuple_entries :
            #tuple.write("%s %s %s\n"%(line, t1, t2))
        tuple_records += tuple_entries
    if len(tuple_records) == 0 :
        print(' no data was actually processed, writing an empty tuple file')
        open(options.tuple_name,'w')
        sys.exit(1)
    print(('alltags', alltags))
    print(' Converting a list of tuples into a numpy recarray')
    if False : 
        # shrink the formats in order to reduce disk space
        format = []
        for k,x in enumerate(tuple_records[0]) :
            if x.__class__ in [int, np.int64] : 
                format.append('<i4')
            else :
                if x.__class__ in [float,np.float64] : 
                    format.append('<f4')
                else :
                    print(('cannot find a format for tag %s'%alltags[k],' class ', x.__class__))
                    sys.exit(1)
        nt = np.rec.fromrecords(tuple_records, dtype={'names': alltags, 'formats':format})
    nt = np.rec.fromrecords(tuple_records, names = alltags)
    print(('writing tuple %s to disk'%options.tuple_name))
    np.save(options.tuple_name, nt)


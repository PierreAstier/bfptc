#!/usr/bin/env python



try :
    import astropy.io.fits as pf
except ImportError :
    import pyfits as pf

    
import numpy as np
import sys
import bfptc.envparams as envparams

from bfptc.filehandlers import *


import argparse


def my_robust_average(a, axis=None, clip=5, mini_output=True):
    """
    axis should be set.
    Returns:
     robust_average if mini_output == True
     else robust_average, sigma 
    """
    w = np.ones_like(a, dtype = np.bool)
    mushape = [d for d in a.shape]
    mushape[axis] = 1
    while True:
        ngood = w.sum(axis=axis)
        print(ngood.shape, ngood.sum())
        ngood_zero = ngood==0 # should never happen
        if ngood_zero.sum() != 0: # debugging
            index = np.where(ngood==0)
            i = index[0][0]
            j = index[1][0]
            print('ERROR: all data discarded :  mu sigma i j ', mu[i,j], sig[i,j], i ,j)
            print(a[:, i,j])
            print(w[:,i,j])
        aw = a*w
        mu = (aw).sum(axis=axis)/ngood
        print('count, average ',ngood.sum(), mu.mean())
        sig = np.sqrt(((aw-mu)**2).sum(axis=axis)/ngood)
        mu.reshape(mushape)
        w = w & (np.abs(a-mu)<=sig*clip)
        if w.sum() == ngood.sum() :
            break
    if mini_output :
        return mu
    return mu,sig
        


#from saunerie.robuststat import robust_average    
#from astropy.stats import sigma_clip

if __name__ == "__main__" :   
    params = envparams.EnvParams()

    # should provide a way to alter that from the command line
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in file_handlers.items()])
    
    usage=" to compute covariances of differences of pairs of flats"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("bias_files", help= " input files ", nargs='+')
    
    parser.add_argument( "-f", "--file-handler", 
                         dest = "file_handler_tag",
                         required = True,
                         help = help_fh_tags)

    parser.add_argument( "-b", "--bias-name", 
                         dest = "bias_file_name",
                         default = "masterbias.fits",
                         help = "name of the output master bias file")

    parser.add_argument( "-s", "--sig-name", 
                         dest = "sig_bias_file_name",
                         default = "masterbias_sig.fits",
                         help = "name of the output master bias *sigma* file")



    options = parser.parse_args()
#    if (len(args) == 0) : 
#        parser.print_help()
#        sys.exit(1)


    try :
        file_handler = file_handlers[options.file_handler_tag]
    except KeyError:
        print('valid values for -f :\n%s',help_fh_tags)
        sys.exit(0)

    #print(options.bias_files)

    params.subtract_bias = False # we obviously do not have it yet !
    print('control parameters\n:',params)
        
    data = {}
    for file in options.bias_files :
        print('reading file %s'%file)
        im = file_handler(file, params)
        ids = im.segment_ids()
        for id in ids :
            ampdata = im.amp_data(id) # just for the size
            ampdata = np.zeros_like(ampdata)
            datasec = im.datasec_bounding_box(id)
            ampdata[datasec] = im.subtract_overscan_and_trim(id, None)
            if id in list(data.keys()) :
                data[id].append(ampdata)
            else :
                data[id] = [ampdata]
    # copy the image as a fits template
    # actually copy in order to remove compression, if any
    output_fits = pf.HDUList()
    output_fits.append(im.im[0]) # copy the main header of the last process image
    output_fits_sig = pf.HDUList()
    output_fits_sig.append(im.im[0])   
    for id,arrays in data.items() :
        pixels = np.array(arrays, dtype=np.float32)
        # free memory, hopefully
        data[id] = []
        print('averaging extension %d'%id)
        # robust average
        # (the home-made local version is much faster than the saunerie one) 
        result, sig = my_robust_average(pixels, axis=0, mini_output=False)
        # sigma_clip is twice as slow as robust_average
        # result = sigma_clip(pixels, axis=0, sigma=5, cenfunc = np.mean)
        # the following is extremely slow
        # result = np.apply_along_axis(clipped_average, 0, pixels)
        output_fits.append(pf.ImageHDU(data = result.astype(np.float32),
                                       header = im.im[id].header))
        output_fits_sig.append(pf.ImageHDU(data = sig.astype(np.float32),
                                           header = im.im[id].header))
    filename = options.bias_file_name
    print('writing master bias to %s'%filename)
    output_fits.writeto(filename, overwrite=True)
    output_fits_sig.writeto(options.sig_bias_file_name, overwrite=True)
        



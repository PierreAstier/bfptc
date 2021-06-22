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


from bfptc.cov_utils import find_mask, fit_back


def simple_mask(im, nsig):
    w = np.ones_like(im)
    count = w.sum()
    mu = np.median(im)
    sigma = np.sqrt(np.median((im-mu)**2))
    for k in range(3) :
        outliers = np.where(np.abs((im-mu)*w)>nsig*sigma)
        w[outliers] = 0
        print('len(outliers)',len(outliers), (1-w).sum())
        if (outliers[0].sum()==0) : break
        sigma = np.sqrt(np.median((im[w!=0]-mu)**2))
        mu = np.median((im[w!=0]))
    return w


if __name__ == "__main__" :   
    params = envparams.EnvParams()

    # should provide a way to alter that from the command line
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in file_handlers.items()])
    
    usage=" assembles a mask from a series of flats"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("flat_files", help= " input files ", nargs='+')
    
    parser.add_argument( "-f", "--file-handler", 
                         dest = "file_handler_tag",
                         required = True,
                         help = help_fh_tags)

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

    params.subtract_bias = True 
    print('control parameters\n:',params)
        
    data = {}
    for file in options.flat_files :
        print('reading file %s'%file)
        im = file_handler(file, params)
        ids = im.segment_ids()
        for id in ids :
            pixels = im.prepare_segment(id)
            # w = find_mask(pixels - fit_back(pixels, 50), params.nsig_image)
            w = simple_mask(pixels , params.nsig_image)
            # here w=1 means OK, and 0 means bad
            # in dead.fits, it is reversed
            print("channel, dead count",im.channel_index(id), (1-w).sum())
            w = (1 - w).astype(np.int32)
            if id in list(data.keys()) :
                data[id] += w
            else :
                data[id] = w
    # copy the image as a fits template
    # actually copy in order to remove compression, if any
    output_fits = pf.HDUList()
    output_fits.append(im.im[0]) # copy the main header of the last process image
    for id, pixels in data.items() :
        # we mask pixels flagged in 1/3 of the images
        threshold = len(options.flat_files)/3
        mask = (pixels > threshold)        
        output_fits.append(pf.ImageHDU(data = mask.astype(np.uint8),
                                       header = im.im[id].header))
        del output_fits[id].header['DATASEC']

    filename = 'dead.fits'
    print('writing master dead to %s'%filename)
    output_fits.writeto(filename, overwrite=True)
        



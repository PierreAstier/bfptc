#!/usr/bin/env python
try:
    import pyfits as pf
except ModuleNotFoundError :
    import astropy.io.fits as pf
import numpy as np
import re

def fortran_to_slice(a,b) :
    if b >= a:
        sx = slice(a-1,b)
    else :
        xmin = b-2
        sx = slice(a-1,xmin if xmin>=0 else None, -1)
    return sx

def convert_region(str):
    """ 
    convert a FITS REGION (string) into a pair of numpy slices
    returns:
        slice_y, slice_x
    """ 
    x = re.match('\[(\d*):(\d*),(\d*):(\d*)\]',str)
    b = [int(x.groups()[i]) for i in range(4)]
    # swap x and y:
    return fortran_to_slice(b[2],b[3]), fortran_to_slice(b[0],b[1])

def overscan_subtract_and_trim(im, xover, xim, yim) :
    """
    subtracts the median (in xover x yim)  of the pedestal
    and trim the image
    """
    pedestal = np.median(im[yim, xover])
    return im[yim, xim]-pedestal

def assemble_image(fh, biasfh=None) :
    """
    arguments: pyfits file handle, output file name.
    if the biasfh is provided, then it is used for bias subtraction.
    assembles the image using DETSIZE, DATASEC, DETSEC and BIASSEC, afetr BIAS usbtraction
    """
    mh = fh[0].header
    sy,sx = detsize  = convert_region(mh['DETSIZE'])
    im = np.ndarray((sy.stop, sx.stop))
    extensions = [i for i in range(len(fh)) if fh[i].header.get('EXTNAME',default='NONE').startswith('CHAN')]
    extensions +=  [i for i in range(len(fh)) if fh[i].header.get('EXTNAME',default='NONE').startswith('Segment')]
    for ext in extensions :
        f = fh[ext]
        #DATASEC = '[11:522,1:2002]'
        #DETSEC  = '[1:512,4004:2003]'
        #BIASSEC = '[523:544,1:2002]'   / Serial overscan region
        dats_y,dats_x  = convert_region(f.header['DATASEC'])
        dets_y, dets_x = convert_region(f.header['DETSEC'])
        try :
            bias_y, bias_x = convert_region(f.header['BIASSEC'])
        except KeyError :
            biasy = dats_y
            bias_x = slice(dats_x.stop, f.data.shape[0])
        rawd = f.data
        if biasfh is None :
            im_ext = overscan_subtract_and_trim(rawd, bias_x, dats_x, dats_y)
        else :
            bias_ext = biasfh[ext]
            bias_data = bias_ext.data
            if bias_data.shape == rawd.shape :
                im_ext = overscan_subtract_and_trim(rawd-bias_data, bias_x, dats_x, dats_y)
            else:
                rawd[datsy,datsx] -= bias_data
                im_ext = overscan_subtract_and_trim(rawd-bias_data, bias_x, dats_x, dats_y)
        im[dets_y, dets_x] = im_ext

    return im


   
if __name__ == "__main__" :
    import sys
    if len(sys.argv) < 3:
        print("usage : %s <mef_image_name> <assembled_image_name>"%sys.argv[0])
        sys.exit(1)
    fh = pf.open(sys.argv[1])
    data = assemble_image(fh)
    #pf.writeto(sys.argv[2], data.astype(np.float32) , header = fh[0])
    pf.writeto(sys.argv[2], data.astype(np.int32), header=fh[0].header, overwrite=True )
    
        
        
        

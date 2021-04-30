#!/pbs/home/a/astier/software/anaconda3/bin/python
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


def assemble_image(fh) :
    """
    arguments: pyfits file handle, output file name.

    """
    mh = fh.im[0].header
    sy,sx = detsize  = convert_region(mh['DETSIZE'])
    im = np.ndarray((sy.stop, sx.stop))
    ids = fh.segment_ids()
    for id in ids :
        detsec = convert_region(fh.im[id].header['DETSEC'])
        imext = fh.prepare_segment(id)
        im[detsec] = imext
    return im


import argparse
from bfptc.filehandlers import *

   
if __name__ == "__main__" :
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in list(file_handlers.items())])

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help= " input file")
    parser.add_argument( "-f", "--file-handler",
                         dest = "file_handler_tag",
                         help = help_fh_tags,
                         default = 'S')

    parser.add_argument( "-o", "--output-file",
                         dest = "output_file",
                         required = True,
                         type = str,
                         help = "output file name ")

    parser.add_argument( "-b", "--bias",
                         dest = "subtract_bias",
                         action= "store_true",
                         help = "subtract bias (masterbias.fits)")


    parser.add_argument( "-n", "--nonlin-correction", 
                         action="store_true",  # default is False
                         dest = "correct_nonlinearity", 
                         help = "correct non linearity (using ./nonlin.pkl)")
    options = parser.parse_args()
    #    if (len(args) == 0) :                                                  
    #        parser.print_help()                                                
    #        sys.exit(1)                                                        


    try :
        file_handler = file_handlers[options.file_handler_tag]
    except KeyError:
        print(('valid values for -f :\n%s',help_fh_tags))
        sys.exit(0)

    
    options.nonlin_corr_file = './nonlin.pkl'
    options.overscan_skip = 10

    # could cook up a params record with the proper fields to trigger the bias subtraction
    fh = file_handler(options.input_file, options)
    data = assemble_image(fh)
    #pf.writeto(sys.argv[2], data.astype(np.float32) , header = fh[0])
    pf.writeto(options.output_file, data.astype(np.float32), header=fh.im[0].header, overwrite=True )
    
        
        
        

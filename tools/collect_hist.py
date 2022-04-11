#!/usr/bin/env python
import astropy.io.fits as pf
import numpy as np
import re


import argparse
import sys
from bfptc.filehandlers import *
import pickle
from bfptc.histogram import Histogram


"""
meant to collect the values of raw ADC codes on a set of images.
"""

   
if __name__ == "__main__" :
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in list(file_handlers.items())])
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list", help= " input file list")
    parser.add_argument( "-f", "--file-handler",
                         dest = "file_handler_tag",
                         help = help_fh_tags,
                         default = 'S')

    parser.add_argument( "-o", "--output_name",
                         dest = "output_name",
                         type = str,
                         default='hist_vals.npy',
                         help = "output file name")




    options = parser.parse_args()
    #    if (len(args) == 0) :                                                  
    #        parser.print_help()                                                
    #        sys.exit(1)                                                        


    try :
        file_handler = file_handlers[options.file_handler_tag]
    except KeyError:
        print(('valid values for -f :\n%s',help_fh_tags))
        sys.exit(0)

    
    file_list = open(options.input_list, 'r')
    histos={}
    for line in file_list.readlines() :
        for file in line.strip().split(' ') :
            print('processing %s'%file)
            fh = file_handler(file, None)
            for id in fh.segment_ids():
                # raw data
                bb = fh.datasec_bounding_box(id)
                # select science data (no overscans)
                pixels = fh.amp_data(id)[bb]
                c_id = fh.channel_index(id)
                if histos.get(c_id) is None : histos[c_id] = Histogram(-0.5,180000.5-1., 180000)
                histos[c_id].fill(pixels)
    # shrink to the non-null spectrum region, just to save disk space
    for h in histos.values():
        h.shrink()
    pickle.dump(histos,open(options.output_name,'wb'))
    
        
        
        

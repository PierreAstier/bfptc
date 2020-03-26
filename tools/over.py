#!/usr/bin/env python 



import numpy as np
import sys


""" Application that creates a tuple to study deffered charge from
serial overscan regions.  The tuple typically contains 5 columns: last
real pixle valu, next pixel, next to next , channel, and som time acquisition time.  It finds the
proper image columns using FileHandler functions """
    
from bfptc.filehandlers import *

def process_file(fh) :
    """
    the argument should be a filehandler
    return a dictionnary (indexed by segment id) of 3 arrays: last pixel, overscan 0  and overscan 1
    """
    if (False) :
        print('#name:\n#sim:\n#s1:\n#s2:\n#pim:\n#p1:\n#p2:\n#ped:\n#slow:\n#shigh:\n#og:\n#time:\n#end')
    extensions = fh.segment_ids()
    res = {}
    for ext in extensions :
        data,over = fh.subtract_overscan_and_trim(ext, None, return_overscan = True)
        assert data.shape[0] == over.shape[0]
        res[ext] = (data[: , -1], over[:,0], over[:,1],data.mean())
    return res

import bfptc.envparams as envparams

import argparse

if __name__ == "__main__" :   
    params = envparams.EnvParams()

    # should provide a way to alter that from the command line
    help_fh_tags = '\n'.join(['         %s : %s'%(key,value) for key,value in file_handlers.items()])
    
    usage=" to collect last physical pixel of each line and overscan pixels, for CTI studies"
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("listfile", help= " list of filenames (several files per line are OK)")
    
    parser.add_argument( "-t", "--tuple_name",
                       dest = "tuple_name", 
                       type = str,
                       help = "output tuple name (default : %(default)s)", 
                       default = "tuple.list"  )
    parser.add_argument( "-f", "--file-handler", 
                         dest = "file_handler_tag",
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
    

    #tuple = open(options.tuple_name,"w")
    #tuple.write('@OPTIONS %s\n'%' '.join(sys.argv))
    #tuple.write('#amp:\n#im:\n#o1:\n#o2:\n#mu : \n#t:\n#end\n')
    params.overscan_skip = 5
    params.subtract_bias = False
    print('control parameters:\n',params)
    
    f = open(options.listfile)
    rows = []
    for i,l in enumerate(f.readlines()) :
        try :
            names = l.split()
        except :
            print("ignore line : %s"%l)
            continue

        for name in names :
            im = file_handler(name,params)
            print("processing %s"%name)
            time = im.time_stamp()
            data = process_file(im)
            for amp,pixels in data.items() :
                im,o1,o2,mu  = pixels
                for k in range(len(im)):
                    rows.append((amp,im[k], o1[k], o2[k], k, mu, time))
    tags = ['amp','im','o1','o2','j','mu','t']
    format = []
    for k,x in enumerate(rows[0]) :
        if x.__class__ in [int, np.int64] : 
            format.append('<i4')
        else :
            if x.__class__ in [float,np.float64] : format.append('<f4')
            else :
                print(('cannot find a format for tag %s'%alltags[k],' class ', x.__class__))
                sys.exit(1)
    nt = np.rec.fromrecords(rows, dtype={'names': tags, 'formats':format})
    print(('writing tuple %s to disk'%options.tuple_name))
    np.save(options.tuple_name, nt)
    #

#!/usr/bin/env python

import numpy as np
import time

if __name__ == "__main__" :
    im = np.random.uniform(size=(2000,2000))
    print('np.fft : %s'%np.fft.__file__)
    print('computing 10 times fft over ',im.shape, 'pixels')
    start = time.clock()
    for k in range(10) :
        fft = np.fft.rfft2(im)
    print('time (for 1): %f'%((time.clock()-start)/10.))

    

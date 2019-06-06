#!/usr/bin/env python

import numpy as np
import sys

if __name__ == "__main__":
    tuples = []
    for f in sys.argv[1:] :
        tuples.append(np.load(f))
    np.save(sys.stdout, np.hstack(tuples))

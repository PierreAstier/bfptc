import numpy as np

def find_groups(x, maxdiff):
    """
    group data into bins, with at most maxdiff distance between bins.
    returns bin indices
    """
    ix = np.argsort(x)
    xsort = np.sort(x)
    index = np.zeros_like(x, dtype=np.int32)
    xc = xsort[0] 
    group = 0
    ng = 1
    for i in range(1,len(ix)) :
        xval = xsort[i]
        if (xval-xc < maxdiff) :
            xc = (ng*xc+xval)/(ng+1)
            ng += 1
            index[ix[i]] = group
        else :
            group+=1
            ng=1
            index[ix[i]] = group
            xc = xval
    return index

def index_for_bins(x, nbins) :
    """
    just builds an index with regular binning
    The result can be fed into bin_data
    """
    bins = np.linspace(x.min(), x.max() + abs(x.max() * 1e-7), nbins + 1)
    return np.digitize(x, bins)


def bin_data(x,y, bin_index, wy=None):
    """
    Bin data (usually for display purposes).
    x and y is the data to bin, bin_index should contain the bin number of each datum, and wy is the inverse of rms of each datum to use when averaging.
    (actual weight is wy**2)

    Returns 4 arrays : xbin (average x) , ybin (average y), wybin (computed from wy's in this bin), sybin (uncertainty on the bin average, considering actual scatter, ignoring weights) 
    """
    if wy is  None : wy = np.ones_like(x)
    bin_index_set = set(bin_index)
    w2 = wy*wy
    xw2 = x*(w2)
    xbin= np.array([xw2[bin_index == i].sum()/w2[bin_index == i].sum() for i in bin_index_set])
    yw2 = y*w2
    ybin= np.array([yw2[bin_index == i].sum()/w2[bin_index == i].sum() for i in bin_index_set])
    wybin = np.sqrt(np.array([w2[bin_index == i].sum() for i in bin_index_set]))
    # not sure about this one...
    #sybin= np.array([yw2[bin_index == i].std()/w2[bin_index == i].sum() for i in bin_index_set])
    sybin= np.array([y[bin_index == i].std()/np.sqrt(np.array([bin_index==i]).sum()) for i in bin_index_set])
    return xbin, ybin, wybin, sybin



import numpy as np

class Histogram:
    def __init__(self, xmin, xmax ,nchan):
        self.min = xmin
        self.max = xmax
        self.nchan = nchan
        self.step = float(xmax-xmin)/float(nchan)
        self.h = np.zeros(nchan)
        self.under = 0
        self.over = 0

    def fill(self,vals):
        chans = np.array(np.floor((vals-self.min)/self.step), dtype=int)
        neg = chans<0
        self.under += neg.sum()
        chans = chans[~neg]
        over = chans>= self.nchan
        self.over += over.sum()
        chans = chans[~over]
        self.h += np.bincount(chans, minlength=self.nchan)

    def bin_centers(self):
        return np.linspace(self.min, self.max-self.step, self.nchan)+self.step*0.5

    def min_max_bins(self):
        filled = np.where(self.h>0)[0]
        return filled[0], filled[-1]

    def shrink(self):
        min, max = self.min_max_bins()
        bin_centers = self.bin_centers()
        self.h = self.h[min:max+1]
        self.nchan = len(self.h)
        self.min = bin_centers[min]-0.5*self.step
        self.max = bin_centers[max]+0.5*self.step

    def rebin(self, factor):
        """
        only integer factors
        """
        assert factor == np.round(factor), "only rebin by integer factors"
        new_nchan = self.nchan//factor
        used_part = new_nchan*factor
        # rounding binning error goes to overflows
        if used_part < self.nchan:
            self.over += self.h[used_part:].sum()
        # sum bins
        self.h = self.h[:used_part].reshape((new_nchan,factor)).sum(axis=1)
        # updates
        self.nchan = new_nchan
        self.step *= factor
        self.max = self.min +self.nchan*self.step


    def drop_starting_bins(self,nbins):
        self.under += self.h[:nbins].sum()
        self.h = self.h[nbins:]
        self.nchan -= nbins
        self.min += nbins*self.step

    def drop_last_bins(self,nbins):
        self.over += self.h[:-nbins].sum()
        self.h = self.h[:-nbins]
        self.nchan -= nbins
        self.max -= nbins*self.step        
        
        

    def sub_range(self, minval, maxval):
        """
        extracts the values between minval and maxval
        return bin abcissa and bon contents
        """
        minbin = int(np.floor((minval-self.min)/self.step))
        maxbin = int(np.ceil((maxval-self.min)/self.step))
        return self.bin_centers()[minbin:maxbin], self.h[minbin:maxbin]

    def sub_range_hist(self, minval, maxval):
        minbin = int(np.floor((minval-self.min)/self.step))
        maxbin = int(np.ceil((maxval-self.min)/self.step))
        bc = self.bin_centers()
        new_xmin = bc[minbin]-0.5*self.step
        new_xmax = bc[maxbin]-0.5*self.step
        res = Histogram(new_xmin,new_xmax, maxbin-minbin)
        res.h = self.h[minbin:maxbin]+0. # want an actual copy
        # should perhaps propagate over and underflows....
        return res

    def xmin(self):
        return self.min
    
    def bin_boundaries(self):
        return np.linspace(self.min, self.max, self.nchan+1, endpoint=True)
        
    def binary_rebin(self):
        """
        when the histogram represents the frequency of ADC codes
        rebin so that it now represetns the frequency with the last bit dropped
        """
        new_nchan = self.nchan//2
        used_part = new_nchan*2
        # rounding binning error goes to overflows
        if used_part < self.nchan:
            self.over += self.h[used_part:].sum()
        # sum bins
        self.h = self.h[:used_part].reshape((new_nchan,2)).sum(axis=1)
        # updates
        bc = self.bin_centers()
        self.nchan = new_nchan
        self.step *= 2
        self.min = bc[0]-0.5*self.step
        self.max = self.min +self.nchan*self.step        
        

    

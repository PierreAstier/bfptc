import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip
import math

"""
CLAP means Cooled Large Amplified Photodiode.
This is a device inserted in the Paris CCD test bench, that delivers
the light flux as a function of time, and this data is stored into our
"image" FITS files. 
The code here aims at integrating the CLAP signal sequance, in order
to deliver the integrated light quantity, in arbitrary but reproducible units. 
"""


def process_one_file(filehandle) :
    exptime = filehandle[0].header['EXPTIME']
    try :
        e = filehandle['CLAP']
        period  = e.header['PERIOD']*20*1e-9 # 20 ns is written in the comment
        # convert to float to make sure that sums behave properly. 
        q, l_on, lb, la, t0, t1 = analyze_clap(np.array(e.data, dtype=float))
        dt = (t1-t0)
        return exptime, q*period, lb, la, dt
    except KeyError:
        return exptime, 0,0,0,0



def clipped_average(d, cut=4.) :
    c, low, upp = sigmaclip(d,cut,cut)
    return c.mean()

# data is just the array of current samples
def analyze_clap(data, debug=False):
    # analyze a signal which looks like an electronic gate:
    # meaning we have to find the transitions and measure the  levels
    # there are no sanity checks. Chi2 of the fits should help
    #
    # rebin, to fight noise/cosmics if any, and make the transitions sharper
    """integrates a wavform that looks like an electronic gate.

    returns q,level during the gate, level before, level after,
    leading edge tie, trailing edge time.
    """
    resamp = 32
    ll = len(data)/resamp
    dd = np.median(data[:resamp*ll].reshape((ll,resamp)), axis=1)
    # differentiate
    der = dd[1:]-dd[:-1]
    # locate peaks in the derivative (i.e. position of the transitions)
    i1 = np.argmax(der)
    i2 = np.argmin(der)
    # +1 is arguable
    start = resamp*(min(i1,i2)+1)
    stop = resamp*(max(i1,i2)+1)
    # done with rebinning.
    # find the levels in the three sections
    margin = 200 # seems OK given our shutter
    w_on = slice(start+margin, stop-margin)
    w_before = slice(margin, start-margin)
    w_after = slice(stop+margin, len(data))
    # default is 4 sigma: OK
    l_on = clipped_average(data[w_on])
    l_before = clipped_average(data[w_before])
    l_after = clipped_average(data[w_after])
    if (debug) : 
        print "transitions indices (start stop)", start,stop
        print "levels before, during, after : ", l_before,l_on,l_after
    #
    # now locate precisely the transitions
    # first transition
    #    sig = np.sqrt(data[w_on].var())
    fit_win_on = slice(start-margin, start+margin,1)
    # see the function "sigmoid" for what parameters mean
    pars = [start,0.1,l_before, l_on]
    popt, pcov = curve_fit(sigmoid, 
                           range(fit_win_on.start, fit_win_on.stop,fit_win_on.step),
                           data[fit_win_on], 
                           p0= pars)
    t_on = popt[0]
    if debug : print "transition pars1 ", popt
    # second transition 
    fit_win_off = slice(stop-margin,stop+margin,1)
    pars = [stop,0.1,l_on, l_after]
    popt, pcov = curve_fit(sigmoid, 
                           range(fit_win_off.start, fit_win_off.stop,
                                 fit_win_off.step),
                           data[fit_win_off], 
                           p0= pars)
    t_off = popt[0]
    if debug : print "transition pars2 ", popt
    
    # cannot use the found times to estimate the charge because the sigmoid
    # does not describe exactly the transitions.
    # we return t_on and t_off, in case somebody wants to study
    # the shutter motions.
    # regarding integrated light, we just integrate :
    #
    pedestal = (l_before+l_after)*0.5
    # integrate the average signal (l_on), after sigma clipping
    q = (fit_win_off.start-fit_win_on.stop)*(l_on-pedestal)
    # add the two wevae fronts, without clipping.  
    q += (data[fit_win_on]-pedestal).sum() + (data[fit_win_off]-pedestal).sum()
    return q, l_on, l_before, l_after, t_on, t_off


def sigmoid(x, x0, slope, left, right):
    return left+(right-left)/(1.+np.exp(-slope*(x-x0)))
    

    

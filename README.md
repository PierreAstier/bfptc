# bfptc

##A package to handle the measurement of Photon Transfer Curves in view of constraining the Brighter-Fatter effect on CCD sensors


This is the code used to produce the analysis presented in the paper https://arxiv.org/abs/1905.08677 . It is published here mostly for documentation purposes.

Here is short description of the file contents:

scripts:
- tools/corr.py  : implements the measurement of variance and covariance in flat pairs. Outputs a ntuple.
- tools/over.py  : puts in a ntuple the content of the last serial pixel and two first overscan pixels of the input fits files
- tools/cattuple.py : concatenates tuples with the same columns into a bigger tuple.

routines:
- py/clap_stuff.py : code that handles the waveform measured by the CLAP (see the paper above)
- py/filehandlers.py : a set of classes that present the data to coor.py and over.py in a uniform way, for various kinds 
          of sensors and FITS formats. If you want to use this code, you'l very likely have to add your own class to this file. 
- py/ptcfit.py : implements the PTC model described in the above paper, and the code that transforms the tuple out 
        of corr.py into a proper structure for fitting       
- py/ptc_utils.py : some PTC anaysis utilities. Non-linearity modeling is here.  
- py/group.py : one set of routines to bin data
- py/pol2d.py :
- py/ptc_plots.py : routines used to generate the plots in the above paper.

The code now contains the material to carry out electrostatic fits
to the data extracted from PTCs. The outcome of teh electrostatic
fit is what you need to correct science image from the BF effect.
elec/integ_et.py : evaluates the needed integrals of the electric field.
elec fit_cov.py : driver to perform the least squares fit and output the results. 
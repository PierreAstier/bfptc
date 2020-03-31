import numpy as np
from .  import pol2d as pol2d




# the same routine could be written using (numpy) MaskedArrays.
# They do not seem that efficient, and we'll use a floating point mask
# to compute its Fourier transform
def find_mask(im, nsig, w=None) :
    if w is None : w = np.ones(im.shape)
    #  mu is used for sake of numerical precision in the sigma
    # computation, and is needed (at full precision) to identify outliers. 
    count = w.sum()
    # different from (w*im).mean()
    mu = (w*im).sum()/count
    # same comment for the variance. 
    sigma = np.sqrt((((im-mu)*w)**2).sum()/count)
    for iter in range(3) :
        outliers = np.where(np.abs((im-mu)*w)>nsig*sigma)
        w[outliers] = 0
        count = w.sum()
        mu = (w*im).sum()/count
        newsig = np.sqrt((((im-mu)*w)**2).sum()/count)
        if (np.abs(sigma-newsig)<0.02*sigma) :
            sigma = newsig
            break
        sigma = newsig
    return w

# a simplified utilty to compute the sky background.

def fit_back(im, stepx, stepy=None) :
    """
    fits a polynomial sky to the image. Misses a mask argument
    """
    if stepy is None : stepy = stepx
    nx = im.shape[0]//stepx
    if nx*stepx<im.shape[0] : nx += 1
    ny = im.shape[1]//stepy
    if ny*stepy<im.shape[1] : ny += 1
    # slice the image by hand in order to accomodate properly the edges
    medians = np.ndarray((nx,ny))
    xpos = np.ndarray(nx)
    ypos = np.ndarray(ny)
    for ix in range(nx) :
        maxx = min(stepx*(ix+1), im.shape[0])
        xpos[ix] = 0.5*(maxx-1+stepx*ix)
        for iy in range(ny):
            maxy = min(stepy*(iy+1), im.shape[1])
            ypos[iy] = 0.5*(maxy-1+stepy*iy)
            medians[ix,iy] = np.median(im[ix*stepx:maxx, iy*stepy:maxy])
    nx,ny = im.shape
    # some sort of reduced coordinates for numerical stability.
    # It does improve things, by several orders of magnitude.
    if True :
        ax=1./float(nx)
        bx = -0.5
        ay = 1./float(ny)
        by = -0.5
    else :
        ax=1.
        bx=0.
        ay=1.
        by=0.
    x,y = np.meshgrid(ax*xpos+bx,ay*ypos+by,indexing='ij')
    p = pol2d.pol2d(x,y,medians,2)
    # now evaluate at all positions of the image
    x,y = np.meshgrid(np.linspace(bx,ax*(nx-1)+bx, num=nx),
                      np.linspace(by,ay*(ny-1)+by, num=ny),indexing='ij')
    return p.eval(x,y)

#

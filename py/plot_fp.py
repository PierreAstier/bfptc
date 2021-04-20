import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.pyplot as pl




def raft_to_fp(raft_name) :
    """
    returns an AffineTransfo that 
    transforms raft coordinates into fp coordinates.
    """
    # raft name is, e.g., R23
    j = int(raft_name[1])
    i = int(raft_name[2])
    # raft size in mm ?
    raft_size = 130
    # there are 5 rafts in both directions. place the center at (0,0)
    transfo = AffineTransfo(1.,0.,0.,1.,float(i-2.5)*raft_size, float(j-2.5)*raft_size)
    if not is_corner_raft(raft_name): return transfo
    # corner rafts are rotated
    # shift to raft center
    shift = AffineTransfo(1,0,0,1, -raft_size*0.5, -raft_size*0.5 )
    if raft_name == 'R00' :
        t = AffineTransfo.Rot180().compose(shift)
    if raft_name == 'R04' :
        t = AffineTransfo.Rot90().compose(shift)
    if raft_name == 'R40' :
        t = AffineTransfo.Rot270().compose(shift)
    if raft_name == 'R44' :
        t = shift
    rot = shift.inverse_transfo().compose(t)
    return transfo.compose(rot)

def is_corner_raft(raft_name) :
    return raft_name in ['R00','R04','R40','R44']

def chip_to_raft(raft_name, chip_name) :
    if is_corner_raft(raft_name) :
        return chip_to_raft_corner(raft_name,chip_name)
    return chip_to_raft_science(chip_name)

def chip_to_raft_science(chip_name) :
    """
    transforms chip coordinates to raft coordinates
    """    
    # chip name is,e.g., S12
    j = int(chip_name[1])
    i = int(chip_name[2])
    assert(i>=0 and i<3 and j>=0 and j<3)
    chip_size=40
    inter_chip=2.5 # from  obs_lsst cameraHeader.yaml
    chip_step = chip_size+inter_chip
    return AffineTransfo(1,0,0,1,i*chip_step,j*chip_step)

def chip_to_raft_corner(raft_name, chip_name):
    """
    Rotations are missing !
    """
    if chip_name == 'SG0' : # slot 10
        dx,dy = +42.5,0
    elif chip_name == 'SG1' : # slot 10
        dx,dy = 0, +42.5
    elif chip_name == 'SW0' :# corresponds to lower half of SR slot 00
        dx,dy = 0, 0
    elif chip_name ==  'SW1' :# corresponds to upper half of SR slot 00
        dx,dy = 0, 0.5*42.5
    else :
        raise ValueError('Chip name %s does not seem to be a corner raft chip'%chip_name)
    return AffineTransfo(1,0,0,1,dx,dy)

def chip_to_fp(chip_id):
    return raft_to_fp(chip_id[:3]).compose(chip_to_raft(chip_id[:3], chip_id[4:]))

def amp_to_chip_pix(chip_name, amp_id):
    """
    chip_name should read RRR_CCC
    amp_id should come from the extname fit key, and should
    be two-characters long.
    """
    try :
        i = amp_id%10
        j = amp_id//10
    except TypeError : # hope it is a string...
        i = int(amp_id[1])
        j = int(amp_id[0])
    assert((i<8) & (j<2)) 
    a22_e2v = [-1,1]
    a11_e2v = [1,-1]
    a22_itl = [-1,1]
    dy_itl=[4000,0]
    dy_e2v=[4004,0]
    
    if ccd_vendor_dict[chip_name[:3]] == 'CORNER' :
        ccd_name = chip_name[4:]
        if ccd_name == 'SG0' or ccd_name == 'SG1':
            return AffineTransfo(-1,0,0,a22_itl[j], 512*(i+1), dy_itl[j])
        if ccd_name == 'SW0' or ccd_name == 'SW1' :
            return AffineTransfo(-1,0,0,1, 508*(i+1), 0)
        raise ValueError(' bad corner raft chip name %s'%ccd_name)
    # done with corner rafts    
    try :
        i = amp_id%10
        j = amp_id//10
    except TypeError : # hope it is a string...
        i = int(amp_id[1])
        j = int(amp_id[0])
    assert((i<8) & (j<2)) 
    a22_e2v = [-1,1]
    a11_e2v = [1,-1]
    a22_itl = [-1,1]
    dy_itl=[4000,0]
    dy_e2v=[4004,0]
    if ccd_vendor_dict[chip_name[:3]] == 'ITL' :
        return AffineTransfo(-1,0,0,a22_itl[j], 508*(i+1), dy_itl[j])
    elif  ccd_vendor_dict[chip_name[:3]] == 'E2V' :
        if j==0:
            return AffineTransfo(a11_e2v[j],0,0,a22_e2v[j], 512*i, dy_e2v[j])
        if j==1 :
            return AffineTransfo(a11_e2v[j],0,0,a22_e2v[j], 512*(i+1), dy_e2v[j])

        

        
def amp_extend(chip_id, amp_id):
    """
    return the extend in pixels of aan amp. 
    chip_id should contain the raft name.
    """
    if ccd_vendor_dict[chip_id[:3]] == 'ITL' :
        return 508,2000
    else :
        return 512, 2002
        
def amp_to_chip_mm(chip_name, amp_id):
    """
    returns the transfo from amp coordinates(pixels) to chip coordinates (mm)
    """
    #
    pix_size = 1e-2 # pixel size in mm
    scale = AffineTransfo(1e-2,0,0,1e-2,0,0)
    t =  amp_to_chip_pix(chip_name, amp_id)
    return scale.compose(t)

def amp_to_fp(chip_id, amp_id):
    return chip_to_fp(chip_id).compose(amp_to_chip_mm(chip_id, amp_id))


def amp_patch_fp(chip_id, amp_id) :
    """
    returns the amp patch in fp coordinates
    as a  matplotlib.patches.Rectangle
    """
    t = amp_to_fp(chip_id, amp_id)
    dx,dy = amp_extend(chip_id, amp_id)
    x,y=0,0
    x0, y0 = t.apply(x, y)
    x1, y1 = t.apply(x + dx, y + dy)
    return Rectangle((x0, y0), x1 - x0, y1 - y0)


from scipy.stats import sigmaclip

def plot_fp(ax, values, z_range=None, cm=pl.cm.hot, sig_clip = None, get_data=None) :
    """
    ax :a matplotlib Axes
    values : a dict of dict of values (indexed by chip and amp respectively)
    appends plotting data to ax.
    z_range : limits of values to plot (None)
    sig_clip : number of sigmas to clip (None)
    """
    # Borrowed the plotting mechanics from
    # https://github.com/lsst-camera-dh/jh-ccs-utils.git
    # /python/focal_plane_plotting.py
    def id(x):
        return x
    if get_data is None : 
        to_plot = values
    else : # The input contains more than what is to be plotted
        # cook up a dictionnary with a single scalar
        to_plot={}
        for chip,chip_data in values.items() :
            to_plot[chip] = { amp:get_data(input) for amp,input in chip_data.items()}
            
    if z_range is None :
        z_values = []
        for chip_val in to_plot.values() :
            z_values.extend(chip_val.values())
        if sig_clip is None :
            z_range = min(z_values), max(z_values)
        else :
            _, zmin,zmax = sigmaclip(z_values, sig_clip,sig_clip)
            z_range=(zmin,zmax)
    print('z_range = ',z_range)
    def mapped_value(val) :
        return max(0, min(1., ((val - z_range[0])
                               /(z_range[1] - z_range[0]))))
    facecolors=[]
    patches = []
    for chip_id, det_val in to_plot.items():
        for amp_id,val in det_val.items() :
            patches.append(amp_patch_fp(chip_id,amp_id))
            facecolors.append(cm(mapped_value(val)))
    pc =  PatchCollection(patches, facecolors=facecolors)
    ax.add_collection(pc)
    # the limits are not set automatically
    ax.set_xlim((-350,350))
    ax.set_ylim((-350,350))
    # cosmetics
    pl.xlabel('y (mm)')
    pl.ylabel('x (mm)')
    ax.set_aspect(1)
    # now the color bar
    norm = pl.Normalize(vmin=z_range[0], vmax=z_range[1])
    sm = pl.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    colorbar=pl.colorbar(sm)

def plot_hist(values, z_range=None, sig_clip = None, get_data=None) :
    
    if get_data is None : 
        to_plot = [x for x in y.values() for y in values.values()]
    else : # The input contains more than what is to be plotted
        # cook up a dictionnary with a single scalar
        to_plot = []
        for chip_data in values.values() :
            to_plot += [get_data(input) for input in chip_data.values()]
    if z_range is not None:
        to_plot = np.array(to_plot)
        index = (to_plot>=z_range[0]) & (to_plot<z_range[1])
        to_plot = to_plot[index]
    pl.figure(figsize=(8,8))
    pl.hist(to_plot)


    
ccd_vendor_dict={'R00': 'CORNER',
                 'R01':'ITL',
   'R02':'ITL',
   'R03':'ITL',
    'R04': 'CORNER',
   'R10':'ITL',
   'R20':'ITL',
   'R11':'E2V',
   'R12':'E2V',
   'R13':'E2V',
   'R14':'E2V',
   'R21':'E2V',
   'R22':'E2V',
   'R23':'E2V',
   'R24':'E2V',
   'R30':'E2V',
   'R31':'E2V',
   'R32':'E2V',
   'R33':'E2V',
   'R34':'E2V',
    'R40':'CORNER',
   'R41':'ITL',
   'R42':'ITL',
                 'R43':'ITL', 'R44':'CORNER'}
        

class AffineTransfo :
    """
    handles affine 2-d coordinate transformations
    application to data, composition, inversion.
    """
    def __init__(self,a11=1, a12=0, a21=1, a22 = 0, dx=0, dy=0):
        self.a11 = a11
        self.a12 = a12
        self.a21 = a21
        self.a22 = a22
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        return "xp = %f + %f*x + %f*y\nyp = %f + %f*x + %f*y"%(self.dx,self.a11,self.a12,self.dy,self.a21,self.a22)


    def Rot90() :
        return AffineTransfo(0,1,-1,0,0,0)

    def Rot180():
        return AffineTransfo(-1,0,0,-1,0,0)

    def Rot270() :
        return AffineTransfo(0,-1,1,0,0,0)
    
    def apply(self, x,y) :
        """
        just applies the transformation
        return the transforms of x and y
        """
        xt = self.dx + self.a11*x + self.a12 *y
        yt = self.dy + self.a21*x + self.a22 *y
        return xt,yt

    def compose(self, right) : 
        """
        return the transformation that does in a single step
        x1,y1 = right.apply(x,y)
        self.apply(x1,y1)
        """
        res = AffineTransfo()
        res.dx, res.dy = self.apply(right.dx, right.dy)
        res.a11 = self.a11*right.a11+self.a12*right.a21
        res.a12 = self.a11*right.a12+self.a12*right.a22
        res.a21 = self.a21*right.a11+self.a22*right.a21
        res.a22 = self.a21*right.a12+self.a22*right.a22
        return res

    def inverse_transfo(self) :
        """
        returns the inverse transfo
        """
        res = AffineTransfo()
        det = self.a11*self.a22-self.a12*self.a21
        # should raise if det = 0
        res.a11 = self.a22/det
        res.a12 = -self.a12/det
        res.a21 = -self.a21/det
        res.a22 = self.a11/det
        # at this time; res.dx,dy =0
        # we compute by how much we are off from 0,0
        rdx,rdy = res.apply(self.dx,self.dy)
        # and compensate
        res.dx = -rdx
        res.dy = -rdy
        return res

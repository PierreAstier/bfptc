from __future__ import print_function

try :
    import astropy.io.fits as pf
except :
    import pyfits as pf

import numpy as np
import scipy.interpolate as interp

try :
    no_clap=False
    import bfstuff.clap_stuff as clap_stuff
    print("found clap handling code")
except ImportError :
    no_clap=True
    print("Did not find clap handling code")
    pass




# All the code below assumes that image indices are [y,x].
# Don't know if this is a good idea. 
""" 
Code that presents data to measurement codes in a way that is
independent of the origin of the data.  
"""



# some utilities to decode the FIST keys describing image regions (DATASEC, ...)

def fortran_to_slice(a,b) :
    if b >= a:
        sx = slice(a-1,b)
    else :
        xmin = b-2
        sx = slice(a-1,xmin if xmin>=0 else None, -1)
    return sx

import re


def convert_region(str):
    """ 
    convert a FITS REGION ([a:b,c:d] string) into a pair of numpy slices
    returns:
        slice_y, slice_x
    """ 
    x = re.match('\[(\d*):(\d*),(\d*):(\d*)\]',str)
    b = [int(x.groups()[i]) for i in range(4)]
    # swap x and y:
    return fortran_to_slice(b[2],b[3]), fortran_to_slice(b[0],b[1])


# if you happen to add one (derived) class, you have to add in to the
# dictionnary at the end of this file

class FileHandler :
    """ 
    This default class should work for simulated data w/o overscan, 
    single segment 
    """
    
    def __init__(self,filename, params) :
        self.filename = filename
        self.im = pf.open(self.filename)
        self.params = params

    
    def segment_ids(self):
        """ 
        by default, returns the extension number in the fits image  
        """
        # handle compressed images.
        h = self.im[0].header
        if h['NAXIS'] == 0:
            return [1]
        return [0]

    def channel_index(self,segment_id) :
        return 0
    
    def get_segment(self, segment_id) :
        """ returns a FITS extension. derived class could do otherwise """ 
        return self.im[segment_id]

    def prepare_segment(self, ext) :
        """ ext is delivered by get_segment """
        return np.array(ext.data, dtype = np.float)

    def sensor_mask(self, ext) :
        """ In case there is a mask associated to the sensor
        """
        return None

    def clip_frame(self, image) :
        return image[self.params.margin.bottom:-self.params.margin.top,
                self.params.margin.left:-self.params.margin.right]

    def ped_stat(self, segment_id) :
        """ some sort of average and sigma of pedestal """
        return 0,0

    def rescale_before_subtraction(self):
        return True

    def other_sensors(self):
        return '',''
    
    def time_stamp(self):
        return 0

    def check_images(self, other) :
        return True

class FileHandlerParisBench(FileHandler):
    """ 
    As its name says. It works to UC-Davis Archon images,
    and perhaps other LSST sources. If the hotodiode extension is not found,
    the content of EXPTIME is returned
    """
    exptime_key_name = "EXPTIME"
    
    def segment_ids(self) :
        extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('CHAN')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('Segm')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('SEGM')]
        return extensions

    def channel_index(self,extension) :
        return extension.header['CHANNEL']
    
    def subtract_overscan_and_trim(self, extension, return_overscan = False) :
        datas_y,datas_x  = convert_region(extension.header['DATASEC'])
        segment = np.array(extension.data, dtype= np.float)
        regions = self.params.regions
        if False:
            # a single value
            pedestal = np.median(segment[datas_y, regions.xover])
        else :
            # pedestal per line
            ped = np.median(segment[datas_y, regions.xover], axis = 1)
            j_values = range(len(ped))
            nodes = np.linspace(0, len(ped), 20)
            spl = interp.splrep(j_values, ped, task=-1, t=nodes[1:-1])
            ped_smoothed = interp.splev(j_values,spl)
            pedestal = ped_smoothed[:,np.newaxis]
        im = segment[datas_y, datas_x]-pedestal
        if return_overscan :
            # overscan starts right after the physcal data
            over = segment[data_y, data_x.stop:] 
            return im, over
        return im
    
    def prepare_segment(self, extension) :
        """
        delivers the segment data, ready for use.
        """
        im = self.subtract_overscan_and_trim(extension)
        channel = self.channel_index(extension)
        if self.params.correct_nonlinearity :
            s = self.params.nonlin_corr[channel]
            im1 = interp.splev(im,s)
            im = im1
        if self.params.correct_deferred_charge :
            self.correct_deferred_charge(im, channel)
        return im

    def ped_stat(self, segment):
        datas_y,dats_x  = convert_region(segment.header['DATASEC'])
        im = np.array(segment.data, dtype= np.float)
        sig = np.std(np.median(im[datas_y, self.params.regions.xover],1))
        ped = np.median(im[datas_y, self.params.regions.xover])
        return ped,sig

    def rescale_before_subtraction(self) :
        is_not_dark = not (self.im[0].header.get('IMGTYPE',default='NONE').lower().startswith('dark'))
        is_not_bias = not (self.im[0].header.get('IMGTYPE',default='NONE').lower().startswith('bias'))
        return is_not_dark and is_not_bias

    def correct_deferred_charge(self, image, channel):
        """ 
        image: physical image (trimmed)
        channel : returned by channel_index()
        """
        delta = self.params.dc_corr[channel](image[:,1:])
        # for simulating deferred charge, signs would be opposite 
        image[:,1:] += delta
        image[:, :-1] -= delta    

    def check_images(self, other) :
        if self.im[0].header[self.exptime_key_name] != other.im[0].header[self.exptime_key_name] :
            print("%s and %s ' have different exposure times ! .... ignoring them"%(self.im1.filename,self.im2.filename))
            return false
        return True

    def other_sensors(self):        
        """
        measure integrated charge from photo diode
        return a measurement and its name (both ascii)
        """
        if not no_clap : # means we found the code
            # poorly written piece of code.
            # should rather integrate the CLAP and raise if absent.
            a1 = clap_stuff.process_one_file(self.im)
            c1 = a1[1]
            if a1[1] == 0 :
                c1 = a1[0] # EXPTIME from extension 0
        else :
            c1 = 0.
        return "%f"%c1, 'c'

    def time_stamp(self):
        i = self.filename.rfind('.')
        return self.filename[i-6:i]


    def channel_index(self,extension) :
        return extension.header['CHANNEL']

    # def check_image : parent class

class FileHandlerESABench(FileHandlerParisBench):
    """ 
    As its name says. images with a single HDU that contains two amps. 
    No info about overscans. 
    """
    exptime_key_name = "INTTIME"

    def __init__(self,filename, params) :
        FileHandler.__init__(self,filename,params)
        self.whole_image = np.array(self.im[0].data, dtype=np.float)
    
    def segment_ids(self) :
        return [0,1]

    def get_segment(self, segment_id) :
        return segment_id
    
    def channel_index(self,amp_id) :
        return amp_id

    def amp_data(self,amp_id):
        nx = self.whole_image.shape[1]
        assert (amp_id == 0 or amp_id ==1), "This codes assumes two amps in a single HDU"
        if amp_id == 0:
            data = self.whole_image[:,:nx/2]
        else :
            data = self.whole_image[:,nx-1:nx/2-1:-1]
        return data

    # boundaries determined using DS9
    image_start_x = 36
    overscan_start_x = 2280
    image_start_y = 6
    image_end_y = 2249

    def subtract_overscan_and_trim(self, extension, return_overscan=False) :
        """ the argument comes from get_segment """
        # black horizontal stripes at the bottom and top
        # trim along y
        data = self.amp_data(extension)[self.image_start_y:self.image_end_y,] 
        # handle overscan now: smooth it along y and subtract 1 value per row
        ped = np.median(data[:, self.overscan_start_x+3:], axis = 1)
        j_values = range(len(ped))
        nodes = np.linspace(0, len(ped), 20)
        spl = interp.splrep(j_values, ped, task=-1, t=nodes[1:-1])
        # 1d array
        ped_smoothed = interp.splev(j_values,spl)
        # 2d array
        ped = ped_smoothed[:,np.newaxis]
        data = data - ped
        # trim along x 
        trimmed = data[:,self.image_start_x: self.overscan_start_x]
        if return_overscan :
            over = data[:, self.overscan_start_x:]
            return trimmed,over
        return trimmed
                    
    def ped_stat(self, amp_id):
        amp_data = self.amp_data(amp_id)[:,11:]
        overscan_region = amp_data[:, self.overscan_start_x+3:]
        # rms over y of the median along x
        sig = np.std(np.median(overscan_region,1))
        ped = np.median(overscan_region) # could be mean
        return ped,sig

    def rescale_before_subtraction(self) :
        # this at least traps biases, I don't think we have darks in the sample
        return (self.im[0].header[self.exptime_key_name] > 0)

    # def correct_deferred_charge(self, image, channel):

    def other_sensors(self):
        """ no photodiode: return integration time """
        exptime =  self.im[0].header[self.exptime_key_name]
        if exptime !=0: exptime += 1.6e-3 # determined from low-signal non-linearity.
        return "%f"%exptime, 'c'

    def time_stamp(self):
        # Flat-Fields-633nmLED-FT-PO500-3ph_0600.fits
        i = self.filename.rfind('.')
        return self.filename[i-4:i]


    def channel_index(self,extension) :
        return extension

        
    # def check_images(self, other) : parent routine OK

# One extension 4 amps
class FileHandlerHSC(FileHandlerESABench):
    """ 
    As its name says. images with a single HDU that contains four amps. 
    info about the raw image layout is encoded in some ad'hoc way in headers 
    """
    exptime_key_name = "EXPTIME"  # To be checked

    def __init__(self,filename, params) :
        FileHandler.__init__(self,filename,params)
        self.whole_image = np.array(self.im[0].data, dtype=np.float)
        chip = int(self.im[0].header['DET-ID'])
        self.dead = None
        dead_name = 'dead/master_dead_%03d.fits.gz'%chip
        try:
            self.dead = pf.open(dead_name)[0].data
        except :
            print('did not find %s'%dead_name)
    
    def segment_ids(self) :
        return [1,2,3,4]

    def get_segment(self, segment_id) :
        return segment_id
    
    def channel_index(self,amp_id) :
        return amp_id
        
    def amp_data(self,amp_id):
        return self.whole_image[self.amp_region(amp_id)]


    def sensor_mask(self, amp_id):
        if self.dead is None:
            return None
        xmin = (amp_id-1)*512
        return self.dead[: , xmin:xmin+512]

    def amp_region(self, amp_id) : 
        assert (amp_id >0 and amp_id<=4), "This codes assumes four amps in a single HDU"
        xmin = int(self.im[0].header["T_EFMN%01d1"%(amp_id)])
        xmax = int(self.im[0].header["T_EFMX%01d1"%(amp_id)])
        ymin = int(self.im[0].header["T_EFMN%01d2"%(amp_id)])
        ymax = int(self.im[0].header["T_EFMX%01d2"%(amp_id)])
        return slice(ymin-1,ymax), slice(xmin-1,xmax)
    
    def overscan_region(self,amp_id) :
        assert (amp_id >0 and amp_id<=4), "This codes assumes four amps in a single HDU"
        xmin = int(self.im[0].header["T_OSMN%01d1"%(amp_id)])
        xmax = int(self.im[0].header["T_OSMX%01d1"%(amp_id)])
        ymin = int(self.im[0].header["T_EFMN%01d2"%(amp_id)])
        ymax = int(self.im[0].header["T_EFMX%01d2"%(amp_id)])
        return slice(ymin-1,ymax), slice(xmin-1,xmax)

    def overscan_data(self,amp_id):
        return self.whole_image[self.overscan_region(amp_id)]
    
    def subtract_overscan_and_trim(self, amp, return_overscan = False) :
        """ 
        the argument comes from get_segment 
        the first read out pixel is the first returned (along x)
        """
        amp_region = self.amp_region(amp)
        data = self.amp_data(amp)
        # handle overscan 
        over_region = self.overscan_region(amp)
        overscan_data = self.whole_image[over_region]
        assert data.shape[0] == overscan_data.shape[0]
        # check if overscan is AFTER or BEFORE the data (along x)
        # mirror data if before
        if over_region[1].start < amp_region[1].stop :
            # print ('mirror amp %d '%amp, 'xx %d %d'%(over_region[1].start , amp_region[1].stop)  )
            data = data[:,::-1]
            overscan_data = overscan_data[:, ::-1]
        ped = np.median(overscan_data[:,3:])
        data = data - ped
        if return_overscan :
            over = overscan_data -ped
            return data,over
        return data
                    
    def ped_stat(self, amp_id):
        amp_region = self.amp_region(amp_id)
        overscan_region = self.overscan_region(amp_id)
        overscan_data = self.whole_image[overscan_region]
        # check if overscan is AFTER or BEFORE the data (along x)
        # mirror data if before
        if overscan_region[1].start < amp_region[1].stop :
            overscan_data = overscan_data[:, ::-1]
        # trim the first pixels 
        overscan_data = overscan_data[:,3:]
        # rms over y of the median along x
        sig = np.std(np.median(overscan_data,1))
        ped = np.median(overscan_data) # could be mean
        return ped,sig

    def rescale_before_subtraction(self) :
        return True

    # def correct_deferred_charge(self, image, channel):

    def other_sensors(self):
        """ no photodiode: return integration time """
        c1 = self.im[0].header[self.exptime_key_name]
        return "%f"%c1, 'c'

    def time_stamp(self):
        exp_id =  self.im[0].header['EXP-ID']
        return int(exp_id[4:10])

    def channel_index(self,extension) :
        return extension

        
    # def check_images(self, other) : parent routine OK

# dictionnary used for user selection
file_handlers={'D':FileHandler, 'P' : FileHandlerParisBench, 'E':FileHandlerESABench, 'H': FileHandlerHSC}

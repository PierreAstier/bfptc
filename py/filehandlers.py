

try :
    import astropy.io.fits as pf
except :
    import pyfits as pf

import numpy as np
import scipy.interpolate as interp
import pickle
import os.path
import glob


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


def spline_smooth_overscan(twod_overscan, nknots=20) :
    """
    medians the array along x, and smoothes using a spline along y.
    returns a 1D array (one value per j value).
    """
    ped = np.median(twod_overscan, axis = 1)
    j_values = list(range(len(ped)))
    nodes = np.linspace(0, len(ped), nknots)
    spl = interp.splrep(j_values, ped, task=-1, t=nodes[1:-1])
    ped_smoothed = interp.splev(j_values,spl)
    return ped_smoothed

# if you happen to add one (derived) class, you have to add in to the
# dictionnary at the end of this file

class FileHandler(object) :
    """ 
    This default class should work for simulated data w/o overscan, 
    single segment 
    """
    # this constructor should be called by all derived classes
    def __init__(self,filename, params) :
        self.filename = filename
        self.im = pf.open(self.filename)
        self.params = params
        # put the corrections into *derived* classes because they are specific
        # and put them into the class, so that they are loaded only once
        if params is not None and hasattr(params,'correct_nonlinearity') and params.correct_nonlinearity and not hasattr(self.__class__,'nonlin_corr') :
            try :
                f = open(params.nonlin_corr_file,'rb')
            except (IOError, OSError) as e :
                print('cannot find %s for non linearity correction'%params.nonlin_corr_file)
                raise
            print('INFO: loading nonlinearity correction %s'%params.nonlin_corr_file) 
            self.__class__.nonlin_corr = pickle.load(f)
            f.close()
        if params is not None and hasattr(params,'correct_deferred_charge') and params.correct_deferred_charge and not hasattr(self.__class__,'dc_corr') :
            try :
                f = open(params.deferred_charge_corr_file,'rb')
            except IOError:
                print('ERROR : cannot open %s for deferred charge correction'%params.deferred_charge_corr_file)
                raise
            print('INFO: loaded deferred charge correction %s'%params.deferred_charge_corr_file) 
            self.__class__.dc_corr = pickle.load(f)
            f.close()

    
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
    
    def __get_fits_extension__(self, segment_id) :
        return self.im[segment_id]

    def prepare_segment(self, ext) :
        """ ext is delivered by segment_ids """
        return np.array(self.__get_fits_extension__(ext).data, dtype = np.float)

    def sensor_mask(self, ext) :
        """ ext is delivered by segment_ids """
        """ In case there is a mask associated to the sensor
        """
        return None

    def clip_frame(self, image) :
        """ image is returned from prepare_segment() """
        return image[self.params.margin_bottom:-self.params.margin_top,
                self.params.margin_left:-self.params.margin_right]

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
    As its name says. It also works for UC-Davis (Archon) images,
    and perhaps other LSST sources. 
    """
    exptime_key_name = "EXPTIME"
    masterbias = None
    dead = None

    def __init__(self,filename, params) :
        # call the base class ctor
        super(FileHandlerParisBench , self).__init__(filename,params)
        # add loading the master bias file (once per run, it is a class
        # attribute), could belong to the base class.
        if params is not None and hasattr(params,'subtract_bias') and params.subtract_bias and FileHandlerParisBench.masterbias is None :
            FileHandlerParisBench.masterbias = pf.open('masterbias.fits')
            print('loaded masterbias.fits for bias subtraction')
        # if dead is missing, we have to accept it because in the pipeline, requiring it here results in a dead lock.
        # dead is (mildly) required for nonlin_tuple, which is needed for dead. Could change the pipeline though
        if FileHandlerParisBench.dead is None and params is not None and hasattr(params,'use_dead') and params.use_dead and os.path.isfile("dead.fits"):
            FileHandlerParisBench.dead = pf.open('dead.fits')
            print('INFO: loaded mask : dead.fits')
        
    
    def segment_ids(self) :
        # iterates over extensions to find which ones contain CCD data
        extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('CHAN')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('Segm')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('SEGM')]
        return extensions

    def channel_index(self, segment_id) :
        """ segment_id is delivered by segment_ids """
        extension = self.__get_fits_extension__(segment_id)
        return extension.header['CHANNEL']

    def chip_id(self) :
        return 0

    def overscan_bounding_box(self, segment_id) :
        """
        segment_id comes from segment_ids
        return two slices (y, x)
        Along x, the overscan starts where datasec stops
        Along y, it covers the full size (used to follow datasec)
        """
        extension = self.__get_fits_extension__(segment_id)
        datas_y,datas_x  = convert_region(extension.header['DATASEC'])
        whole_extension_shape = extension.data.shape
        # Along x, the overscan starts where datasec stops
        # Along y, it covers the full size (used to follow datasec)
        bb = datas_y, slice(datas_x.stop, whole_extension_shape[0])
        return bb

    def datasec_bounding_box(self, segment_id) :
        """
        segment_id is obtained from segment_ids
        return two slices (y, x)
        """
        extension = self.__get_fits_extension__(segment_id)
        data_y,data_x  = convert_region(extension.header['DATASEC'])
        return data_y, data_x

    def amp_data(self, segment_id) :
        """
        segment_id is obtained from segment_ids
        returns the whole amp data (with pre/over-scans)
        """
        return np.array(self.__get_fits_extension__(segment_id).data, dtype= np.float)
    
    def subtract_overscan_and_trim(self, segment_id, bias, return_overscan = False) :
        pixels = self.amp_data(segment_id)
        # figure out the overscan bounding box
        oy,ox = self.overscan_bounding_box(segment_id)
        full_obb = oy,ox
        # skip a few columns, "contaminated" by deferred signals.
        obb = oy, slice(ox.start + self.params.overscan_skip, ox.stop)
        if False:
            # a single value
            pedestal = np.median(pixels[obb])
        else :
            # pedestal per line
            pedestal = spline_smooth_overscan(pixels[obb])
            pedestal = pedestal[:,np.newaxis]
        datasec_y, datasec_x = self.datasec_bounding_box(segment_id)
        im = pixels[datasec_y, datasec_x]-pedestal
        if bias is not None:
            bias_data = bias.data
            # a few sanity checks
            assert bias_data.shape == pixels.shape, " Expect bias and current images  to have the same raw size %s "%self.filename
            bias_datasec = convert_region(bias.header['DATASEC'])
            assert bias_datasec == (datasec_y, datasec_x), " Datasec for bias and image %s are different"%self.filename
            im -= bias_data[datasec_y, datasec_x]
        if return_overscan :
            return im, pixels[full_obb]-pedestal, None
        return im
        
    def correct_nonlin(self, im, channel, chip) :
        s = self.nonlin_corr[channel]
        return interp.splev(im,s)

    def correct_deferred_charge(self, image, channel, chip_id):
        """ 
        image: physical image (trimmed)
        chip_id is ignored in this routine (there is a single chip on the bench)
        channel : returned by channel_index()
        "image" is altered "in place"
        """
        delta = self.dc_corr[channel](image[:,1:])
        # for simulating deferred charge, signs would be opposite 
        image[:,1:] += delta
        image[:, :-1] -= delta
    
    def prepare_segment(self, segment_id) :
        """
        segment_id comes from segment_ids
        delivers the segment data, ready for use.
        """
        bias = None
        if self.masterbias is not None :   
            bias = self.masterbias[segment_id] 
        im = self.subtract_overscan_and_trim(segment_id, bias)
        channel = self.channel_index(segment_id)
        if hasattr(self.params,'correct_nonlinearity') and self.params.correct_nonlinearity :
            im = self.correct_nonlin(im, channel, self.chip_id()) 
        if hasattr(self.params,'correct_deferred_charge') and self.params.correct_deferred_charge :
            self.correct_deferred_charge(im, channel, self.chip_id())
        return im

    def ped_stat(self, segment_id):
        # this routine is only used to provide diagnostics, not values
        # for pedestal subtraction
        extension = self.__get_fits_extension__(segment_id)
        full_obb = self.overscan_bounding_box(segment_id)
        # skip the first columns, as done when subtracting
        oy,ox = full_obb
        obb = oy, slice(ox.start + self.params.overscan_skip, ox.stop)
        overscan = np.array(extension.data, dtype= np.float)[obb]
        sig = np.std(np.median(overscan,axis=1))
        ped = np.median(overscan)
        return ped,sig

    def rescale_before_subtraction(self) :
        """
        works for both Paris and SLAC
        """
        is_not_dark = not (self.im[0].header.get('IMGTYPE',default='NONE').lower().startswith('dark'))
        is_not_bias = not (self.im[0].header.get('IMGTYPE',default='NONE').lower().startswith('bias'))
        return is_not_dark and is_not_bias

    def check_images(self, other) :
        if self.im[0].header[self.exptime_key_name] != other.im[0].header[self.exptime_key_name] :
            print("%s and %s ' have different exposure times ! .... ignoring them"%(self.filename, other.filename))
            return False
        return True

    def other_sensors(self):        
        """
        measure integrated charge from photo diode
        return a measurement and its name (both ascii)
        """
        c1 = 0.
        if not no_clap : # means we found the code
            # poorly written piece of code.
            # should rather integrate the CLAP and raise if absent.
            try :
                a1 = clap_stuff.process_one_file(self.im)
                c1 = a1[1]
                if a1[1] == 0 :
                    c1 = a1[0] # EXPTIME from extension 0
            except TypeError :
                pass

        return "%f"%c1, 'c'

    def time_stamp(self):
        i = self.filename.rfind('.')
        return self.filename[i-6:i]

    # def check_image : parent class

import os
from scipy.stats import sigmaclip

# used for the analysis of the SLAC photodiode timeline
def clipped_average(d, cut=4.) :
    c, low, upp = sigmaclip(d,cut,cut)
    return c.mean()

class SlacBot(FileHandlerParisBench) :
    """
    Try to change as little as possible w.r.t Paris/Davis
    """
    exptime_key_name = "EXPTIME"

    
    def segment_ids(self) :
        # iterates over extensions to find which ones contain CCD data
        extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('CHAN')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('Segment')]
        if len(extensions) == 0:
          extensions = [i for i,ext in enumerate(self.im) if ext.header.get('EXTNAME',default='NONE').startswith('SEGM')]
        return extensions

        
    def channel_index(self, ext):
        """
        ext should come from segment_ids().
        channel_index is the one that will go to the output tuple.
        SLAC people use the EXTNAME key (which reads Segment%d)  so, we do the same.
        We consider these numbers as decimal (although they are rather octal).
        """
        return int(re.search('\d+',self.im[ext].header['EXTNAME']).group())


    def chip_id(self) :
        return self.im[0].header['RAFTBAY'].strip()+'_'+self.im[0].header['CCDSLOT'].strip()


    def overscan_bounding_box(self, segment_id) :
        """
        serial overscan. extends in y into the // overscan
        segment_id comes from segment_ids
        return two slices (y, x)
        """
        extension = self.__get_fits_extension__(segment_id)
        datas_y,datas_x  = convert_region(extension.header['DATASEC'])
        whole_extension_shape = extension.data.shape
        # Along x, the overscan starts where datasec stops
        # Along y, the overscan follow datasec
        bb = slice(datas_y.start, whole_extension_shape[0]), slice(datas_x.stop, whole_extension_shape[1])
        return bb

    def parallel_overscan_bounding_box(self, segment_id) :
        """
        segment_id comes from segment_ids
        return two slices (y, x)
        """
        extension = self.__get_fits_extension__(segment_id)
        datas_y,datas_x  = convert_region(extension.header['DATASEC'])
        whole_extension_shape = extension.data.shape
        # Along x, the way the following routine 
        # is written imposes to start at 0
        # Along y, the // overscan follow datasec
        bb = slice(datas_y.stop,whole_extension_shape[0]), slice(0, whole_extension_shape[1])
        return bb



    def subtract_overscan_and_trim(self, segment_id, bias, return_overscan = False) :
        """
        returns the overscan-sutracted data, possibily with master bias subtraction.
        Currently, this routine implements a 2D overscan subtraction.
        also returns serial and parallel overscans if requested 
        """
        # probably, the behavior should be controlled by the parameters.
        pixels = self.amp_data(segment_id)
        # figure out the overscan bounding box
        oy,ox = self.overscan_bounding_box(segment_id)
        full_obb = oy,ox
        # skip a few columns, "contaminated" by deferred signals.
        obb = oy, slice(ox.start + self.params.overscan_skip, ox.stop)
        if False:
            # a single value
            pedestal = np.median(pixels[obb])
        else :
            # serial pedestal per line
            serial_pedestal = spline_smooth_overscan(pixels[obb])
            extension = self.__get_fits_extension__(segment_id)
            # print(extension.header['EXTNAME'], ' serial mean :', serial_pedestal.mean())
            serial_pedestal = serial_pedestal[:,np.newaxis]
            oyp, oxp = self.parallel_overscan_bounding_box(segment_id)
            p_overscan_bb = oyp,oxp
            p_overscan = pixels[p_overscan_bb]-serial_pedestal[oyp,:]
            p_overscan_values = p_overscan # if returned to the caller
            # ignore the first // overscan lines
            p_overscan = p_overscan[2:,:].mean(axis= 0)
            p_overscan = p_overscan[np.newaxis, :] 

            
        datasec_y, datasec_x = self.datasec_bounding_box(segment_id)
        # the second correction assumes that oxp.start == 0
        im = pixels[datasec_y, datasec_x]-serial_pedestal[datasec_y,:] - p_overscan[:,datasec_x]
        # print('image mean before bias subtraction',im.mean())
        if bias is not None:
            bias_data = bias.data
            # a few sanity checks
            assert bias_data.shape == pixels.shape, " Expect bias and current images  to have the same raw size %s "%self.filename
            bias_datasec = convert_region(bias.header['DATASEC'])
            assert bias_datasec == (datasec_y, datasec_x), " Datasec for bias and image %s are different"%self.filename
            im -= bias_data[datasec_y, datasec_x]
            # print('image mean after bias subtraction',im.mean())
        if return_overscan :
            return im, (pixels[full_obb] - serial_pedestal)[datasec_y,:], p_overscan_values[:,datasec_x]
        return im



    # try with the parent class (i.e. ParisBench) one.
    """
    def subtract_overscan_and_trim(self, extension, return_overscan = False) :
        datas_y,datas_x  = convert_region(extension.header['DATASEC'])
        segment = np.array(extension.data, dtype= np.float)
        xover = slice(datas_x.stop+ self.params.overscan_skip, segment.shape[1])
        if False:
            # a single value
            pedestal = np.median(segment[datas_y, xover])
        else :
            # pedestal per line
            ped_smoothed = spline_smooth_overscan(segment[datas_y, xover])
            pedestal = ped_smoothed[:,np.newaxis]
        im = segment[datas_y, datas_x]-pedestal
        if return_overscan :
            # overscan starts right after the physical data
            over = segment[data_y, data_x.stop:] 
            return im, over
        return im
    """
    
    def correct_nonlin(self, im, channel, chip) :
        """
        the (nonlin.pkl) file in principle contains a dictionnary of 
        dictionnaries of (scipy.interpolate) splines. The file is loaded 
        by the main program, and provided through "params" to the constructor.
        chip is cooked up by chip_id()
        """
        # the sckpy splines consists of 3 elements: knots, coeffs, degree
        # s = self.nonlin_corr[chip][channel]
        # use one file per chip
        try : 
            s = self.nonlin_corr[channel]
        except KeyError : # nonlin correction not available for this channel
            return im
        t,c,k = s
        # the spline value is ridiculous when out of the training domain
        # cook up something reasonnable when this happens
        below = (im <= t[0])
        above = (im >= t[-1])
        # val_below = interp.splev(t[0], s)
        val_above = interp.splev(t[-1], s)
        # 
        # the spline encodes the *correction* only,
        correction = interp.splev(im, s)
        #correction[below] = val_below
        correction[below] = 0
        correction[above] = val_above
        return im+correction

    def sensor_mask(self, amp_id):
        if self.dead is None:
            return None
        return self.dead[amp_id].data


    def correct_deferred_charge(self, image, channel, chip):
        """ 
        image: physical image (trimmed)
        channel : returned by channel_index()
        chip: returned by chip_id()
        "image" is altered "in place"
        """
        try :
            # the correction is implemented as a polynomial
            # p = self.dc_corr[chip][channel]
            # use one file per chip, easier in production:
            s = self.dc_corr[channel]
        except KeyError :
            print('WARNING :missing deferred charge corr for chip', chip,' channel',channel)
            return
        t,c,k = s
        xmin = t[0]
        # patch the model above and below the "training interval"
        valmin = interp.splev(xmin, s)
        xmax = t[-1]
        valmax = interp.splev(xmax, s)
        im_source = image[:,1:]
        delta = interp.splev(im_source, s)
        index = (im_source < xmin)
        delta[index] = valmin
        index = (im_source > xmax)
        delta[index] = valmax
        # for simulating deferred charge, signs would be opposite 
        image[:,1:] += delta
        image[:, :-1] -= delta

    def photodiode_filename(self) :
        """
        find out the photodiode file name in the exposure directory, for this image.
        """
        path = self.filename
        if os.path.islink(path) :
            path = os.path.dirname(self.filename)+'/'+os.readlink(path)
        # photodiode file name : at some point the date and time was added into the file name.
        l = glob.glob(os.path.dirname(path)+'/Photodiode_Readings*.txt')
        if len(l) != 1 : 
            print ("Several or no matches for the photo diode file in directory %s"%path)
            return os.path.dirname(path)+'/Photodiode_Readings*.txt' # just for printout, it does not exist
        return l[0]

    def my_diode_integral(self) :
        """
        measure integrated charge from photo diode
        return a measurement and its name (both ascii)
        There is a photodiode at SLAC but the data is not stored in the 
        fits files
        """
        filename = self.photodiode_filename()
        try :
            d = np.loadtxt(filename) 
            t = d[:,0]
            I = d[1:,1]
            dt = t[1:]-t[:-1]
            I = I*dt
            # numerical derivative
            der = I[1:]-I[:-1]
            # search for peaks
            i1 = np.argmax(der)
            i2 = np.argmin(der)
            start = (min(i1,i2)+1)
            stop = (max(i1,i2)+1)
            margin = 2
            # robust average before and after
            if start-margin>0 :
                w_before = slice(0,start-margin)
                val_before = clipped_average(I[w_before])
                n_before = start-margin
            else :
                val_before=0
                n_before=0
            if stop+margin<len(I):
                w_after = slice(stop+margin,len(I))
                val_after = clipped_average(I[w_after])
                n_after = len(I)-stop-margin
            if (n_before+n_after>0) :
                ped = (n_before*val_before+n_after*val_after)/(n_after+n_before)
            else :
                print("WARNING: unable to determine a pedestal on %s"%filename)
                ped = 0
            integral = (I-ped).sum() # all time interval are equal in practice
            return integral
        except IOError :
            print('Could not find %s, just hoping it is usesless'%filename)
            return -1

    def other_sensors(self):
        exptime = self.im[0].header[self.exptime_key_name] 
        if exptime != 0:
            # photodiode file name
            filename = self.photodiode_filename()
            try :
                # Borrowed from mondiode_value in eotest
                t,I = np.loadtxt(filename).transpose() 
                Ithresh = (min(I) + max(I))/5 + min(I)
                int_no_sub = sum((I[1:] + I[:-1])/2*(t[1:] - t[:-1]))
                zero = np.median(I[np.where(I < Ithresh)])
                I_sub= I-zero
                int_sub = sum((I_sub[1:] + I_sub[:-1])/2*(t[1:] - t[:-1]))
                # my cook-up 
                # numerical derivative
                der = I[1:]-I[:-1]
                # search for peaks
                i1 = np.argmax(der)
                i2 = np.argmin(der)
                sampling_period = np.median(t[1:] - t[:-1])
                int_trunc = sum(I[max(i1-2,0): min(i2+2,len(I))])*sampling_period
                # i1 and i2 are shifted by 1, and i2 is excluded from the following:
                # assume that the rise is 2 samples, as well as the fall.
                current = np.mean(I[i1+3:i2-2])                               
            except IOError :
                print('Could not find %s, just hoping it is usesless'%filename)
                int_sub = -1
        else : # exptime is zero
            int_no_sub = 0
            current = 0
            int_trunc = 0
            
        # the following line was used to compare the algorithms. Seem to be similar, on average.
        # return [integral, self.my_diode_integral(), exptime] , [ 'd', 'd0', 'expt']
        return [int_no_sub, exptime, int_trunc, current*exptime] , [ 'd', 'expt','dt','c']

    def time_stamp(self):
        seqnum = self.im[0].header['SEQNUM']
        # dayobs is yyyymmdd, dop the year so that it fits on 32 bits
        dayobs = self.im[0].header['DAYOBS'][4:]
        return int('%s%04d'%(dayobs,seqnum))

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

    def channel_index(self,amp_id) :
        return amp_id

    def amp_data(self,amp_id):
        nx = self.whole_image.shape[1]
        assert (amp_id == 0 or amp_id ==1), "This codes assumes two amps in a single HDU"
        if amp_id == 0:
            data = self.whole_image[:,:nx//2]
        else :
            data = self.whole_image[:,nx-1:nx//2-1:-1]
        return data

    # boundaries determined using DS9
    image_start_x = 36
    overscan_start_x = 2280
    image_start_y = 6
    image_end_y = 2249

    def subtract_overscan_and_trim(self, extension, bias, return_overscan=False) :
        """ 
        the extension number comes from get_segment 
        bias is not used (yet?)
        """
        # black horizontal stripes at the bottom and top
        # trim along y
        data = self.amp_data(extension)[self.image_start_y:self.image_end_y,] 
        # handle overscan now: smooth it along y and subtract 1 value per row
        ped = np.median(data[:, self.overscan_start_x+3:], axis = 1)
        j_values = list(range(len(ped)))
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
            return trimmed,over, None
        return trimmed
                    
    def ped_stat(self, amp_id):
        amp_data = self.amp_data(amp_id)[:,11:]
        overscan_region = amp_data[:, self.overscan_start_x+self.params.overscan_skip:]
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

# the file contains two polynomial per channel
    def correct_deferred_charge(self, image, channel, chip):
        """ 
        image: physical image (trimmed)
        channel : returned by channel_index()
        for Plato, the correction is two polynomials for two pixels
        The data file contains a dictionnary of list of np.polynomials
        chip is ignored
        """
        [corr_fun1, corr_fun2] = self.dc_corr[channel]
        delta1 = np.polyval(corr_fun1,image[:,1:])
        delta2 = np.polyval(corr_fun2,image[:,2:])
        # for simulating deferred charge, signs would be opposite 
        image[:,1:] += delta1
        image[:, :-1] -= delta1
        image[:,2:] += delta2
        image[:, :-2] -= delta2

        
    # def check_images(self, other) : parent routine OK

# One extension 4 amps
class FileHandlerHSC(FileHandlerESABench):
    """ 
    As its name says. Images with a single HDU that contains four amps. 
    info about the raw image layout is encoded in some ad'hoc way in headers. 
    """
    exptime_key_name = "EXPTIME"  # To be checked
    dead = None

    def __init__(self,filename, params) :
        FileHandler.__init__(self,filename,params)
        self.whole_image = np.array(self.im[0].data, dtype=np.float)
        self.chip = int(self.im[0].header['DET-ID'])
        # handle dead, if required
        dead_name = 'dead/master_dead_%03d.fits.gz'%self.chip
        if FileHandlerHSC.dead is None and params is not None and hasattr(params,'use_dead') and params.use_dead :
            FileHandlerHSC.dead = pf.open(dead_name)[0].data
            print('INFO: loaded mask %s'%dead_name)
        # nonlin and deferred charge corrections are loaded in the base class constructor. 
    
    def segment_ids(self) :
        return [1,2,3,4]

    def channel_index(self,amp_id) :
        return amp_id
        
    def amp_data(self,amp_id):
        return self.whole_image[self.amp_region(amp_id)]

    def chip_id(self) :
        return self.im[0].header['DET-ID']
    
    def correct_nonlin(self, im, channel, chip) :
        try :
            #  handle 2 different cases: one file contains corrections for all chips, 
            # or the file is chip_specific
            p = self.nonlin_corr[channel]
            if p.__class__ == dict :
                p = self.nonlin_corr[chip][channel]
        except KeyError :
            print('WARNING :missing nonlin corr for chip %d channel %d'%(chip,channel))
            return im
        return np.polyval(p,im)
    
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
    
    def subtract_overscan_and_trim(self, amp, bias, return_overscan = False) :
        """ 
        the first argument comes from segment_ids
        the first read out pixel is the first returned (along x)
        bias is not used (yet?)
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
            return data,over, None
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

    def correct_deferred_charge(self, image, channel, chip):
        """ 
        image: physical image (trimmed)
        channel : returned by channel_index()
        chip is ignored.
        """
        """
        spline = self.dc_corr[channel]
        delta = interp.splev(image[:,1:],spline)
        """
        poly = self.dc_corr[channel]
        delta = np.polyval(poly, image[:,1:])
        # for simulating deferred charge, signs would be opposite 
        image[:,1:] += delta
        image[:, :-1] -= delta
    
    def rescale_before_subtraction(self) :
        return True

    # def correct_deferred_charge(self, image, channel):

    def other_sensors(self):
        """ no photodiode: return integration time """
        c1 = self.im[0].header[self.exptime_key_name]
        return c1, 'c'

    def time_stamp(self):
        #exp_id =  self.im[0].header['EXP-ID']
        #return int(exp_id[4:])
        return  self.im[0].header['MJD']


    def channel_index(self,extension) :
        return extension

        
    # def check_images(self, other) : parent routine OK

# dictionnary used for user selection
file_handlers={'D':FileHandler, 'P' : FileHandlerParisBench, 'E':FileHandlerESABench, 'H': FileHandlerHSC, 'S': SlacBot}

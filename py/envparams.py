
import yaml
import os

class EnvParams(object) :
    """
    stores parameters relevant for covariance measurements.
    default values are in the constructor, and can be overridden
    via a yaml file provided via the BFPARAMS environment variable.
    """
    class margin :
        pass
    def __init__(self) :
        self.margin_bottom = self.margin_top = 20
        self.margin_right = self.margin_left = 20
        self.debug = 0
        self.maxrange = 40
        self.overscan_skip = 5
        self.subtract_bias = False
        self.deferred_charge_corr_file = 'cte.pkl'
        self.nonlin_corr_file = 'nonlin.pkl'
        self.correct_nonlinearity = False
        self.correct_deferred_charge = False
        self.nsig_image = 5
        self.nsig_diff = 4
        self.subtract_sky_before_clipping = False
        self.use_dead = False
        dc_name = os.getenv('BFPARAMS')        
        if  dc_name is not None :
            try :
                stream = open(dc_name,'r')
            except IOError:                
                raise IOError('could not open %s obtained via the BFPARAMS env var'%dc_name)
            obj_read = yaml.load(stream)
            print('INFO : just read configuration file file %s'%dc_name)
            # yaml just returns a dictionnary 
            for name,value in obj_read.items():
                #if not name.startswith('__') and not callable(getattr(obj_read, name)):
                setattr(self, name, value)
            
    def __str__(self) :
        res = '{\n'
        for key,value in self.__dict__.items() :
            res+= '%s : %s\n'%(key,value)
        return res+'}' 
        



        

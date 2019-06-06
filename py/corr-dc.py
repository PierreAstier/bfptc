import pickle


class CorrDC :
    def __init__(self, pkl_filename) :
        """
        The pickle file is supposed to contain the charge deferred in the
        next (series) pixel as a function of content.
        We expect to find in the file a list of callables, 
        indexed by channel number.
        """
        f  = open(pkl_filename, 'rb')
        self.funcs = pickle.load(f)
        f.close()

    def __call__(self, image, channel):
        """
        corrects the input (bias subtracted) image for deferred charge and 
        returns the corrected image
        """
        delta = self.funcs[channel](image[: , 1:])
        # charge conservation is built-in :
        image[: , 1: ] -= delta
        image[: , :-1] += delta
        return image

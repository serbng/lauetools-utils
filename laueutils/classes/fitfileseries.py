import numpy as np
import multiprocess as mp
from pandas import DataFrame

from . import FitFile
#from ..visualization import strain

def _parse_fitfile(file_path):
    try:
        return FitFile(file_path)
    except IOError:
        return None

class FitFileSeries:
    
    def __init__(self, file_paths, workers=8):
        with mp.Pool(workers) as pool:
            self.fitfiles = pool.starmap(_parse_fitfile, zip(file_paths), chunksize=1)
        
        # It will turn useful when building overall class attributes
        for fitfile in self.fitfiles:
            if fitfile is not None:
                self._first_existing_fitfile = fitfile
                break        
        
        self._excluded_attributes = ["filename", "corfile", "software", "timestamp", "peaklist", "CCDdict"]
        self._inherit_attributes()                        
    
    
    def _collect(self, attr, match_length=True, match_data_shape=True, padding=np.NaN):
        # Pad to length means where there is not a fitfile put something
        # pad_with means where the is not a fitfile put this value
        # match_shape means where there is not a fitfile infer the shape of the data and fill with 'pad_with'
        """Shape is (nb_files, *data_shape). Examples:
        euler_angles                   : shape = (nb_files, 3,  )
        deviatoric_strain_crystal_frame: shape = (nb_files, 3, 3)
        mean_pixel_deviation           : shape = (nb_files, 1,  )
        """
        if attr not in self._first_existing_fitfile.__dict__.keys():
            raise AttributeError(f"Attribute not in FitFile object")
        
        if match_data_shape:
            data_shape = getattr(self._first_existing_fitfile, attr).shape
        else:
            data_shape = (1,)
        
        values = []
        for fitfile in self.fitfiles:
            if fitfile is None:
                if not match_length:
                    continue
                values.append(np.full(data_shape, padding))
            else:
                values.append(getattr(fitfile, attr))
        
        if match_data_shape:
            return np.stack(values)
        
        return values
            
    def _peak_property(self, miller_index: str, position: int):
        peak_property = np.full(self.nb_files, np.NaN)
        for file_index in range(self.nb_files):
            try:
                peak_property[file_index] = (self.fitlist[file_index]).peak[miller_index][position]
            except (AttributeError, KeyError):
                pass
        return peak_property
   
    # ----- class functions to access the spot parameters for  given miller indices -----
    
    def x_position(self, miller_index: str) -> np.ndarray:        
        return self._peak_property(miller_index, 7)
    
    def y_position(self, miller_index: str) -> np.ndarray:        
        return self._peak_property(miller_index, 8)
    
    def ttheta(self, miller_index: str) -> np.ndarray:
        return self._peak_property(miller_index, 5)
    
    def chi(self, miller_index: str) -> np.ndarray:       
        return self._peak_property(miller_index, 6)
    
    def pixel_deviation(self, miller_index: str) -> np.ndarray:         
        return self._peak_property(miller_index, 11)
    
    def intensity(self, miller_index: str) -> np.ndarray:      
        return self._peak_property(miller_index, 1)

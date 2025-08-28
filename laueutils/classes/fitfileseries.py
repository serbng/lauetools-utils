import numpy as np
import multiprocess as mp
import pandas as pd
from . import FitFile
#from ..visualization import strain

def _parse_fitfile(file_path):
    try:
        return FitFile(file_path)
    except (TypeError, IOError, OSError):
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
                
        try:
            self._first_existing_fitfile
        except AttributeError: 
            raise ValueError("None of the provided fit files could be loaded")
        
        self._excluded_attributes = ["filename", "corfile", "software", "timestamp", "peaklist", "CCDdict"]    
    
    def _collect(self, attr, match_length=True, match_data_shape=True, padding=np.nan):
        """Go through each fit file and retrieve the value 

        Args:
            attr (str): name of the attribute of the FitFile class
            match_length (bool, optional): Where the fit file doesn't exist, put "padding" to match the length of the output to the number of fit files. Defaults to True.
            match_data_shape (bool, optional): Where the fit file doesn't exist, put a numpy array with the shape that the attribute is supposed to have and fill it with "padding". Defaults to True.
            padding (float, optional): Value to use for the padding. Defaults to np.nan.

        Raises:
            AttributeError: When attr is not an attribute of class FitFile

        Returns:
            result (numpy.ndarray | list): Array containing the result if match_data_shape=True, else a list.
        """
        if attr not in self._first_existing_fitfile.__dict__.keys():
            raise AttributeError(f"Attribute not in FitFile object")
        
        data_shape = (1,)
        if match_data_shape:
            attr_value =  getattr(self._first_existing_fitfile, attr)
            if hasattr(attr_value, "shape") and attr_value.shape != ():   # MODIFIED: avoid () scalar shapes
                data_shape = attr_value.shape
            
        values = []
        for fitfile in self.fitfiles:
            if fitfile is None:
                if not match_length:
                    continue
                values.append(np.full(data_shape, padding))
            else:
                val = getattr(fitfile, attr)                              # MODIFIED: store in variable
                if match_data_shape:
                    arr = np.asarray(val)                                 # MODIFIED: convert to array
                    if arr.shape == ():                                   # MODIFIED: handle scalars
                        arr = np.full(data_shape, arr)
                    elif arr.shape != data_shape:                         # MODIFIED: try broadcast
                        try:
                            arr = np.broadcast_to(arr, data_shape).copy()
                        except Exception as e:
                            raise ValueError(
                                f"Attribute '{attr}' has shape {arr.shape} "
                                f"but cannot be broadcast to target shape {data_shape}."
                            ) from e
                    values.append(arr)                                    # MODIFIED: append normalized
                else:
                    values.append(val)                                    # unchanged
        
        if match_data_shape: # shapes of elements are the same, stacking allowed
            return np.stack(values)
        
        # List of values, will have different shapes
        return values
    
    @property
    def number_indexed_spots(self):
        return self._collect("number_indexed_spots")

    @property
    def mean_pixel_deviation(self):
        return self._collect("mean_pixel_deviation")
    
    @property
    def euler_angles(self):
        return self._collect("euler_angles")
    
    @property
    def UB(self):
        return self._collect("UB")
    
    @property
    def B0(self):
        return self._collect("B0")
    
    @property
    def UBB0(self):
        return self._collect("UBB0")
    
    @property
    def deviatoric_strain_crystal_frame(self):
        return self._collect("deviatoric_strain_crystal_frame")
    
    @property
    def deviatoric_strain_sample_frame(self):
        return self._collect("deviatoric_strain_sample_frame")
    
    @property
    def new_lattice_parameters(self):
        return self._collect("new_lattice_parameters")
    
    @property
    def a_prime(self):
        return self._collect("a_prime")
    
    @property
    def b_prime(self):
        return self._collect("b_prime")
    
    @property
    def c_prime(self):
        return self._collect("c_prime")
    
    @property
    def astar_prime(self):
        return self._collect("astar_prime")
    
    @property
    def bstar_prime(self):
        return self._collect("bstar_prime")
    
    @property
    def cstar_prime(self):
        return self._collect("cstar_prime")
    
    def track_hkl(self, h: int, k: int, l: int) -> pd.DataFrame:
        """_summary_

        Args:
            h (int): _description_
            k (int): _description_
            l (int): _description_

        Raises:
            TypeError: _description_

        Returns:
            pd.DataFrame: _description_
        """
        if not all([isinstance(index, int) for index in (h,k,l)]):
            raise TypeError("All Miller indices must be integers")
        
        matches = []
        columns = list(self._first_existing_fitfile.peaklist.columns)
        columns.append("spot idx")
        M = len(columns)
        
        for i, ff in enumerate(self.fitfiles):
            if ff is not None:
                match = ff.peaklist.query(f"h=={h} and k=={k} and l=={l}")
            
            if len(match)==0 or ff is None:
                match =  pd.DataFrame(
                    data=np.full((1, M), np.nan), 
                    columns=columns, 
                    index=[i]
                )
                # Even though every field is a nan, let's put the correct miller indices values
                # Remember that the index is the image number, i.e. i
                match.loc[i, ["h", "k", "l"]] = h, k, l
                matches.append(match)
                continue
            
            # I expect only one match, so I create one new column called spot idx
            # and I store in row 0 the value of the index of the dataframe (that
            # is the spot idx) and set the current index to the image number
            match.insert(M-1, "spot idx", match.index.values)
            match.index = [i]
            matches.append(match)
            
        return pd.concat(matches).convert_dtypes()

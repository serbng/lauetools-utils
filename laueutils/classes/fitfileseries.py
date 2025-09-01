import numpy as np
import multiprocess as mp
import pandas as pd
from . import FitFile
#from ..visualization import strain

def _parse_fitfile(file_path):
    try:
        return FitFile(file_path)
    except (TypeError, IOError, OSError):  # MODIFIED: (TypeError, IOError, OSError)
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
        
        #self._excluded_attributes = ["filename", "corfile", "software", "timestamp", "peaklist", "CCDdict"]    
    
    def _collect(self, attr, padding=np.nan):
        """
        Gather `attr` from each FitFile in `self.fitfiles`.
    
        Returns
        -------
        np.ndarray
            If values are numeric:
              - scalar attribute  -> shape (N,)
              - array attribute   -> shape (N, *target_shape)
        list
            If values are non-numeric, returns a Python list (with None
            for missing fitfiles).
        """
        # Validate attribute existence on the first valid FitFile
        if attr not in self._first_existing_fitfile.__dict__:
            raise AttributeError(f"Attribute '{attr}' not in FitFile object.")
    
        sample_val = getattr(self._first_existing_fitfile, attr)
        sample_arr = np.asarray(sample_val)
        is_numeric = (sample_arr.dtype != object) and np.issubdtype(sample_arr.dtype, np.number)
    
        if is_numeric:
            # --- numeric path ---
            if sample_arr.shape == ():
                # Scalar case -> 1D array (N,)
                out_vals = []
                for ff in self.fitfiles:
                    if ff is None:
                        out_vals.append(padding)
                    else:
                        val = getattr(ff, attr)
                        arr = np.asarray(val)
                        if arr.shape != ():
                            raise ValueError(
                                f"Attribute '{attr}' expected scalar (), got {arr.shape}."
                            )
                        # Extract scalar (arr.item() handles numpy scalars cleanly)
                        out_vals.append(arr.item())
                return np.asarray(out_vals)
    
            else:
                # Array case -> stack to (N, *target_shape)
                target_shape = sample_arr.shape
                out_arrays = []
                for ff in self.fitfiles:
                    if ff is None:
                        out_arrays.append(np.full(target_shape, padding))
                    else:
                        arr = np.asarray(getattr(ff, attr))
                        if arr.shape != target_shape:
                            raise ValueError(
                                f"Attribute '{attr}' expected shape {target_shape}, "
                                f"got {arr.shape}."
                            )
                        out_arrays.append(arr)
                return np.stack(out_arrays)
    
        else:
            # --- non-numeric path -> list with None for missing ---
            out = []
            for ff in self.fitfiles:
                out.append(None if ff is None else getattr(ff, attr))
            return out
        
    @property
    def number_indexed_spots(self):
        return self._collect("number_indexed_spots") # MODIFIED: flatten (n,) instead of (n,1)

    @property
    def mean_pixel_deviation(self):
        return self._collect("mean_pixel_deviation") # MODIFIED: flatten (n,) instead of (n,1)


    @property
    def boa(self):
        return self._collect("boa")

    @property
    def coa(self):
        return self._collect("coa")
    
    
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

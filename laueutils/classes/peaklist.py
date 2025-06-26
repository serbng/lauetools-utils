import numpy as np
import time
from pandas import DataFrame

class PeakList:
    
    _COLUMNS = [
        "x_position",
        "y_position",
        "intensity",
        "intensity_sub",
        "fwhm_major",
        "fwhm_minor",
        "inclination",
        "x_deviation",
        "y_deviation",
        "intensity_background",
        "intensity_maximum"
    ]
    
    _COLUMNS_LONG_FMT = [
        "image_number"
        "x_position",
        "y_position",
        "intensity",
        "intensity_sub",
        "fwhm_major",
        "fwhm_minor",
        "inclination",
        "x_deviation",
        "y_deviation",
        "intensity_background",
        "intensity_maximum"
    ]
    
    def __init__(self, data=None, format='short', columns=None, **kwargs):
        if format == 'long':
            self._COLUMNS = self._COLUMNS_LONG_FMT
            
        if columns is not None: #Overwriting columns
            self._COLUMNS = columns
            
        self._idx_map = {name: i for i, name in enumerate(self._COLUMNS)}
            
        if data is None:
            self._data = np.empty((0,len(self._COLUMNS)))
            
        elif isinstance(data, str):
            self._data = np.loadtxt(data, **kwargs)
            if self._data.shape[1] != len(self._COLUMNS):
                raise ValueError(
                    f"Expected {len(self._COLUMNS)} columns, got {self._data.shape[1]}."
                )
            
        elif isinstance(data, np.ndarray):
            if data.ndim != 2 or data.shape[1] != len(self._COLUMNS):
                raise ValueError(
                    f"The array must have shape (N, {len(self._COLUMNS)})."
                    f"Got {data.shape}"
                )
            self._data = data
        else:
            raise TypeError("type(data) must be in {None, str, np.ndarray}")
    
    @property
    def image_number(self):
        return self._data[:, self._idx_map["image_number"]]

    @image_number.setter
    def y_position(self, values):
        self._data[:, self._idx_map["image_number"]] = values
    
    @property
    def x_position(self):
        return self._data[:, self._idx_map["x_position"]]

    @x_position.setter
    def x_position(self, values):
        self._data[:, self._idx_map["x_position"]] = values
    
    @property
    def y_position(self):
        return self._data[:, self._idx_map["y_position"]]

    @y_position.setter
    def y_position(self, values):
        self._data[:, self._idx_map["y_position"]] = values

    @property
    def intensity(self):
        return self._data[:, self._idx_map["intensity"]]

    @intensity.setter
    def intensity(self, values):
        self._data[:, self._idx_map["intensity"]] = values

    @property
    def intensity_sub(self):
        return self._data[:, self._idx_map["intensity_sub"]]

    @intensity_sub.setter
    def intensity_sub(self, values):
        self._data[:, self._idx_map["intensity_sub"]] = values

    @property
    def fwhm_major(self):
        return self._data[:, self._idx_map["fwhm_major"]]

    @fwhm_major.setter
    def fwhm_major(self, values):
        self._data[:, self._idx_map["fwhm_major"]] = values

    @property
    def fwhm_minor(self):
        return self._data[:, self._idx_map["fwhm_minor"]]

    @fwhm_minor.setter
    def fwhm_minor(self, values):
        self._data[:, self._idx_map["fwhm_minor"]] = values

    @property
    def inclination(self):
        return self._data[:, self._idx_map["inclination"]]

    @inclination.setter
    def inclination(self, values):
        self._data[:, self._idx_map["inclination"]] = values

    @property
    def x_deviation(self):
        return self._data[:, self._idx_map["x_deviation"]]

    @x_deviation.setter
    def x_deviation(self, values):
        self._data[:, self._idx_map["x_deviation"]] = values

    @property
    def y_deviation(self):
        return self._data[:, self._idx_map["y_deviation"]]

    @y_deviation.setter
    def y_deviation(self, values):
        self._data[:, self._idx_map["y_deviation"]] = values

    @property
    def intensity_background(self):
        return self._data[:, self._idx_map["intensity_background"]]

    @intensity_background.setter
    def intensity_background(self, values):
        self._data[:, self._idx_map["intensity_background"]] = values

    @property
    def intensity_maximum(self):
        return self._data[:, self._idx_map["intensity_maximum"]]

    @intensity_maximum.setter
    def intensity_maximum(self, values):
        self._data[:, self._idx_map["intensity_maximum"]] = values
        
    def __len__(self):
        return len(self._data)
    
    def __str__(self):
        return DataFrame(self._data, columns=self._COLUMNS).to_string()
    
    def __repr__(self):
        with np.printoptions(precision=3):
            rstring = f"PeakList(num_peaks = {len(self.data)})\n{self._data}"
        return rstring
    
    def __getitem__(self, key):
        # NumPy-like access to data. Ex.: peaklist[i,j], peaklist[:,2], etc...
        return self._data[key]
    
    def __setitem__(self, key, value):
        # NumPy-like data setting. Ex.: peaklist[i,j] = 1100.
        self._data[key] = value
        
    # -----    Peak manipulation methods     -----
    def add_peak(self, peak_data):
        # Add one or multiple peaks to the list
        peak_array = np.array(peak_data)
        
        if peak_array.ndim == 1 and len(peak_array) != len(self._COLUMNS):
            raise ValueError(
                f"Expected {len(self._COLUMNS)} columns, got {len(peak_array)}."
            )
        
        if peak_array.ndim == 2 and peak_array.shape[1] != len(self._COLUMNS):
            raise ValueError(
                f"The array must have shape (N, {len(self._COLUMNS)})."
                f"Got {peak_array.shape}"
            )
            
        self._data = np.vstack([self._data, peak_array])
        
    def remove_peak(self, index):
        # Remove peak with given index from the list
        # I don't do any checks, if an index is out of bounds np.delete will say it
        # In this way this will work whether index is a list of indices or a single index
        self._data = np.delete(self._data, index, axis=0)
       
    # Handy method
    @property
    def peak_positions(self):
        return np.vstack((self.x_position, self.y_position)).T
        
    def sort_by(self, attribute, descending=True):
        index = self._idx_map[attribute]
        order = np.argsort(self._data[:, index])
        if descending:
            order = order[::-1]
        self._data = self._data[order]
    
    
    def filter_by(self, mask, return_fit_params=False, inplace=False):
        """Filter the peaks according to a given rule(s). Only the peaks that match all the filters are returned.

        Parameters
        ----------
        mask  dict: Contains (attribute_name[str], rule[func]) as (key, value) pairs. 
                    The function needs to be applicable element-wise to numpy arrays and to return booleans.
                    Examples:
                    >>> pl = PeakList(filepath)
                    >>> mask = {'intensity': lambda x: x > 10000} 
                    >>> pl.filter_by(mask)
                    
                    will return only the peaks with a fitted intensity larger than 10k.

                    >>> pl = PeakList(filepath)
                    >>> def is_in_interval(x, cen, side):
                    ...     return (x > cen - side/2) & (x < cen + side/2)
                    >>> mask = {'x_position': lambda x: is_in_interval(x, 1000, 10),
                    ...         'y_position': lambda x: is_in_interval(x, 1000, 10)}
                    >>> pl.filter_by(mask)
                              
                    will return the spots whose position is within a square centered at pixel (1000, 1000) with side 10
                       
                    If no key matches an attribute AttributeError is raised.
                    If there is at least one key that matches an attribute, the others are ignored and the filtering is performed.
                    
        return_fit_params  bool: If False, only the peak positions are returned.
        inplace            bool: If True the class will be overwritten with the filtered values, even if none are found.
        
        Return value
        ----------
        If overwrite is False, returns a pandas.DataFrame, eventually empty.
        if overwrite is True, nothing is returned.
        """
        # Sir, if dictionary is empty, that's a no no
        if len(mask) < 1:
            raise(ValueError, 'mask dictionary must contain at least one entry')
        
        # If the filters don't match the attributes or, that's also a no no
        if len(set(list(mask.keys())) & set(self._COLUMNS)) == 0:
            raise(AttributeError, 'No attributes match your filters')
            
        # If an item in mask is not present in self._COLUMNS warn the user and remove it
        elif not all([mask in self._COLUMNS for mask in mask.keys()]):
            print("Warning, not all input filters match an attribute.")
            unused_masks = list(set(mask.keys() - set(self._COLUMNS)))
            for to_remove in unused_masks:
                print(f'Ignoring {to_remove}')
                del mask[to_remove]
        
        # Initializing array with True
        matching_indices = np.ones(len(self._data), dtype=bool)
        for attr, rule in mask.items():
            matching_indices &= rule(getattr(self, attr))

        # If matching_indices is full of False, i.e. if nothing matches
        # write proper result and exit
        if not any(matching_indices):
            return None
        
        # If I want to overwrite the attributes I take a shorter route
        if inplace:
            self._data = self._data[matching_indices]
            return
        
        if return_fit_params:
            return self._data[matching_indices]
        else:
            x_idx, y_idx = self._idx_map['x_position'], self._idx_map['y_position']
            return np.vstack((self._data[matching_indices, x_idx],
                              self._data[matching_indices, y_idx])).T
        
    def savetxt(self, filename: str, compact=False):
        padding = 2 # this is because the header starts with '# '
        header  = f'File created at {time.asctime()} with laueutils/classes/peaklist.py\n'
        header += f'Number of peaks {len(self._data)}\n'
        
        if compact: # minimum formatter width to have the values well stacked together
            fmts = '%7.2f'
        else: # The values are aligned with the columns
            fmts   = [] 
            for column in self._COLUMNS:
                header += f'{column} '
                fmts.append(f'%{len(column)}.2f')
        
            # modify first formatter to shift first columns $pad caracters to the left
            fmts[0] = f'%{len(self._COLUMNS[0]) + padding}.2f'
            
        np.savetxt(filename, self._data, header=header, fmt=fmts)
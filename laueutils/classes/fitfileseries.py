import os
import numpy as np
import multiprocess as mp
#to_remove
import itertools
# ----
from tqdm import tqdm
from .classes import FitFile

def _parse_fitfile(folderpath: str,
                   file_index: int, 
                   nbdigits_filename: int,
                   file_prefix: str = 'img_', 
                   file_suffix: str = '_g0.fit'):
    
                filename = f"{file_prefix}{file_index:0>{nbdigits_filename}d}{file_suffix}" 
                return FitFile(os.path.join(folderpath, filename), verbose = False)

class FitFileSeries:
    
    def __init__(self, folderpath: str, 
                       file_indices: tuple, 
                       nbdigits_filename: int = 4,
                       file_prefix: str = 'img_', 
                       file_suffix: str = '_g0.fit',
                       use_multiprocessing: bool = True):
        
        nb_files = file_indices[1] - file_indices[0]
        if nb_files <= 0:
            raise(ValueError,'File indices should be in increasing order')
            
        else:
            self.folderpath   = folderpath
            self.file_indices = file_indices
            self.nb_files     = nb_files
            
            # Parse folder of fit files 
            # The result is a list of Fitfile objects in the attribute self.fitlist            
            self._parse_folder(folderpath, 
                               file_indices, 
                               nbdigits_filename,
                               file_prefix, 
                               file_suffix,
                               use_multiprocessing)
            
            # Build attributes based on the single fit files
            self._inherit_attributes()
    
    def _parse_folder(self, folderpath: str, 
                            file_indices: tuple, 
                            nbdigits_filename: int,
                            file_prefix: str, 
                            file_suffix: str,
                            use_multiprocessing: bool):
        
        ###################################
        ##### MULTIPROCESSING VERSION #####
        ###################################
        
        if use_multiprocessing:
            nb_cpus = mp.cpu_count()
            print(f"Using {nb_cpus} cpus.")
            
            _parse_args = zip(itertools.repeat(folderpath),
                                      range(*file_indices),
                                      itertools.repeat(nbdigits_filename),
                                      itertools.repeat(file_prefix),
                                      itertools.repeat(file_suffix))
            
            with mp.Pool(nb_cpus) as pool:
                self.fitlist = pool.starmap(_parse_fitfile,
                                            tqdm(_parse_args, 
                                                 total = len(range(*file_indices)),
                                                 desc  = 'Parsing fit files'),
                                            chunksize = 1)
                
            if not mp.active_children():
                print("Done!")
                
        ###################################
        ####### SINGLE CORE VERSION #######
        ###################################
            
        else:
            print("Using a single cpu.")
            fitfile_list = []

            for file_index in tqdm(range(*file_indices), 
                                   desc  = 'Parsing fit files'):
                fitfile = _parse_fitfile(folderpath,
                                         file_index,
                                         nbdigits_filename,
                                         file_prefix,
                                         file_suffix)
                
                fitfile_list.append(fitfile)
            
            self.fitlist = fitfile_list

    def _inherit_attributes(self):
        """Look through the attributes of class fitfile and create corresponding attributes for fitfileseries containing the values of given fitfile attribute for each parsed fitfile"""
        # List of attributes from a fitfile loaded on the fitfileseries class
        # Value chosen to be in the center of the scan
        for i, fitfile in enumerate(self.fitlist):
            try:
                fitfile.deviatoric_strain_crystal_frame
                self.fitfile_index = i
                break
            except AttributeError:
                continue
                
        fitfile_attr_list = list(vars(self.fitlist[self.fitfile_index]).keys())
        # Remove attributes that are not needed for fitfileseries
        attributes = self._exclude_attributes(fitfile_attr_list, ['corfile',
                                                                  'timestamp',
                                                                  'software',
                                                                  'element'])
        for attr in attributes:
            setattr(self, attr, self._collect_attribute(attr))
        
    def _collect_attribute(self, attr: str) -> list:
        """Shape is (nb_files, *data_shape). Examples:
        euler_angles                   : shape = (nb_files, 3,  )
        deviatoric_strain_crystal_frame: shape = (nb_files, 3, 3)
        mean_pixel_deviation           : shape = (nb_files, 1,  )
        """
        # Type and shape of the data in the attribute
        try:
            attr_value = getattr(self.fitlist[self.fitfile_index], attr)
        except AttributeError:
            raise(AttributeError, f"Can't get the attribute {attr} from the fitfiles")
        
        # Set proper container for the values of the attribute depending
        # On the type    
        if isinstance(attr_value, (str, dict, list)):
            # Container is a list full of None
            values = [None] * self.nb_files
        elif isinstance(attr_value, (int, float, np.float64, np.ndarray)):
            # Container is a np.ndarray
            try:
                attr_shape = attr_value.shape
            except AttributeError:
                # attr_value is a scalar, which does not have the shape attr
                attr_shape = (1,)
                
            values = np.full((self.nb_files, *attr_shape), np.NaN)
        
        for file_number in range(self.nb_files): 
            try: 
                values[file_number] = getattr(self.fitlist[file_number], attr)
            except AttributeError:
                pass # If attribute is not found, default value is good
            
        return values
        
    def _exclude_attributes(self, attributes: list, exclusions: list):
        for excluded_attribute in exclusions:
            try:
                attributes.remove(excluded_attribute)
            except ValueError:
                pass # This means that the string was not in the list which
                     # can happen if fitfile obj didn't ready anything
        return attributes

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
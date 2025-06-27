import numpy as np

from LaueTools.CrystalParameters import Prepare_Grain
from LaueTools.lauecore import SimulateLaue_full_np
from LaueTools.dict_LaueTools import dict_Materials, dict_CCD

def simulate(material, orientation, calibration_parameters, **kwargs):
    """Simulate a Laue pattern given the material, its orientation matrix and the calibration parameters.

    Parameters
    ----------
    material                        (str): Name of the material. Should match a key inside the LaueTools materials dictionary.
                                           If the material is not present, it is possible to pass to the function your own.
                                           See the 'materials_dictionary' in the keyword arguments.
    orientation              (np.ndarray): Array with shape (3,3). UB matrix.
    calibration_parameters  (list[float]): 5-parameter list containing the calibration parameters: detector distance, xcen,
                                           ycen, xbet, xgam.

    Returns
    ----------
    result           (np.ndarray): Array with shape (N, 4), where N is the number of peaks. The columns contain respectively
                                   x_position, y_position, twotheta, chi.

    Keyword arguments
    ----------
    Emin                  (float):
    Emax                  (float):
    materials_dictionary   (dict):
    camera_label            (str):
    detector_diameter     (float):
    """
    
    # Simulation parameters
    Emin = kwargs.pop("Emin",  5)
    Emax = kwargs.pop("Emax", 25)
    material_dictionary = kwargs.pop("material_dictionary", dict_Materials)
    simulation_parameters = Prepare_Grain(material, orientation, dictmaterials=material_dictionary)
    
    # Camera-related parameters
    camera_label      = kwargs.pop("camera_label", "sCMOS")
    detector_diameter = kwargs.pop("detector_diameter", 148.1212) * 1.75 # Detector diameter is not present in the dict_CCD["sCMOS"] for some reason
    pixel_size        = dict_CCD[camera_label][1]
    frame_shape       = dict_CCD[camera_label][0]
    
    # Actual simulation
    result = SimulateLaue_full_np(simulation_parameters, Emin, Emax, calibration_parameters, # mandatory positional arguments
                                  detectordiameter = detector_diameter,
                                  pixelsize        = pixel_size,
                                  dim              = frame_shape,
                                  dictmaterials    = material_dictionary,
                                  kf_direction     = 'Z>0', # default value
                                  removeharmonics  = 0)     # default value
    
    # discarding 2theta, chi, miller indices, reflection energy
    twotheta   = result[0]
    chi        = result[1]
    x_position = result[3]
    y_position = result[4]
    
    # Some simulated reflection fall outside of the detector, remove them
    to_keep  = np.ones(len(x_position), dtype=bool)
    to_keep &= (x_position > 0) & (x_position < frame_shape[1])
    to_keep &= (y_position > 0) & (y_position < frame_shape[0])
    

    twotheta   = twotheta[to_keep]
    chi        = chi[to_keep]
    x_position = x_position[to_keep]
    y_position = y_position[to_keep]
    
    return np.vstack((x_position, y_position, twotheta, chi)).T
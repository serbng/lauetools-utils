import numpy as np
from LaueTools.indexingSpotsSet import spotsset
#from dict_LaueTools import dict_CCD as camera_dictionary

DEFAULT_MATCHING_RATES      = [50, 60, 80]
DEFAULT_MATCHING_ANGLE_TOLS = [0.5, 0.2, 0.1]
DEFAULT_LUT_MAX_INDEX       = 4

def refinement_dict_from_kwargs(**kwargs):
    """ Build the refinement dictionary from kwargs and return eventual remaining ones"""
    refinement_dict = {
        # I use kwargs.get for the entries that are also an input of the indexing function
        'AngleTolLUT':                kwargs.pop("lut_angle_tol", 0.5),   # Tolerance angle [deg]
        'nlutmax'    :                kwargs.get("lut_max_index", DEFAULT_LUT_MAX_INDEX), # Maximum miller index checked in the LUT
        'list matching tol angles':   kwargs.get("matching_angle_tols", DEFAULT_MATCHING_ANGLE_TOLS),
        'central spots indices':      kwargs.pop("spot_set_A", [0]),  # spots set A 
        #number of most intense spot candidate to have a recognisable distance
        'NBMAXPROBED':                kwargs.pop("spot_set_B", 10),
        'MATCHINGRATE_ANGLE_TOL':     kwargs.pop("MATCHINGRATE_ANGLE_TOL", 0.2),
        'MinimumMatchingRate':        kwargs.pop("MinimumMatchingRate", 5),
        'MinimumNumberMatches':       kwargs.pop("nb_matches_min", 15),
        'UseIntensityWeights':        kwargs.pop("UseIntensityWeights", False),
        'nbSpotsToIndex':             kwargs.pop("nb_spots_max", 10000),
        'MATCHINGRATE_THRESHOLD_IAL': kwargs.pop("MATCHINGRATE_THRESHOLD_IAL", 2)
    }

    return refinement_dict, kwargs

def index(peaks, material, **kwargs):
    """ Wrapper of LaueTools.indexingSpotsSet.spotsset.IndexSpotsSet for easier indexation.

    Parameters
    ----------

    Keyword arguments
    ----------
    
    """
    spotset = spotsset()

    # Set mandatory positional arguments for the LaueTools indexer
    if isinstance(peaks, str):
        spotset.importdatafromfile(peaks)
        
    elif isinstance(peaks, np.ndarray):
        if calibration_dictionary not in kwargs:
            raise(ValueError, "If a list of peaks is given, the calibration dictionary must be provided")
        # I need to manually fill the parameters of the class regarding the calibration parameters

        # Copied (excluding the existance checks) from
        # LaueTools.indexingSpotsSet.spotsset.IndexSpotsSet.importdatafromfile()
        # Lines 446 - 481
        calibration_dictionary     = kwargs.pop("calibration_dictionary")
        spotset.CCDcalibdict       = calibration_dictionary
        spotset.CCDLabel           = calibration_dictionary["CCDLabel"]
        spotset.pixelsize          = calibration_dictionary["pixelsize"]
        spotset.framedim           = calibration_dictionary["framedim"]
        spotset.dim                = calibration_dictionary["framedim"]
        spotset.detectordiameter   = calibration_dictionary["kf_direction"]
        spotset.detectorparameters = calibration_dictionary["CCDCalibParams"]
        spotset.nbspots            = len(peaks)        

    refinement_dict, indexation_kwargs = refinement_dict_from_kwargs(**kwargs)
    
    Emin     = kwargs.pop("Emin",  5)
    Emax     = kwargs.pop("Emax", 28)
    database = kwargs.pop("database", None)
    
    # Additional arguments
    indexation_kwargs.update(
        {
            "angletol_list":      kwargs.pop("matching_angle_tols", DEFAULT_MATCHING_ANGLE_TOLS),
            "MatchingRate_List":  kwargs.pop("matching_rates",      DEFAULT_MATCHING_RATES),
            "nbGrainstoFind":     kwargs.pop("nb_grains", 1),
            "dirnameout_fitfile": kwargs.pop("outputdir", None),
            "n_LUT":              kwargs.pop("lut_max_index", DEFAULT_LUT_MAX_INDEX)
        }
    )

    spotset.IndexSpotSet(peaks, material, Emin, Emax, refinement_dict, database, **indexation_kwargs)
    


def check_orientation(peaks, material, orientation, **kwargs):
    kwargs.update(
        {"spot_set_B": 0,
         "previousResults": [1, [orientation], 5, 5]}
    )

    try:
        index(peaks, material, **kwargs)
    except:
        pass

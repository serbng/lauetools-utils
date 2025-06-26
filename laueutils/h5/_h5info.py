import h5py
import numpy as np

AUTORESHAPE = False
GLOBAL_SHAPE         = None
GLOBAL_RESHAPE_ORDER = 'C'

def autoreshape_on(shape):
    AUTORESHAPE  = True
    GLOBAL_SHAPE = shape
    
def autoreshape_off():
    AUTORESHAPE = False
    
def reshape_order(order='C'):
    if order in ['C', 'F', 'A']:
        GLOBAL_RESHAPE_ORDER=order
    else:
        ValueError("Unrecognized reshape order. Must be in {'C', 'F', 'A'} or 'auto'." + f"Got {order}.")

def _get_h5field(h5source, field):
    if not isinstance(h5source, h5py._hl.files.File):
        # If h5source is not a file object, try to open it
        with h5py.File(h5source) as h5file:
            h5field = h5file[field]
    else:
        h5field = h5source[field]
    
    return h5field

def measurement(h5source, name, scan_number=1):
    # I don't add any more checks, 
    # if measurement is not a string or not a valid key you will have a KeyError
    # It's not affected by AUTORESHAPE
    return _get_h5field(h5source, f"{scan_number}.1/{name}")[:]
    
def scan_positions(h5source, scan_number=1, relative=True):
    """Get the motor positions

    Args:
        h5path (_type_): _description_
        relative (bool, optional): _description_. Defaults to True.
    """
    x_motor = measurement(h5source, scan_number=scan_number, measurement="xech")
    y_motor = measurement(h5source, scan_number=scan_number, measurement="yech")
    
    if relative:
        x_motor -= x_motor[0]
        y_motor -= y_motor[0]
        
    if AUTORESHAPE:
        x_motor = x_motor.reshape(GLOBAL_SHAPE, order=GLOBAL_RESHAPE_ORDER)
        y_motor = y_motor.reshape(GLOBAL_SHAPE, order=GLOBAL_RESHAPE_ORDER)
    
    return x_motor, y_motor
        
def fluorescence(h5source, scan_number=1, info=False):
    data = measurement(h5source, "xia", scan_number=scan_info)
    
    if info:
        pass
    
    if AUTORESHAPE:
        data = data.reshape(GLOBAL_SHAPE, order=GLOBAL_RESHAPE_ORDER)
    
    return data

def xeol(h5source, scan_number=1, info=False):
    data = measurement(h5source, 'qepro1', scan_number=scan_number)
    
    if info:
        pass
    
    if AUTORESHAPE:
        data = data.reshape(GLOBAL_SHAPE, order=GLOBAL_RESHAPE_ORDER)

def shape(h5source, scan_number=1):
    info_dict = scan_info(h5source, scan_number=scan_number)
    return (info_dict["motor1_npoints"], 
            info_dict["motor2_npoints"])

def scan_info(h5source, scan_number=1, display=False):
    title = _get_h5field(h5source, f"{scan_number}.1/title")
    title_sections = title.split()
    
    info_dict = {
        "scan_type"     :       title_sections[0],
        "slow_axis"     :       title_sections[1],
        "motor1_ipos"   : float(title_sections[2]),
        "motor1_fpos"   : float(title_sections[3]),
        "motor1_npoints":   int(title_sections[4]) + 1,
        "fast_axis"     :       title_sections[5],
        "motor2_ipos"   : float(title_sections[6]),
        "motor2_fpos"   : float(title_sections[7]),
        "motor2_npoints":   int(title_sections[8]) + 1,
        "count_time"    :       title_sections[9]
    }
    
    for key, value in info_dict.items():
        print(f"{key:}")
    
    return info_dict
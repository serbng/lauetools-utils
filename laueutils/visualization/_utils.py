import numpy as np
import matplotlib as mpl
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

def cbarticks_zero_centered(vmin, vmax): 
    """
    Generate a TwoSlopeNorm normalization (centered at zero) and corresponding ticks for the colorbar. 
    If vmin and vmax have the same sign, the colorbar will be onesided.
    
    Parameters
    ----------
    vmin [float]: Minimum value for the data.
    vmax [float]: Maximum value for the data.
        
    Returns
    -------
    norm   [TwoSlopeNorm]: Normalization object with center zero.
    cbar_ticks [np.array]: Array of ticks for the colorbar.
    """
    if vmax < vmin:
        raise ValueError("vmin and vmax must be in ascending order")
    
    factors = [0.25, 0.5, 0.75, 1]
    positive_ticks = np.array([vmax*f for f in factors])
    negative_ticks = np.array([vmin*f for f in factors[::-1]])
    
    # If vmin and vmax have the same sign, adjust the onesided colorbar
    if vmin >= 0:
        vmin = -1e-8
        negative_ticks = np.array([])
    if vmax <= 0:
        vmax = +1e-8
        positive_ticks = np.array([])
        
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
    cbar_ticks = np.concatenate((negative_ticks, 0, positive_ticks), axis=None)
    
    return norm, cbar_ticks

def cbarticks_scientific():
    """
    Return proper format to display the colorbar ticks with scientific notation

    Example:
    ...
    fig, ax = plt.subplots()
    image = ax.imshow(data)
    format = cbarticks_scientific()
    plt.colorbar(image, format=format)
    # Or
    # draw_colorbar(image, format=format)
    """
    format = ScalarFormatter(useMathText=True)
    format.set_powerlimits((0, 0))
    return format # to use in format **kwarg of plt.colorbar()

def draw_colorbar(image, width=5, pad=0.1, **kwargs):
    """
    Make colorbars with the same width for a given image
    
    Parameters
    -----------
    image  [matplotlib.cm.ScalarMappable]: Ex.: AxesImage, ContourSet, QuadMesh... 
                                           Object returned by any matplotlib plotting function
                                           that supports a colorbar
    axis           [matplotlib.axes.Axes]: where the plot is located
    
    Keyword arguments
    -----------
    scientific                     [bool]: Format ticks using scientific notation
    """
    # scientific = kwargs.pop('scientific', None)
    # if scientific:
    #     format=cbarticks_scientific()
    
    divider = make_axes_locatable(image.axes)
    cbar_ax = divider.append_axes('right', f'{width}%', pad = pad)
    orient  = kwargs.pop('orientation', 'vertical')
   
    colorbar(image, cax=cbar_ax, orientation=orient, **kwargs)

def apply_filters(data, filters):
    """
    Applies filters to the data, setting values to np.nan 
    if they do not meet the specified condition.

    Parameters:
        data (array-like): Array of data to be filtered.
        filters (tuple): A tuple containing:
            - A list or array of numbers to evaluate.
            - A threshold value for filtering.

    Returns:
        np.ndarray: Array with filtered values set to np.nan.
    """
    if filters is None:
        return data  # Return data unchanged if no filters are provided.

    condition, threshold = filters    
    data = np.where(condition < threshold, np.nan, data)
    return data

def limits_meanNstd(data, std_mul):
    """
    Returns the tuple (mean - std_mul * std, mean + std_mul * std) computed on data values.
    
    Parameters
    ----------
    data    [np.ndarray]: array of values, can contain np.nan.
    std_mul      [float]: std scaling value.
          
    Returns
    -------
    limits       [tuple]: The computed range values.
    """
    if std_mul == 0:
        return None, None

    mean_val = np.nanmean(data)
    std_val  = np.nanstd(data)
    return mean_val-std_mul*std_val, mean_val+std_mul*std_val

def apply_style(style):
    if style not in ["jupyter_dark", "jupyter_light", "default", "atom_dark", "jupyter_night"]:
        print("Don't know that style")
        return
    
    #mpl.rcParams["font.family"] = "monospace"
    #mpl.rcParams["font.monospace"] = ["FreeMono"]
    
    if style == "jupyter_dark":
        mpl.style.use('dark_background')
        mpl.rcParams['figure.facecolor'] = '#111111'
        mpl.rcParams['axes.facecolor']   = '#FFFFFF'

    elif style == "jupyter_night":
        mpl.style.use('dark_background')
        mpl.rcParams['figure.facecolor'] = '#04080C'
        mpl.rcParams['axes.facecolor']   = '#FFFFFF'
    
    elif style in ["jupyter_light", "default"]:
        mpl.style.use('default')
        
    elif style == "atom_dark":
        style_params = {
        'text.color':         '#ACB2BE',
        'xtick.color':        '#ACB2BE', # color of the ticks
        'xtick.labelcolor':   'inherit', # color of the tick labels or inherit from xtick.color
        'ytick.color':        '#ACB2BE', # color of the ticks
        'ytick.labelcolor':   'inherit', # color of the tick labels or inherit from xtick.color
        'axes.facecolor':     '#2D3139', # axes background color
        'axes.edgecolor':     '#ACB2BE', # axes edge color
        'axes.linewidth':     0.8      , # edge line width
        'axes.grid':          False    , # display grid or not
        'axes.grid.axis':     'both'   , # which axis the grid should apply to
        'axes.grid.which':    'major'  , # grid lines at {major, minor, both} ticks
        'axes.titlelocation': 'center' , # alignment of the title: {left, right, center}
        'axes.titlesize':     'large'  , # font size of the axes title
        'axes.titleweight':   'normal' , # font weight of title
        'axes.titlecolor':    '#ACB2BE', # color of the axes title, auto falls back to text.color as default value
        'axes.labelcolor':    '#ACB2BE',
        'figure.titlesize':   'large'  , # size of the figure title (``Figure.suptitle()``)
        'figure.titleweight': 'normal' , # weight of the figure title
        'figure.labelsize':   'large'  , # size of the figure label (``Figure.sup[x|y]label()``)
        'figure.labelweight': 'normal' , # weight of the figure label
        'figure.figsize':     (6.4, 4.8),# figure size in inches
        'figure.dpi':         100,       # figure dots per inch
        'figure.facecolor':   '#292C33', # figure face color
        'figure.edgecolor':   'white'  , # figure edge color
        'figure.frameon':     True       # enable figure frame
        }
        mpl.style.use(style_params)

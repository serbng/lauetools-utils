import numpy as np
import h5py
import fabio
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

from laueutils.classes.fitfileseries import FitFile, FitFileSeries
from ._utils import draw_colorbar

def __gaussian__(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

#        --- USE THE .H5 FILE TO READ MOTOR POSITIONS ---
#
# A minimal working example on how to use in an equivalent way the h5
# file is:
#
# import h5py
# with h5py.File('path/to/h5') as f:
#    xech = f['scan_nb/measurement/xech'][:]
#    yech = f['scan_nb/measurement/yech'][:]
#    
#    xech = ((xech - xech[0]) * 1e3).reshape(nb_rows,nb_cols)
#    yech = ((yech - yech[0]) * 1e3).reshape(nb_rows,nb_cols)

def display_image(image_path: str, ROI: tuple = None, **kwargs) -> None:
    """Plot an image.
    
    Parameters
    ----------
    image_path str       : Full path to the image
    ROI        tuple[int]: Subset of pixels inside the image to plot. The format is:
                           (x_position, y_position, x_boxsize, y_boxsize)
    
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.imshow
    """
    image_data = fabio.open(image_path).data
    
    if ROI is not None:
        xindex1 = ROI[0] - (ROI[2] // 2)
        xindex2 = ROI[0] + (ROI[2] // 2)
        yindex1 = ROI[1] - (ROI[3] // 2)
        yindex2 = ROI[1] + (ROI[3] // 2)
                
        image_data = image_data[xindex1:xindex2, yindex1:yindex2]
    
    ax = plt.gca()
    image = ax.imshow(image_data, **kwargs)
    draw_colorbar(image, ax)
    ax.set_title(image_path.split('/')[-1])
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_aspect('equal')    

# def plot_pattern(data_source, axis=None, **kwargs):
#     """Plot Laue pattern peak positions
    
#     Parameters
#     ----------
#     data_source: Can be
#                  * Full path to .dat file 
#                  * DatFile object
#                  * numpy.ndarray (n, 2) containing (x, y) positions as coloumns
                 
#     Keyword arguments
#     ----------
#     kwargs: passed to matplotlib.pyplot.scatter
#     """
#     if isinstance(data_source, str):
#         data = PeakList(data_source)
#         peaks_data = np.vstack((data.x_position, 
#                                 data.y_position)).T
#     elif isinstance(data_source, PeakList):
#         peaks_data = np.vstack((data_source.x_position, 
#                                 data_source.y_position)).T
#     elif isinstance(data_source, np.ndarray):
#         peaks_data = data_source
#     else:
#         raise ValueError(
#             f'Unrecognized data_source type. Type must be str or numpy.array, given {type(data_source)}'
#             )
    
#     if axis is None:
#         fig, axis = plt.subplots(figsize = (6,6))
#     axis.scatter(peaks_data[:,0], peaks_data[:,1], **kwargs)
#     axis.set_xlabel('X pixel')
#     axis.set_ylabel('Y pixel')
#     axis.set_aspect('equal')
#     axis.invert_yaxis()
#     print(f'Number of spots: {peaks_data.shape[0]}')

def plot_indexed_pattern(data_source, 
                         axis = None, 
                         fontsize: int = 6, **kwargs):
    """Plot Laue pattern peak positions with corresponding [h,k,l] label
    
    Parameters
    ----------
    data_source: Can be
                 * Full path to .fit file
                 * FitFile object
                 * Dictionary whose keys are in the form 'h k l' and values are the rows of
                   the .fit file containing, in order
                   spot_index intensity h k l 2theta Chi Xexp Yexp Energy GrainIndex PixDev
                   
    axis  matplotlib.axes.Axes: matplotlib axis onto which to plot. Defaults to None.
    fontsize               int: font size of the labels.
                   
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.scatter
    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(6,6))
        
    if isinstance(data_source, str):
        ff = FitFile(data_source)
        peaks_data = ff.peak
    elif isinstance(data_source, FitFile):
        peaks_data = data_source.peak
    elif isinstance(data_source, dict):
        peaks_data = data_source
    else:
        raise(TypeError, f'Unrecognized type of data_source. Expected either str, dict, FitFile. Given {type(data_source)}')
    
    for key in peaks_data.keys():
        x_position = peaks_data[key][7]
        y_position = peaks_data[key][8]
        
        axis.scatter(x_position, y_position, **kwargs)
        
        #TO DO: IMPROVE THE DECISION BASED ON THE FRAMESIZE FROM CCDlabel IN DICTLAUETOOLS
        bottom_left_quadrant  = x_position <  1024 and y_position >= 1024
        bottom_right_quadrant = x_position >= 1024 and y_position >= 1024
        top_left_quadrant     = x_position <  1024 and y_position <  1024
        top_right_quadrant    = x_position >= 1024 and y_position <  1024
        
        xshift, yshift = 0, 0
        
        if bottom_left_quadrant:
            ha, va = 'right', 'bottom'
            xshift = -5
            yshift = -5
        elif bottom_right_quadrant:
            ha, va = 'left', 'bottom'
            xshift =  5
            yshift = -5
        elif top_left_quadrant:
            ha, va = 'right', 'top'
            xshift = -5
            yshift = +5
        elif top_right_quadrant:
            ha, va = 'left', 'top'
            xshift =  5
            yshift = +5
        
        axis.text(x_position + xshift, 
                  y_position + yshift, 
                  f'[{key}]', 
                  ha = ha, va = va, fontsize = fontsize, color = kwargs.get('color', 'black'))
        
    axis.set_xlim(0,2016)
    axis.set_ylim(0,2018)
    axis.invert_yaxis()
    axis.set_aspect('equal')
    
def number_indexed_spots(sample_x: np.ndarray, 
                         sample_y: np.ndarray,
                         ffs: FitFileSeries, 
                         axis = None,
                         reshape_order = 'C',
                         **kwargs):
    """Plot the number of indexed spots for all positions in a scan.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    axis  matplotlib.axes.Axes: matplotlib axis onto which to plot. Defaults to None
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    if axis is None:
        fig, axis = plt.subplots(figsize = (5,5))
        
    data_shape = sample_x.shape
    data = ffs.number_indexed_spots.reshape(data_shape, order = reshape_order)
    
    image = axis.pcolormesh(sample_x, sample_y, data, **kwargs)
    draw_colorbar(image, axis)

    axis.set_xlabel('Position [µm]')
    axis.set_ylabel('Position [µm]')
    axis.set_aspect('equal')
    axis.set_title('Number of indexed spots')
    
def mean_pixel_deviation(sample_x: np.ndarray, 
                         sample_y: np.ndarray,
                         ffs: FitFileSeries, 
                         axis = None,
                         reshape_order = 'C',
                         **kwargs):
    """Plot the mean pixel deviation for all positions in a scan.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    axis  matplotlib.axes.Axes: matplotlib axis onto which to plot. Defaults to None
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    if axis is None:
        fig, axis = plt.subplots(figsize = (5,5))
        
    data_shape = sample_x.shape
    data = ffs.mean_pixel_deviation.reshape(data_shape, order = reshape_order)
    
    image = axis.pcolormesh(sample_x, sample_y, data, **kwargs)
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes('right', '5%', pad = 0.1)
    plt.colorbar(image, cax = cbar_ax, orientation = 'vertical')

    axis.set_xlabel('Position [µm]')
    axis.set_ylabel('Position [µm]')
    axis.set_aspect('equal')
    axis.set_title('Mean pixel deviation')
        

def peak_position(sample_x: np.ndarray, 
                  sample_y: np.ndarray, 
                  ffs: FitFileSeries, 
                  miller_index: str, 
                  space: str = 'camera',
                  relative_position: bool = False,
                  reshape_order = 'C',
                  **kwargs) -> tuple:
    """Plot the (x, y) peak position for a given [h,k,l] reflection throughout the scan. When in an
    image the specified reflection is not found, no color is added.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    miller_index           str: Miller index of the reflection, in the form 'h k l'.
    space                  str: Must be in {'camera', 'twothetachi'}. Specifies whether to represent
                                the peak position in the (x, y) camera plane or in the (2theta, chi)
                                space. Defaults to 'camera'.
    relative_position     bool: Remove the mean value from the data.
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    fig, ax = plt.subplots(1,2, figsize = (9, 5))
    
    data_shape = sample_x.shape
    if space == 'camera':
        data0 = ffs.x_position(miller_index).reshape(data_shape, order = reshape_order)
        data1 = ffs.y_position(miller_index).reshape(data_shape, order = reshape_order)
        titles = ['X pixel', 'Y pixel']
        
    elif space == 'twothetachi':
        data0 = ffs.chi(   miller_index).reshape(data_shape)
        data1 = ffs.ttheta(miller_index).reshape(data_shape)
        titles = ['chi', '2theta']
    else:
        raise NameError("Invalid plot type. Accepted type arguments for 'space' are either 'pixel' or 'twothetachi'")
        
    if relative_position:
        data0 -= np.nanmean(data0)
        data1 -= np.nanmean(data1)
        
    for axis, datum, title in zip(ax, [data0, data1], titles):
        image = axis.pcolormesh(sample_x, sample_y, datum, **kwargs)
        axis.set_xlabel('Position [µm]')
        axis.set_ylabel('Position [µm]')
        axis.set_aspect('equal')
        axis.set_title(title)
        
        divider = make_axes_locatable(axis)
        cbar_ax = divider.append_axes('right', '5%', pad = 0.1)
        plt.colorbar(image, cax = cbar_ax, orientation = 'vertical')

    fig.tight_layout()

def peak_intensity(sample_x: np.ndarray, 
                   sample_y: np.ndarray, 
                   ffs: FitFileSeries, 
                   miller_index: str, 
                   reshape_order = 'C',
                   **kwargs) -> tuple:
    """Plot the intensity of a given [h,k,l] reflection throughout the scan. When in an image the
    specified reflection is not found, no color is added.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    miller_index           str: Miller index of the reflection, in the form 'h k l'.
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    fig, axis = plt.subplots(figsize = (5,5))
    
    data_shape = sample_x.shape
    data = ffs.intensity(miller_index).reshape(data_shape, order = reshape_order)
    
    image = axis.pcolormesh(sample_x, sample_y, data, **kwargs)
    axis.set_xlabel('Position [µm]')
    axis.set_ylabel('Position [µm]')
    axis.set_aspect('equal')
    axis.set_title(f'[{miller_index}] peak intensity')
    
    divider = make_axes_locatable(axis)
    cbar_ax = divider.append_axes('right', '5%', pad = 0.1)
    plt.colorbar(image, cax = cbar_ax, orientation = 'vertical')
    
def euler_angles(ffs: FitFileSeries, **kwargs) -> tuple:
    """Plot the Euler angles phi, theta, psi, of the crystal linearly along the scan.
    
    Parameters
    ----------
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    miller_index           str: Miller index of the reflection, in the form 'h k l'.

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.scatter        
    """
    fig, axes = plt.subplots(1,3, figsize = (12, 4))
    
    data  = [ffs.euler_angles[:, i] for i in range(3)] 
    titles = ['Phi', 'Theta', 'Psi']
    
    for axis, datum, title in zip(axes, data, titles):
        axis.scatter(np.arange(0, ffs.nb_files), datum, **kwargs)
        axis.set_xlabel('Image index')
        axis.set_ylabel('Angle [deg]')
        axis.set_title(title)
    
    fig.tight_layout()
    
    return fig, axes

def euler_angles_2d(sample_x: np.ndarray, 
                    sample_y: np.ndarray, 
                    ffs: FitFileSeries, 
                    reshape_order = 'C', 
                    **kwargs):
    """Plot the Euler angles phi, theta, psi, of the crystal along the scan.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    fig, axes = plt.subplots(1,3, figsize = (12, 4))
    
    data_shape = sample_x.shape
    data   = [ffs.euler_angles[:,i].reshape(data_shape, order = reshape_order) for i in range(3)] 
    titles = ['Phi', 'Theta', 'Psi']
    
    for axis, datum, title in zip(axes, data, titles):
        image = axis.pcolormesh(sample_x, sample_y, datum, **kwargs)
        axis.set_xlabel('Position [µm]')
        axis.set_ylabel('Position [µm]')
        axis.set_aspect('equal')
        axis.set_title(title)
        
        divider = make_axes_locatable(axis)
        cbar_ax = divider.append_axes('right', '5%', pad = 0.1)
        plt.colorbar(image, cax = cbar_ax, orientation = 'vertical')
    
    fig.suptitle('Euler Angles')
    fig.tight_layout()

def strain_map(sample_x: np.ndarray, 
               sample_y: np.ndarray, 
               ffs: FitFileSeries,
               multiplier:  float = 1e4, 
               scale:         str = 'default',
               reshape_order: str = 'C',
               summary:      bool = False,
               **kwargs):
    """Plot the six strain components εxx, εyy, εzz, εxy, εxz, εyz.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_x        np.ndarray: 2D mesh of the x positions of the yech motor
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    multiplier           float: Value multiplied to each strain component. Defaults to 1e4
    scale                  str: Must be in {'default', 'mean3sigma', 'uniform', 'other'}.
                                * 'default'    lets matplotlib lets matplotlib decide the scales for
                                               each plot: each plot will have a colorbar between its
                                               minimum and maximum value.
                                * 'mean3sigma' each plot will have a colorbar between mean - 3sigma
                                               and mean + 3sigma.
                                               The value 3 can be substituted with any value and is
                                               parsed as a float. Example 'mean4.5sigma' will work.
                                * 'uniform'    The same colorbar will be applied to all components.
                                               the limits are between +- max(np.abs(components)).
                                * 'other'      Custom values for the colorbar limits will be applied
                                               to all plots. vmin and vmax kwargs must be provided.
    reshape_order          str: Must be in {'C', 'F', 'A'}. Refer to numpy.reshape documentation. 
                                Defaults to 'C'.
                                Notes:
                                The data extracted from the FitFileSeries object is reshaped to
                                match the sample_x shape. This variable is passed to numpy.reshape
                                to specify how to reshape it.
                                https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    summary               bool: Print a table summarizing some notable values for each component.
                                (Mean, STD, Mean+3*STD, Mean-3*STD, Max, Min)
                                
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.pcolormesh        
    """
    
    fig, axes = plt.subplots(2,3, figsize = (11, 6))
    
    normalized_strain = ffs.deviatoric_strain_crystal_frame * multiplier
    voigt_indices     = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    strain_components = [normalized_strain[:, index[0], index[1]] for index in voigt_indices]
    titles = ['ε$_{xx}$', 'ε$_{yy}$', 'ε$_{zz}$', 'ε$_{xy}$', 'ε$_{xz}$', 'ε$_{yz}$']
    
    # indices of the xx, yy, zz, xy, xz, yz components in the flattened matrix
    indices = [0, 4, 8, 1, 2, 5]
    means    = np.nanmean(normalized_strain, axis = 0).flatten()[indices]
    std_devs = np.nanstd( normalized_strain, axis = 0).flatten()[indices]
    maxima   = np.nanmax( normalized_strain, axis = 0).flatten()[indices]
    minima   = np.nanmin( normalized_strain, axis = 0).flatten()[indices]
    
    if summary:
        # Print table with strain components ranges
        table_col  = ['µ', 'σ', 'µ - 3σ', 'µ + 3σ', 'Minimum', 'Maximum']
        table_row  = ['ε_xx', 'ε_yy', 'ε_zz', 'ε_xy', 'ε_xz', 'ε_yz']
        table_data = np.vstack((means, 
                               std_devs, 
                               means - 3*std_devs,
                               means + 3*std_devs,
                               minima,
                               maxima)).T

        print(format_table(table_col, table_row, table_data))
    
    if scale == 'default':
        plotvmax = np.repeat(None, 6)
        plotvmin = plotvmax
    
    elif scale.startswith('mean') and scale.endswith('sigma'):
        std_mul  = float(scale.split('mean')[-1].split('sigma')[0]) 
        plotvmax = [means[i] + std_mul*std_devs[i] for i in range(6)]
        plotvmin = [means[i] - std_mul*std_devs[i] for i in range(6)]
        
    elif scale == 'uniform':
        plotvmax = np.abs([minima.min(), maxima.max()]).max()
        plotvmin = - plotvmax

        plotvmin = np.repeat(plotvmin, 6)
        plotvmax = np.repeat(plotvmax, 6)
    
    elif scale == 'other':
        try:
            plotvmin = np.repeat(kwargs.pop('vmin'), 6)
            plotvmax = np.repeat(kwargs.pop('vmax'), 6)
        except KeyError:
            raise KeyError("If you select scale = 'other' you must specify vmin and vmax")
    
    else:
        raise Exception("scale must be in the list ['default', 'meanNsigma', 'uniform', 'other']")
    
    for axidx, data, title, plotmin, plotmax in zip(np.ndindex(axes.shape), 
                                                    strain_components, 
                                                    titles, 
                                                    plotvmin, 
                                                    plotvmax):
        
        image = axes[axidx].pcolormesh(sample_x, 
                                       sample_y, 
                                       data.reshape(sample_x.shape, order = reshape_order), 
                                       vmin = plotmin, vmax = plotmax, **kwargs)
        
        divider = make_axes_locatable(axes[axidx])
        cbar_ax = divider.append_axes('right', '5%', pad = 0.1)
        plt.colorbar(image, cax = cbar_ax, orientation = 'vertical')
        if axidx[1] == 0:
            axes[axidx].set_ylabel('Position [µm]')
        if axidx[0] == axes.shape[0] - 1:
            axes[axidx].set_xlabel('Position [µm]')
        axes[axidx].set_title(title)
        axes[axidx].set_aspect('equal')
        
    fig.tight_layout()
    

def strain_histogram(ffs: FitFileSeries, 
                     multiplier: float = 1e4,
                     fit: bool = False, **kwargs):
    """Plot the histogram of the six strain components εxx, εyy, εzz, εxy, εxz, εyz.
    
    Parameters
    ----------
    ffs          FitFileSeries: Objected containing the information of a folder of parsed .fit files
    multiplier           float: Value multiplied to each strain component. Defaults to 1e4
    fit                   bool: Fit the histogram with a gaussian. The fit parameters will be printed
                                as the title of each subplot.
    
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.hist   
    """
    
    normalized_strain = ffs.deviatoric_strain_crystal_frame * multiplier
    voigt_indices     = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    strain_components = [normalized_strain[:, index[0], index[1]] for index in voigt_indices]
    
    xlabels = ['ε$_{xx}$', 'ε$_{yy}$', 'ε$_{zz}$', 'ε$_{xy}$', 'ε$_{xz}$', 'ε$_{yz}$']
    
    fig, axes = plt.subplots(2, 3, figsize = (11, 7))
    
    for axidx, data, xlabel in zip(np.ndindex(axes.shape), 
                                   strain_components, 
                                   xlabels):
        
        counts, bins, patches = axes[axidx].hist(data, **kwargs)
        
        if fit:       
            # Compute the bins centers. Used when evaluating the fitting function (__gaussian__)
            bin_centers = bins[:-1] + np.diff(bins)/2
            # Compute the fit_params [A, mu, sigma], returned covariance is trashed   
            fit_params, _ = curve_fit(__gaussian__, bin_centers, counts, p0 = (50, 0, 2))
            
            # Plot result
            xlims = axes[axidx].get_xlim()
            xvals = np.linspace(xlims[0], xlims[1], 200)
            
            axes[axidx].plot(xvals, __gaussian__(xvals, *fit_params), color = 'red', linewidth = 2)
            axes[axidx].set_title(f'A = {fit_params[0]:.1f}, μ = {fit_params[1]:.2f}, σ = {fit_params[2]:.2f}')
    
        axes[axidx].set_xlabel(xlabel)
        if axidx[1] == 0:
            axes[axidx].set_ylabel('Counts')
        
    fig.tight_layout()

def LatticeParamsMap(xech: np.ndarray, 
                     yech: np.ndarray, 
                     ffs: FitFileSeries, **kwargs):
    
    pass
    
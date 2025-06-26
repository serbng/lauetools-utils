import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit
from ..classes import FitFileSeries
from ..utils import tabular_data
from ._utils import (draw_colorbar, 
                     cbarticks_zero_centered, 
                     cbarticks_scientific,
                     limits_meanNstd)

def __gaussian__(x, A, mu, sigma):
    return A * np.exp(- (x - mu)**2 / (2 * sigma**2))

def strain_map(x, y, strain_tensors, **kwargs):
    """
    Plot six strain component maps (exx eyy, ezz, eₓxy, exz, and eyz) using a list of 3×3 strain tensors.
    
    Parameters
    ----------
    x                  [np.ndarray]: x-coordinate values (2D grid NxM).
    y                  [np.ndarray]: y-coordinate values (2D grid NxM).
    strain_tensors list[np.ndarray]: list of 3×3 strain tensors corresponding to each grid point.
    
    Keyword Arguments
    -----------------
    multiplier            [float]: Factor to multiply each strain component.  Default 1.0.
    reshape_order {'C', 'F', 'A'}: Order for numpy.reshape to match the grid shape. Default 'C'.
        scale {'mean[sigma_value]sigma', 'uniform', 'default'}: default 'default'
            Color scaling mode:
              - 'mean[sigma_value]sigma': Set limits as mean ± (sigma_value × std). E.g., 'mean3sigma'.
              - 'uniform': Apply same limits to all components, computed as ±max(|component|) across all components.
    cbar_width            [float]: Width of the colorbar, in percentage with respect to the axes width. Default 5
    cbar_norm              [bool]: If True, center the colorbar at zero. Default False
    ncols                   [int]: Number of columns in the subplots
    nrows                   [int]: Number of rows in the subplots
    figsize       tuple[int, int]: Size in inches of the figure
        
    Additional kwargs are passed to matplotlib.pyplot.pcolormesh.
    
    Returns
    -------
    fig   [matplotlib.figure.Figure]: The figure object containing the subplots.
    axes                [np.ndarray]: Flattened array of subplot axes.
    """
    # Extract custom keyword arguments.
    multiplier     = kwargs.pop('multiplier', 1.0)
    reshape_order  = kwargs.pop('reshape_order', 'C')
    scale          = kwargs.pop('scale', 'default')
    cbar_width     = kwargs.pop('cbar_width', 5)
    cbar_norm      = kwargs.pop('cbar_norm', False)
    figsize        = kwargs.pop('figsize', (12,10))
    ncols          = kwargs.pop('ncols', 3)
    nrows          = kwargs.pop('nrows', 2)
    cbar_ticks     = None # if cbar_norm = True, this value will be overwritten
    
    # Ensure the number of strain tensors matches the grid size.
    npoints = len(strain_tensors)
    if npoints != x.size:
        raise ValueError("Length of strain_tensors does not match the total number of grid points in x/y.")
    
    # Define the strain component keys and their respective indices.
    components_index = {'xx': (0, 0), 'yy': (1, 1),
                        'zz': (2, 2), 'xy': (0, 1),
                        'xz': (0, 2), 'yz': (1, 2)}
    
    # Extract and reshape each strain component.
    components = {}
    for comp, index in components_index.items():
        components[comp] = np.array([tensor[index] * multiplier for tensor in strain_tensors])
        components[comp] = components[comp].reshape(x.shape, order=reshape_order) 
    

    if scale == 'uniform':
        global_max = max([np.nanmax(np.abs(component[comp])) for comp in components_index.keys()])
        vmin, vmax = (-global_max, global_max)

    # Create a figure with six subplots arranged in 2 rows x 3 columns.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    # Loop over each strain component and generate a plot.
    for i, (component, value) in enumerate(components.items()):
        # Start with a copy of the remaining kwargs for pcolormesh.
        singleplot_kwargs = kwargs.copy()
        
        if scale == 'default':
            vmin = kwargs.get('vmin', np.nanmin(value))
            vmax = kwargs.get('vmax', np.nanmax(value))
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})
            
        if scale.startswith('mean') and scale.endswith('sigma'):
            try:
                std_mul = float(scale[len('mean'): -len('sigma')])
            except ValueError:
                std_mul = 3.0
            vmin, vmax = limits_meanNstd(value, std_mul)
            singleplot_kwargs.update({'vmin': vmin, 'vmax': vmax})
            
        # If cbar_norm is enabled, center the colorbar using a custom norm and put vmin and vmax to None
        if cbar_norm:
            # Determine normalization limits either from provided values or data.  
            norm, cbar_ticks = cbarticks_zero_centered(vmin, vmax)
            singleplot_kwargs['norm'] = norm
            singleplot_kwargs['vmin'] = None
            singleplot_kwargs['vmax'] = None
        
        # Create the pcolormesh plot.
        mesh = axes[i].pcolormesh(x, y, value, **singleplot_kwargs)
        axes[i].set_title(f"$\\varepsilon_{{{component}}}$")
        axes[i].set_aspect('equal')
                           
        draw_colorbar(mesh, width=cbar_width, ticks=cbar_ticks)
    
    fig.suptitle(f"Deviatoric strain components ($\\times 10^{{{-np.log10(multiplier):3.1f}}}$)")
    fig.tight_layout()
    return fig, axes

def strain4map(sample_x: np.ndarray, 
               sample_y: np.ndarray, 
               ffs: FitFileSeries,
               multiplier: float = 1e4, 
               scale: str = 'default',
               reshape_order: str = 'C',
               summary: bool = False,
               cbar_norm: bool = False,
               **kwargs):
    """Plot four strain components: (εxx+εyy)/2, εzz, εxy, (εxz+εyz)/2.
    
    Parameters
    ----------
    sample_x        np.ndarray: 2D mesh of the x positions of the xech motor
    sample_y        np.ndarray: 2D mesh of the y positions of the yech motor
    ffs          FitFileSeries: Object containing the information of a folder of parsed .fit files
    multiplier           float: Value multiplied to each strain component. Defaults to 1e4
    scale                  str: Defines the colorbar scale. Options: 'default', 'meanNsigma', 'uniform', 'other'.
    reshape_order          str: Defines the reshape order for the data array. Defaults to 'C'.
    summary               bool: If True, prints a summary table with statistical values.
    cbar_norm             bool: If True, centers the colorbar at 0.
    
    Keyword arguments
    ----------
    **kwargs : dict - passed to matplotlib.pyplot.pcolormesh
    
        Additional options:
        - cbar_width : str, optional
            Defines the colorbar size as a percentage of the plot. Default is '10%'.
    
    Returns:
    --------
    tuple
        (fig, axes) matplotlib figure and axis objects.
    """
    
    cbar_width = kwargs.pop('cbar_width', '10')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 grid for four components
    
    normalized_strain = ffs.deviatoric_strain_crystal_frame * multiplier
    
    # Extracting strain components
    ε_xx = normalized_strain[:, 0, 0]
    ε_yy = normalized_strain[:, 1, 1]
    ε_zz = normalized_strain[:, 2, 2]
    ε_xy = normalized_strain[:, 0, 1]
    ε_xz = normalized_strain[:, 0, 2]
    ε_yz = normalized_strain[:, 1, 2]
    
    # Computing the required strain components
    strain_components = [
        (ε_xx + ε_yy) / 2,   # Mean in-plane strain
        ε_zz,                # Out-of-plane strain
        ε_xy,                # Shear component
        (ε_xz + ε_yz) / 2    # Mean out-of-plane shear
    ]
    
    titles = [r'$(\varepsilon_{xx} + \varepsilon_{yy})/2$', r'$\varepsilon_{zz}$', r'$\varepsilon_{xy}$', r'$(\varepsilon_{xz} + \varepsilon_{yz})/2$']
    

    means = np.nanmean(strain_components, axis=1)
    std_devs = np.nanstd(strain_components, axis=1)
    maxima = np.nanmax(strain_components, axis=1)
    minima = np.nanmin(strain_components, axis=1)

    if summary:
        table_col = ['µ', 'σ', 'µ - 3σ', 'µ + 3σ', 'Min', 'Max']
        table_row = ['(εxx+εyy)/2', 'εzz', 'εxy', '(εxz+εyz)/2']
        table_data = np.vstack((means, 
                                std_devs, 
                                means - 3 * std_devs,
                                means + 3 * std_devs,
                                minima,
                                maxima)).T

        print(tabular_data(table_col, table_row, table_data))

    # Define color scale ranges
    if scale == 'default':
        plotvmax = [None] * 4
        plotvmin = [None] * 4
    
    elif scale.startswith('mean') and scale.endswith('sigma'):
        std_mul = float(scale.split('mean')[-1].split('sigma')[0]) 
        plotvmax = [means[i] + std_mul * std_devs[i] for i in range(4)]
        plotvmin = [means[i] - std_mul * std_devs[i] for i in range(4)]
        
    elif scale == 'uniform':
        plotvmax = np.abs([minima.min(), maxima.max()]).max()
        plotvmin = -plotvmax

        plotvmin = [plotvmin] * 4
        plotvmax = [plotvmax] * 4
    
    elif scale == 'other':
        try:
            plotvmin = [kwargs.pop('vmin')] * 4
            plotvmax = [kwargs.pop('vmax')] * 4
        except KeyError:
            raise KeyError("If you select scale = 'other', you must specify vmin and vmax")
    
    else:
        raise ValueError("scale must be in the list ['default', 'meanNsigma', 'uniform', 'other']")

    # Plot each strain component
    for axidx, data, title, plotmin, plotmax in zip(np.ndindex(axes.shape), 
                                                    strain_components, 
                                                    titles, 
                                                    plotvmin, 
                                                    plotvmax):

        norm = TwoSlopeNorm(vmin=plotmin, vcenter=0, vmax=plotmax) if cbar_norm else None
        cbar_ticks = np.round([plotmin+0.1, plotmin/2, 0, plotmax/2, plotmax-0.1], 1) if cbar_norm else None
        
        image = axes[axidx].pcolormesh(sample_x, 
                                       sample_y, 
                                       data.reshape(sample_x.shape, order=reshape_order), 
                                       vmin=None if cbar_norm else plotmin, 
                                       vmax=None if cbar_norm else plotmax,
                                       norm=norm,
                                       **kwargs)
        
        draw_colorbar(image, cbar_width, ticks=cbar_ticks)

        if axidx[1] == 0:
            axes[axidx].set_ylabel('Position [µm]')
        if axidx[0] == axes.shape[0] - 1:
            axes[axidx].set_xlabel('Position [µm]')
        axes[axidx].set_title(title)
        axes[axidx].set_aspect('equal')
        
    fig.tight_layout()

    return fig, axes
    
    

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


def plot_strain_for_slice(ffs: FitFileSeries, sample_x: np.ndarray, slice_range: tuple):
    """
    Plots the mean strain values (εzz and (εxx + εyy)/2) per column 
    for a given range of x positions.

    Parameters
    ----------
    ffs : FitFileSeries
        Object containing the information of a folder of parsed .fit files.
    sample_x : np.ndarray
        2D mesh of the x positions of the xech motor.
    slice_range : tuple
        Range of indices to slice the x positions and strain data.
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects.
    """
    
    normalized_strain = ffs.deviatoric_strain_crystal_frame 
    
    # Extracting strain components and reshaping
    ε_xx = normalized_strain[:, 0, 0].reshape(sample_x.shape, order='F')
    ε_yy = normalized_strain[:, 1, 1].reshape(sample_x.shape, order='F')
    ε_zz = normalized_strain[:, 2, 2].reshape(sample_x.shape, order='F')

    # Computing the required strain components
    mean_in_plane_strain = (ε_xx + ε_yy) / 2  # Mean in-plane strain
    strain_components = [mean_in_plane_strain, ε_zz]  # List of required strain components
    
    # Compute mean strain components per column
    strain_mean = [np.nanmean(component, axis=0) for component in strain_components]
    
    # Extract x positions and strain values for the given slice range
    x = sample_x[0, slice_range[0]:slice_range[1]]  # Taking first row since x positions are the same for each column
    epsilon_xx_yy_vals = strain_mean[0][slice_range[0]:slice_range[1]]
    epsilon_zz_vals = strain_mean[1][slice_range[0]:slice_range[1]]

    # Plot experimental strain values
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, epsilon_zz_vals * 100, label=r'Experimental $\epsilon_{zz}$', color='orange', linewidth=2)
    ax.scatter(x, epsilon_xx_yy_vals * 100, label=r'Experimental $(\epsilon_{xx} + \epsilon_{yy}) / 2$', color='purple', linewidth=2)

    # Print mean and standard deviation
    print(f'Mean εzz: {np.mean(epsilon_zz_vals * 1e4):.2f}')
    print(f'Std εzz: {np.std(epsilon_zz_vals * 100):.2f}')
    print(f'Mean (εxx + εyy)/2: {np.mean(epsilon_xx_yy_vals * 1e4):.2f}')
    print(f'Std (εxx + εyy)/2: {np.std(epsilon_xx_yy_vals * 100):.2f}')
    
    # Axis labels and formatting
    ax.set_xlabel('Position (µm)')
    ax.set_ylabel('Strain Shell Region (%)')
    ax.legend()
    ax.grid()
    
    plt.show()
    
    return fig, ax

import ipywidgets as widgets
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker
import numpy as np
from IPython.display import display
import h5py

from ._utils import draw_colorbar


def plotfluoh5(h5path: str, 
               pathfluo: str, 
               sample_x: np.ndarray, 
               sample_y: np.ndarray,
               normalize: bool = True, 
               figsize: tuple = None, 
               scale: str = 'mean2sigma', 
               file_indices: tuple = None,
               reshape_order: str = 'F',
               **kwargs) -> np.array:
    """
    Plots fluorescence intensity data from an HDF5 file.
    
    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathfluo : str
        Path inside the HDF5 file where fluorescence data is stored.
    sample_x : np.ndarray
        2D mesh of x positions of the sample.
    sample_y : np.ndarray
        2D mesh of y positions of the sample.
    normalize : bool, optional
        If True, normalizes data using monitor values. Default is True.
    size : tuple, optional
        Figure size in inches (width, height). Default is (10, 5).
    scale : str, optional
        Controls the color scale of the plot. Must be one of:
        - 'default': Uses matplotlib default scale.
        - 'meanNsigma': Sets colorbar limits to mean ± N*sigma (e.g., 'mean2sigma').
        - 'other': Requires 'vmin' and 'vmax' as kwargs for custom limits.
    file_indices : tuple, optional
        If provided, selects a subset of data using indices (start, end).
    reshape_order : str, optional
        Order parameter for numpy.reshape. Must be in {'C', 'F', 'A'}. Default is 'C'.
    
    **kwargs : dict
        Additional keyword arguments for `pcolormesh`.
        
        The following keyword arguments can also be used for the colorbar (`cbar`): 
            - sci_notation : bool, optional. Default is False
                If True sets scientific notation in the colorbar scale
            - cbar_size: str, optional. Default is '10%'
                Defines the size of the colorbar as a percentage of the plot. Example: `cbar_size='20%'`. 
    
    Returns:
    --------
    np.array
        The reshaped fluorescence intensity data.
    matplotlib.figure.Figure
        The created figure.
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    cbar_width   = kwargs.pop('cbar_width', 5)


    with h5py.File(h5path, 'r') as h5f:
        # Load fluorescence data and normalize if needed
        data  = h5f[pathfluo][()].astype(np.float64) 
        dataset = pathfluo.split('/')[0]

        if normalize:
            mon = h5f[f'{dataset}/measurement/mon'][()]
            data /= mon / 10**2
        
        if file_indices is not None:
            data = data[file_indices[0]:file_indices[1]]
    
    mean  = np.nanmean(data)
    sigma = np.nanstd(data)
    
    print(f"""
    Data Shape: {sample_x.shape}
    Mean of Fluorescence Intensity: {mean:.4f}
    Standard deviation: {sigma:.4f}""")
    
    # Determine color scale
    if scale == 'default':
        plotvmin = None
        plotvmax = None
    elif scale.startswith('mean') and scale.endswith('sigma'):
        multiplier = float(scale.split('mean')[-1].split('sigma')[0])
        plotvmin = max(0, mean - multiplier * sigma)
        plotvmax = mean + multiplier * sigma
    elif scale == 'other':
        try:
            plotvmin = kwargs.pop('vmin') 
            plotvmax = kwargs.pop('vmax')
        except KeyError:
            raise KeyError("If scale='other', you must specify vmin and vmax.")
    else:
        raise ValueError("scale must be in ['default', 'meanNsigma', 'other']")
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(sample_x, sample_y, 
                        data.reshape(sample_x.shape, order=reshape_order),
                        vmin=plotvmin, vmax=plotvmax, **kwargs)
    
    ax.set_xlabel('Position [µm]')
    ax.set_ylabel('Position [µm]')
    ax.set_aspect('equal')
    
    
    draw_colorbar(im, width = cbar_width)
    #fig.tight_layout()
    
    return data, fig, ax



def plotxeolh5(h5path: str, 
               pathxeol: str, 
               sample_x: np.ndarray, 
               sample_y: np.ndarray,
               channel: int = None, 
               roi: tuple = None,
               pathref: str = None, 
               ref_range: tuple = None, 
               normalize: bool = True, 
               figsize: tuple = None, 
               scale: str = 'mean2sigma', 
               reshape_order: str = 'F',
               sci_notation=False,
               **kwargs) -> np.array:
    """
    Plots XEOL intensity data from an HDF5 file.
    
    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathxeol : str
        Path inside the HDF5 file where XEOL data is stored.
    sample_x : np.ndarray
        2D mesh of x positions of the sample.
    sample_y : np.ndarray
        2D mesh of y positions of the sample.
    channel : int, optional
        Selected channel for analysis. Required if `roi` is not provided.
    roi : tuple, optional
        Region of interest in the spectrum (start, end indices). If provided, sums over this range.
    pathref : str, optional
        Path inside the HDF5 file to the reference data for subtraction.
    ref_range : tuple, optional
        Range in the spectrum used to calculate reference intensity.
    normalize : bool, optional
        If True, normalizes data using monitor values. Default is True.
    size : tuple, optional
        Figure size in inches (width, height). Default is (10, 5).
    scale : str, optional
        Controls the color scale of the plot. Must be one of:
        - 'default': Uses matplotlib default scale.
        - 'meanNsigma': Sets colorbar limits to mean ± N*sigma (e.g., 'mean2sigma').
        - 'other': Requires 'vmin' and 'vmax' as kwargs for custom limits.
    reshape_order : str, optional
        Order parameter for numpy.reshape. Must be in {'C', 'F', 'A'}. Default is 'C'.
    sci_notation : bool, optional
        If True sets scientific notation for the colorbar scale.
    **kwargs : dict
        Additional keyword arguments for `pcolormesh`.
        
        The following keyword arguments can also be used for the colorbar (`cbar`): 
            - sci_notation : bool, optional
                If True sets scientific notation in the colorbar scale
            - cbar_size: str, optional
                Defines the size of the colorbar as a percentage of the plot. Example: `cbar_size='20%'`. 
    
    Returns:
    --------
    np.array
        The reshaped XEOL intensity data.
    matplotlib.figure.Figure
        The created figure.
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    sci_notation = kwargs.pop('sci_notation', False)
    cbar_width = kwargs.pop('cbar_width', 5)

    if channel is None and roi is None:
        raise TypeError("'channel' (int) or 'roi' (tuple) need to be provided")
    
    with h5py.File(h5path, 'r') as h5f:
        all_data = h5f[pathxeol][()].astype(np.float64)
        dataset = pathxeol.split('/')[0]
        
        if pathref is not None:
            ref = h5f[pathref][0]
        elif ref_range is not None:
            ref = np.mean(all_data[:, ref_range[0]:ref_range[1]])
        else:
            ref = 0
        
        all_data -= ref
        
        if roi is not None:
            data = np.array([np.sum(i[roi[0]:roi[1] + 1]) for i in all_data])
            wavelen = (h5f[f'{dataset}/measurement/qepro_det1'][0, roi[0]],
                       h5f[f'{dataset}/measurement/qepro_det1'][0, roi[1]])
        else:
            data = all_data[:, channel]
            wavelen = h5f[f'{dataset}/measurement/qepro_det1'][0, channel]
        
        if normalize:
            mon = h5f[f'{dataset}/measurement/mon'][()]
            data /= mon / 10**4
    
    mean = np.nanmean(data)
    sigma = np.nanstd(data)
    
    print(f"""
    Data Shape: {sample_x.shape}
    Mean of XEOL Intensity: {mean:.2f}
    Standard deviation: {sigma:.2f}""")
    
    if scale == 'default':
        plotvmin, plotvmax = None, None
    elif scale.startswith('mean') and scale.endswith('sigma'):
        multiplier = float(scale.split('mean')[-1].split('sigma')[0])
        plotvmin = max(0, mean - multiplier * sigma)
        plotvmax = mean + multiplier * sigma
    elif scale == 'other':
        plotvmin = kwargs.pop('vmin', None)
        plotvmax = kwargs.pop('vmax', None)
        if plotvmin is None or plotvmax is None:
            raise KeyError("If scale='other', you must specify vmin and vmax.")
    else:
        raise ValueError("scale must be in ['default', 'meanNsigma', 'other']")
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(sample_x, sample_y, data.reshape(sample_x.shape, order=reshape_order),
                        vmin=plotvmin, vmax=plotvmax, **kwargs)
    
    ax.set_xlabel('Position [µm]')
    ax.set_ylabel('Position [µm]')
    ax.set_aspect('equal')
    
    if roi is not None:
        ax.set_title(f'{wavelen[0]:.0f}-{wavelen[1]:.0f} nm')
    else:
        ax.set_title(f'{wavelen:.0f} nm')
    
    draw_colorbar(im, width=cbar_width)
    
    return data, fig, ax


def interactive_xeol_plot(h5path: str, 
                           pathxeol: str,
                           pathfluo: str,  
                           sample_x: np.ndarray, 
                           sample_y: np.ndarray, 
                           reshape_order: str = 'F'):
    """
    Creates an interactive XEOL visualization from an HDF5 file.

    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathxeol : str
        Path inside the HDF5 file where XEOL data is stored.
    pathfluo : str
        Path inside the HDF5 file where fluorescence data is stored.
    sample_x : np.ndarray
        2D mesh of x positions of the sample.
    sample_y : np.ndarray
        2D mesh of y positions of the sample.
    reshape_order : str, optional
        Order parameter for numpy.reshape. Must be in {'C', 'F', 'A'}. Default is 'F'.
    """
    shape = sample_x.shape
    with h5py.File(h5path, "r") as h5f:
        xeol_data = h5f[pathxeol][()]
        dataset = pathxeol.split('/')[0]
        fluoGa = h5f[pathfluo][()][:].reshape(shape, order=reshape_order)
        mon = h5f[f"{dataset}/measurement/mon"][:].reshape(shape, order=reshape_order)
        wave = h5f[f"{dataset}/measurement/qepro_det1"][:][0]
    
    row_slider = widgets.IntSlider(value=0, min=0, max=shape[0] - 1, step=1, description='Row')
    col_slider = widgets.IntSlider(value=0, min=0, max=shape[1] - 1, step=1, description='Col')
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 14), gridspec_kw={'height_ratios': [1, 5]})
    
    def update_plot(row, col):
        ax[0].cla()
        ax[1].cla()
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('Position [μm]')
        ax[0].set_ylabel('Position [μm]')
        # ax[0].set_title('Ga fluorescence')
        
        ax[1].set_xlim(0, 1044)
        ax[1].set_ylim(0, 1000)
        ax[1].set_xlabel('Channels')
        ax[1].set_ylabel('Intensity (a.u.)')
        
        ax[0].pcolormesh(sample_x, sample_y, fluoGa / mon, cmap='inferno')
        
        x_pos = sample_x[0, col]
        y_pos = sample_y[row, 0]
        ax[0].hlines(y_pos, sample_x.min(), sample_x.max(), color='blue')
        ax[0].vlines(x_pos, sample_y.min(), sample_y.max(), color='blue')
        
        file_index = col * shape[0] + row
        ax[1].plot(xeol_data[file_index] / 7)
        
        inten = xeol_data[file_index].max()
        ch = np.argmax(xeol_data[file_index])
        wv = round(wave[ch], 1)
        
    row_slider.observe(lambda change: update_plot(change.new, col_slider.value), 'value')
    col_slider.observe(lambda change: update_plot(row_slider.value, change.new), 'value')
    
    def on_key(event):
        row, col = row_slider.value, col_slider.value
        if event.key == 'down':
            row = max(row - 1, row_slider.min)
        elif event.key == 'up':
            row = min(row + 1, row_slider.max)
        elif event.key == 'left':
            col = max(col - 1, col_slider.min)
        elif event.key == 'right':
            col = min(col + 1, col_slider.max)
        
        row_slider.value = row
        col_slider.value = col
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    display(row_slider, col_slider)
    update_plot(row_slider.value, col_slider.value)
    
    return fig, ax


def analyze_xeol_emission(h5path: str, 
                           pathxeol: str,
                           sample_x: np.ndarray, 
                           #sample_y: np.ndarray=None, 
                           col_range:tuple,
                           row_range:tuple,
                           emission_range:tuple = (380,440)
                           ) -> tuple:
    """
    Analyzes XEOL emission data, performs a linear fit, and visualizes results.

    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathxeol : str
        Path inside the HDF5 file where XEOL data is stored.
    shape : tuple
        Shape of the dataset for reshaping.

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects.
    """
    with h5py.File(h5path, "r") as h5f:
        dataset = pathxeol.split('/')[0]
        wave = h5f[f"{dataset}/measurement/qepro_det1"][:][0]
        xeol_data = h5f[pathxeol][()]
    
    shape = sample_x.shape
    emission_data = np.full((shape), np.nan)
    
    for col in range(*col_range): 
        for row in range(*row_range): 
            file_index = col * shape[0] + row
            ch = np.argmax(xeol_data[file_index])
            wv = round(wave[ch], 1)
            
            if emission_range[0] < wv < emission_range[1]: 
                emission_data[col][row] = wv
    
    wavelength_mean = np.nanmean(emission_data, axis=1)
    
    y = wavelength_mean[col_range[0]:col_range[1]]
    x = sample_x[0][col_range[0]:col_range[1]] - sample_x[0][col_range[0]]
    
    print(f'Initial and final position in the wire: {sample_x[0][col_range[0]]:.1f} - {sample_x[0][col_range[1]]:.1f} microns')
    
    # Linear Fit
    linear_coeffs = np.polyfit(x, y, 1)
    y_linear_fit = np.polyval(linear_coeffs, x)
    slope = linear_coeffs[0]
    
    rmse_linear = np.sqrt(np.mean((y - y_linear_fit)**2))
    ss_res_linear = np.sum((y - y_linear_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    
    print("=== Linear Fit ===")
    print(f"Slope (Δλ): {slope:.4f}")
    print(f"RMSE: {rmse_linear:.4f}")
    print(f"R²: {r2_linear:.4f}")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c='darkorange', label='Exp. Data', linewidth=2)
    ax.plot(x, y_linear_fit, '--', label=f'Linear Fit \nR²: {r2_linear:.4f}', linewidth=2)
    
    text_position_x = 0.72 * (x.max() - x.min()) + x.min()
    text_position_y = 0.85 * (y.max() - y.min()) + y.min()
    ax.text(text_position_x, text_position_y, f"Δλ = {slope:.1f} nm/µm", fontsize=12, color='black', 
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    
    ax.set_xlabel('Shell Length (µm)')
    ax.set_ylabel('Mean of Max. Emission (nm)')
    ax.legend()
    ax.grid()
    plt.show()
    
    return fig, ax

def interactive_xeol_channel_plot(h5path: str, 
                                   pathxeol: str, 
                                   sample_x: np.ndarray, 
                                   sample_y: np.ndarray,
                                   plot_deviation = 5,
                                   pathref: str = None, 
                                   normalize: bool = True, 
                                   size: tuple = (8, 5)) -> None:
    """
    Creates an interactive plot for XEOL data visualization across different channels.
    
    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathxeol : str
        Path inside the HDF5 file where XEOL data is stored.
    sample_x : np.ndarray
        2D mesh of x positions of the sample.
    sample_y : np.ndarray
        2D mesh of y positions of the sample.
    pathref : str, optional
        Path to the reference data for subtraction.
    normalize : bool, optional
        If True, normalizes data using monitor values. Default is True.
    size : tuple, optional
        Figure size in inches (width, height). Default is (8, 5).
    """
    
    with h5py.File(h5path, "r") as h5f:
        xeol_data = h5f[pathxeol][()].astype(np.float64)
        dataset = pathxeol.split('/')[0]
        wave = h5f[f'{dataset}/measurement/qepro_det1'][:][0]
        
        if pathref is not None:
            ref = h5f[pathref][0]
            xeol_data -= ref
    
    channel_slider = widgets.IntSlider(value=0, min=0, max=1044, step=1, description='Channel')
    fig_interact_2, ax_interact_2 = plt.subplots(figsize=size)
    
    def update_plot(channel):
        xeol = xeol_data[:, channel]
        wavelen = wave[channel]
        
        if normalize:
            with h5py.File(h5path, "r") as h5f:
                mon = h5f[f'{dataset}/measurement/mon'][()]
            xeol /= mon / 10**2
        
        mean = np.nanmean(xeol)
        sigma = np.nanstd(xeol)

        plotvmin = max(0, mean - plot_deviation * sigma)
        plotvmax = mean + plot_deviation * sigma
        
        ax_interact_2.cla()
        ax_interact_2.set_title(f'{np.round(wavelen, 0)} nm\nMax. Int: {xeol.max():.0f}')
        ax_interact_2.set_aspect('equal')
        ax_interact_2.set_xlabel('Position [μm]')
        ax_interact_2.set_ylabel('Position [μm]')
        ax_interact_2.pcolormesh(sample_x, sample_y, xeol.reshape(sample_x.shape, order='F'), 
                                 vmin=plotvmin, vmax=plotvmax, cmap='magma_r')
    
    channel_slider.observe(lambda x: update_plot(x.new), 'value')
    
    def on_key(event):
        channel = channel_slider.value
        if event.key == 'left':
            channel = max(channel - 1, channel_slider.min)
        elif event.key == 'right':
            channel = min(channel + 1, channel_slider.max)
        
        channel_slider.value = channel
    
    fig_interact_2.canvas.mpl_connect('key_press_event', on_key)
    display(channel_slider)
    update_plot(channel_slider.value)
    
    

def analyze_xeol_emission(h5path: str, 
                           pathxeol: str,
                           sample_x: np.ndarray, 
                           col_range: tuple,
                           row_range: tuple,
                           emission_range: tuple = (380, 440)) -> tuple:
    """
    Analyzes XEOL emission data, extracts peak emissions within a specified range,
    performs a linear fit, and visualizes results.

    Parameters:
    -----------
    h5path : str
        Path to the HDF5 file.
    pathxeol : str
        Path inside the HDF5 file where XEOL data is stored.
    sample_x : np.ndarray
        2D array representing the sample x positions.
    col_range : tuple
        Range of columns to consider for analysis (start, end).
    row_range : tuple
        Range of rows to consider for analysis (start, end).
    emission_range : tuple, optional
        Wavelength range (start, end) for emission filtering, default is (380, 440).

    Returns:
    --------
    tuple
        (fig, ax) matplotlib figure and axis objects with the linear fit visualization.
    """
    with h5py.File(h5path, "r") as h5f:
        dataset = pathxeol.split('/')[0]
        wave = h5f[f"{dataset}/measurement/qepro_det1"][:][0]
        xeol_data = h5f[pathxeol][()]
    
    shape = sample_x.shape
    emission_data = np.full(shape, np.nan)
    
    for col in range(*col_range): 
        for row in range(*row_range):
            file_index = col * shape[0] + row
            ch = np.argmax(xeol_data[file_index])
            wv = round(wave[ch], 1)
            
            if emission_range[0] < wv < emission_range[1]: 
                emission_data[row, col] = wv
    
    wavelength_mean = np.nanmean(emission_data, axis=0)
    
    y = wavelength_mean[col_range[0]:col_range[1]]
    x = sample_x[0][col_range[0]:col_range[1]] - sample_x[0][col_range[0]]
    
    print(f'Initial and final position in the wire: {sample_x[0][col_range[0]]:.1f} - {sample_x[0][col_range[1]]:.1f} microns')
    
    # Linear Fit
    linear_coeffs = np.polyfit(x, y, 1)
    y_linear_fit = np.polyval(linear_coeffs, x)
    slope = linear_coeffs[0]
    
    rmse_linear = np.sqrt(np.mean((y - y_linear_fit)**2))
    ss_res_linear = np.sum((y - y_linear_fit)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_linear = 1 - (ss_res_linear / ss_tot)
    
    print("=== Linear Fit ===")
    print(f"Slope (Δλ): {slope:.4f}")
    print(f"RMSE: {rmse_linear:.4f}")
    print(f"R²: {r2_linear:.4f}")
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, c='darkorange', label='Exp. Data', linewidth=2)
    ax.plot(x, y_linear_fit, '--', label=f'Linear Fit \nR²: {r2_linear:.4f}', linewidth=2)
    
    text_position_x = 0.72 * (x.max() - x.min()) + x.min()
    text_position_y = 0.85 * (y.max() - y.min()) + y.min()
    ax.text(text_position_x, text_position_y, f"Δλ = {slope:.1f} nm/µm", fontsize=12, color='black', 
            bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    
    ax.set_xlabel('Shell Length (µm)')
    ax.set_ylabel('Mean of Max. Emission (nm)')
    ax.legend()
    ax.grid()
    plt.show()
    
    return fig, ax


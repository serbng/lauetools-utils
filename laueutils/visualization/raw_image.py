import fabio
import numpy as np
import matplotlib.pyplot as plt
import multiprocess as mp

from ..utils.chunks import linear_chunks # for mosaic
from ._utils import draw_colorbar

def display_image(data, roi=None, **kwargs):
    """Plot an image.
    
    Parameters
    ----------
    image_path        (str): Full path to the image
    ROI        (tuple[int]): Subset of pixels inside the image to plot. The format is:
                             (x_position, y_position, x_boxsize, y_boxsize)
    
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.imshow
    """
    if isinstance(data, str):
        with fabio.open(data) as f:
            image_data = f.data
    elif isinstance(data, np.ndarray):
        image_data = data
    else:
        raise TypeError(
            "data type must be in {str, np.ndarray}." + f" Got {type(data)}"
        )
    
    if roi is not None:
        y1 = roi[0] - (roi[2] // 2)
        y2 = roi[0] + (roi[2] // 2)
        x1 = roi[1] - (roi[3] // 2)
        x2 = roi[1] + (roi[3] // 2)
                
        image_data = image_data[x1:x2, y1:y2]
    
    ax = plt.gca()
    image = ax.imshow(image_data, **kwargs)
    draw_colorbar(image)
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_aspect('equal')

def _mosaic_row(row_paths, roi_indices):
    """Worker function in mosaic.

    Parameters
    ----------
    row_paths      (list[str]): list of paths to the files of the images of one row of the mosaic.
    roi_indices   (tuple[int]): (x1, x2, y1, y2) that will be used as a slice of the image data.

    Returns
    ----------
    row_data      [np.ndarray]: Array of shape (x2-x1, (y2-y1)*len(row_paths)) with the data of a row.
    """
    x1, x2, y1, y2 = roi_indices
    row_data = []
    for path in row_paths:
        # Fetch data whether it exists or not
        try:
            with fabio.open(path) as image:
                image_data = image.data
            # Cropping
            image_data = image_data[x1:x2, y1:y2]
        except(IndexError, IOError):
            roi_boxsize = (x2-x1, y2-y1)
            image_data = np.zeros(roi_boxsize)

        row_data.append(image_data)
        
    row_data = np.hstack(row_data)
    # Now row_data is a matrix containing the data of a row
    # <-- len(row)*roi_boxsize[1] -->
    # *-----*-----*   .....   *-----*  
    # |     |     |           |     |  roi_boxsize[0]
    # |     |     |           |     |  
    # *-----*-----*   .....   *-----*  
    return row_data
    
def mosaic(paths, num_rows, num_cols, roi_center, roi_boxsize, workers=4):
    """Stitch together the same ROI of different images to create a mosaic.
    
    The images are stitched together row by row. So, if ´´´num_cols=10´´´, the images are read in chunks of 10 and
    put in a row. At the end the rows are stacked on top of each other.
    
    Parameters
    ----------
    paths        (list[str]): List of paths to the images used to build the mosaic.
    num_rows           (int): When the images come from a 2D scan, number of rows of the scan.
    num_cols           (int): When the images come from a 2D scan, number of coloumns of the scan.
    roi_center  (tuple[int]): Position on the detector to track.
    roi_boxsize (tuple[int]): Size of the ROI. It is the side of a square centered at ´´´roi_center´´´.
    workers            (int): (optional) Default to 4. Number of cpus to use to speed up the process.


    Returns
    ----------
    mosaic      (np.ndarray): Array of shape (num_rows*roi_boxsize[1], num_cols*roi_boxsize[0]). Result of the mosaic.
    """
    y1 = int(roi_center[0] - roi_boxsize[0] // 2)
    y2 = int(roi_center[0] + roi_boxsize[0] // 2)
    x1 = int(roi_center[1] - roi_boxsize[1] // 2)
    x2 = int(roi_center[1] + roi_boxsize[1] // 2)
    row_indices = linear_chunks(num_rows * num_cols, num_cols)
    # Ex.:
    # row_indices = linear_chunks(81 * 81, 81)
    # row_indices
    # [[   0,    1,    2,    3,    4, ...,   80],
    #  [  81,   82,   83,   84,   85, ...,  161].
    #  ...
    #  [6479, 6480, 6481, 6482, 6483, ..., 6560]]
    # Each row_indices contains the indices of the files corresponding to a row
    row_paths = [ [paths[i] for i in columns] for columns in row_indices]
    with mp.Pool(workers) as pool:
        mosaic_rows = pool.starmap(
            lambda row: _mosaic_row(row, (x1, x2, y1, y2)),
            row_paths,
            chunksize=1
        )
    # mosaic_rows is now a list of arrays containing the data of the rows
    # concatenating them vertically creates the mosaic
    mosaic = np.vstack(mosaic_rows)
    
    return mosaic    

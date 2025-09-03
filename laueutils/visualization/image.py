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

def _mosaic_line(line_paths, roi_indices, line_direction):
    """Worker function in mosaic.

    Parameters
    ----------
    line_paths     (list[str]): list of paths to the files of the images of one line of the mosaic.
    roi_indices   (tuple[int]): (x1, x2, y1, y2) that will be used to slice the image data.
    line_direction       (str): Must be in {'horizontal', 'vertical'}. Whether to treat the line as
                                as a row or a columns

    Returns
    ----------
    row_data      [np.ndarray]: Array with the data of a row/column.
    """
    if line_direction not in ('vertical', 'horizontal'):
        raise(ValueError, "line_durectuib must be in {'horizontal', 'vertical'}. " + f"Got {line_direction}.")
        
    x1, x2, y1, y2 = roi_indices
    line_data = []
    for path in line_paths:
        # Fetch data whether it exists or not
        try:
            with fabio.open(path) as image:
                image_data = image.data
            # Cropping
            image_data = image_data[x1:x2, y1:y2]
        except(IndexError, IOError):
            roi_boxsize = (x2-x1, y2-y1)
            image_data = np.zeros(roi_boxsize)

        line_data.append(image_data)

    if line_direction == 'horizontal':
        # Now line_data is a matrix containing the data of a row
        # <--len(line_paths)*roi_boxsize[1]-->
        # *-----*-----*     .....     *-----*  
        # |     |     |               |     |  roi_boxsize[0]
        # |     |     |               |     |  
        # *-----*-----*     .....     *-----*
        return np.hstack(line_data)
    if line_direction == 'vertical':
        # Now line_data is a matrix containing the data of a columns
        # <-- roi_boxsize[0] -->
        # *-----*
        # |     |  data coming from:
        # |     |  line_paths[-1]
        # *-----*  
        # |     |  
        # |     |   len(line_paths) * roi_boxsize[1]
        # *-----*
        # .     .
        # .     .
        # *-----*  
        # |     |  data coming from:
        # |     |  line_paths[0]
        # *-----* 
        return np.vstack(line_data[::-1])

def mosaic(paths, num_rows, num_cols, roi_center, roi_boxsize, scan_direction='horizontal', workers=4):
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
    scan_direction     (str): (optional) Defaults to 'horizontal'. Specify if the scan is done row by row or column by column.
    workers            (int): (optional) Default to 4. Number of cpus to use to speed up the process.


    Returns
    ----------
    mosaic      (np.ndarray): Array of shape (num_rows*roi_boxsize[1], num_cols*roi_boxsize[0]). Result of the mosaic.
    """
    y1 = int(roi_center[0] - roi_boxsize[0] // 2)
    y2 = int(roi_center[0] + roi_boxsize[0] // 2)
    x1 = int(roi_center[1] - roi_boxsize[1] // 2)
    x2 = int(roi_center[1] + roi_boxsize[1] // 2)

    if scan_direction == 'horizontal':
        chunk_size = num_cols
    elif scan_direction =='vertical':
        chunk_size = num_rows
    else:
        raise(ValueError, "scan_direction must be in {'horizontal', 'vertical'}. " + f"Got {scan_direction}.")
    
    line_indices = linear_chunks(num_rows * num_cols, chunk_size)
    # Ex.:
    # line_indices = linear_chunks(81 * 81, 81)
    # line_indices
    # [[   0,    1,    2,    3,    4, ...,   80],
    #  [  81,   82,   83,   84,   85, ...,  161].
    #  ...
    #  [6479, 6480, 6481, 6482, 6483, ..., 6560]]
    # Each element of line_indices is a list of indices corresponding to the files of a line
    row_paths = [ [paths[i] for i in line] for line in line_indices]
    with mp.Pool(workers) as pool:
        mosaic_lines = pool.starmap(
            lambda paths: _mosaic_line(paths, (x1, x2, y1, y2), line_direction=scan_direction),
            zip(row_paths),
            chunksize=1
        )

    if scan_direction == 'horizontal':
        mosaic = np.vstack(mosaic_lines)

    if scan_direction == 'vertical':
        mosaic = np.hstack(mosaic_lines)
    
    return mosaic    

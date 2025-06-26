import fabio
import numpy as np
import matplotlib.pyplot as plt
from graintools.utils import linear_chunks # for mosaic
from graintools.visualization._utils import draw_colorbar

def display_image(data, roi=None, **kwargs):
    """Plot an image.
    
    Parameters
    ----------
    image_path      [str]: Full path to the image
    ROI        tuple[int]: Subset of pixels inside the image to plot. The format is:
                           (x_position, y_position, x_boxsize, y_boxsize)
    
    Keyword arguments
    ----------
    kwargs: passed to matplotlib.pyplot.imshow
    """
    if isinstance(data, str):
        with fabio.open(data) as f:
            image_data = f.data
    if isinstance(data, np.ndarray):
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
    draw_colorbar(image, ax)
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_aspect('equal')
    
def mosaic(paths, num_rows, num_cols, roi_center, roi_boxsize, verbose=True):
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
    #Each row_indices contains the indices of the files corresponding to a row
    mosaic_data = []
    for row in row_indices:
        # I go through each column in the row, and I append a matrix with shape
        # roi_boxsize[1], roi_boxsize[0] and I concatenate horizontally at the end
        row_data = []
        for image_index in row:
            try:
                if verbose:
                    print(f'Opening image {paths[image_index]}', end='\r')
                try:
                    with fabio.open(paths[image_index]) as image:
                        image_data = image.data
                    # Cropping
                    image_data = image_data[x1:x2, y1:y2]
                except IOError:
                    image_data = np.zeros(roi_boxsize)
            except IndexError:
                image_data = np.zeros(roi_boxsize)
                
            row_data.append(image_data)
        row_data = np.hstack(row_data)
        # Now row_data is a matrix containing the data of a row
        # <-- len(row)*roi_boxsize[1] -->
        # *-----*-----*   .....   *-----*  
        # |     |     |           |     |  roi_boxsize[0]
        # |     |     |           |     |  
        # *-----*-----*   .....   *-----*  
        # I append it to the overall mosaic_data
        # and at the very end i concatenate vertically 
        mosaic_data.append(row_data)
    
    return np.vstack(mosaic_data)    
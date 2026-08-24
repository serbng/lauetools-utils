from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING
from pathlib import Path
from functools import partial
import multiprocessing as mp
import numpy as np

if TYPE_CHECKING:
    import plotly.graph_objects as go

from lauexplore._utils import linear_chunks
from lauexplore.scan import Scan
from lauexplore.image import ROI, read
from lauexplore.plots import heatmap, mosaic_hovermenu

def _get_mosaic_line(
        linepaths: Iterable[str | Path], 
        roi_indices: tuple[int, int, int, int], 
        direction: str
    ) -> np.ndarray:
    """Worker function in mosaic.

    Parameters
    ----------
    line_paths  (list[str]): list of paths to the files of the images of one line of the mosaic.
    roi_indices (tuple[int]): (x1, x2, y1, y2) that will be used to slice the image data.
    direction         (str): Must be in {'horizontal', 'vertical'}. Whether to treat the line as as a row or a columns

    Returns
    ----------
    row_data      [np.ndarray]: Array with the data of a row/column.
    """
    if direction not in ('vertical', 'horizontal'):
        raise ValueError("Direction must be in {'horizontal', 'vertical'}. " + f"Got {direction}.")
        
    x1, x2, y1, y2 = roi_indices
    line_data = []
    for path in linepaths:
        # Fetch data whether it exists or not
        try:
            image_data = read(path)[y1:y2, x1:x2]
        except(IndexError, IOError):
            roi_boxsize = (y2-y1, x2-x1)
            image_data = np.zeros(roi_boxsize, dtype=np.uint16)

        line_data.append(image_data)

    if direction == 'horizontal':
        # Now line_data is a matrix containing the data of a row
        # <--len(line_paths)*roi_boxsize[1]-->
        # *-----*-----*     .....     *-----*  
        # |     |     |               |     |  roi_boxsize[0]
        # |     |     |               |     |  
        # *-----*-----*     .....     *-----*
        return np.hstack(line_data)
    if direction == 'vertical':
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
    
def _get_tile(
        filepath: str | Path,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        fill_dtype: type = np.uint16
    ) -> np.ndarray:
    try:
        img = read(filepath)
        tile = img[y1:y2, x1:x2]
        return tile
    except Exception:
        return np.zeros((y2-y1, x2-x1), dtype=fill_dtype)

def build_mosaic(
        filepaths: Iterable[str | Path], 
        nbrows: int, 
        nbcols: int, 
        roi_center: tuple[int, int],
        roi_boxsize: tuple[int, int],
        direction: str = 'horizontal', 
        workers: int = 8,
        chunksize: int = 256
    ) -> np.ndarray:
    """Stitch together the same ROI of different images to create a mosaic.
    
    The images are stitched together row by row. So, if ´num_cols=10´, the images are read in chunks of 10 and
    put in a row. At the end the rows are stacked on top of each other.
    
    Parameters
    ----------
    paths        (list[str]): List of paths to the images used to build the mosaic.
    num_rows           (int): When the images come from a 2D scan, number of rows of the scan.
    num_cols           (int): When the images come from a 2D scan, number of coloumns of the scan.
    roi_center  (tuple[int]): Position on the detector to track.
    roi_boxsize (tuple[int]): Size of the ROI. It is the side of a square centered at ´roi_center´.
    direction          (str): (optional) Defaults to 'horizontal'. Specify if the scan is done row by row or column by column.
    workers            (int): (optional) Default to 4. Number of cpus to use to speed up the process.


    Returns
    ----------
    mosaic      (np.ndarray): Array of shape (num_rows*roi_boxsize[1], num_cols*roi_boxsize[0]). Result of the mosaic.
    """
    if direction == 'horizontal':
        # Row-major acquisition: columns change fastest
        # i -> (r, c) = (i // nbcols, i % nbcols)
        def slot(idx, tile_h, tile_w):
            r, c = divmod(idx, nbcols)
            return slice(r*tile_h, (r+1)*tile_h), slice(c*tile_w, (c+1)*tile_w)
        
    elif direction =='vertical':
        # Column-major acquisition: rows change fastest
        # i -> (r, c) = (i % nbrows, i // nbrows)
        def slot(idx, tile_h, tile_w):
            c, r = divmod(idx, nbrows)
            # reversed column
            # when accessing column, start from the bottom
            rc = nbcols - 1 - c 
            return slice(rc*tile_h, (rc+1)*tile_h), slice(r*tile_w, (r+1)*tile_w)
    else:
        raise ValueError("Direction must be in {'horizontal', 'vertical'}. " + f"Got {direction}.")
    
    
    x1 = int(roi_center[0] - roi_boxsize[0]//2)
    x2 = int(roi_center[0] + roi_boxsize[0]//2)
    y1 = int(roi_center[1] - roi_boxsize[1]//2)
    y2 = int(roi_center[1] + roi_boxsize[1]//2)
    tile_h, tile_w = (y2 - y1), (x2 - x1)
    
    H, W = nbrows * tile_h, nbcols * tile_w
            
    mosaic_data = np.zeros((H, W), dtype=np.uint16)
    
    # Worker function is _get_tile with partially specified parameters
    worker = partial(_get_tile, x1=x1, x2=x2, y1=y1, y2=y2)
    chunksize = min(chunksize, len(filepaths) // workers)
    
    with mp.Pool(workers) as pool:
        tiles = pool.map(worker, filepaths, chunksize=chunksize)

    for idx, tile in enumerate(tiles):
        rows, cols = slot(idx, tile_h, tile_w)
        mosaic_data[rows, cols] = tile
        
    return mosaic_data   

@dataclass
class Mosaic:
    data: np.ndarray
    roi: ROI
    scan: Scan
    
    @classmethod
    def from_files(cls, filepaths: Iterable[str | Path], roi: ROI, scan: Scan, workers: int = 16):
        mosaic_data = build_mosaic(
            filepaths,
            scan.nbypoints,
            scan.nbxpoints,
            roi.center,
            roi.boxsize,
            scan.direction,
            workers
        )
        
        return cls(mosaic_data, roi, scan)
    
    def plot(self,
            width: int = 800,
            height: int = 800,
            zmin: float | None = 1000,
            zmax: float | None = None,
            title: str | None = None,
            xlabel: str | None = None,
            ylabel: str | None = None,
            colorscale: str = "gray",
            log10: bool = False,
            cbartitle: str | None = None
        ) -> "go.Figure":
        data = np.flipud(self.data)
        
        ny, nx = data.shape
        customdata, hovertemplate = mosaic_hovermenu(self.scan, self.roi)
        
        x = np.arange(nx)
        y = np.arange(ny)
        
        mosaic_plot = heatmap(
            data, x, y,
            customdata=customdata,
            hovertemplate=hovertemplate,
            width = width,
            height = height,
            zmin = zmin,
            zmax = zmax,
            title = title or f"Mosaic. (X, Y) = ({self.roi.center}, boxsize = ({self.roi.boxsize}))",
            xlabel = xlabel or "X pixel",
            ylabel = ylabel or "Y pixel",
            colorscale = colorscale,
            log10 = log10,
            cbartitle = cbartitle or "counts",
        )
        
        return mosaic_plot

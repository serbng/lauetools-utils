import numpy as np
from sklearn.neighbors import KDTree

def subtract(peaklist1, peaklist2, tol=1.0):
    """Remove the peaks in peaklist2 from peaklist1

    Parameters
    ----------
    peaklist1 [np.ndarray]: peak array, should be at least (N, 2) where the first two columns are (x, y) coordinates
    peaklist2 [np.ndarray]: peak array, should be at least (N, 2) where the first two columns are (x, y) coordinates
    tol            [float]: for distance recognition.
    
    Returns
    ----------
    result   [np.ndarray]: result with the same number of columns as peaklist1
    """
    if len(peaklist2) == 0:
        return peaklist1

    # Extract coordinate columns (adjust if necessary for more dimensions)
    peak_positions1 = peaklist1[:, :2]
    peak_positions2 = peaklist2[:, :2]

    # Build a KDTree for the positions of peaks in the first list.
    tree = KDTree(peak_positions1)

    # Query all points in peak_positions2 at once; returns an array of indices for each query point.
    indices, distances = tree.query_radius(peak_positions2, r=tol, sort_results=True, return_distance=True)
    
    # Flatten the list of arrays and get unique indices to remove
    if indices.size > 0:  # if any query returned matches
        to_remove = np.unique(np.concatenate(indices))
    else:
        to_remove = np.array([])
        
    if len(to_remove) == 0:
        return peaklist1
    else:
        return np.delete(peaklist1, to_remove, axis=0)
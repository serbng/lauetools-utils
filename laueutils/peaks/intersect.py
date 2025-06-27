import numpy as np
from sklearn.neighbors import KDTree

def intersect(peaklist1, peaklist2, tol=1.0):
    """Return a peaklist with the peaks that are both in peaklist1 and peaklist2.

    Parameters
    ----------
    peaklist1 [np.ndarray]: peak array, should be at least (N, 2) where the first two columns are (x, y) coordinates
    peaklist2 [np.ndarray]: peak array, should be at least (N, 2) where the first two columns are (x, y) coordinates
    tol            [float]: for distance recognition.
    
    Returns
    ----------
    result   [np.ndarray]: result with the same number of columns as peaklist1
    """
    # If there are no peaks in the second list, there is no intersection.
    if len(peaklist2) == 0:
        return np.empty((0, peaklist1.shape[1]))

    # Extract the coordinate columns (assumes the first two columns are (x, y))
    peak_positions1 = peaklist1[:, :2]
    peak_positions2 = peaklist2[:, :2]

    # Build a KDTree for the positions in peaklist1.
    tree = KDTree(peak_positions1)

    # Query all points in peak_positions2 at once; returns an array of candidate index arrays.
    indices, distances = tree.query_radius(peak_positions2, r=tol, sort_results=True, return_distance=True)
    
    # If any query returned matches, we flatten the list of index arrays and take the unique indices.
    if indices.size > 0:
        intersection_idx = np.unique(np.concatenate(indices))
    else:
        # No matches were found; return an empty array with appropriate number of columns.
        return np.empty((0, peaklist1.shape[1]))

    return peaklist1[intersection_idx]
import numpy as np
from sklearn.neighbors import KDTree

def _build_kdtrees(dataset):
    kdtrees = []
    for peaklist in dataset:
        if len(peaklist) > 0:
            tree = KDTree(peaklist[:,:2])
        else:
            tree = None
        kdtrees.append(tree)
    return kdtrees

def remove_duplicates(peaklist, tol=0.5):
    """_summary_

    Args:
        peaklist (_type_): _description_
        tol (float, optional): _description_. Defaults to 0.5.
    """
    if len(peaklist) == 0:
        return peaklist
    
    peak_positions = peaklist[:,:2]
    
    tree = KDTree(peak_positions)
    used = np.zeros(len(peaklist), dtype=bool)
    unique_positions = []
    
    for i, peak in enumerate(peak_positions):
        if used[i]:
            continue
        idx, dist = tree.query_radius(peak.reshape(1,-1), r=tol, return_distance=True, sort_results=True)
        neighbors = idx[0]
        used[neighbors] = True
        
        chosen_index = neighbors[0]
        unique_positions.append(peaklist[chosen_index, :])
    
    return np.array(unique_positions)

    
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
    
def track(peaklists, positions, tol, use_intensity=False):
    """For each peaklist return the peak closest, within the given tolerance, to the specified position
    
    Parameters
    ----------
        peaklists list[np.ndarray]: list of peaklists, each of them must be a numpy.ndarray with shape
                                    at least (N, 2), where the first two columns are the (x, y) coord-
                                    inates. 
                                    If use_intensity=True, the shape must be at least (N, 3).
                                    The peaklists are assumed to have the same shape.
        position      [array-like]: A single (x, y) position or an array-like of positions with shape
                                    (P, 2). Each will be tracked independently
        tol                [float]: for distance recognition
        use_intensity       [bool]: when more than a match is found around 'position', use the intensity
                                    value to decide which peak to keep. If True, it is assumed that the
                                    third column of the peaklist contains the value of the intensity

    Returns:
    ----------
        result        [np.ndarray]: array of shape (len(peaklists), M) where M is the number of columns
                                    of each peaklist.
    """
    # Ensure positions is at least 2D (so a single (x, y) becomes shape (1,2))
    positions = np.atleast_2d(positions)
    num_positions = positions.shape[0]
    
    # Number of columns in each peaklist (e.g. 2 or 3)
    M = peaklists[0].shape[1]
    num_peaklists = len(peaklists)
    
    # Build one KDTree for each peaklist (your _build_kdtrees function should provide these)
    kdtrees = _build_kdtrees(peaklists)
    
    # Preallocate a 3D array for results: dimensions (num_peaklists, num_positions, M)
    results = np.full((num_peaklists, M, num_positions), np.nan)
    
    # Loop over each image (peaklist) and its corresponding KDTree.
    for i, (peaklist, tree) in enumerate(zip(peaklists, kdtrees)):
        if tree is None or len(peaklist) == 0:
            # List is empty -> nothing to find -> NaN's are good
            continue
        
        # Query the tree for all positions at once
        idx_list, dist_list = tree.query_radius(positions, r=tol, return_distance=True, sort_results=True)
        
        # Loop over the results for each position
        for j, matching_indices in enumerate(idx_list):
            if len(matching_indices) == 0: # No match is found. NaN's are good
                continue
            
            matching_peaks = peaklist[matching_indices]
            
            # If multiple candidates are found and we are not on the first image
            # and use_intensity is enabled, use the previous image's tracked intensity
            if len(matching_peaks) > 1:
                
                if use_intensity and i > 0: # Try to use the intensity
                    I_prev = results[i-1, 2, j]
                    
                    if np.isnan(I_prev): # I_prev is nan, just use the closest
                        chosen_peak = matching_peaks[0]
                        
                    else: # I can take the closest in intensity
                        I_diffs = np.abs(matching_peaks[:, 2] - I_prev)
                        chosen_peak = matching_peaks[np.argmin(I_diffs)]
                        
                else: # Don't use intensity, just take the closest
                    chosen_peak = matching_peaks[0]
                    
            else: # Only one match
                chosen_peak = matching_peaks
            
            results[i, :, j] = chosen_peak
    
    # Organize output: if one tracking position was provided, return a (num_peaklists, M) array;
    # otherwise, return a list of arrays, each corresponding to one position.
    if num_positions == 1:
        return results[:, :, 0]
    else:
        return [results[:, :, j] for j in range(num_positions)]
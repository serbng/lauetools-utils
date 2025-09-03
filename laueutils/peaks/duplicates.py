import numpy as np
from sklearn.neighbors import KDTree

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
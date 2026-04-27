import numpy as np


def compute_zlimits(
    data: np.ndarray,
    scale: str = "uniform",
    multiplier: float = 1e4,
) -> tuple[float, float, float]:
    """
    Compute (zmin, zmax, zmid) automatically for strain components.

    Parameters
    ----------
    data : np.ndarray
        1D strain component array.
    scale : str
        - 'uniform'  → symmetric around 0
        - 'default'  → min, max, mean
        - 'meanNsigma' (e.g. 'zero3sigma') - sigma around 0
        - 'meanMsigma' (e.g. 'mean3sigma') - sigma around Mean
    multiplier : float
        Scaling factor for visualization.

    Returns
    -------
    (zmin, zmax, zmid)
    """

    data = data * multiplier

    # Symmetric range around zero
    if scale == "uniform":
        gmax = np.nanmax(np.abs(data))
        return -gmax, gmax, 0.0

    # mean ± N sigma
    if scale.startswith("zero") and scale.endswith("sigma"):
        try:
            N = float(scale[4:-5])
        except ValueError:
            print('invalid scale name.')
            print('Setting scale as mean3sigma (i.e. mean value ± 3*std)')
            N = 3.0
        mean = np.nanmean(data)
        std = np.nanstd(data)

        zmin = mean - N * std
        zmax = mean + N * std

        # Enforce sign if degenerate
        if zmin >= 0:
            zmin = -1e-4
        if zmax <= 0:
            zmax = 1e-4

        return zmin, zmax, 0.0

    if scale.startswith("mean") and scale.endswith("sigma"):
        try:
            M = float(scale[4:-5])
        except ValueError:
            print('invalid scale name.')
            print('Setting scale as mean3sigma (i.e. mean value ± 3*std)')
            M = 3.0
        mean = np.nanmean(data)
        std = np.nanstd(data)

        zmin = mean - M * std
        zmax = mean + M * std

        # Enforce sign if degenerate
        if zmin >= 0:
            zmin = -1e-4
        if zmax <= 0:
            zmax = 1e-4

        return zmin, zmax, mean

    # raw min/max
    if scale == "default":
        return np.nanmin(data), np.nanmax(data), np.nanmean(data)

    raise ValueError(f"Unknown scale mode '{scale}'")
    
# ===========================================================
# COLORSCALE SHIFT (nonlinear midpoint centering)
# ===========================================================

def nonlinear_colorscale(base_colorscale, zmin, zmax, zmid):
    """
    Build a *nonlinear* colorscale so that the midpoint color is located
    exactly at the value `zmid` within (zmin, zmax).

    This keeps the full resolution of the original colorscale and simply
    reassigns the positions of each color stop.

    Parameters
    ----------
    base_colorscale : list
        Original colorscale used by Plotly (e.g., px.colors.diverging.balance)
    zmin, zmax : float
        Lower and upper bounds of the data range.
    zmid : float or None
        The desired "center" value of the colorscale. If None, no shift applied.

    Returns
    -------
    list
        A new Plotly-compatible colorscale with remapped positions.
    """

    if zmid is None or zmin is None or zmax is None:
        return base_colorscale

    # Normalized target midpoint ∈ [0,1]
    p_mid = (zmid - zmin) / (zmax - zmin)
    p_mid = float(np.clip(p_mid, 0.0, 1.0))

    new_scale = []
    for pos, color in base_colorscale:

        # Re-center the positions nonlinearly
        if pos < 0.5:
            new_pos = p_mid * (pos / 0.5)
        else:
            new_pos = p_mid + (pos - 0.5) * ((1 - p_mid) / 0.5)

        new_scale.append([float(np.clip(new_pos, 0.0, 1.0)), color])

    new_scale.sort(key=lambda x: x[0])
    return new_scale
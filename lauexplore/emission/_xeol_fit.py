"""Multi-Gaussian batch fit for XEOL spectra.

Usage
-----
from lauexplore.emission._xeol_fit import fit_spectra_xeol, plot_gaussian_maps

p0 = [
    450,  375,  5,
    100,  400,  5,
     50,  419,  3,
    3000, 450,  8,
]
df_fit = fit_spectra_xeol(xeol, wl_range=(360, 475), p0=p0, workers=8)
fig    = plot_gaussian_maps(df_fit, xeol, n_components=4)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


# ── Gaussian helpers ──────────────────────────────────────────────────────────

def _gaussian(x: np.ndarray, amp: float, cen: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


def _multi_gaussian(x: np.ndarray, *params) -> np.ndarray:
    y = np.zeros_like(x, dtype=float)
    for k in range(len(params) // 3):
        y += _gaussian(x, params[3*k], params[3*k+1], params[3*k+2])
    return y


# ── Single-point fit ──────────────────────────────────────────────────────────

def _fit_one(args: tuple) -> dict:
    point_idx, x, y, p0, bounds, n_comp, x_mm, y_mm = args
    row: dict = {
        'point_idx':   point_idx,
        'x_mm':        x_mm,
        'y_mm':        y_mm,
        'fit_success': False,
        'r2_total':    np.nan,
    }
    for k in range(n_comp):
        for col in ('amp', 'cen', 'fwhm', 'r2'):
            row[f'{col}_{k}'] = np.nan

    try:
        popt, _ = curve_fit(
            _multi_gaussian, x, y, p0=p0, bounds=bounds, maxfev=10_000
        )
    except Exception:
        return row

    y_fit  = _multi_gaussian(x, *popt)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - y_fit) ** 2))

    row['fit_success'] = True
    row['r2_total']    = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    for k in range(n_comp):
        amp, cen, sig = popt[3*k], popt[3*k+1], popt[3*k+2]
        row[f'amp_{k}']  = float(amp)
        row[f'cen_{k}']  = float(cen)
        row[f'fwhm_{k}'] = float(2.355 * abs(sig))

        # R² per component: quality of component k against
        # (data − all other components)
        comp_k   = _gaussian(x, amp, cen, sig)
        isolated = y - (y_fit - comp_k)
        ss_t = float(np.sum((isolated - isolated.mean()) ** 2))
        ss_r = float(np.sum((isolated - comp_k) ** 2))
        row[f'r2_{k}'] = max(0.0, 1.0 - ss_r / ss_t) if ss_t > 0 else np.nan

    return row


# ── Batch fit ─────────────────────────────────────────────────────────────────

def fit_spectra_xeol(
    xeol,
    wl_range: tuple[float, float],
    p0: list[float],
    bounds: tuple | None = None,
    workers: int = 8,
) -> pd.DataFrame:
    """Fit a sum of Gaussians to every spectrum in the XEOL scan.

    Parameters
    ----------
    xeol      : XEOL object (.spectra, .wl_array, .scan)
    wl_range  : (wl_min_nm, wl_max_nm) fit window
    p0        : initial params [amp0, cen0, sigma0, amp1, cen1, sigma1, ...]
    bounds    : (lower, upper) bound lists; if None, auto-built from p0:
                amp ∈ [0, 1e6], centre ± 15 nm, sigma ∈ [1, 20] nm
    workers   : threads (scipy releases the GIL during curve_fit)

    Returns
    -------
    pd.DataFrame — columns: point_idx, x_mm, y_mm, fit_success, r2_total,
                   amp_k, cen_k, fwhm_k, r2_k  (for each component k)
    """
    wl     = xeol.wl_array
    mask   = (wl >= wl_range[0]) & (wl <= wl_range[1])
    x      = wl[mask]
    n_comp = len(p0) // 3

    if bounds is None:
        lower, upper = [], []
        for k in range(n_comp):
            lower += [0.0,  p0[3*k+1] - 15.0, 1.0]
            upper += [1e6,  p0[3*k+1] + 15.0, 20.0]
        bounds = (lower, upper)

    xp = xeol.scan.xpoints * 1e3
    yp = xeol.scan.ypoints * 1e3

    # Pre-extract windowed spectra — each worker receives a small 1-D array,
    # not the full spectra matrix, keeping pickling overhead minimal.
    Y = xeol.spectra[:, mask].astype(float)

    args_list = []
    for idx in range(xeol.scan.length):
        ii, jj = xeol.scan.index_to_ij(idx)
        args_list.append((idx, x, Y[idx], p0, bounds, n_comp,
                          float(xp[ii]), float(yp[jj])))

    rows: list = [None] * xeol.scan.length
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in tqdm(pool.map(_fit_one, args_list), total=len(args_list), desc='Fitting'):
            rows[r['point_idx']] = r

    return pd.DataFrame(rows).sort_values('point_idx').reset_index(drop=True)


# ── Heatmaps ──────────────────────────────────────────────────────────────────

_COMP_COLS = ['amp', 'cen', 'fwhm', 'r2']
_COMP_LABELS = {
    'amp':  'Amplitude (counts)',
    'cen':  'Centre (nm)',
    'fwhm': 'FWHM (nm)',
    'r2':   'R² component',
}
_COMP_CMAPS = {
    'amp':  'viridis',
    'cen':  'RdBu_r',
    'fwhm': 'plasma',
    'r2':   'RdYlGn',
}


def plot_gaussian_maps(
    df: pd.DataFrame,
    xeol,
    n_components: int,
    percentile_clip: tuple[float, float] = (2, 98),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """Spatial heatmaps of Gaussian fit parameters.

    Layout: one row per component (amp | cen | fwhm | r²)
            + a summary row (r²_total | fit_success rate).
    """
    from lauexplore.plots.base import _as_grid

    n_rows = n_components + 1
    n_cols = 4
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3.0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    xp     = xeol.scan.xpoints * 1e3
    yp     = xeol.scan.ypoints * 1e3
    extent = [float(xp[0]), float(xp[-1]), float(yp[0]), float(yp[-1])]
    ok     = df['fit_success'].astype(bool)

    def _to_grid(col: str) -> np.ndarray:
        flat = np.full(xeol.scan.length, np.nan)
        flat[df.loc[ok, 'point_idx'].values] = df.loc[ok, col].values
        return _as_grid(flat, xeol.scan)

    def _show(ax, grid: np.ndarray, title: str, cmap: str,
              lo: float | None = None, hi: float | None = None) -> None:
        vals = grid[~np.isnan(grid)]
        if lo is None:
            lo = float(np.nanpercentile(vals, percentile_clip[0])) if len(vals) else 0
        if hi is None:
            hi = float(np.nanpercentile(vals, percentile_clip[1])) if len(vals) else 1
        im = ax.imshow(grid, origin='lower', aspect='equal',
                       extent=extent, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation='none')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x (mm)', fontsize=7)
        ax.set_ylabel('y (mm)', fontsize=7)
        ax.tick_params(labelsize=7)

    for k in range(n_components):
        for c, key in enumerate(_COMP_COLS):
            col = f'{key}_{k}'
            if col not in df.columns:
                axes[k, c].axis('off')
                continue
            lo = hi = None
            _show(axes[k, c], _to_grid(col),
                  f'Comp {k} — {_COMP_LABELS[key]}', _COMP_CMAPS[key], lo, hi)

    # Summary row: r2_total and fit success rate
    _show(axes[-1, 0], _to_grid('r2_total'), 'R² total', 'RdYlGn')

    success_flat = np.zeros(xeol.scan.length)
    success_flat[df.loc[ok, 'point_idx'].values] = 1.0
    _show(axes[-1, 1], _as_grid(success_flat, xeol.scan),
          'Fit success', 'RdYlGn', 0.0, 1.0)

    for c in range(2, n_cols):
        axes[-1, c].axis('off')

    fig.suptitle(
        f'Multi-Gaussian fit  —  {n_components} components  '
        f'({ok.sum()}/{len(df)} points converged)',
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_mean_comparison(
    df: pd.DataFrame,
    xeol,
    wl_range: tuple[float, float],
    n_components: int,
    figsize: tuple[float, float] = (9, 4),
) -> plt.Figure:
    """Mean spectrum vs mean multi-Gaussian fit, with individual components.

    For each component k, the mean curve is computed by averaging the fitted
    Gaussian curves across all converged points — more accurate than using
    mean parameters directly.
    """
    wl   = xeol.wl_array
    mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
    x    = wl[mask]

    ok      = df['fit_success'].astype(bool)
    df_ok   = df[ok]
    indices = df_ok['point_idx'].values

    # Mean raw spectrum over converged points only
    mean_spec = xeol.spectra[indices][:, mask].astype(float).mean(axis=0)

    # Mean curve per component: average Gaussian curves (not average params)
    mean_comps = []
    for k in range(n_components):
        amp_col  = f'amp_{k}'
        cen_col  = f'cen_{k}'
        fwhm_col = f'fwhm_{k}'
        if amp_col not in df_ok.columns:
            continue
        curves = np.array([
            _gaussian(x, row[amp_col], row[cen_col], row[fwhm_col] / 2.355)
            for _, row in df_ok[[amp_col, cen_col, fwhm_col]].iterrows()
        ])
        mean_comps.append(curves.mean(axis=0))

    mean_total = sum(mean_comps)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, mean_spec,   color='black', lw=1.5, label='Mean spectrum', zorder=3)
    ax.plot(x, mean_total,  color='red',   lw=1.5, linestyle='--',
            label='Mean total fit', zorder=3)

    colors = plt.cm.tab10.colors
    for k, curve in enumerate(mean_comps):
        ax.fill_between(x, curve, alpha=0.3, color=colors[k % 10], label=f'Comp {k}')
        ax.plot(x, curve, color=colors[k % 10], lw=0.8)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.set_title(
        f'Mean spectrum vs fit  ({len(df_ok)} converged points)',
        fontsize=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py

from ipywidgets import IntRangeSlider, IntSlider, VBox
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from lauexplore.scan import Scan
from lauexplore import plots
from lauexplore._parsers import _h5


@dataclass
class XEOL:
    """
    Container for analyzing and plotting visible-light emission (XEOL).

    All spectral parameters are given in wavelength (nm):

    - channel : wavelength (nm) of interest
    - roi     : (start_nm, end_nm) to integrate
    - ref_data : can be range inside each spectro (start_nm, end_nm) where the mean value of this range 
                will be used for subtraction or a given value (float) for directly subtration
    """

    spectra: np.ndarray                 # (Npoints, Nchannels) — raw on input, treated after __post_init__
    scan: Scan
    wl_array: np.ndarray  # (Nchannels,)
    channel: float | None = None        # always in nm
    roi: tuple[float, float] | None = None  # always in nm
    wavelength: float | tuple[float, float] | None = None
    normalize_to_monitor: bool = True
    norm_zone: tuple[float, float] | None = None
    ref_data: tuple[int, int] | int | float | np.ndarray | None = None
    data: np.ndarray | None = None
    spectra_raw: np.ndarray | None = None  # untouched, set from `spectra` input

    def __post_init__(self):
        wl = self.wl_array

        if self.spectra_raw is None:
            self.spectra_raw = self.spectra

        # --- reference subtraction ---
        if isinstance(self.ref_data, tuple):
            start_ref, end_ref = self.ref_data
            ref_idx0 = int(np.abs(wl - start_ref).argmin())
            ref_idx1 = int(np.abs(wl - end_ref).argmin())
            ref = np.mean(self.spectra_raw[:, ref_idx0:ref_idx1], axis=1)[:, None]
        elif isinstance(self.ref_data, (int, float, np.ndarray)):
            ref = self.ref_data
        else:
            ref = 0

        spectra = self.spectra_raw - ref
        spectra = np.where(spectra < 0, 0, spectra)

        # --- monitor normalisation (whole spectrum, per point) ---
        if self.normalize_to_monitor:
            spectra = spectra * 1e5 / self.scan.monitor_data[:, None]

        # --- norm_zone normalisation (whole spectrum, per point) ---
        if self.norm_zone is not None:
            z0, z1 = self.norm_zone
            i0 = int(np.abs(wl - z0).argmin())
            i1 = int(np.abs(wl - z1).argmin())
            dead_map = np.sum(spectra[:, i0:i1 + 1], axis=1)[:, None]
            # TODO: rescale by dead_map's order of magnitude to avoid collapsing
            # intensities by several decades (disabled for now — would change the
            # scale of existing processed maps). Apply later.
            # dead_median = np.median(dead_map)
            # norm_scale = 10 ** np.round(np.log10(dead_median)) if dead_median > 0 else 1.0
            # spectra = spectra / dead_map * norm_scale
            spectra = spectra / dead_map

        self.spectra = spectra

        if self.data is not None:
            return
        if self.roi is None and self.channel is None:
            return

        if self.roi is not None:
            start_nm, end_nm = self.roi
            idx0 = int(np.abs(wl - start_nm).argmin())
            idx1 = int(np.abs(wl - end_nm).argmin())
            data = np.sum(spectra[:, idx0:idx1 + 1], axis=1)
            if self.wavelength is None:
                self.wavelength = self.roi
        else:
            idx = int(np.abs(wl - self.channel).argmin())
            data = spectra[:, idx].copy()
            if self.wavelength is None:
                self.wavelength = float(wl[idx])

        self.data = data

    # ------------------------------------------------------------------
    @classmethod
    def from_h5(
        cls,
        filepath: str | Path,
        scan_number: int = 1,
        *,
        channel: float | None = None,             # nm
        roi: tuple[float, float] | None = None,   # nm
        normalize_to_monitor: bool = True,
        norm_zone: tuple[float, float] | None = None,  # nm
        ref_path: str | None = None,
        ref_data: tuple[int, int] | int | float | None = None,  # nm
    ) -> "XEOL":

        if channel is None and roi is None:
            raise ValueError("Provide either a spectral `channel` (nm) or a `roi=(start_nm,end_nm)`.")

        filepath = Path(filepath)

        # --- scan info ---
        scan = Scan.from_h5(filepath, scan_number)

        with h5py.File(filepath, "r") as h5f:
            # raw spectra (Npoints, Nchannels)
            spectra_raw = np.array(_h5.get_xeol(h5f, scan_number))

            # wavelength calibration (nm)
            wl_array = h5f[f"{scan_number}.1/measurement/qepro_det1"][0]

            if ref_path is not None:
                ref_data = h5f[ref_path][0]

        # __post_init__ handles: reference subtraction, monitor normalisation,
        # norm_zone normalisation, and `data` (ROI/channel) extraction.
        return cls(
            spectra=spectra_raw,
            wl_array=wl_array,
            scan=scan,
            channel=channel,
            roi=roi,
            normalize_to_monitor=normalize_to_monitor,
            norm_zone=norm_zone,
            ref_data=ref_data,
        )

    # ------------------------------------------------------------------
    def plot(
        self,
        *,
        width: int = 600,
        height: int = 600,
        zmin: float | None = None,
        zmax: float | None = None,
        percentile: float = 0.1,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        colorscale: str = "Viridis",
        log10: bool = False,
        cbartitle: str | None = None,
    ) -> go.Figure:

        if title is None:
            if isinstance(self.wavelength, tuple):
                w0, w1 = self.wavelength
                title = f"XEOL {w0:.0f}–{w1:.0f} nm"
            else:
                title = f"XEOL {self.wavelength:.0f} nm"

        z = plots.base._as_grid(self.data, self.scan)
        if zmin is None:
            zmin = float(np.nanpercentile(self.data, percentile))
        if zmax is None:
            zmax = float(np.nanpercentile(self.data, 100 - percentile))
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3

        customdata, hover = plots.scan_hovermenu(self.scan)

        fig = plots.base.heatmap(
            z, x, y,
            customdata=customdata,
            hovertemplate=hover,
            width=width,
            height=height,
            zmin=zmin,
            zmax=zmax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            colorscale=colorscale,
            log10=log10,
            cbartitle=cbartitle,
        )
        return fig

    # ------------------------------------------------------------------
    # INTERACTIVE: matplotlib + ipywidgets
    # ------------------------------------------------------------------

    def interactive_plot(
        self,
        *,
        figsize: tuple[float, float] = (12, 5),
        zmin: float | None = None,
        zmax: float | None = None,
        percentile: float = 2,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        cmap: str = "viridis",
        log10: bool = False,
        cbartitle: str | None = None,
    ):
        """Interactive XEOL visualization (Jupyter, requires %matplotlib widget).

        - Left panel: 2-D emission map; click any pixel to show its spectrum.
        - Right panel: spectrum at the clicked scan point.
        - Slider controls the wavelength ROI used to build the map.

        With ``log10=True`` the map shows log10 of the integrated intensity, so
        ``zmin``/``zmax`` are then given in log units.
        """
        if self.scan is None:
            raise ValueError("scan must not be None.")
        if self.wl_array is None:
            raise ValueError("XEOL object has no wavelength calibration (`wl_array`).")

        from IPython.display import display as _display

        wl = self.wl_array
        xp = self.scan.xpoints * 1e3   # (nbxpoints,) — unique x grid positions
        yp = self.scan.ypoints * 1e3   # (nbypoints,) — unique y grid positions

        # Flat position arrays (one entry per scan point, in scan order)
        # needed to find the nearest point on click.
        x_flat = np.empty(self.scan.length)
        y_flat = np.empty(self.scan.length)
        for k in range(self.scan.length):
            ii, jj = self.scan.index_to_ij(k)
            x_flat[k] = xp[ii]
            y_flat[k] = yp[jj]

        # ----- initial map data -----
        if self.roi is not None:
            init_w0, init_w1 = self.roi
        else:
            init_w0 = init_w1 = float(self.channel)

        def _z_flat(w0, w1):
            # self.spectra is already treated (ref subtraction + monitor +
            # norm_zone), so this is a plain ROI integration.
            idx0 = int(np.abs(wl - w0).argmin())
            idx1 = int(np.abs(wl - w1).argmin())
            z = np.sum(self.spectra[:, idx0:idx1 + 1], axis=1).astype(float)
            if log10:
                # non-positive sums would turn into -inf and drag the percentile
                # limits below down with them, so they are dropped instead
                z = np.log10(np.where(z > 0, z, np.nan))
            return z

        z_init = _z_flat(init_w0, init_w1)
        grid   = plots.base._as_grid(z_init, self.scan)

        lo = zmin if zmin is not None else float(np.nanpercentile(z_init, percentile))
        hi = zmax if zmax is not None else float(np.nanpercentile(z_init, 100 - percentile))

        # extent: [xmin, xmax, ymin, ymax] in mm; _as_grid returns (ny, nx)
        extent = [float(xp[0]), float(xp[-1]), float(yp[0]), float(yp[-1])]

        auto_title = title or (
            f"XEOL {init_w0:.0f}–{init_w1:.0f} nm"
            if self.roi is not None else f"XEOL {init_w0:.0f} nm"
        )

        # ----- figure -----
        plt.close('xeol_interactive')
        with plt.ioff():
            fig, (ax_map, ax_spec) = plt.subplots(
                1, 2, figsize=figsize, num='xeol_interactive'
            )

        im = ax_map.imshow(
            grid, origin='lower', aspect='equal',
            extent=extent, cmap=cmap, vmin=lo, vmax=hi,
            interpolation='none',
        )
        cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
        if cbartitle is not None:
            cbar.set_label(cbartitle)
        ax_map.set_title(auto_title)
        ax_map.set_xlabel(xlabel if xlabel is not None else 'x (mm)')
        ax_map.set_ylabel(ylabel if ylabel is not None else 'y (mm)')

        (marker,) = ax_map.plot([], [], '+', color='white', ms=14, mew=2, zorder=5)

        spec_line, = ax_spec.plot(wl, self.spectra[0, :], color='C0')
        ax_spec.set_xlabel('Wavelength (nm)')
        ax_spec.set_ylabel('Intensity (a.u.)')
        spec_text = ax_spec.set_title('Click map to show spectrum')
        fig.tight_layout()

        # ----- wavelength slider -----
        if self.roi is not None:
            slider_wl = IntRangeSlider(
                value=[int(init_w0), int(init_w1)],
                min=int(wl.min()), max=int(wl.max()), step=1,
                description='λ (nm)', continuous_update=False,
                layout={'width': '700px'},
            )

            def on_wl_change(change):
                w0, w1 = slider_wl.value
                z = _z_flat(w0, w1)
                im.set_data(plots.base._as_grid(z, self.scan))
                im.set_clim(np.nanpercentile(z, percentile),
                            np.nanpercentile(z, 100 - percentile))
                ax_map.set_title(f'XEOL {w0:.0f}–{w1:.0f} nm')
                fig.canvas.draw_idle()
        else:
            slider_wl = IntSlider(
                value=int(init_w0),
                min=int(wl.min()), max=int(wl.max()), step=1,
                description='λ (nm)', continuous_update=False,
                layout={'width': '700px'},
            )

            def on_wl_change(change):
                w = slider_wl.value
                z = _z_flat(w, w)
                im.set_data(plots.base._as_grid(z, self.scan))
                im.set_clim(np.nanpercentile(z, percentile),
                            np.nanpercentile(z, 100 - percentile))
                ax_map.set_title(f'XEOL {w:.0f} nm')
                fig.canvas.draw_idle()

        slider_wl.observe(on_wl_change, names='value')

        # ----- click handler -----
        def _on_click(event):
            if event.inaxes is not ax_map or event.xdata is None:
                return
            dists     = np.hypot(x_flat - event.xdata, y_flat - event.ydata)
            point_idx = int(np.argmin(dists))
            spec      = self.spectra[point_idx, :]
            spec_line.set_ydata(spec)
            ax_spec.relim()
            ax_spec.autoscale_view()
            spec_text.set_text(f'Point {point_idx}  '
                               f'({x_flat[point_idx]:.3f}, {y_flat[point_idx]:.3f}) mm')
            marker.set_data([event.xdata], [event.ydata])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('button_press_event', _on_click)

        _display(VBox([slider_wl, fig.canvas]))

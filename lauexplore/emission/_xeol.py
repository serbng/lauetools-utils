from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py

from ipywidgets import IntRangeSlider, VBox, Output

import plotly.graph_objects as go

from lauexplore.scan import Scan
from lauexplore import plots
from lauexplore._parsers import _h5

@dataclass
class XEOL:
    """
    Container for analyzing and plotting visible-light emission (XEOL).

    Supports:
    - selecting a single spectral channel
    - summing over a Region Of Interest (roi = (start, end))
    - interactive wavelength slider to choose ROI in nm

    The interface mirrors the Fluorescence class for consistency.
    """

    spectra: np.ndarray          # full spectra per scan point (Npoints, Nchannels)
    data: np.ndarray             # extracted intensity (1D flattened)
    wavelength: float | tuple[float, float] | None
    scan: Scan
    wl_array: np.ndarray | None = None
    channel: int | None = None
    roi: tuple[int, int] | None = None
    normalize_to_monitor: bool = True

    # ------------------------------------------------------------------
    @classmethod
    def from_h5(
        cls,
        filepath: str | Path,
        scan_number: int = 1,
        *,
        channel: int | None = None,
        roi: tuple[int, int] | None = None,
        normalize_to_monitor: bool = True,
        ref_path: str | None = None,
        ref_range: tuple[int, int] | None = None,
    ) -> "XEOL":

        if channel is None and roi is None:
            raise ValueError("Provide either a spectral `channel` or a `roi=(start,end)`.")

        filepath = Path(filepath)

        # -- Load scan info ----------------------------------------
        scan = Scan.from_h5(filepath, scan_number)
        mon = scan.monitor_data

        with h5py.File(filepath, "r") as h5f:

            # Load raw spectra (Npoints, Nchannels)
            spectra = _h5.get_xeol(h5f, scan_number)

            # Reference subtraction
            if ref_path is not None:
                ref = h5f[ref_path][0]
            elif ref_range is not None:
                ref = np.mean(spectra[:, ref_range[0]:ref_range[1]])
            else:
                ref = 0.0

            spectra = spectra - ref

            # Load wavelength calibration
            wl_array = h5f[f"{scan_number}.1/measurement/qepro_det1"][0]

        # -----------------------------------------------------------
        # Extract intensity (channel or ROI)
        # -----------------------------------------------------------

        if roi is not None:
            start, end = roi
            data = np.sum(spectra[:, start:end+1], axis=1)
            wavelength = (wl_array[start], wl_array[end])
        else:
            data = spectra[:, channel]
            wavelength = wl_array[channel]

        # Normalize
        if normalize_to_monitor:
            data = data / mon

        return cls(
            spectra=spectra,
            data=data,
            wavelength=wavelength,
            wl_array=wl_array,
            scan=scan,
            channel=channel,
            roi=roi,
            normalize_to_monitor=normalize_to_monitor,
        )

    def plot(
        self,
        *,
        width: int = 600,
        height: int = 600,
        zmin: float | None = None,
        zmax: float | None = None,
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

        # Convert to grid
        z = plots.base._as_grid(self.data, self.scan)
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
    # INTERACTIVE SLIDER PLOT (nm range)
    # ------------------------------------------------------------------

    def interactive_plot(
        self,
        *,
        width: int = 1000,
        height: int = 500,
        zmin: float | None = None,
        zmax: float | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        colorscale: str = "Viridis",
        log10: bool = False,
        cbartitle: str | None = None,
    ):
        """
        Interactive XEOL ROI selection using a wavelength slider (nm).
        Parameters forwarded to heatmap():
        ---------------------------------------------------
        width : int
        height : int
        zmin, zmax : float or None
        title : str or None
        xlabel, ylabel : str or None
        colorscale : str
        log10 : bool
        cbartitle : str or None
        """

        if self.scan is None:
            raise ValueError("scan must not be None.")

        if not hasattr(self, "wl_array") or self.wl_array is None:
            raise ValueError(
                "XEOL object has no wavelength calibration (`wl_array`). "
                "Provide it manually or use XEOL.from_h5()."
            )

        wl = self.wl_array  # (Nchannels,)
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3
        customdata, hover = plots.scan_hovermenu(self.scan)

        # -----------------------------
        # Slider in wavelength (nm)
        # -----------------------------
        slider = IntRangeSlider(
            value=[int(wl.min()), int(wl.min()) + 50],
            min=int(wl.min()),
            max=int(wl.max()),
            step=1,
            description="λ (nm)",
            continuous_update=False,
            layout={'width': '700px'},
        )

        out = Output()

        # -----------------------------
        # Update function
        # -----------------------------
        def update_plot(change):
            with out:
                out.clear_output()

                # Slider → wavelength range (nm)
                w0, w1 = slider.value
                idx0 = np.argmin(np.abs(wl - w0))
                idx1 = np.argmin(np.abs(wl - w1))

                # Integrate spectra in ROI
                z_flat = np.sum(self.spectra[:, idx0:idx1+1], axis=1)

                if self.normalize_to_monitor:
                    z_flat = z_flat / self.scan.monitor_data

                # Reshape to 2D
                z = plots.base._as_grid(z_flat, self.scan)

                # Auto title if user didn't pass one
                dynamic_title = (
                    title if title is not None
                    else f"XEOL {w0:.0f}–{w1:.0f} nm"
                )

                fig = plots.base.heatmap(
                    z,
                    x,
                    y,
                    customdata=customdata,
                    hovertemplate=hover,
                    width=width,
                    height=height,
                    zmin=zmin,
                    zmax=zmax,
                    title=dynamic_title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    colorscale=colorscale,
                    log10=log10,
                    cbartitle=cbartitle,
                )

                fig.show()

        # -----------------------------
        # Connect slider + initial draw
        # -----------------------------
        slider.observe(update_plot, names="value")
        update_plot(None)

        return VBox([slider, out])

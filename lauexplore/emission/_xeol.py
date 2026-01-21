from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py

from ipywidgets import IntRangeSlider, IntSlider, VBox
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

    spectra: np.ndarray                 # (Npoints, Nchannels)
    scan: Scan
    wl_array: np.ndarray  # (Nchannels,)
    channel: float | None = None        # always in nm
    roi: tuple[float, float] | None = None  # always in nm
    wavelength: float | tuple[float, float] | None = None
    normalize_to_monitor: bool = True
    data: np.ndarray | None = None

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
        ref_path: str | None = None,
        ref_data: tuple[int, int] | int | float | None = None,  # nm
    ) -> "XEOL":

        if channel is None and roi is None:
            raise ValueError("Provide either a spectral `channel` (nm) or a `roi=(start_nm,end_nm)`.")

        filepath = Path(filepath)

        # --- scan info ---
        scan = Scan.from_h5(filepath, scan_number)
        mon = scan.monitor_data

        with h5py.File(filepath, "r") as h5f:
            # raw spectra (Npoints, Nchannels)
            spectra = np.array(_h5.get_xeol(h5f, scan_number))

            # wavelength calibration (nm)
            wl_array = h5f[f"{scan_number}.1/measurement/qepro_det1"][0]

            # --- reference subtraction ---
            if ref_path is not None:
                ref = h5f[ref_path][0]

            elif isinstance(ref_data, tuple):
                start_ref, end_ref = ref_data
                ref_idx0 = int(np.abs(wl_array - start_ref).argmin())
                ref_idx1 = int(np.abs(wl_array - end_ref).argmin())
                # mean over reference window, per scan point
                ref = np.mean(spectra[:, ref_idx0:ref_idx1], axis=1)[:, None]
                
            elif isinstance(ref_data, (int,float)): 
                ref = ref_data

            else:
                ref = 0
                
            spectra = spectra - ref
            spectra = np.where(spectra < 0, 0, spectra)

        # --- extract intensity for mapping ---
        if roi is not None:
            start_nm, end_nm = roi
            idx0 = int(np.abs(wl_array - start_nm).argmin())
            idx1 = int(np.abs(wl_array - end_nm).argmin())
            data = np.sum(spectra[:, idx0:idx1 + 1], axis=1)
            wavelength = (start_nm, end_nm)
        else:
            idx = int(np.abs(wl_array - channel).argmin())
            data = spectra[:, idx]
            wavelength = wl_array[idx]

        if normalize_to_monitor:
            data = data*1e5 / mon

        return cls(
            spectra=spectra,
            wavelength=wavelength,
            wl_array=wl_array,
            scan=scan,
            channel=channel,
            roi=roi,
            normalize_to_monitor=normalize_to_monitor,
            data = data
        )

    # ------------------------------------------------------------------
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
    # INTERACTIVE: ipywidgets + FigureWidget (rápido e estável)
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
        Interactive XEOL visualization (Jupyter):

        - IntRangeSlider (λ in nm) controla o ROI → atualiza apenas o mapa.
        - IntSlider (point index) controla o espectro → atualiza apenas o trace de espectro.
        - Usa go.FigureWidget para atualizar em tempo real, sem recriar a figura.
        """

        if self.scan is None:
            raise ValueError("scan must not be None.")
        if self.wl_array is None:
            raise ValueError("XEOL object has no wavelength calibration (`wl_array`).")

        wl = self.wl_array
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3
        customdata, hover = plots.scan_hovermenu(self.scan)

        # ----- initial ROI in nm -----
        if self.roi is not None:
            init_w0, init_w1 = self.roi
        else:
            init_w0 = init_w1 = float(self.channel)

        idx0_init = int(np.abs(wl - init_w0).argmin())
        idx1_init = int(np.abs(wl - init_w1).argmin())

        z_flat_init = np.sum(self.spectra[:, idx0_init:idx1_init + 1], axis=1)
        if self.normalize_to_monitor:
            z_flat_init = z_flat_init / self.scan.monitor_data
        z_init = plots.base._as_grid(z_flat_init, self.scan)

        # ----- build figure once (FigureWidget) -----
        base_fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.55, 0.45],
            subplot_titles=[
                (title if title is not None else f"XEOL {init_w0:.0f}–{init_w1:.0f} nm"),
                "Spectrum",
            ],
        )

        fig = go.FigureWidget(base_fig)

        # LEFT: heatmap (um único trace)
        heatmap_fig = plots.base.heatmap(
            z_init,
            x,
            y,
            customdata=customdata,
            hovertemplate=hover,
            width=width // 2,
            height=height,
            zmin=zmin,
            zmax=zmax,
            title=None,
            xlabel=xlabel,
            ylabel=ylabel,
            colorscale=colorscale,
            log10=log10,
            cbartitle=cbartitle,
        )
        for tr in heatmap_fig.data:
            fig.add_trace(tr, row=1, col=1)
            self.data = tr.z.ravel()

        # guarda referência ao trace de mapa (assumindo 1 trace do heatmap)
        heat_trace = fig.data[0]
        

        # RIGHT: spectrum trace
        spec_trace = go.Scatter(
            x=wl,
            y=self.spectra[0, :],
            mode="lines",
            name=None,
        )
        fig.add_trace(spec_trace, row=1, col=2)

        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
        fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=2)
        fig.update_layout(width=width, height=height)
        

        # ----- sliders -----
        slider_roi = IntRangeSlider(
            value=[int(init_w0), int(init_w1)],
            min=int(wl.min()),
            max=int(wl.max()),
            step=1,
            description="λ (nm)",
            continuous_update=False,
            layout={'width': '700px'},
        )

        slider_idx = IntSlider(
            value=0,
            min=0,
            max=self.spectra.shape[0] - 1,
            step=1,
            description="Point index",
            continuous_update=True,
            layout={'width': '400px'},
        )

        # ----- callbacks -----

        def on_roi_change(change):
            w0, w1 = slider_roi.value
            idx0 = int(np.abs(wl - w0).argmin())
            idx1 = int(np.abs(wl - w1).argmin())

            z_flat = np.sum(self.spectra[:, idx0:idx1 + 1], axis=1)
            if self.normalize_to_monitor:
                z_flat = z_flat / self.scan.monitor_data
            z_new = plots.base._as_grid(z_flat, self.scan)

            # atualiza somente o mapa
            heat_trace.z = z_new
            fig.layout.annotations[0].text = f"XEOL {w0:.0f}–{w1:.0f} nm"

        def on_index_change(change):
            idx = slider_idx.value
            new_y = self.spectra[idx, :]

            fig.data[-1].y = new_y
            fig.layout.annotations[1].text = f"Spectrum {idx}"

        slider_roi.observe(on_roi_change, names="value")
        slider_idx.observe(on_index_change, names="value")

        return VBox([slider_roi, slider_idx, fig])

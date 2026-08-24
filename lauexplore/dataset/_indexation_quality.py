from dataclasses import dataclass
from typing import Literal
import numpy as np
import plotly.graph_objects as go
from plotly.express.colors import get_colorscale

from lauexplore.scan import Scan
from lauexplore.dataset import Dataset
from lauexplore.plots.base import _as_grid
from lauexplore.plots.composite import tiles
from lauexplore.plots.hovermenus import scan_hovermenu
from lauexplore._utils import compute_zlimits, nonlinear_colorscale

@dataclass
class IndexationQuality:
    """
    Container for visualizing Laue indexation-quality metrics.

    You may provide either:
    - `scan` directly, OR
    - a dataset containing a scan (via from_dataset).

    Available plots:
    ----------------
    - plot_nspots(): number of indexed spots
    - plot_mean_dev(): mean pixel deviation
    """

    number_indexed_spots: np.ndarray
    mean_pixel_deviation: np.ndarray

    _scan: Scan | None = None
    dataset: Dataset | None = None

    @property
    def scan(self) -> Scan:
        """Return the Scan object, either from _scan or dataset.scan."""
        if self._scan is not None:
            return self._scan
        if self.dataset is not None and self.dataset.scan is not None:
            return self.dataset.scan
        raise ValueError(
            "No scan available. Provide `scan=` or use "
            "IndexationQuality.from_dataset(dataset)."
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
    ) -> "IndexationQuality":
        """
        Create an IndexationQuality container from a Dataset instance.
        """
        if dataset.number_indexed_spots is None:
            raise ValueError("Dataset missing `number_indexed_spots`.")
        if dataset.mean_pixel_deviation is None:
            raise ValueError("Dataset missing `mean_pixel_deviation`.")

        return cls(
            number_indexed_spots=np.asarray(dataset.number_indexed_spots),
            mean_pixel_deviation=np.asarray(dataset.mean_pixel_deviation),
            dataset=dataset,
            _scan=None,
        )

    def _plot_component(
        self,
        data: np.ndarray,
        title: str,
        *,
        colorscale: str = "viridis",
        width: int = 500,
        height: int = 500,
        cbar_title: str | None = None,
        cbar_width: int = 20,
        cbar_padding: int = 0,
        mask: np.ndarray | None = None,
        multiplier: float = 1.0,
        scale: str | None = None,
        zmin: float | None = None,
        zmax: float | None = None,
        zmid: float | None = None,
    ) -> go.Figure:
        """
        Internal helper for plotting a single map (nspots or mean_dev).
        """
        scan = self.scan
        data = data.copy()

        if mask is not None:
            data = np.where(mask, data, np.nan)

        n_total = data.size
        n_used = int(np.count_nonzero(np.isfinite(data)))
        n_masked = n_total - n_used
        print(
            f"{title}: mean={np.nanmean(data):.2f}, sigma={np.nanstd(data):.2f}, "
            f"used={n_used}/{n_total} px, masked={n_masked} px"
        )

        # ---- compute automatic z-limits ----
        if scale is not None:
            zmin, zmax, zmid = compute_zlimits(data, scale=scale, multiplier=multiplier)

        # ---- grid conversion ----
        grid = _as_grid(data * multiplier, scan)

        # ---- hover menu ----
        customdata, hovertemplate = scan_hovermenu(scan)

        # ---- build single heatmap ----
        fig = tiles(
            [grid],
            x=scan.xpoints * 1e3,
            y=scan.ypoints * 1e3,
            nrows=1,
            ncols=1,
            colorscale=colorscale,
            width=width,
            height=height,
            customdata=customdata,
            hovertemplate=hovertemplate,
            cbar_title=cbar_title,
            cbar_width=cbar_width,
            cbar_padding=cbar_padding,
            subplot_titles=[title],
        )

        # ---- apply z-limits and nonlinear midpoint shift ----
        base_colorscale = get_colorscale(colorscale)
        trace = fig.data[0]
        if zmin is not None:
            trace.update(zmin=zmin)
        if zmax is not None:
            trace.update(zmax=zmax)
        if zmid is not None:
            shifted = nonlinear_colorscale(
                base_colorscale, zmin=zmin, zmax=zmax, zmid=zmid
            )
            trace.update(colorscale=shifted)

        fig.update_layout(width=width, height=height, title=title)
        return fig

    def plot_nspots(
        self,
        **kwargs,
    ) -> go.Figure:
        """
        Plot map of the number of indexed spots.

        Parameters
        ----------
        kwargs : forwarded to _plot_component
        """
        return self._plot_component(
            self.number_indexed_spots,
            title="Number of indexed spots",
            **kwargs,
        )

    def plot_mean_dev(
        self,
        **kwargs,
    ) -> go.Figure:
        """
        Plot map of the mean pixel deviation (in pixels).

        Parameters
        ----------
        kwargs : forwarded to _plot_component
        """
        return self._plot_component(
            self.mean_pixel_deviation,
            title="Mean pixel deviation (px)",
            **kwargs,
        )

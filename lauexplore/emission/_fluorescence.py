from pathlib import Path
from dataclasses import dataclass
import numpy as np
import h5py
import plotly.graph_objects as go

from lauexplore.scan import Scan
from lauexplore._parsers import _h5
from lauexplore import plots

@dataclass
class Fluorescence:
    data: np.ndarray
    material: str
    scan: Scan | None = None
    
    @classmethod
    def from_h5(cls, 
            filepath: str | Path, 
            material: str, 
            scan_number: int = 1,
            normalize_to_monitor=True,
        ):
        with h5py.File(filepath) as h5f:
            data = _h5.get_fluo(h5f, material, scan_number)
            
        scan = Scan.from_h5(filepath, scan_number)
        if normalize_to_monitor: 
            mon = scan.monitor_data
            data  = data/mon
            
        return cls(data, material, scan)
    
    def plot(self,
            width: int = 600,
            height: int = 600,
            zmin: float | None = None,
            zmax: float | None = None,
            title: str | None = None,
            xlabel: str | None = None,
            ylabel: str | None = None,
            colorscale: str = "Viridis",
            log10: bool = False,
            cbartitle: str | None = None
        ) -> go.Figure:
        if self.scan is None:
            raise ValueError("scan must not be None.")
        
        z = plots.base._as_grid(self.data, self.scan)
        x = self.scan.xpoints * 1e3
        y = self.scan.ypoints * 1e3
        
        customdata, hovertemplate = plots.scan_hovermenu(self.scan)
        
        fluoplot = plots.base.heatmap(
            z, x, y,
            customdata=customdata,
            hovertemplate=hovertemplate,
            width = width,
            height = height,
            zmin = zmin,
            zmax = zmax,
            title = title or f"Fluorescence plot – {self.material} ({self.scan.nbxpoints} × {self.scan.nbxpoints})",
            xlabel = xlabel,
            ylabel = ylabel,
            colorscale = colorscale,
            log10 = log10,
            cbartitle = cbartitle,
        )
        
        return fluoplot

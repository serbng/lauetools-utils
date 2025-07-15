import numpy as np
from pandas import DataFrame
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

def indexation_plot(peaks: DataFrame):
    # ================ Data to display in the hover menu ================
    # numpy.ndarray containing data for the tooltip
    # data will be Xexp, Yexp, h, k, l, intensity, energy, spot_index
    custom_data = np.hstack((
                peaks[['Xexp', 'Yexp', 'h','k','l','Intensity','Energy']].values,
                np.atleast_2d(peaks.index.values).T # column vector with spot_index values
            ))
    
    # ===================== How to display the menu =====================
    hover_template = (
            "(%{customdata[0]}, %{customdata[1]})<br>"
            "Miller indices: [%{customdata[2]}, %{customdata[3]}, %{customdata[4]}]<br>"
            "I = %{customdata[5]:8.2f} [cts]<br>"
            "E = %{customdata[6]:6.3f} [keV]<br>"
            "spot index = %{customdata[7]}"
        )
    
    # ========== Create the figure widget with the two plots ===========
    fig = go.FigureWidget(
        data=[
            go.Scatter(
                x=peaks['Xexp'], y=peaks['Yexp'],
                mode='markers',
                name='Experimental',
                hovertemplate=hover_template,
                customdata=custom_data
            ),
            go.Scatter(
                x=peaks.get('Xtheo', []), y=peaks.get('Ytheo', []),
                mode='markers',
                name='Theoretical',
                visible=False,
                marker=dict(symbol='star', size=10, color='rgba(0,0,0,0)', line=dict(color='red', width=2))
            )
        ],
        layout=go.Layout(
            title='Indexed peak positions',
            width=800,
            height=700,
            xaxis=dict(title='Xexp', range=[0,2018]),
            yaxis=dict(title='Yexp', range=[2018,0]),
            showlegend=True
        )
    )
    # Grab the references to the traces
    exp_scatter, theo_scatter = fig.data

    # =================== Create the navigation widgets ====================
    toggle_button = widgets.ToggleButton(
        value=False,
        description='Camera positions ↔ Scattering angles',
        tooltip='Switch between (X, Y) and (2θ, χ) plots',
        layout=widgets.Layout(width='250px')
    )
    show_theo = widgets.Checkbox(
        value=False,
        description='Show Theoretical spots',
        layout=widgets.Layout(width='400px')
    )

    # ======================== Callback definition =========================
    def on_change(*args):
        if not toggle_button.value:
            # Camera space plot
            exp_x, exp_y = 'Xexp','Yexp'
            theo_x, theo_y = 'Xtheo','Ytheo'
            xlabel, ylabel = 'x pixel','y pixel'
            xlim, ylim = [0, 2018], [2018, 0]
        else:
            # Angle space plot
            exp_x, exp_y = '2θexp','χexp'
            theo_x, theo_y = '2θtheo','χtheo'
            xlabel, ylabel = '2θ','χ'
            xlim, ylim = [40, 140], [-40, 40]
        
        with fig.batch_update():
            exp_scatter.x = peaks[exp_x]
            exp_scatter.y = peaks[exp_y]
            fig.layout.xaxis.title = xlabel
            fig.layout.yaxis.title = ylabel
            fig.layout.xaxis.range = xlim
            fig.layout.yaxis.range = ylim

            # now handle theoretical overlay: only if user checked it AND both theo cols exist
            if show_theo.value and theo_x in peaks.columns and theo_y in peaks.columns:
                theo_scatter.x = peaks[theo_x]
                theo_scatter.y = peaks[theo_y]
                theo_scatter.visible = True
            else:
                theo_scatter.visible = False

    # ====================== Link callback to buttons ======================
    toggle_button.observe(on_change, names='value')
    show_theo.observe(on_change, names='value')

    # ============================== Display ===============================
    ui = widgets.HBox([toggle_button, show_theo])
    display(ui, fig)
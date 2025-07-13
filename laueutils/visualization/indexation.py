import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display

def indexation_plot(df):

    # 1) Create a FigureWidget with two traces up front
    fig = go.FigureWidget(
        data=[
            go.Scatter(
                x=df['Xexp'], y=df['Yexp'],
                mode='markers',
                name='Experimental',
                hovertemplate=(
                    "(%{customdata[0]}, %{customdata[1]})<br>"
                    "Miller indices: [%{customdata[2]}, %{customdata[3]}, %{customdata[4]}]<br>"
                    "I = %{customdata[5]:8.2f} [cts]<br>"
                    "E = %{customdata[6]:6.3f} [keV]<br>"
                    "spot index = %{customdata[7]}"
                ),
                customdata= np.hstack(
                    (
                        df[['Xexp', 'Yexp', 'h','k','l','Intensity','Energy']].values,
                        np.atleast_2d(df.index.values).T
                    )
                )
            ),
            
            go.Scatter(
                x=df.get('Xtheo', []), y=df.get('Ytheo', []),
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

    # Grab references to the traces
    scatter_exp, scatter_theo = fig.data

    # 2) Create the widgets
    toggle_btn = widgets.ToggleButton(
        value=False,
        description='Camera positions ↔ Scattering angles',
        tooltip='Toggle Experimental axes',
        layout=widgets.Layout(width='250px')
    )
    show_theo = widgets.Checkbox(
        value=False,
        description='Show Theoretical spots',
        layout=widgets.Layout(width='400px')
    )

    # 3) Define the callback
    def on_change(*args):
        if not toggle_btn.value:
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

        if exp_x not in df.columns or exp_y not in df.columns:
            return
        
        with fig.batch_update():
            scatter_exp.x = df[exp_x]
            scatter_exp.y = df[exp_y]
            fig.layout.xaxis.title = xlabel
            fig.layout.yaxis.title = ylabel
            fig.layout.xaxis.range = xlim
            fig.layout.yaxis.range = ylim

            # now handle theoretical overlay: only if user checked it AND both theo cols exist
            if show_theo.value and theo_x in df.columns and theo_y in df.columns:
                scatter_theo.x = df[theo_x]
                scatter_theo.y = df[theo_y]
                scatter_theo.visible = True
            else:
                scatter_theo.visible = False

    # 4) Wire up + display
    toggle_btn.observe(on_change, names='value')
    show_theo.observe(on_change, names='value')

    ui = widgets.HBox([toggle_btn, show_theo])
    display(ui, fig)
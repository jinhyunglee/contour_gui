''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np
import os

from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput
from bokeh.plotting import figure


class Visualizer(object):

    def __init__(self):
        # Read files
        self.root = "data"
        self.templates = np.load(os.path.join(self.root, "templates_222_SVD5run.npy"))

        # set up plots
        self.plot = figure(plot_height=400, plot_width=400, title="my sine wave",
                tools="crosshair,pan,reset,save,wheel_zoom",
                x_range=[0, 4*np.pi], y_range=[-2.5, 2.5])


        # Set up data
        N = 200
        x = np.linspace(0, 4*np.pi, N)
        y = np.sin(x)
        self.source = ColumnDataSource(data=dict(x=x, y=y))
 
        self.plot.line('x', 'y', source=self.source, line_width=3, line_alpha=0.6)

        # Set up widgets
        self.text = TextInput(title="title", value='my sine wave')
        self.offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
        self.amplitude = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
        self.phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
        self.freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.1)

        # Set up layouts and add to document
        inputs = widgetbox(
                self.text, self.offset, self.amplitude, self.phase, self.freq)

        self.text.on_change('value', self.update_title)
        for w in [self.offset, self.amplitude, self.phase, self.freq]:
            w.on_change('value', self.update_data)

        curdoc().add_root(row(inputs, self.plot, width=800))
        curdoc().title = "Sliders"

    # Set up callbacks
    def update_title(self, attrname, old, new):
        self.plot.title.text = self.text.value

    def update_data(self, attrname, old, new):

        # Get the current slider values
        a = self.amplitude.value
        b = self.offset.value
        w = self.phase.value
        k = self.freq.value

        # Generate the new curve
        x = np.linspace(0, 4*np.pi, N)
        y = a*np.sin(k*x + w) + b

        self.source.data = dict(x=x, y=y)


if __name__ == "__main__":
    vis = Visualizer()

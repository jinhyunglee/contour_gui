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
import scipy
import scipy.spatial

from bokeh.io import curdoc
from bokeh.layouts import column, row, widgetbox
from bokeh.models import ColumnDataSource, TapTool
from bokeh.models.widgets import Slider, TextInput, Select
from bokeh.plotting import figure
from bokeh.models.glyphs import Patches, MultiLine, Ellipse
from bokeh.models import Range1d
from bokeh.models.callbacks import CustomJS

# read data
root = "data"
templates = np.load(os.path.join(root, "templates_222_SVD5run.npy"))
templates = templates.transpose([2, 1, 0])
geometry = np.loadtxt("data/ej49_geometry1.txt")
sta = np.load("data/spike_train_STA_data.npz")
contours = np.load("data/contours.npy")
contours[:, 5] = contours[:, 5] * -1

class WaveForms(object):

    def __init__(self, wave_forms):
        self.wave_forms = wave_forms
        self.n_unit, self.n_channel, self.n_times = self.wave_forms.shape

        self.ptp = self.wave_forms.ptp(-1).max(-1)
        self.active_chans = self.wave_forms.ptp(-1) > 2.
        self.pairwise_dist()
        self.similar_units = self.pdist.argsort(axis=1)

    def pairwise_dist(self):
        self.pdist = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(
                self.wave_forms.reshape([self.n_unit, -1])))

    def get_template_lines(self, unit, scale=8., squeeze=0.8):
        x_range = np.arange(self.n_times) * squeeze
        x = [x_range + geometry[i, 0] for i in range(self.n_channel)]
        y = [self.wave_forms[unit, i, :] * scale + geometry[i, 1] for i in range(self.n_channel)]
        return x, y

    def get_error_bar_data(self, scale=8., squeeze=0.8):
        x_range = np.arange(self.n_times) * squeeze
        xpts = np.array([x_range[0], x_range[-1], x_range[-1], x_range[0]])
        ypts = np.array([-1, -1, 1, 1]) * scale
        xs = [xpts + xx for xx in geometry[:, 0]]
        ys = [ypts + yy for yy in geometry[:, 1]]
        return xs, ys


wf = WaveForms(templates)

colors = ['dodgerblue', 'darkorange', 'forestgreen', 'red']


# set up trace plots
off_idx = contours[:, 0] < 0
on_idx = np.logical_not(off_idx)
contours[np.isnan(contours)]=1000.
midget_idx = contours[:, 3] * contours[:, 4] < 1.
parasol_idx = np.logical_not(midget_idx)

cell_idx = []
cell_idx.append(np.logical_and(on_idx, midget_idx))
cell_idx.append(np.logical_and(off_idx, midget_idx))
cell_idx.append(np.logical_and(on_idx, parasol_idx))
cell_idx.append(np.logical_and(off_idx, parasol_idx))

plot_tiles = []
tile_glyphs = []
source_tiles = []
title_tiles = ["ON Midget", "OFF Midget", "ON Parasol", "OFF Parasol"]
for i in range(4):
    plot_tiles.append(figure(
        plot_height=400, plot_width=200, title=title_tiles[i],
        tools="pan,reset,wheel_zoom,tap",
        x_range=(0, 32), y_range=(0, 64)))
    source_tiles.append(ColumnDataSource(data=dict(
        x=contours[cell_idx[i], 1], y=contours[cell_idx[i], 2],
        w=contours[cell_idx[i], 3] * 2, h=contours[cell_idx[i], 4] * 2,
        a=contours[cell_idx[i], 5])))
    glyph_tiles = plot_tiles[-1].ellipse(
            x="x", y="y", width="w", height="h", angle="a",
            source=source_tiles[i], fill_alpha=0.)
    #plot_tiles[i].add_glyph(source_tiles[i], glyph_tiles)
    tile_glyphs.append(glyph_tiles)

TOTAL = 4
show_units = wf.similar_units[0, :TOTAL]

contour_sources = []
contour_glyphs = []
for i, unit in enumerate(show_units):
    c_ = contours[unit:unit + 1, :]

    contour_sources.append(ColumnDataSource(dict(
        x=c_[:, 1], y=c_[:, 2], w=c_[:, 3] * 2,
        h=c_[:, 4] * 2, a=c_[:, 5])))
    contour_glyphs.append(Ellipse(
        x="x", y="y", width="w", height="h", angle="a",
        fill_color=colors[i], fill_alpha=0.5))
    plot_tiles[0].add_glyph(contour_sources[-1], contour_glyphs[-1]) 

# Set up trace plots
plot = figure(plot_height=400, plot_width=800, title="Spatio-Temporal Trace",
              tools="pan,reset,wheel_zoom")

plot_rf = figure(plot_height=400, plot_width=200, title="RF Main Unit",
              tools="pan,reset,wheel_zoom", x_range=(0, 32), y_range=(0, 64))
plot_rf_sim = figure(plot_height=400, plot_width=200, title="RF Similar Unit",
              tools="pan,reset,wheel_zoom", x_range=(0, 32), y_range=(0, 64))

rf_source = ColumnDataSource(data=dict(d=[sta['STA_spatial'][0][1].reshape([64, 32])]))
plot_rf.image(image='d', x=0, y=0, dw=32, dh=64, source=rf_source, palette="Viridis256")

rf_sim_source = ColumnDataSource(data=dict(d=[sta['STA_spatial'][wf.similar_units[0, 1]][1].reshape([64, 32])]))
plot_rf_sim.image(image='d', x=0, y=0, dw=32, dh=64, source=rf_sim_source, palette="Viridis256")


# Spatio temporal sources
trace_sources = []
trace_glyphs = []

TOTAL = 4
show_units = wf.similar_units[0, :TOTAL]
for i, unit in enumerate(show_units):
    x, y = wf.get_template_lines(unit=unit)
    trace_sources.append(ColumnDataSource(data=dict(x=x, y=y)))
    trace_glyphs.append(MultiLine(
        xs="x", ys="y", line_width=2, line_alpha=0.7, line_color=colors[i]))
    plot.add_glyph(trace_sources[-1], trace_glyphs[-1])


exs, eys = wf.get_error_bar_data()
error_bar_source = ColumnDataSource(dict(
    xs=exs, ys=eys))

glyph = Patches(xs="xs", ys="ys", fill_alpha=0.2, line_alpha=0.0)
plot.add_glyph(error_bar_source, glyph)


# Set up widgets
widget_sortby = Select(
        title="Sorty by:", value="Unit ID",
        options=["Peak to Peak", "Unit ID"]) 
widget_unit = Slider(title="Unit", value=0.0, start=0.0, end=wf.n_unit - 1)
widget_simby = Select(
        title="Similarty by:", value="Trace",
        options=["Trace", "RF"])
widget_sim_unit = Slider(title="Sim Unit #", value=1.0, start=1.0, end=10)
widget_squeeze = Slider(title="squeeze", value=.8, start=0.5, end=2., step=.1)
widget_scale = Slider(title="scale", value=8.0, start=1.0, end=20.)


def update_data(attrname, old, new):
    # Get the current slider values
    unit = int(widget_unit.value)
    if widget_sortby.value == "Peak to Peak":
        unit = wf.ptp.argsort()[unit]
    sim_order = int(widget_sim_unit.value)
    sim_unit = wf.similar_units[unit, sim_order]
    s = widget_scale.value
    squeeze = widget_squeeze.value

    # Generate the new curve
    x, y = wf.get_template_lines(unit=unit, scale=s, squeeze=squeeze)
    exs, eys = wf.get_error_bar_data(scale=s, squeeze=squeeze)
    trace_sources[0].data = dict(x=x, y=y)
    error_bar_source.data = dict(xs=exs, ys=eys)
    #plot.y_range = Range1d(-ptp[0], ptp[0])
    plot.title.text = "Unit {}".format(unit)
    rf_source.data = dict(d=[sta['STA_spatial'][unit][1].reshape([64, 32])])

    c_ = contours[unit:unit + 1, :]
    contour_sources[0].data = dict(
        x=c_[:, 1], y=c_[:, 2], w=c_[:, 3] * 2,
        h=c_[:, 4] * 2, a=c_[:, 5])

    update_sim_data(attrname, old, new)

def update_sim_data(attrname, old, new):
    # Get the current slider values
    unit = int(widget_unit.value)
    sim_order = int(widget_sim_unit.value)
    sim_unit = wf.similar_units[unit, sim_order]
    s = widget_scale.value
    squeeze = widget_squeeze.value

    # Generate the new curve
    x, y = wf.get_template_lines(unit=sim_unit, scale=s, squeeze=squeeze)
    trace_sources[1].data = dict(x=x, y=y)
    for i in range(2, TOTAL):
        trace_sources[i].data = dict(x=[], y=[])
    #plot.y_range = Range1d(-ptp[0], ptp[0])
    plot.title.text = "Unit {}".format(unit)
    rf_sim_source.data = dict(d=[sta['STA_spatial'][sim_unit][1].reshape([64, 32])])

    c_ = contours[sim_unit:sim_unit + 1, :]
    contour_sources[1].data = dict(
        x=c_[:, 1], y=c_[:, 2], w=c_[:, 3] * 2,
        h=c_[:, 4] * 2, a=c_[:, 5])
    for i in range(2, TOTAL):
        contour_sources[i].data =dict(x=[], y=[], w=[], h=[], a=[])
 

for w in [widget_sortby, widget_unit, widget_sim_unit,  widget_scale, widget_squeeze]:
    w.on_change('value', update_data)

for w in [widget_sim_unit]:
    w.on_change('value', update_sim_data)


# Set up layouts and add to document
inputs = widgetbox(
        widget_sortby, widget_simby, widget_unit, widget_sim_unit,
        widget_scale, widget_squeeze)

row1 = row(inputs, plot, width=1200)
row2 = row(plot_tiles + [plot_rf, plot_rf_sim], width=800)
layout = column(row1, row2)

def callback(attr, old, new):
    print (new)

for i in range(4):
    tile_glyphs[i].data_source.selected.on_change('indices', callback)


curdoc().add_root(layout)
curdoc().title = "Sliders"


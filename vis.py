import sys
import numpy as np
import os
import scipy
import scipy.spatial
import itertools

from bokeh.io import curdoc, reset_output
from bokeh.layouts import column, row, widgetbox
from bokeh.models import (ColumnDataSource, TapTool, LinearColorMapper, 
                         Range1d)
from bokeh.models.widgets import Slider, TextInput, Select
from bokeh.plotting import figure
from bokeh.models.glyphs import Patches, MultiLine, Ellipse
from bokeh.models.callbacks import CustomJS

#if __name__== "__main__":

# read data
root = sys.argv[1]

# geometry
geometry = np.loadtxt("/Users/jinhyunglee/Desktop/contour_gui/512coords.txt")

# templates
templates = np.load(os.path.join(root, "templates.npy"))
templates = templates.transpose([2, 1, 0])
print ("TEMPLATES (# units, # channels, # times): ", templates.shape)

# spatial sta
spatial = np.load(os.path.join(root,'STA_spatial.npy'))[:, :, :, 1]
print ("SPATIAL RF (# units, stim size 0, stim size 1): ", spatial.shape)

# temporal fits (time courses)
tcs = np.load(os.path.join(root, 'STA_temporal.npy')).transpose(0, 2, 1)
print ("TIME COURSES RF (# units, 3 colours, # frames): ", tcs.shape)

# input classification
labels = np.load(os.path.join(root, 'labels.npy'))
classification = []
for j in range(np.max(labels)+1):
    classification.append(np.where(labels==j)[0])
idx_no_rf = np.load(os.path.join(root, 'idx_no_rf.npy'))
idx_multi_rf = np.load(os.path.join(root, 'idx_multi_rf.npy'))
classification.append(idx_no_rf)
classification.append(idx_multi_rf)


# contour info
contours = np.load(os.path.join(root, "gaussian_fits.npy"))
contours[:, 5] = contours[:, 5] * -1

print ("DONE LOADING")


# ****************************************************
# ****************************************************
# *********** GENERATE TEMPLATE PLOTS ****************
# ****************************************************
# ****************************************************

class WaveForms(object):

    def __init__(self, wave_forms):
        print (" loading waveforms)")
        self.wave_forms = wave_forms
        self.n_unit, self.n_channel, self.n_times = self.wave_forms.shape

        self.ptp = self.wave_forms.ptp(-1).max(-1)
        self.active_chans = self.wave_forms.ptp(-1) > 2.
        #self.pairwise_dist()
        #self.similar_units = self.pdist.argsort(axis=1)
        self.similar_units = []
        self.vis_chan = self.update_vis_chan()

    def update_vis_chan(self, value=2.):
        self.vis_chan = self.wave_forms.ptp(-1) > value
        return self.vis_chan

    def pairwise_dist(self):
        self.pdist = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(
                self.wave_forms.reshape([self.n_unit, -1])))

    def get_template_lines(self, unit, ptp=2., scale=8., squeeze=0.8):
        x_range = np.arange(self.n_times) * squeeze
        #idx = range(self.n_channel)
        idx = np.where(self.vis_chan[unit])[0]
        x = [x_range + geometry[i, 0] for i in idx]
        y = [self.wave_forms[unit, i, :] * scale + geometry[i, 1] for i in idx]
        return x, y

    def get_error_bar_data(self, scale=8., squeeze=0.8):
        x_range = np.arange(self.n_times) * squeeze
        xpts = np.array([x_range[0], x_range[-1], x_range[-1], x_range[0]])
        ypts = np.array([-1, -1, 1, 1]) * scale
        xs = [xpts + xx for xx in geometry[:, 0]]
        ys = [ypts + yy for yy in geometry[:, 1]]
        return xs, ys


# ****************************************************
# ****************************************************
# **************** GET RF PARAMETERS *****************
# ****************************************************
# ****************************************************

class ReceptiveField(object):
    print ("loading rfs")

    def __init__(self, contours, spatial):
        self.contours = contours
        self.contours[np.isnan(self.contours)]=1000.
        self.spatial = spatial

    def get_ellipse_source(self, idx):
        dic1 = dict(
                x=self.contours[idx, 1], y=self.contours[idx, 2],
                w=self.contours[idx, 3] * 2, h=self.contours[idx, 4] * 2,
                a=self.contours[idx, 5])

        return dic1


# ****************************************************
# ****************************************************
# ************* GENERATE WIDGET BOXES ****************
# ****************************************************
# ****************************************************

class SpikeCanvas(object):

    def __init__(self, rec_field, wave_form, n_unit, contour_data, tcs, root):
        self.colors = [
                'dodgerblue', 'darkorange', 'forestgreen', 'red', 'orchid',
                'mediumspringgreen']
                
        self.root_dir = root
        self.rec_field = rec_field
        self.wave_form = wave_form
        self.n_unit = n_unit
        self.contour_data = []
        self.tcs = tcs
        self.n_cells = tcs.shape[0]
        
        
        # append garbage canvas plot also
        for k in range(len(contour_data)):
            self.contour_data.append(contour_data[k])
        self.contour_data.append(np.empty((0),'int64'))
        self.contour_data = np.array(self.contour_data)

        self.plots = {}
        self.glyphs = {}
        self.sources = {}
        self.selected_unit = 0
        self.current_class = 0
        
        # The first set of units to be displayed.
        self.show_units = [0,1] #self.wave_form.similar_units[0, :self.n_unit]
       
        self.make_plots_tiles()
        self.make_plots_contours()
        self.make_plots_traces()
        self.make_plots_receptive_fields()
        self.make_plots_tcs()
        self.make_widgets2()

    def make_plots_tiles(self):
        """Initializing the RF contour tiles."""
        #print ("FUNCTION: make_plots_tiles")
        name = "tiles"
        titles = ['ON Parasol', 'OFF Parasol', 'ON Midget', 'OFF Midget', 'Large ON', 'Large OFF',
                    'Small Bistratified', 'Other', 'No RF', 'Multi RF']
        self.titles = titles
        self.plots[name] = []
        self.glyphs[name] = []
        self.sources[name] = []
        for i in range(len(titles)):
            print ("plotting: ", titles[i])
            g = figure(
                plot_height=400, plot_width=200, title=titles[i],
                tools="pan,reset,wheel_zoom,tap",
                x_range=(0, 32), y_range=(0, 64))
            
            self.plots[name].append(g)
           
            # 
            idx = self.contour_data[i]
            #print (i, "selected cells: ", idx)
            
            # load elipse parameter array (from Gaussian Params) as dictionary of 5 entries
            dic = self.rec_field.get_ellipse_source(idx=idx)
            
            # 
            self.sources[name].append(ColumnDataSource(dic))
            
            # draw elipse
            glyph = self.plots[name][i].ellipse(
                    x="x", y="y", width="w", height="h", angle="a",
                    source=self.sources[name][i], fill_alpha=0.)
                    
            #plot_tiles[i].add_glyph(source_tiles[i], glyph_tiles)
            self.glyphs[name].append(glyph)

   
    def make_plots_contours(self):
        """Initializing the RF contours of selected plots units."""
        #print ("FUNCTION: make_plots_contours")
        name = "contours"
        self.glyphs[name] = []
        self.sources[name] = []

        for i, unit in enumerate(self.show_units): 
            dic = self.rec_field.get_ellipse_source(idx=[unit])
            self.sources[name].append(ColumnDataSource(dic))
            self.glyphs[name].append(Ellipse(
                x="x", y="y", width="w", height="h", angle="a",
                fill_color=self.colors[i], fill_alpha=0.5))


    def make_plots_traces(self):
        # Set up trace plots
        name = "traces"
        self.glyphs[name] = []
        self.sources[name] = []

        self.plots[name] = figure(
            plot_height=400, plot_width=800, title="Spatio-Temporal Trace",
            tools="pan,reset,wheel_zoom")

        exs, eys = self.wave_form.get_error_bar_data()
        self.sources["error"] = ColumnDataSource(dict(
            xs=exs, ys=eys))
        self.glyphs["error"] = Patches(
                xs="xs", ys="ys", fill_alpha=0.2, line_alpha=0.0)
        self.plots[name].add_glyph(
                self.sources["error"], self.glyphs["error"])

        for i, unit in enumerate(self.show_units):
            x, y = self.wave_form.get_template_lines(unit=unit)
            self.sources[name].append(ColumnDataSource(
                data=dict(x=x, y=y)))
            self.plots[name].multi_line(
                    xs="x", ys="y", legend="# {}".format(i + 1),
                    line_width=2, line_alpha=0.7, line_color=self.colors[i],
                    source=self.sources[name][-1])

        self.plots[name].legend.click_policy = "hide"

    def make_plots_tcs(self):
        """Initialize receptive fields images."""
        name = "tcs"
        self.glyphs[name] = []
        self.sources[name] = []

        self.plots[name] = figure(
            plot_height=400, plot_width=800, title="Time course fits",
            tools="pan,reset,wheel_zoom")

        x = np.arange(30)
        y = np.zeros(30)
        clrs = ['black','red','green','blue']

        # plot y=0 line
        x_array = []
        y_array = []
        y_array.append(y)
        for k in range(4):
            x_array.append(x)
        for i, unit in enumerate([0]):
            for p in range(3):
                y = self.tcs[unit][p]
                y_array.append(y)
        
        #colors = factor_cmap('ur', palette=Category20b_20, factors=4) 
        print (" len(y_array): ", len(y_array))
        print (" len(x_array): ", len(x_array))
        
        self.sources['tcs'].append(ColumnDataSource(
             data=dict(color=clrs, x=x_array, y=y_array)))
             
        self.plots[name].multi_line(
                    xs="x", ys="y", 
                    line_width=2, line_alpha=1.0, line_color='color',
                    source=self.sources[name][-1]) # add the data just appended       
             
        # clrs=['red','green','blue']
        # for i, unit in enumerate([0]):
            # #x, y = self.wave_form.get_template_lines(unit=unit)
            # x = np.arange(30)
            # for p in range(3):
                # y = self.tcs[unit][p]
                # self.sources[name].append(ColumnDataSource(
                    # data=dict(x=x, y=y)))
                
                # self.plots[name].line(
                        # x="x", y="y", legend="# {}".format(i + 1),
                        # line_width=2, line_alpha=0.7, line_color=clrs[p],
                        # source=self.sources[name][-1]) # add the data just appended

        # self.plots[name].legend.click_policy = "hide"
                

    def make_plots_receptive_fields(self):
        """Initialize receptive fields images."""
        name = "rf"
        self.plots[name] = []
        self.sources[name] = []

        for i, unit in enumerate(self.show_units):
            title = "RF Unit {}".format(unit)
            self.plots[name].append(figure(
                plot_height=400, plot_width=200, title=title,
                tools="pan,reset,wheel_zoom",
                x_range=(0, 32), y_range=(0, 64)))

            img = self.rec_field.spatial[unit]
            vmax = np.max(np.abs(img))
            self.sources[name].append(ColumnDataSource(
                    data=dict(d=[img])))
            
            color_mapper = LinearColorMapper(palette="Viridis256", low=-vmax, high=vmax)
            self.plots[name][-1].image(color_mapper=color_mapper,
                image='d', x=0, y=0, dw=32, dh=64,
                source=self.sources[name][-1])

               
    def update_traces(self, units, scale, squeeze):

        """Update the spatio-temporal traces."""
        n = len(units)
        for i, unit in enumerate(units[:self.n_unit]):
            x, y = self.wave_form.get_template_lines(
                    unit=unit, scale=scale, squeeze=squeeze)
            self.sources["traces"][i].data = dict(x=x, y=y)
        # Turn off the rest.
        for i in range(n, self.n_unit):
            self.sources["traces"][i].data = dict(x=[], y=[])
               
    def update_tcs(self, units, scale, squeeze):

        x = np.arange(30)
        y = np.zeros(30)
        clrs = ['black','red','green','blue']

        # plot y=0 line
        x_array = []
        y_array = []
        y_array.append(y)
        for k in range(4):
            x_array.append(x)
        for i, unit in enumerate(units):
            for p in range(3):
                y = self.tcs[unit][p]
                y_array.append(y)
        
        self.sources['tcs'][0].data = dict(color=clrs, 
                                x=x_array, y=y_array)
                        
        
    def update_errors(self, scale, squeeze):

        """Update the error bars on the trace plot."""
        exs, eys = self.wave_form.get_error_bar_data(
                scale=scale, squeeze=squeeze)
        self.sources["error"].data = dict(xs=exs, ys=eys)

    def update_rfs(self, units):
        #print ("FUNCTION: update_rfs")
        """Update receptive fields images."""
        n = len(units)
        for i, unit in enumerate(units[:self.n_unit]):
            self.sources["rf"][i].data = dict(d=[self.rec_field.spatial[unit]])
            self.plots["rf"][i].title.text = "RF Unit {}".format(unit)
            
        # Turn off the rest.
        for i in range(n, self.n_unit):
            self.sources["rf"][i].data = dict(
                    d=[self.rec_field.spatial[0] * 0])
            self.plots["rf"][i].title.text = ""

    def update_contours2(self, units):
        #print ("FUNCTION: update_contours2")
        """Update the contour plot ellipses for selected units."""
        n = len(units)
        
        for i, unit in enumerate(units[:self.n_unit]):
            dic = self.rec_field.get_ellipse_source(idx=[unit])
            self.sources["contours"][i].data = dic
        
        # Turn off the rest.
        for i in range(n, self.n_unit):
            self.sources["contours"][i].data = dict()

    def move_cell(self,  attrname, old, new):

        # get info on calsses
        current_class = self.current_class
        target_class = self.titles.index(self.widgets["moveto"].value)
        
        # update current class
        current_cells = self.contour_data[current_class]
        idx_local = np.where(current_cells==self.selected_unit)[0]
        updated_current = np.delete(current_cells, idx_local)
        self.contour_data[current_class] = updated_current
        
        # update target class
        target_cells = self.contour_data[target_class]
        updated_target = np.append(target_cells, self.selected_unit)
        self.contour_data[target_class] = updated_target
        
        # reset everythign after reassigning cells
        main_functions(self.rec_field, self.wave_form, self.contour_data, self.tcs, self.root_dir)

    def save_contours(self, attrname, old, new):
        #print ("SAVING", self.root_dir+'/cell_type_vec.npy') 
        print ("SAVING classification result") 
        
        new_label = np.ones(self.n_cells, 'int32')*-10
        for ii, id_ in enumerate(self.contour_data[:len(self.titles)]):
            new_label[id_] = ii
        new_label
        
        np.save(os.path.join(self.root_dir,'labels_updated.npy'), new_label)

    def load_contours(self, attrname, old, new):
        print ("LOADING", self.root_dir+'/cell_type_vec.npy') 
        self.contour_data = np.load(self.root_dir+'/cell_type_vec.npy',allow_pickle=True)
        main_functions(self.rec_field, self.wave_form, self.contour_data, self.tcs, self.root_dir)


    # ******************************************
    # ********* SLIDERS & SELECTORS ************
    # ******************************************
    def make_widgets2(self):
        """Initialize widgets."""
        self.widgets = {}
        self.widgets["unit"] = Slider(
                title="Unit", value=0.0, start=0.0,
                end=self.wave_form.n_unit - 1)
        
        # make list of options of where to move
        options = []
        options.append("")
        options.extend(self.titles)

        self.widgets["moveto"] = Select(
                title="Move to:", value="",
                options=options)
        self.widgets['moveto'].on_change("value", self.move_cell)
                
        self.widgets["save"] = Select(
                title="Save contours:", value="",
                options=["", "SAVE"])
        self.widgets['save'].on_change("value", self.save_contours)
                
        self.widgets["load"] = Select(
                title="Load contours:", value="",
                options=["", "LOAD"])
        self.widgets['load'].on_change("value", self.load_contours)
              
        self.widgets["squeeze"] = Slider(
                title="squeeze", value=.8, start=0.5, end=2., step=.1)
        self.widgets["scale"] = Slider(
                title="scale", value=8.0, start=1.0, end=20.)                
                

    def update_data(self, units, scale, squeeze):
        """The callback for the contour selection"""
        self.update_traces(units=units, scale=scale, squeeze=squeeze)
        self.update_tcs(units=units, scale=scale, squeeze=squeeze)
        self.update_errors(scale=scale, squeeze=squeeze)
        self.update_contours2(units)
        self.update_rfs(units)

    def main_callback(self, attrname, old, new):
        unit = int(self.widgets["unit"].value)
        squeeze = self.widgets["squeeze"].value
        scale = self.widgets["scale"].value
     
        units = unit# self.wave_form.similar_units[unit, :self.n_unit]
        self.update_data(units, scale, squeeze)

    # ****************************************************
    # ****************************************************
    # *********** CONTOUR CLASS CALLBACKS ****************
    # ****************************************************
    # ****************************************************

    def callback0(self, attrname, old, new):
        idx = self.contour_data[0]
        self.selected_unit = idx[new]
        self.current_class = 0
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback1(self, attrname, old, new):
        idx = self.contour_data[1]
        self.selected_unit = idx[new]
        self.current_class = 1
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback2(self, attrname, old, new):
        idx = self.contour_data[2]
        self.selected_unit = idx[new]
        self.current_class = 2
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback3(self, attrname, old, new):
        idx = self.contour_data[3]
        self.selected_unit = idx[new]
        self.current_class = 3
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback4(self, attrname, old, new):
        idx = self.contour_data[4]
        self.selected_unit = idx[new]
        self.current_class = 4
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback5(self, attrname, old, new):
        idx = self.contour_data[5]
        self.selected_unit = idx[new]
        self.current_class = 5
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback6(self, attrname, old, new):
        idx = self.contour_data[6]
        self.selected_unit = idx[new]
        self.current_class = 6
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback7(self, attrname, old, new):
        idx = self.contour_data[7]
        self.selected_unit = idx[new]
        self.current_class = 7
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)

    def callback8(self, attrname, old, new):
        idx = self.contour_data[8]
        self.selected_unit = idx[new]
        self.current_class = 8
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)
        
    def callback9(self, attrname, old, new):
        idx = self.contour_data[9]
        self.selected_unit = idx[new]
        self.current_class = 9
        self.update_data(self.selected_unit, squeeze=0.8, scale=8.)
        
def main_functions(rec_field, wf, contour_data, tcs, root):
    
    curdoc().clear()

    print ("********* RESTARTING MAIN FUNCTIONS *******") 
    canvas = SpikeCanvas(
            rec_field=rec_field, wave_form=wf, n_unit=2, 
            contour_data=contour_data,
            root=root,
            tcs=tcs)
            
    canvas.glyphs["tiles"][0].data_source.selected.on_change(
            'indices', canvas.callback0)
    canvas.glyphs["tiles"][1].data_source.selected.on_change(
            'indices', canvas.callback1)
    canvas.glyphs["tiles"][2].data_source.selected.on_change(
            'indices', canvas.callback2)
    canvas.glyphs["tiles"][3].data_source.selected.on_change(
            'indices', canvas.callback3)
    canvas.glyphs["tiles"][4].data_source.selected.on_change(
            'indices', canvas.callback4)
    canvas.glyphs["tiles"][5].data_source.selected.on_change(
            'indices', canvas.callback5)
    canvas.glyphs["tiles"][6].data_source.selected.on_change(
            'indices', canvas.callback6)
    canvas.glyphs["tiles"][7].data_source.selected.on_change(
            'indices', canvas.callback7)
    canvas.glyphs["tiles"][8].data_source.selected.on_change(
            'indices', canvas.callback8)
    canvas.glyphs["tiles"][9].data_source.selected.on_change(
            'indices', canvas.callback9)

    for w in canvas.widgets.values():
        print ("widget callback..................")
        w.on_change('value', canvas.main_callback)

    # Set up layouts and add to document
    wid = canvas.widgets
    inputs = widgetbox(
            wid["moveto"], wid['save'], wid['load'], wid["unit"], 
            wid["squeeze"], wid["scale"])
            
    rows = []
    rows.append(row(canvas.plots["tiles"], width=800))
    rows.append(row(canvas.plots["rf"], width = 200*2))
    rows.append(row(canvas.plots['tcs'], width=800))
    rows.append(row(inputs, canvas.plots["traces"], width=1200))
    layout = column(rows)
    
    curdoc().add_root(layout)
    curdoc().title = "Sliders"

# ****************************************************
# ****************************************************
# ************** INITALIZE ALL BOXES *****************
# ****************************************************
# ****************************************************
# Bokeh layouts and callback functions start here.
wf = WaveForms(templates)
#spatial = sta["STA_spatial"][:, 1, :].reshape([-1, 64, 32])

rec_field = ReceptiveField(
    contours=contours, spatial=spatial)

main_functions(rec_field, wf, classification, tcs, root)

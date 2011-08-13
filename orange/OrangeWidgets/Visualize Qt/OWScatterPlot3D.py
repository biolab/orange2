"""<name> 3D Scatterplot</name>
"""

from OWWidget import *
from plot.owplot3d import *

import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

import OWGUI
import OWToolbars
import OWColorPalette
import orngVizRank
from OWkNNOptimization import *
from orngScaleScatterPlotData import *

import numpy

TooltipKind = enum('NONE', 'VISIBLE', 'ALL') # Which attributes should be displayed in tooltips?

class ScatterPlot(OWPlot3D, orngScaleScatterPlotData):
    def __init__(self, parent=None):
        OWPlot3D.__init__(self, parent)
        orngScaleScatterPlotData.__init__(self)

    def set_data(self, data, subsetData=None, **args):
        orngScaleScatterPlotData.setData(self, data, subsetData, **args)

class OWScatterPlot3D(OWWidget):
    settingsList = ['plot.show_legend', 'plot.symbol_size', 'plot.show_x_axis_title', 'plot.show_y_axis_title',
                    'plot.show_z_axis_title', 'plot.show_legend', 'plot.use_2d_symbols',
                    'plot.transparency', 'plot.show_grid', 'plot.pitch', 'plot.yaw', 'plot.use_ortho',
                    'plot.show_chassis', 'plot.show_axes',
                    'auto_send_selection', 'auto_send_selection_update',
                    'jitter_size', 'jitter_continuous']
    contextHandlers = {"": DomainContextHandler("", ["x_attr", "y_attr", "z_attr"])}
    jitter_sizes = [0.0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]

    def __init__(self, parent=None, signalManager=None, name="Scatter Plot 3D"):
        OWWidget.__init__(self, parent, signalManager, name, True)

        self.inputs = [("Examples", ExampleTable, self.set_data, Default), ("Subset Examples", ExampleTable, self.set_subset_data)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable)]

        self.x_attr = 0
        self.y_attr = 0
        self.z_attr = 0

        self.x_attr_discrete = False
        self.y_attr_discrete = False
        self.z_attr_discrete = False

        self.color_attr = None
        self.size_attr = None
        self.shape_attr = None
        self.label_attr = None

        self.alpha_value = 255

        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.main_tab = OWGUI.createTabPage(self.tabs, 'Main')
        self.settings_tab = OWGUI.createTabPage(self.tabs, 'Settings', canScroll=True)

        self.x_attr_cb = OWGUI.comboBox(self.main_tab, self, "x_attr", box="X-axis attribute",
            tooltip="Attribute to plot on X axis.",
            callback=self.on_axis_change
            )

        self.y_attr_cb = OWGUI.comboBox(self.main_tab, self, "y_attr", box="Y-axis attribute",
            tooltip="Attribute to plot on Y axis.",
            callback=self.on_axis_change
            )

        self.z_attr_cb = OWGUI.comboBox(self.main_tab, self, "z_attr", box="Z-axis attribute",
            tooltip="Attribute to plot on Z axis.",
            callback=self.on_axis_change
            )

        self.color_attr_cb = OWGUI.comboBox(self.main_tab, self, "color_attr", box="Point color",
            tooltip="Attribute to use for point color",
            callback=self.on_axis_change)

        # Additional point properties (labels, size, shape).
        additional_box = OWGUI.widgetBox(self.main_tab, 'Additional Point Properties')
        self.size_attr_cb = OWGUI.comboBox(additional_box, self, "size_attr", label="Point size:",
            tooltip="Attribute to use for pointSize",
            callback=self.on_axis_change,
            indent=10,
            emptyString='(Same size)',
            )

        self.shape_attr_cb = OWGUI.comboBox(additional_box, self, "shape_attr", label="Point shape:",
            tooltip="Attribute to use for pointShape",
            callback=self.on_axis_change,
            indent=10,
            emptyString='(Same shape)',
            )

        self.label_attr_cb = OWGUI.comboBox(additional_box, self, "label_attr", label="Point label:",
            tooltip="Attribute to use for pointLabel",
            callback=self.on_axis_change,
            indent=10,
            emptyString='(No labels)'
            )

        self.plot = ScatterPlot(self)
        self.vizrank = OWVizRank(self, self.signalManager, self.plot, orngVizRank.SCATTERPLOT3D, "ScatterPlot3D")
        self.optimization_dlg = self.vizrank

        self.optimization_buttons = OWGUI.widgetBox(self.main_tab, 'Optimization dialogs', orientation='horizontal')
        OWGUI.button(self.optimization_buttons, self, "VizRank", callback=self.vizrank.reshow,
            tooltip='Opens VizRank dialog, where you can search for interesting projections with different subsets of attributes',
            debuggingEnabled=0)

        box = OWGUI.widgetBox(self.settings_tab, 'Point properties')
        ss = OWGUI.hSlider(box, self, "plot.symbol_scale", label="Symbol scale",
            minValue=1, maxValue=20,
            tooltip="Scale symbol size",
            callback=self.on_checkbox_update,
            )
        ss.setValue(5)

        OWGUI.hSlider(box, self, "plot.transparency", label="Transparency",
            minValue=10, maxValue=255,
            tooltip="Point transparency value",
            callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        self.jitter_size = 0
        self.jitter_continuous = False
        box = OWGUI.widgetBox(self.settings_tab, "Jittering Options")
        self.jitter_size_combo = OWGUI.comboBox(box, self, 'jitter_size', label='Jittering size (% of size)'+'  ',
            orientation='horizontal',
            callback=self.update_plot,
            items=self.jitter_sizes,
            sendSelectedValue=1,
            valueType=float)
        OWGUI.checkBox(box, self, 'jitter_continuous', 'Jitter continuous attributes',
            callback=self.update_plot,
            tooltip='Does jittering apply also on continuous attributes?')

        self.dark_theme = False

        box = OWGUI.widgetBox(self.settings_tab, 'General settings')
        OWGUI.checkBox(box, self, 'plot.show_x_axis_title',   'X axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_y_axis_title',   'Y axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_z_axis_title',   'Z axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_legend',         'Show legend',    callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.use_ortho',           'Use ortho',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.use_2d_symbols',      '2D symbols',     callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'dark_theme',               'Dark theme',     callback=self.on_theme_change)
        OWGUI.checkBox(box, self, 'plot.show_grid',           'Show grid',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_axes',           'Show axes',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_chassis',        'Show chassis',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.draw_point_cloud',    'Point cloud',    callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.hide_outside',        'Hide outside',   callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        self.auto_send_selection = True
        self.auto_send_selection_update = False
        self.plot.selection_changed_callback = self.selection_changed_callback
        self.plot.selection_updated_callback = self.selection_updated_callback
        box = OWGUI.widgetBox(self.settings_tab, 'Auto Send Selected Data When...')
        OWGUI.checkBox(box, self, 'auto_send_selection', 'Adding/Removing selection areas',
            callback = self.on_checkbox_update, tooltip='Send selected data whenever a selection area is added or removed')
        OWGUI.checkBox(box, self, 'auto_send_selection_update', 'Moving selection areas',
            callback = self.on_checkbox_update, tooltip='Send selected data when a user moves or resizes an existing selection area')

        self.zoom_select_toolbar = OWToolbars.ZoomSelectToolbar(self, self.main_tab, self.plot, self.auto_send_selection,
            buttons=(1, 4, 5, 0, 6, 7, 8))
        self.connect(self.zoom_select_toolbar.buttonSendSelections, SIGNAL('clicked()'), self.send_selections)
        self.connect(self.zoom_select_toolbar.buttonSelectRect, SIGNAL('clicked()'), self.change_selection_type)
        self.connect(self.zoom_select_toolbar.buttonSelectPoly, SIGNAL('clicked()'), self.change_selection_type)
        self.connect(self.zoom_select_toolbar.buttonZoom, SIGNAL('clicked()'), self.change_selection_type)
        self.connect(self.zoom_select_toolbar.buttonRemoveLastSelection, SIGNAL('clicked()'), self.plot.remove_last_selection)
        self.connect(self.zoom_select_toolbar.buttonRemoveAllSelections, SIGNAL('clicked()'), self.plot.remove_all_selections)
        self.toolbarSelection = None

        self.tooltip_kind = TooltipKind.NONE
        box = OWGUI.widgetBox(self.settings_tab, "Tooltips Settings")
        OWGUI.comboBox(box, self, 'tooltip_kind', items = [
            'Don\'t Show Tooltips', 'Show Visible Attributes', 'Show All Attributes'])

        self.plot.mouseover_callback = self.mouseover_callback
        self.shown_attr_indices = []

        self.main_tab.layout().addStretch(100)
        self.settings_tab.layout().addStretch(100)

        self.mainArea.layout().addWidget(self.plot)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.plot.save_to_file)

        self.loadSettings()
        self.plot.update_camera()

        self.data = None
        self.subsetData = None
        self.data_array_jittered = None
        self.resize(1100, 600)

    def mouseover_callback(self, index):
        if self.tooltip_kind == TooltipKind.VISIBLE:
            self.plot.show_tooltip(self.get_example_tooltip(self.data[index], self.shown_attr_indices))
        elif self.tooltip_kind == TooltipKind.ALL:
            self.plot.show_tooltip(self.get_example_tooltip(self.data[index]))

    def get_example_tooltip(self, example, indices=None, max_indices=20):
        if indices and type(indices[0]) == str:
            indices = [self.attr_name_index[i] for i in indices]
        if not indices:
            indices = range(len(self.data.domain.attributes))

        if example.domain.classVar:
            classIndex = self.attr_name_index[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)

        text = '<b>Attributes:</b><br>'
        for index in indices[:max_indices]:
            attr = self.attr_name[index]
            if attr not in example.domain:  text += '&nbsp;'*4 + '%s = ?<br>' % (attr)
            elif example[attr].isSpecial(): text += '&nbsp;'*4 + '%s = ?<br>' % (attr)
            else:                           text += '&nbsp;'*4 + '%s = %s<br>' % (attr, str(example[attr]))

        if len(indices) > max_indices:
            text += '&nbsp;'*4 + ' ... <br>'

        if example.domain.classVar:
            text = text[:-4]
            text += '<hr><b>Class:</b><br>'
            if example.getclass().isSpecial(): text += '&nbsp;'*4 + '%s = ?<br>' % (example.domain.classVar.name)
            else:                              text += '&nbsp;'*4 + '%s = %s<br>' % (example.domain.classVar.name, str(example.getclass()))

        if len(example.domain.getmetas()) != 0:
            text = text[:-4]
            text += '<hr><b>Meta attributes:</b><br>'
            for key in example.domain.getmetas():
                try: text += '&nbsp;'*4 + '%s = %s<br>' % (example.domain[key].name, str(example[key]))
                except: pass
        return text[:-4]

    def selection_changed_callback(self):
        if self.plot.selection_type == SelectionType.ZOOM:
            indices = self.plot.get_selection_indices()
            if len(indices) < 1:
                self.plot.remove_all_selections()
                return
            # TODO: refactor this properly
            if self.data_array_jittered:
                X, Y, Z = self.data_array_jittered
            else:
                X, Y, Z = self.data_array[:, self.x_attr],\
                          self.data_array[:, self.y_attr],\
                          self.data_array[:, self.z_attr]
            X = [X[i] for i in indices]
            Y = [Y[i] for i in indices]
            Z = [Z[i] for i in indices]
            min_x, max_x = numpy.min(X), numpy.max(X)
            min_y, max_y = numpy.min(Y), numpy.max(Y)
            min_z, max_z = numpy.min(Z), numpy.max(Z)
            self.plot.set_new_zoom(min_x, max_x, min_y, max_y, min_z, max_z)
        else:
            if self.auto_send_selection:
                self._send_selections()

    def selection_updated_callback(self):
        if self.plot.selection_type != SelectionType.ZOOM and self.auto_send_selection_update:
            self._send_selections()

    def _send_selections(self):
        # TODO: implement precise get_selection_indices
        indices = self.plot.get_selection_indices()
        if len(indices) < 1:
            return

        selected_indices = [1 if i in indices else 0
                            for i in range(len(self.data))]
        unselected_indices = [1-i for i in selected_indices]
        selected = self.plot.rawData.selectref(selected_indices)
        unselected = self.plot.rawData.selectref(unselected_indices)

        if len(selected) == 0:
            selected = None
        if len(unselected) == 0:
            unselected = None

        self.send('Selected Examples', selected)
        self.send('Unselected Examples', unselected)

    def change_selection_type(self):
        if self.toolbarSelection < 3:
            selection_type = [SelectionType.ZOOM, SelectionType.RECTANGLE, SelectionType.POLYGON][self.toolbarSelection]
            self.plot.set_selection_type(selection_type)

    def set_data(self, data=None):
        self.closeContext("")
        self.data = data
        self.plot.set_data(data, self.subsetData)
        self.x_attr_cb.clear()
        self.y_attr_cb.clear()
        self.z_attr_cb.clear()
        self.color_attr_cb.clear()
        self.size_attr_cb.clear()
        self.shape_attr_cb.clear()
        self.label_attr_cb.clear()

        self.discrete_attrs = {}

        if self.data is not None:
            self.all_attrs = data.domain.variables + data.domain.getmetas().values()
            self.candidate_attrs = [attr for attr in self.all_attrs if attr.varType in [Discrete, Continuous]]

            self.attr_name_index = {}
            for i, attr in enumerate(self.all_attrs):
                self.attr_name_index[attr.name] = i

            self.attr_name = {}
            for i, attr in enumerate(self.all_attrs):
                self.attr_name[i] = attr.name

            self.color_attr_cb.addItem('(Same color)')
            self.size_attr_cb.addItem('(Same size)')
            self.shape_attr_cb.addItem('(Same shape)')
            self.label_attr_cb.addItem('(No labels)')
            icons = OWGUI.getAttributeIcons() 
            for (i, attr) in enumerate(self.candidate_attrs):
                self.x_attr_cb.addItem(icons[attr.varType], attr.name)
                self.y_attr_cb.addItem(icons[attr.varType], attr.name)
                self.z_attr_cb.addItem(icons[attr.varType], attr.name)
                self.color_attr_cb.addItem(icons[attr.varType], attr.name)
                self.size_attr_cb.addItem(icons[attr.varType], attr.name)
                self.label_attr_cb.addItem(icons[attr.varType], attr.name)
                if attr.varType == orange.VarTypes.Discrete:
                    self.discrete_attrs[len(self.discrete_attrs)+1] = (i, attr)
                    self.shape_attr_cb.addItem(icons[orange.VarTypes.Discrete], attr.name)

            array, c, w = self.data.toNumpyMA()
            if len(c):
                array = numpy.hstack((array, c.reshape(-1,1)))
            self.data_array = array

            self.x_attr, self.y_attr, self.z_attr = numpy.min([[0, 1, 2],
                                                               [len(self.candidate_attrs) - 1]*3
                                                              ], axis=0)
            self.color_attr = 0
            self.shown_attr_indices = [self.x_attr, self.y_attr, self.z_attr, self.color_attr]
            self.openContext('', data)

    def set_subset_data(self, data=None):
        self.subsetData = data

    def handleNewSignals(self):
        self.update_plot()
        self.send_selections()

    def saveSettings(self):
        OWWidget.saveSettings(self)

    def sendReport(self):
        self.startReport('%s [%s - %s - %s]' % (self.windowTitle(), self.attr_name[self.x_attr],
                                                self.attr_name[self.y_attr], self.attr_name[self.z_attr]))
        self.reportSettings('Visualized attributes',
                            [('X', self.attr_name[self.x_attr]),
                             ('Y', self.attr_name[self.y_attr]),
                             ('Z', self.attr_name[self.z_attr]),
                             self.color_attr and ('Color', self.attr_name[self.color_attr]),
                             self.label_attr and ('Label', self.attr_name[self.label_attr]),
                             self.shape_attr and ('Shape', self.attr_name[self.shape_attr]),
                             self.size_attr  and ('Size', self.attr_name[self.size_attr])])
        self.reportSettings('Settings',
                            [('Symbol size', self.plot.symbol_scale),
                             ('Transparency', self.plot.transparency),
                             ("Jittering", self.jitter_size),
                             ("Jitter continuous attributes", OWGUI.YesNo[self.jitter_continuous])
                             ])
        self.reportSection('Plot')
        self.reportImage(self.plot.save_to_file_direct, QSize(400, 400))

    def send_selections(self):
        if self.data == None:
            return
        indices = self.plot.get_selection_indices()
        selected = [1 if i in indices else 0 for i in range(len(self.data))]
        unselected = map(lambda n: 1-n, selected)
        selected = self.data.selectref(selected)
        unselected = self.data.selectref(unselected)
        self.send('Selected Examples', selected)
        self.send('Unselected Examples', unselected)

    def on_axis_change(self):
        if self.data is not None:
            self.update_plot()

    def on_theme_change(self):
        if self.dark_theme:
            self.plot.theme = DarkTheme()
        else:
            self.plot.theme = LightTheme()

    def on_checkbox_update(self):
        self.plot.updateGL()

    def update_plot(self):
        if self.data is None:
            return

        self.x_attr_discrete = self.y_attr_discrete = self.z_attr_discrete = False

        if self.candidate_attrs[self.x_attr].varType == Discrete:
            self.x_attr_discrete = True
        if self.candidate_attrs[self.y_attr].varType == Discrete:
            self.y_attr_discrete = True
        if self.candidate_attrs[self.z_attr].varType == Discrete:
            self.z_attr_discrete = True

        X, Y, Z, mask = self.get_axis_data(self.x_attr, self.y_attr, self.z_attr)

        color_discrete = shape_discrete = size_discrete = False

        if self.color_attr > 0:
            color_attr = self.candidate_attrs[self.color_attr - 1]
            C = self.data_array[:, self.color_attr - 1]
            if color_attr.varType == Discrete:
                color_discrete = True
                palette = OWColorPalette.ColorPaletteHSV(len(color_attr.values))
                colors = [palette[int(value)] for value in C.ravel()]
                colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
                palette_colors = [palette[i] for i in range(len(color_attr.values))]
            else:
                palette = OWColorPalette.ColorPaletteBW()
                maxC, minC = numpy.max(C), numpy.min(C)
                C = (C - minC) / (maxC - minC)
                colors = [palette[value] for value in C.ravel()]
                colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
        else:
            colors = 'b'

        if self.size_attr > 0:
            size_attr = self.candidate_attrs[self.size_attr - 1]
            S = self.data_array[:, self.size_attr - 1]
            if size_attr.varType == Discrete:
                size_discrete = True
                sizes = [v+1. for v in S]
            else:
                min, max = numpy.min(S), numpy.max(S)
                sizes = [(v - min) / (max-min) for v in S]
        else:
            sizes = 1.

        shapes = None
        if self.shape_attr > 0:
            i, shape_attr = self.discrete_attrs[self.shape_attr]
            if shape_attr.varType == Discrete:
                shape_discrete = True
                shapes = self.data_array[:, i]

        labels = None
        if self.label_attr > 0:
            label_attr = self.candidate_attrs[self.label_attr - 1]
            labels = self.data_array[:, self.label_attr - 1]
            if label_attr.varType == Discrete:
                value_map = {key: label_attr.values[key] for key in range(len(label_attr.values))}
                labels = [value_map[value] for value in labels]

        self.plot.clear()

        if self.plot.show_legend:
            legend_keys = {}
            color_attr = color_attr if self.color_attr > 0 and color_discrete else None
            size_attr = size_attr if self.size_attr > 0 and size_discrete else None
            shape_attr = shape_attr if self.shape_attr > 0 and shape_discrete else None

            single_legend = [color_attr, size_attr, shape_attr].count(None) == 2
            if single_legend:
                legend_join = lambda name, val: val
            else:
                legend_join = lambda name, val: name + '=' + val 

            if color_attr != None:
                num = len(color_attr.values)
                val = [[], [], [1.]*num, [Symbol.RECT]*num]
                var_values = getVariableValuesSorted(self.data.domain[self.attr_name_index[color_attr.name]])
                for i in range(num):
                    val[0].append(legend_join(color_attr.name, var_values[i]))
                    c = palette_colors[i]
                    val[1].append([c.red()/255., c.green()/255., c.blue()/255., 1.])
                legend_keys[color_attr] = val

            if shape_attr != None:
                num = len(shape_attr.values)
                if legend_keys.has_key(shape_attr):
                    val = legend_keys[shape_attr]
                else:
                    val = [[], [(0, 0, 0, 1)]*num, [1.]*num, []]
                var_values = getVariableValuesSorted(self.data.domain[self.attr_name_index[shape_attr.name]])
                val[3] = []
                val[0] = []
                for i in range(num):
                    val[3].append(i)
                    val[0].append(legend_join(shape_attr.name, var_values[i]))
                legend_keys[shape_attr] = val

            if size_attr != None:
                num = len(size_attr.values)
                if legend_keys.has_key(size_attr):
                    val = legend_keys[size_attr]
                else:
                    val = [[], [(0, 0, 0, 1)]*num, [], [Symbol.RECT]*num]
                val[2] = []
                val[0] = []
                var_values = getVariableValuesSorted(self.data.domain[self.attr_name_index[size_attr.name]])
                for i in range(num):
                    val[0].append(legend_join(size_attr.name, var_values[i]))
                    val[2].append(0.1 + float(i) / len(var_values))
                legend_keys[size_attr] = val
        else:
            legend_keys = {}

        for val in legend_keys.values():
            for i in range(len(val[1])):
                self.plot.legend.add_item(val[3][i], val[1][i], val[2][i], val[0][i])

        self.plot.scatter(X, Y, Z, colors, sizes, shapes, labels)
        self.plot.set_x_axis_title(self.candidate_attrs[self.x_attr].name)
        self.plot.set_y_axis_title(self.candidate_attrs[self.y_attr].name)
        self.plot.set_z_axis_title(self.candidate_attrs[self.z_attr].name)

        def create_discrete_map(attr_index):
            values = self.candidate_attrs[attr_index].values
            return {key: value for key, value in enumerate(values)}

        if self.candidate_attrs[self.x_attr].varType == Discrete:
            self.plot.set_x_axis_map(create_discrete_map(self.x_attr))
        if self.candidate_attrs[self.y_attr].varType == Discrete:
            self.plot.set_y_axis_map(create_discrete_map(self.y_attr))
        if self.candidate_attrs[self.z_attr].varType == Discrete:
            self.plot.set_z_axis_map(create_discrete_map(self.z_attr))

    def get_axis_data(self, x_index, y_index, z_index):
        array = self.data_array
        X, Y, Z = array[:, x_index], array[:, y_index], array[:, z_index]

        if self.jitter_size > 0:
            X, Y, Z = map(numpy.copy, [X, Y, Z])
            x_range = numpy.max(X)-numpy.min(X)
            y_range = numpy.max(Y)-numpy.min(Y)
            z_range = numpy.max(Z)-numpy.min(Z)
            if self.x_attr_discrete or self.jitter_continuous:
                X += (numpy.random.random(len(X))-0.5) * (self.jitter_size * x_range / 100.)
            if self.y_attr_discrete or self.jitter_continuous:
                Y += (numpy.random.random(len(Y))-0.5) * (self.jitter_size * y_range / 100.)
            if self.z_attr_discrete or self.jitter_continuous:
                Z += (numpy.random.random(len(Z))-0.5) * (self.jitter_size * z_range / 100.)
            self.data_array_jittered = (X, Y, Z)
        return X, Y, Z, None

    def showSelectedAttributes(self):
        val = self.vizrank.getSelectedProjection()
        if not val: return
        if self.data.domain.classVar:
            self.attr_color = self.attr_name_index[self.data.domain.classVar.name]
        if not self.plot.have_data:
            return
        attr_list = val[3]
        if attr_list and len(attr_list) == 3:
            self.x_attr = self.attr_name_index[attr_list[0]]
            self.y_attr = self.attr_name_index[attr_list[1]]
            self.z_attr = self.attr_name_index[attr_list[2]]

        #if self.graph.dataHasDiscreteClass and (self.vizrank.showKNNCorrectButton.isChecked() or self.vizrank.showKNNWrongButton.isChecked()):
        #    kNNExampleAccuracy, probabilities = self.vizrank.kNNClassifyData(self.graph.createProjectionAsExampleTable([self.graph.attributeNameIndex[self.attrX], self.graph.attributeNameIndex[self.attrY]]))
        #    if self.vizrank.showKNNCorrectButton.isChecked(): kNNExampleAccuracy = ([1.0 - val for val in kNNExampleAccuracy], "Probability of wrong classification = %.2f%%")
        #    else: kNNExampleAccuracy = (kNNExampleAccuracy, "Probability of correct classification = %.2f%%")
        #else:
        #    kNNExampleAccuracy = None
        #self.graph.insideColors = insideColors or self.classificationResults or kNNExampleAccuracy or self.outlierValues
        #self.graph.updateData(self.attrX, self.attrY, self.attrColor, self.attrShape, self.attrSize, self.attrLabel)
        self.update_plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWScatterPlot3D()
    data = orange.ExampleTable("../../doc/datasets/iris")
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()

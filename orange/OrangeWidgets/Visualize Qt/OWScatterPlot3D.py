'''
<name>Scatterplot 3D</name>
<icon>icons/ScatterPlot.png</icon>
<priority>2001</priority>
'''

from OWWidget import *
from plot.owplot3d import *
from plot.owtheme import ScatterLightTheme, ScatterDarkTheme
from plot.owplotgui import OWPlotGUI
from plot.owplot import OWPlot
from plot import OWPoint

import orange
Discrete = orange.VarTypes.Discrete
Continuous = orange.VarTypes.Continuous

from Orange.preprocess.scaling import get_variable_values_sorted

import OWGUI
import OWToolbars
import orngVizRank
from OWkNNOptimization import *
from orngScaleScatterPlotData import *

import numpy

TooltipKind = enum('NONE', 'VISIBLE', 'ALL')

class ScatterPlot(OWPlot3D, orngScaleScatterPlotData):
    def __init__(self, parent=None):
        self.parent = parent
        OWPlot3D.__init__(self, parent)
        orngScaleScatterPlotData.__init__(self)

        self._theme = ScatterLightTheme()
        self.show_grid = True
        self.show_chassis = True

        self.animate_plot = False

    def set_data(self, data, subset_data=None, **args):
        if data == None:
            return
        args['skipIfSame'] = False
        orngScaleScatterPlotData.set_data(self, data, subset_data, **args)
        OWPlot3D.set_plot_data(self, self.scaled_data, self.scaled_subset_data)
        OWPlot3D.initializeGL(self)

    def update_data(self, x_attr, y_attr, z_attr,
                    color_attr, symbol_attr, size_attr, label_attr):
        if self.data == None:
            return
        self.before_draw_callback = self.before_draw

        color_discrete = symbol_discrete = size_discrete = False

        color_index = -1
        if color_attr != '' and color_attr != '(Same color)':
            color_index = self.attribute_name_index[color_attr]
            if self.data_domain[color_attr].varType == Discrete:
                color_discrete = True
                self.discrete_palette.setNumberOfColors(len(self.data_domain[color_attr].values))

        symbol_index = -1
        num_symbols_used = -1
        if symbol_attr != '' and symbol_attr != 'Same symbol)' and\
           len(self.data_domain[symbol_attr].values) < len(Symbol) and\
           self.data_domain[symbol_attr].varType == Discrete:
            symbol_index = self.attribute_name_index[symbol_attr]
            symbol_discrete = True
            num_symbols_used = len(self.data_domain[symbol_attr].values)

        size_index = -1
        if size_attr != '' and size_attr != '(Same size)':
            size_index = self.attribute_name_index[size_attr]
            if self.data_domain[size_attr].varType == Discrete:
                size_discrete = True

        label_index = -1
        if label_attr != '' and label_attr != '(No labels)':
            label_index = self.attribute_name_index[label_attr]

        x_index = self.attribute_name_index[x_attr]
        y_index = self.attribute_name_index[y_attr]
        z_index = self.attribute_name_index[z_attr]

        x_discrete = self.data_domain[x_attr].varType == Discrete
        y_discrete = self.data_domain[y_attr].varType == Discrete
        z_discrete = self.data_domain[z_attr].varType == Discrete

        colors = []
        if color_discrete:
            for i in range(len(self.data_domain[color_attr].values)):
                c = self.discrete_palette[i]
                colors.append(c)

        data_scale = [self.attr_values[x_attr][1] - self.attr_values[x_attr][0],
                      self.attr_values[y_attr][1] - self.attr_values[y_attr][0],
                      self.attr_values[z_attr][1] - self.attr_values[z_attr][0]]
        data_translation = [self.attr_values[x_attr][0],
                            self.attr_values[y_attr][0],
                            self.attr_values[z_attr][0]]
        data_scale = 1. / numpy.array(data_scale)
        if x_discrete:
            data_scale[0] = 0.5 / float(len(self.data_domain[x_attr].values))
            data_translation[0] = 1.
        if y_discrete:
            data_scale[1] = 0.5 / float(len(self.data_domain[y_attr].values))
            data_translation[1] = 1.
        if z_discrete:
            data_scale[2] = 0.5 / float(len(self.data_domain[z_attr].values))
            data_translation[2] = 1.

        self.clear()

        attr_indices = [x_index, y_index, z_index]
        if color_index > -1:
            attr_indices.append(color_index)
        if size_index > -1:
            attr_indices.append(size_index)
        if symbol_index > -1:
            attr_indices.append(symbol_index)
        if label_index > -1:
            attr_indices.append(label_index)

        valid_data = self.getValidList(attr_indices)
        self.set_valid_data(valid_data)

        self.set_shown_attributes(x_index, y_index, z_index,
            color_index, symbol_index, size_index, label_index,
            colors, num_symbols_used,
            x_discrete, y_discrete, z_discrete,
            data_scale, data_translation)

        ## Legend
        def_color = QColor(150, 150, 150)
        def_symbol = 0
        def_size = 10

        if color_discrete:
            num = len(self.data_domain[color_attr].values)
            values = get_variable_values_sorted(self.data_domain[color_attr])
            for ind in range(num):
                self.legend().add_item(color_attr, values[ind], OWPoint(def_symbol, self.discrete_palette[ind], def_size))

        if symbol_index != -1:
            num = len(self.data_domain[symbol_attr].values)
            values = get_variable_values_sorted(self.data_domain[symbol_attr])
            for ind in range(num):
                self.legend().add_item(symbol_attr, values[ind], OWPoint(ind, def_color, def_size))

        if size_discrete:
            num = len(self.data_domain[size_attr].values)
            values = get_variable_values_sorted(self.data_domain[size_attr])
            for ind in range(num):
                self.legend().add_item(size_attr, values[ind], OWPoint(def_symbol, def_color, 6 + round(ind * 5 / len(values))))

        if color_index != -1 and self.data_domain[color_attr].varType == Continuous:
            self.legend().add_color_gradient(color_attr, [("%%.%df" % self.data_domain[color_attr].numberOfDecimals % v) for v in self.attr_values[color_attr]])

        self.legend().max_size = QSize(400, 400)
        self.legend().set_floating(True)
        self.legend().set_orientation(Qt.Vertical)
        if self.legend().pos().x() == 0:
            self.legend().setPos(QPointF(100, 100))
        self.legend().update_items()
        self.legend().setVisible(self.show_legend)

        ## Axes
        self.set_axis_title(Axis.X, x_attr)
        self.set_axis_title(Axis.Y, y_attr)
        self.set_axis_title(Axis.Z, z_attr)

        if x_discrete:
            self.set_axis_labels(Axis.X, get_variable_values_sorted(self.data_domain[x_attr]))
        if y_discrete:
            self.set_axis_labels(Axis.Y, get_variable_values_sorted(self.data_domain[y_attr]))
        if z_discrete:
            self.set_axis_labels(Axis.Z, get_variable_values_sorted(self.data_domain[z_attr]))

        self.update()

    def before_draw(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.projection.data(), dtype=float))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixd(numpy.array(self.modelview.data(), dtype=float))

        if self.show_grid:
            self.draw_grid()
        if self.show_chassis:
            self.draw_chassis()

    def draw_chassis(self):
        self.qglColor(self._theme.axis_values_color)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, 0x00FF)
        glDisable(GL_DEPTH_TEST)
        glLineWidth(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        edges = [self.x_axis, self.y_axis, self.z_axis,
                 self.x_axis+self.unit_z, self.x_axis+self.unit_y,
                 self.x_axis+self.unit_z+self.unit_y,
                 self.y_axis+self.unit_x, self.y_axis+self.unit_z,
                 self.y_axis+self.unit_x+self.unit_z,
                 self.z_axis+self.unit_x, self.z_axis+self.unit_y,
                 self.z_axis+self.unit_x+self.unit_y]
        glBegin(GL_LINES)
        for edge in edges:
            start, end = edge
            glVertex3f(*start)
            glVertex3f(*end)
        glEnd()
        glDisable(GL_LINE_STIPPLE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def draw_grid(self):
        cam_in_space = numpy.array([
          self.camera[0]*self.camera_distance,
          self.camera[1]*self.camera_distance,
          self.camera[2]*self.camera_distance
        ])

        def _draw_grid(axis0, axis1, normal0, normal1, i, j):
            self.qglColor(self._theme.grid_color)
            for axis, normal, coord_index in zip([axis0, axis1], [normal0, normal1], [i, j]):
                start, end = axis.copy()
                start_value = self.map_to_data(start.copy())[coord_index]
                end_value = self.map_to_data(end.copy())[coord_index]
                values, _ = loose_label(start_value, end_value, 7)
                for value in values:
                    if not (start_value <= value <= end_value):
                        continue
                    position = start + (end-start)*((value-start_value) / float(end_value-start_value))
                    glBegin(GL_LINES)
                    glVertex3f(*position)
                    glVertex3f(*(position-normal*1.))
                    glEnd()

        glDisable(GL_DEPTH_TEST)
        glLineWidth(1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        planes = [self.axis_plane_xy, self.axis_plane_yz,
                  self.axis_plane_xy_back, self.axis_plane_yz_right]
        axes = [[self.x_axis, self.y_axis],
                [self.y_axis, self.z_axis],
                [self.x_axis+self.unit_z, self.y_axis+self.unit_z],
                [self.z_axis+self.unit_x, self.y_axis+self.unit_x]]
        normals = [[numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0, 0,-1]), numpy.array([ 0,-1, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([-1, 0, 0])],
                   [numpy.array([0,-1, 0]), numpy.array([ 0, 0,-1])]]
        coords = [[0, 1],
                  [1, 2],
                  [0, 1],
                  [2, 1]]
        visible_planes = [plane_visible(plane, cam_in_space) for plane in planes]
        xz_visible = not plane_visible(self.axis_plane_xz, cam_in_space)
        if xz_visible:
            _draw_grid(self.x_axis, self.z_axis, numpy.array([0,0,-1]), numpy.array([-1,0,0]), 0, 2)
        for visible, (axis0, axis1), (normal0, normal1), (i, j) in\
             zip(visible_planes, axes, normals, coords):
            if not visible:
                _draw_grid(axis0, axis1, normal0, normal1, i, j)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

class OWScatterPlot3D(OWWidget):
    settingsList = ['plot.show_legend', 'plot.symbol_size', 'plot.show_x_axis_title', 'plot.show_y_axis_title',
                    'plot.show_z_axis_title', 'plot.show_legend', 'plot.use_2d_symbols',
                    'plot.alpha_value', 'plot.show_grid', 'plot.pitch', 'plot.yaw', 'plot.use_ortho',
                    'plot.show_chassis', 'plot.show_axes',
                    'auto_send_selection', 'auto_send_selection_update',
                    'plot.jitter_size', 'plot.jitter_continuous', 'dark_theme']
    contextHandlers = {'': DomainContextHandler('', ['x_attr', 'y_attr', 'z_attr'])}
    jitter_sizes = [0.0, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50]

    def __init__(self, parent=None, signalManager=None, name='Scatter Plot 3D'):
        OWWidget.__init__(self, parent, signalManager, name, True)

        self.inputs = [('Examples', ExampleTable, self.set_data, Default), ('Subset Examples', ExampleTable, self.set_subset_data)]
        self.outputs = [('Selected Examples', ExampleTable), ('Unselected Examples', ExampleTable)]

        self.x_attr = ''
        self.y_attr = ''
        self.z_attr = ''

        self.x_attr_discrete = False
        self.y_attr_discrete = False
        self.z_attr_discrete = False

        self.color_attr = ''
        self.size_attr = ''
        self.symbol_attr = ''
        self.label_attr = ''

        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.main_tab = OWGUI.createTabPage(self.tabs, 'Main')
        self.settings_tab = OWGUI.createTabPage(self.tabs, 'Settings', canScroll=True)

        self.x_attr_cb = OWGUI.comboBox(self.main_tab, self, 'x_attr', box='X-axis attribute',
            tooltip='Attribute to plot on X axis.',
            callback=self.on_axis_change,
            sendSelectedValue=1,
            valueType=str)

        self.y_attr_cb = OWGUI.comboBox(self.main_tab, self, 'y_attr', box='Y-axis attribute',
            tooltip='Attribute to plot on Y axis.',
            callback=self.on_axis_change,
            sendSelectedValue=1,
            valueType=str)

        self.z_attr_cb = OWGUI.comboBox(self.main_tab, self, 'z_attr', box='Z-axis attribute',
            tooltip='Attribute to plot on Z axis.',
            callback=self.on_axis_change,
            sendSelectedValue=1,
            valueType=str)

        self.color_attr_cb = OWGUI.comboBox(self.main_tab, self, 'color_attr', box='Point color',
            tooltip='Attribute to use for point color',
            callback=self.on_axis_change,
            sendSelectedValue=1,
            valueType=str)

        # Additional point properties (labels, size, symbol).
        additional_box = OWGUI.widgetBox(self.main_tab, 'Additional Point Properties')
        self.size_attr_cb = OWGUI.comboBox(additional_box, self, 'size_attr', label='Point size:',
            tooltip='Attribute to use for point size',
            callback=self.on_axis_change,
            indent=10,
            emptyString='(Same size)',
            sendSelectedValue=1,
            valueType=str)

        self.symbol_attr_cb = OWGUI.comboBox(additional_box, self, 'symbol_attr', label='Point symbol:',
            tooltip='Attribute to use for point symbol',
            callback=self.on_axis_change,
            indent=10,
            emptyString='(Same symbol)',
            sendSelectedValue=1,
            valueType=str)

        self.label_attr_cb = OWGUI.comboBox(additional_box, self, 'label_attr', label='Point label:',
            tooltip='Attribute to use for pointLabel',
            callback=self.on_axis_change,
            indent=10,
            emptyString='(No labels)',
            sendSelectedValue=1,
            valueType=str)

        self.plot = ScatterPlot(self)
        self.vizrank = OWVizRank(self, self.signalManager, self.plot, orngVizRank.SCATTERPLOT3D, 'ScatterPlot3D')
        self.optimization_dlg = self.vizrank

        self.optimization_buttons = OWGUI.widgetBox(self.main_tab, 'Optimization dialogs', orientation='horizontal')
        OWGUI.button(self.optimization_buttons, self, 'VizRank', callback=self.vizrank.reshow,
            tooltip='Opens VizRank dialog, where you can search for interesting projections with different subsets of attributes',
            debuggingEnabled=0)

        box = OWGUI.widgetBox(self.settings_tab, 'Point properties')
        ss = OWGUI.hSlider(box, self, 'plot.symbol_scale', label='Symbol scale',
            minValue=1, maxValue=20,
            tooltip='Scale symbol size',
            callback=self.on_checkbox_update)
        ss.setValue(8)

        OWGUI.hSlider(box, self, 'plot.alpha_value', label='Transparency',
            minValue=10, maxValue=255,
            tooltip='Point transparency value',
            callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        box = OWGUI.widgetBox(self.settings_tab, 'Jittering Options')
        self.jitter_size_combo = OWGUI.comboBox(box, self, 'plot.jitter_size', label='Jittering size (% of size)'+'  ',
            orientation='horizontal',
            callback=self.handleNewSignals,
            items=self.jitter_sizes,
            sendSelectedValue=1,
            valueType=float)
        OWGUI.checkBox(box, self, 'plot.jitter_continuous', 'Jitter continuous attributes',
            callback=self.handleNewSignals,
            tooltip='Does jittering apply also on continuous attributes?')

        self.dark_theme = False

        box = OWGUI.widgetBox(self.settings_tab, 'General settings')
        OWGUI.checkBox(box, self, 'plot.show_x_axis_title',   'X axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_y_axis_title',   'Y axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_z_axis_title',   'Z axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_legend',         'Show legend',    callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.use_ortho',           'Use ortho',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.use_2d_symbols',      '2D symbols',     callback=self.update_plot)
        OWGUI.checkBox(box, self, 'dark_theme',               'Dark theme',     callback=self.on_theme_change)
        OWGUI.checkBox(box, self, 'plot.show_grid',           'Show grid',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_axes',           'Show axes',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_chassis',        'Show chassis',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.hide_outside',        'Hide outside',   callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        self.gui = OWPlotGUI(self)
        gui = self.gui
        self.zoom_select_toolbar = gui.zoom_select_toolbar(self.main_tab, buttons=gui.default_zoom_select_buttons)
        self.connect(self.zoom_select_toolbar.buttons[gui.SendSelection], SIGNAL("clicked()"), self.send_selection)
        self.connect(self.zoom_select_toolbar.buttons[gui.Zoom], SIGNAL("clicked()"), self._set_behavior_zoom)
        self.connect(self.zoom_select_toolbar.buttons[gui.SelectionOne], SIGNAL("clicked()"), self._set_behavior_replace)
        self.connect(self.zoom_select_toolbar.buttons[gui.SelectionAdd], SIGNAL("clicked()"), self._set_behavior_add)
        self.connect(self.zoom_select_toolbar.buttons[gui.SelectionRemove], SIGNAL("clicked()"), self._set_behavior_remove)

        self.tooltip_kind = TooltipKind.NONE
        box = OWGUI.widgetBox(self.settings_tab, 'Tooltips Settings')
        OWGUI.comboBox(box, self, 'tooltip_kind', items = [
            'Don\'t Show Tooltips', 'Show Visible Attributes', 'Show All Attributes'])

        self.plot.mouseover_callback = self.mouseover_callback

        self.main_tab.layout().addStretch(100)
        self.settings_tab.layout().addStretch(100)

        self.mainArea.layout().addWidget(self.plot)
        self.connect(self.graphButton, SIGNAL('clicked()'), self.plot.save_to_file)

        self.loadSettings()
        self.plot.update_camera()
        self.on_theme_change()

        self._set_behavior_replace()

        self.data = None
        self.subset_data = None
        self.resize(1100, 600)

    def _set_behavior_zoom(self):
        self.plot.unselect_all_points()
        self.plot.zoom_into_selection = True

    def _set_behavior_add(self):
        self.plot.set_selection_behavior(OWPlot.AddSelection)

    def _set_behavior_replace(self):
        self.plot.set_selection_behavior(OWPlot.ReplaceSelection)

    def _set_behavior_remove(self):
        self.plot.set_selection_behavior(OWPlot.RemoveSelection)

    def mouseover_callback(self, index):
        if self.tooltip_kind == TooltipKind.VISIBLE:
            self.plot.show_tooltip(self.get_example_tooltip(self.data[index], self.shown_attrs))
        elif self.tooltip_kind == TooltipKind.ALL:
            self.plot.show_tooltip(self.get_example_tooltip(self.data[index]))

    def get_example_tooltip(self, example, indices=None, max_indices=20):
        if indices and type(indices[0]) == str:
            indices = [self.plot.attribute_name_index[i] for i in indices]
        if not indices:
            indices = range(len(self.data.domain.attributes))

        if example.domain.classVar:
            classIndex = self.plot.attribute_name_index[example.domain.classVar.name]
            while classIndex in indices:
                indices.remove(classIndex)

        text = '<b>Attributes:</b><br>'
        for index in indices[:max_indices]:
            attr = self.plot.data_domain[index].name
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

    def set_data(self, data=None):
        self.closeContext()
        self.vizrank.clearResults()
        same_domain = self.data and data and\
            data.domain.checksum() == self.data.domain.checksum()
        self.data = data
        if not same_domain:
            self.init_attr_values()
        self.openContext('', data)

    def init_attr_values(self):
        self.x_attr_cb.clear()
        self.y_attr_cb.clear()
        self.z_attr_cb.clear()
        self.color_attr_cb.clear()
        self.size_attr_cb.clear()
        self.symbol_attr_cb.clear()
        self.label_attr_cb.clear()

        self.discrete_attrs = {}

        if not self.data:
            return

        self.color_attr_cb.addItem('(Same color)')
        self.label_attr_cb.addItem('(No labels)')
        self.symbol_attr_cb.addItem('(Same symbol)')
        self.size_attr_cb.addItem('(Same size)')

        icons = OWGUI.getAttributeIcons() 
        for metavar in [self.data.domain.getmeta(mykey) for mykey in self.data.domain.getmetas().keys()]:
            self.label_attr_cb.addItem(icons[metavar.varType], metavar.name)

        for attr in self.data.domain:
            if attr.varType in [Discrete, Continuous]:
                self.x_attr_cb.addItem(icons[attr.varType], attr.name)
                self.y_attr_cb.addItem(icons[attr.varType], attr.name)
                self.z_attr_cb.addItem(icons[attr.varType], attr.name)
                self.color_attr_cb.addItem(icons[attr.varType], attr.name)
                self.size_attr_cb.addItem(icons[attr.varType], attr.name)
            if attr.varType == Discrete and len(attr.values) < len(Symbol):
                self.symbol_attr_cb.addItem(icons[attr.varType], attr.name)
            self.label_attr_cb.addItem(icons[attr.varType], attr.name)

        self.x_attr = str(self.x_attr_cb.itemText(0))
        if self.y_attr_cb.count() > 1:
            self.y_attr = str(self.y_attr_cb.itemText(1))
        else:
            self.y_attr = str(self.y_attr_cb.itemText(0))

        if self.z_attr_cb.count() > 2:
            self.z_attr = str(self.z_attr_cb.itemText(2))
        else:
            self.z_attr = str(self.z_attr_cb.itemText(0))

        if self.data.domain.classVar and self.data.domain.classVar.varType in [Discrete, Continuous]:
            self.color_attr = self.data.domain.classVar.name
        else:
            self.color_attr = ''

        self.symbol_attr = self.size_attr = self.label_attr = ''
        self.shown_attrs = [self.x_attr, self.y_attr, self.z_attr, self.color_attr]

    def set_subset_data(self, data=None):
        self.subset_data = data

    def handleNewSignals(self):
        self.plot.set_data(self.data, self.subset_data)
        self.vizrank.resetDialog()
        self.update_plot()
        self.send_selection()

    def saveSettings(self):
        OWWidget.saveSettings(self)

    def sendReport(self):
        self.startReport('%s [%s - %s - %s]' % (self.windowTitle(), self.x_attr, self.y_attr, self.z_attr))
        self.reportSettings('Visualized attributes',
                            [('X', self.x_attr),
                             ('Y', self.y_attr),
                             ('Z', self.z_attr),
                             self.color_attr and ('Color', self.color_attr),
                             self.label_attr and ('Label', self.label_attr),
                             self.symbol_attr and ('Symbol', self.symbol_attr),
                             self.size_attr  and ('Size', self.size_attr)])
        self.reportSettings('Settings',
                            [('Symbol size', self.plot.symbol_scale),
                             ('Transparency', self.plot.alpha_value),
                             ('Jittering', self.jitter_size),
                             ('Jitter continuous attributes', OWGUI.YesNo[self.jitter_continuous])
                             ])
        self.reportSection('Plot')
        self.reportImage(self.plot.save_to_file_direct, QSize(400, 400))

    def send_selection(self):
        if self.data == None:
            return
        selected = self.plot.get_selected_indices()
        if selected == None or len(selected) != len(self.data):
            return
        unselected = numpy.logical_not(selected)
        selected = self.data.selectref(list(selected))
        unselected = self.data.selectref(list(unselected))
        self.send('Selected Examples', selected)
        self.send('Unselected Examples', unselected)

    def on_axis_change(self):
        if self.data is not None:
            self.update_plot()

    def on_theme_change(self):
        if self.dark_theme:
            self.plot.theme = ScatterDarkTheme()
        else:
            self.plot.theme = ScatterLightTheme()

    def on_checkbox_update(self):
        self.plot.update()

    def update_plot(self):
        if self.data is None:
            return

        self.plot.update_data(self.x_attr, self.y_attr, self.z_attr,
                              self.color_attr, self.symbol_attr, self.size_attr,
                              self.label_attr)

    def showSelectedAttributes(self):
        val = self.vizrank.getSelectedProjection()
        if not val: return
        if self.data.domain.classVar:
            self.attr_color = self.data.domain.classVar.name
        if not self.plot.have_data:
            return
        attr_list = val[3]
        if attr_list and len(attr_list) == 3:
            self.x_attr = attr_list[0]
            self.y_attr = attr_list[1]
            self.z_attr = attr_list[2]

        self.update_plot()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = OWScatterPlot3D()
    data = orange.ExampleTable('../../doc/datasets/iris')
    w.set_data(data)
    w.handleNewSignals()
    w.show()
    app.exec_()

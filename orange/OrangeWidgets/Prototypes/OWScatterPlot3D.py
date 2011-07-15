"""<name> 3D Scatterplot</name>
"""

from OWWidget import *
from owplot3d import *

import OWGUI
import OWColorPalette

import numpy

class OWScatterPlot3D(OWWidget):
    settingsList = ['plot.show_legend']
    contextHandlers = {"": DomainContextHandler("", ["xAttr", "yAttr", "zAttr"])}
 
    def __init__(self, parent=None, signalManager=None, name="Scatter Plot 3D"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Subset Examples", ExampleTable, self.setSubsetData)]
        self.outputs = []

        self.x_attr = 0
        self.y_attr = 0
        self.z_attr = 0

        self.color_attr = None
        self.size_attr = None
        self.shape_attr = None
        self.label_attr = None

        self.symbol_scale = 5
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
            indent = 10,
            emptyString = '(Same size)',
            )

        self.shape_attr_cb = OWGUI.comboBox(additional_box, self, "shape_attr", label="Point shape:",
            tooltip="Attribute to use for pointShape",
            callback=self.on_axis_change,
            indent = 10,
            emptyString = '(Same shape)',
            )

        self.label_attr_cb = OWGUI.comboBox(additional_box, self, "label_attr", label="Point label:",
            tooltip="Attribute to use for pointLabel",
            callback=self.on_axis_change,
            indent = 10,
            emptyString = '(No labels)'
            )

        self.plot = OWPlot3D(self)

        box = OWGUI.widgetBox(self.settings_tab, 'Point properties')
        OWGUI.hSlider(box, self, "plot.symbol_scale", label="Symbol scale",
            minValue=1, maxValue=10,
            tooltip="Scale symbol size",
            callback=self.on_checkbox_update
            )

        OWGUI.hSlider(box, self, "plot.transparency", label="Transparency",
            minValue=10, maxValue=255,
            tooltip="Point transparency value",
            callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        box = OWGUI.widgetBox(self.settings_tab, 'General settings')
        OWGUI.checkBox(box, self, 'plot.show_x_axis_title',   'X axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_y_axis_title',   'Y axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_z_axis_title',   'Z axis title',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.show_legend',         'Show legend',    callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.ortho',               'Use ortho',      callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.filled_symbols',      'Filled symbols', callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.face_symbols',        'Face symbols',   callback=self.on_checkbox_update)
        OWGUI.checkBox(box, self, 'plot.grid',                'Show grid',      callback=self.on_checkbox_update)
        OWGUI.rubber(box)

        self.main_tab.layout().addStretch(100)
        self.settings_tab.layout().addStretch(100)

        self.mainArea.layout().addWidget(self.plot)

        self.data = None
        self.subsetData = None
        self.resize(1000, 600)

    def setData(self, data=None):
        self.closeContext("")
        self.data = data
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
            self.axis_candidate_attrs = [attr for attr in self.all_attrs
                if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]

            self.color_attr_cb.addItem('(Same color)')
            self.size_attr_cb.addItem('(Same size)')
            self.shape_attr_cb.addItem('(Same shape)')
            self.label_attr_cb.addItem('(No labels)')
            icons = OWGUI.getAttributeIcons() 
            for (i, attr) in enumerate(self.axis_candidate_attrs):
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
                                                               [len(self.axis_candidate_attrs) - 1]*3
                                                              ], axis=0)
            self.color_attr = max(len(self.axis_candidate_attrs) - 1, 0)
            self.openContext('', data)

    def setSubsetData(self, data=None):
        self.subsetData = data

    def handleNewSignals(self):
        self.update_plot()

    def on_axis_change(self):
        if self.data is not None:
            self.update_plot()

    def on_checkbox_update(self):
        self.plot.updateGL()

    def update_plot(self):
        if self.data is None:
            return

        x_ind, y_ind, z_ind = self.get_axes_indices()
        X, Y, Z, mask = self.get_axis_data(x_ind, y_ind, z_ind)

        color_legend_items = []
        if self.color_attr > 0:
            color_attr = self.axis_candidate_attrs[self.color_attr - 1]
            C = self.data_array[:, self.color_attr - 1]
            if color_attr.varType == orange.VarTypes.Discrete:
                palette = OWColorPalette.ColorPaletteHSV(len(color_attr.values))
                colors = [palette[int(value)] for value in C.ravel()]
                colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
                palette_colors = [palette[i] for i in range(len(color_attr.values))]
                color_legend_items = [[Symbol.TRIANGLE, [c.red()/255., c.green()/255., c.blue()/255., 1], 1, title]
                    for c, title in zip(palette_colors, color_attr.values)]
            else:
                palette = OWColorPalette.ColorPaletteBW()
                maxC, minC = numpy.max(C), numpy.min(C)
                C = (C - minC) / (maxC - minC)
                colors = [palette[value] for value in C.ravel()]
                colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
        else:
            colors = 'b'

        if self.size_attr > 0:
            size_attr = self.axis_candidate_attrs[self.size_attr - 1]
            S = self.data_array[:, self.size_attr - 1]
            if size_attr.varType == orange.VarTypes.Discrete:
                sizes = [(v + 1) * len(size_attr.values) / (11 - self.symbol_scale) for v in S]
            else:
                min, max = numpy.min(S), numpy.max(S)
                sizes = [(v - min) * self.symbol_scale / (max-min) for v in S]
        else:
            sizes = 1

        shapes = None
        if self.shape_attr > 0:
            i, shape_attr = self.discrete_attrs[self.shape_attr]
            if shape_attr.varType == orange.VarTypes.Discrete:
                # Map discrete attribute to [0...num shapes-1]
                shapes = self.data_array[:, i]
                num_shapes = 0
                unique_shapes = {}
                for shape in shapes:
                    if shape not in unique_shapes:
                        unique_shapes[shape] = num_shapes
                        num_shapes += 1
                shapes = [unique_shapes[value] for value in shapes]

        labels = None
        if self.label_attr > 0:
            label_attr = self.axis_candidate_attrs[self.label_attr - 1]
            labels = self.data_array[:, self.label_attr - 1]

        self.plot.clear()

        num_symbols = len(Symbol)
        if self.shape_attr > 0:
            _, shape_attr = self.discrete_attrs[self.shape_attr]
            titles = list(shape_attr.values)
            for i, title in enumerate(titles):
                if i == num_symbols-1:
                    title = ', '.join(titles[i:])
                self.plot.legend.add_item(i, (0,0,0,1), 1, '{0}={1}'.format(shape_attr.name, title))
                if i == num_symbols-1:
                    break

        if color_legend_items:
            for item in color_legend_items:
                self.plot.legend.add_item(*item)

        self.plot.scatter(X, Y, Z, colors, sizes, shapes, labels)
        self.plot.set_x_axis_title(self.axis_candidate_attrs[self.x_attr].name)
        self.plot.set_x_axis_title(self.axis_candidate_attrs[self.y_attr].name)
        self.plot.set_x_axis_title(self.axis_candidate_attrs[self.z_attr].name)

    def get_axis_data(self, x_ind, y_ind, z_ind):
        array = self.data_array
        X, Y, Z = array[:, x_ind], array[:, y_ind], array[:, z_ind]
        return X, Y, Z, None

    def get_axes_indices(self):
        return self.x_attr, self.y_attr, self.z_attr

if __name__ == "__main__":
  app = QApplication(sys.argv)
  w = OWScatterPlot3D()
  data = orange.ExampleTable("../../doc/datasets/iris")
  w.setData(data)
  w.handleNewSignals()
  w.show()
  app.exec_()

"""<name> 3D Scatterplot</name>
"""

from OWWidget import *
from OWGraph3D import *

import OWGUI
import OWColorPalette

import numpy

class OWScatterPlot3D(OWWidget):
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
        self.label_attr = None

        self.point_size = 5
        self.alpha_value = 255

        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.general_tab = OWGUI.createTabPage(self.tabs, 'General')
        self.settings_tab = OWGUI.createTabPage(self.tabs, 'Settings')

        self.x_attr_cb = OWGUI.comboBox(self.general_tab, self, "xAttr", box="X-axis Attribute",
            tooltip="Attribute to plot on X axis.",
            callback=self.onAxisChange
            )

        self.y_attr_cb = OWGUI.comboBox(self.general_tab, self, "yAttr", box="Y-axis Attribute",
            tooltip="Attribute to plot on Y axis.",
            callback=self.onAxisChange
            )

        self.z_attr_cb = OWGUI.comboBox(self.general_tab, self, "zAttr", box="Z-axis Attribute",
            tooltip="Attribute to plot on Z axis.",
            callback=self.onAxisChange
            )

        self.color_attr_cb = OWGUI.comboBox(self.general_tab, self, "colorAttr", box="Point color",
            tooltip="Attribute to use for point color",
            callback=self.onAxisChange)

        # Additional point properties (labels, size, shape).
        additional_box = OWGUI.widgetBox(self.general_tab, 'Additional Point Properties')
        self.size_attr_cb = OWGUI.comboBox(additional_box, self, "sizeAttr", box="Point Size:",
            tooltip="Attribute to use for pointSize",
            callback=self.onAxisChange,
            indent = 10,
            emptyString = '(Same size)',
            valueType = str
            )

        self.shape_attr_cb = OWGUI.comboBox(additional_box, self, "shapeAttr", box="Point Shape:",
            tooltip="Attribute to use for pointShape",
            callback=self.onAxisChange,
            )

        self.label_attr_cb = OWGUI.comboBox(additional_box, self, "labelAttr", box="Point Label:",
            tooltip="Attribute to use for pointLabel",
            callback=self.onAxisChange,
            )


        OWGUI.hSlider(self.settings_tab, self, "point_size", box="Max. point size",
            minValue=1, maxValue=10,
            tooltip="Maximum point size",
            callback=self.onAxisChange
            )

        OWGUI.hSlider(self.settings_tab, self, "alpha_value", box="Transparency",
            minValue=10, maxValue=255,
            tooltip="Point transparency value",
            callback=self.onAxisChange)

        # TODO: jittering options
        # TODO: find out what's with the TODO above
        # TODO: add ortho/perspective checkbox (or perhaps not?)
        # TODO: add grid enable/disable options

        OWGUI.rubber(self.general_tab)

        self.graph = OWGraph3D(self)
        self.mainArea.layout().addWidget(self.graph)

        self.data = None
        self.subsetData = None
        self.resize(800, 600)

    def setData(self, data=None):
      self.closeContext("")
      self.data = data
      self.x_attr_cb.clear()
      self.y_attr_cb.clear()
      self.z_attr_cb.clear()
      self.color_attr_cb.clear()
      self.size_attr_cb.clear()
      self.shape_attr_cb.clear()
      if self.data is not None:
        self.allAttrs = data.domain.variables + data.domain.getmetas().values()
        self.axisCandidateAttrs = [attr for attr in self.allAttrs if attr.varType in [orange.VarTypes.Continuous, orange.VarTypes.Discrete]]

        self.color_attr_cb.addItem("<None>")
        self.size_attr_cb.addItem("<None>")
        self.shape_attr_cb.addItem("<None>")
        icons = OWGUI.getAttributeIcons() 
        for attr in self.axisCandidateAttrs:
            self.x_attr_cb.addItem(icons[attr.varType], attr.name)
            self.y_attr_cb.addItem(icons[attr.varType], attr.name)
            self.z_attr_cb.addItem(icons[attr.varType], attr.name)
            self.color_attr_cb.addItem(icons[attr.varType], attr.name)
            self.size_attr_cb.addItem(icons[attr.varType], attr.name)

        array, c, w = self.data.toNumpyMA()
        if len(c):
          array = numpy.hstack((array, c.reshape(-1,1)))
        self.dataArray = array

        self.x_attr, self.y_attr, self.z_attr = numpy.min([[0, 1, 2], [len(self.axisCandidateAttrs) - 1]*3], axis=0)
        self.color_attr = max(len(self.axisCandidateAttrs) - 1, 0)

        self.openContext("", data)

    def setSubsetData(self, data=None):
      self.subsetData = data

    def handleNewSignals(self):
      self.updateGraph()

    def onAxisChange(self):
      if self.data is not None:
        self.updateGraph()

    def updateGraph(self):
      if self.data is None:
        return

      xInd, yInd, zInd = self.getAxesIndices()
      X, Y, Z, mask = self.getAxisData(xInd, yInd, zInd)

      if self.color_attr > 0:
        color_attr = self.axisCandidateAttrs[self.color_attr - 1]
        C = self.dataArray[:, self.color_attr - 1]
        if color_attr.varType == orange.VarTypes.Discrete:
          palette = OWColorPalette.ColorPaletteHSV(len(color_attr.values))
          colors = [palette[int(value)] for value in C.ravel()]
          colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
        else:
          palette = OWColorPalette.ColorPaletteBW()
          maxC, minC = numpy.max(C), numpy.min(C)
          C = (C - minC) / (maxC - minC)
          colors = [palette[value] for value in C.ravel()]
          colors = [[c.red()/255., c.green()/255., c.blue()/255., self.alpha_value/255.] for c in colors]
      else:
        colors = "b"

      if self.size_attr > 0:
        size_attr = self.axisCandidateAttrs[self.size_attr - 1]
        S = self.dataArray[:, self.size_attr - 1]
        if size_attr.varType == orange.VarTypes.Discrete:
          sizes = [(v + 1) * len(size_attr.values) / (11 - self.point_size) for v in S]
        else:
          min, max = numpy.min(S), numpy.max(S)
          sizes = [(v - min) * self.point_size / (max-min) for v in S]
      else:
        sizes = 1

      self.graph.clear()
      self.graph.scatter(X, Y, Z, colors, sizes)
      self.graph.set_x_axis_title(self.axisCandidateAttrs[self.x_attr].name)
      self.graph.set_x_axis_title(self.axisCandidateAttrs[self.y_attr].name)
      self.graph.set_x_axis_title(self.axisCandidateAttrs[self.z_attr].name)

    def getAxisData(self, xInd, yInd, zInd):
      array = self.dataArray
      X, Y, Z = array[:, xInd], array[:, yInd], array[:, zInd]
      return X, Y, Z, None

    def getAxesIndices(self):
      return self.x_attr, self.y_attr, self.z_attr

if __name__ == "__main__":
  app = QApplication(sys.argv)
  w = OWScatterPlot3D()
  data = orange.ExampleTable("../../doc/datasets/iris")
  w.setData(data)
  w.handleNewSignals()
  w.show()
  app.exec_()

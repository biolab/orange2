#
# OWHist.py
#
# the base for network histograms

import math
import numpy

import Orange
import OWGUI

from OWWidget import *
from OWGraph import *
from OWHist import *

class OWNxHist():

    def __init__(self, parent=None, type=0):
        self.parent = parent

    def addHistogramControls(self, parent=None):
        # set default settings
        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
        self.netOption = 0
        self.dstWeight = 0
        self.kNN = 0
        self.andor = 0
        self.matrix = None
        self.excludeLimit = 1
        self.percentil = 0

        if parent is None:
            parent = self.controlArea

        boxGeneral = OWGUI.widgetBox(parent, box="Distance boundaries")

        ribg = OWGUI.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)
        OWGUI.lineEdit(ribg, self, "spinLowerThreshold", "Lower", orientation='horizontal', callback=self.changeLowerSpin, valueType=float, enterPlaceholder=True, controlWidth=100)
        OWGUI.lineEdit(ribg, self, "spinUpperThreshold", "Upper    ", orientation='horizontal', callback=self.changeUpperSpin, valueType=float, enterPlaceholder=True, controlWidth=100)
        ribg.layout().addStretch(1)
        #ribg = OWGUI.radioButtonsInBox(boxGeneral, self, "andor", [], orientation='horizontal', callback = self.generateGraph)
        #OWGUI.appendRadioButton(ribg, self, "andor", "OR", callback = self.generateGraph)
        #b = OWGUI.appendRadioButton(ribg, self, "andor", "AND", callback = self.generateGraph)
        #b.setEnabled(False)
        #ribg.hide(False)

        ribg = OWGUI.widgetBox(boxGeneral, None, orientation="horizontal", addSpace=False)
        OWGUI.spin(ribg, self, "kNN", 0, 1000, 1, label="kNN   ", orientation='horizontal', callback=self.generateGraph, callbackOnReturn=1, controlWidth=100)
        OWGUI.doubleSpin(ribg, self, "percentil", 0, 100, 0.1, label="Percentil", orientation='horizontal', callback=self.setPercentil, callbackOnReturn=1, controlWidth=100)
        ribg.layout().addStretch(1)
        # Options
        self.attrColor = ""
        ribg = OWGUI.radioButtonsInBox(parent, self, "netOption", [], "Options", callback=self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "All vertices", callback=self.generateGraph)
        hb = OWGUI.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Large components only. Min nodes:", insertInto=hb, callback=self.generateGraph)
        OWGUI.spin(hb, self, "excludeLimit", 1, 100, 1, callback=(lambda h=True: self.generateGraph(h)))
        OWGUI.appendRadioButton(ribg, self, "netOption", "Largest connected component only", callback=self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Connected component with vertex")
        self.attribute = None
        self.attributeCombo = OWGUI.comboBox(parent, self, "attribute", box="Filter attribute", orientation='horizontal')#, callback=self.setVertexColor)

        ribg = OWGUI.radioButtonsInBox(parent, self, "dstWeight", [], "Distance -> Weight", callback=self.generateGraph)
        hb = OWGUI.widgetBox(ribg, None, orientation="horizontal", addSpace=False)
        OWGUI.appendRadioButton(ribg, self, "dstWeight", "Weight := distance", insertInto=hb, callback=self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "dstWeight", "Weight := 1 - distance", insertInto=hb, callback=self.generateGraph)

        self.label = ''
        self.searchString = OWGUI.lineEdit(self.attributeCombo.box, self, "label", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.generateGraph)

        if str(self.netOption) != '3':
            self.attributeCombo.box.setEnabled(False)

    def setPercentil(self):
        self.spinLowerThreshold = self.histogram.minValue
        # flatten matrix, sort values and remove identities (self.matrix[i][i])
        vals = sorted(sum(self.matrix, ()))[self.matrix.dim:]
        ind = int(len(vals) * self.percentil / 100)
        self.spinUpperThreshold = vals[ind]
        self.generateGraph()

    def enableAttributeSelection(self):
        self.attributeCombo.box.setEnabled(True)

    def setSearchStringTimer(self):
        self.searchStringTimer.stop()
        self.searchStringTimer.start(750)

    def setMatrix(self, data):
        if data == None: return

        self.matrix = data
        # draw histogram
        data.matrixType = orange.SymMatrix.Symmetric
        values = data.getValues()
        #print "values:",values
        self.histogram.setValues(values)

        low = min(values)
        upp = max(values)
        self.spinLowerThreshold = self.spinUpperThreshold = math.floor(low - (0.03 * (upp - low)))
        self.generateGraph()
        self.attributeCombo.clear()
        vars = []
        if (self.matrix != None):
            if hasattr(self.matrix, "items"):

                if isinstance(self.matrix.items, orange.ExampleTable):
                    vars = list(self.matrix.items.domain.variables)

                    metas = self.matrix.items.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)

        self.icons = self.createAttributeIconDict()

        for var in vars:
            try:
                self.attributeCombo.addItem(self.icons[var.varType], unicode(var.name))
            except:
                print "error adding ", var, " to the attribute combo"

    def changeLowerSpin(self):
        self.percentil = 0

        if self.spinLowerThreshold < self.histogram.minValue:
            self.spinLowerThreshold = self.histogram.minValue
        elif self.spinLowerThreshold > self.histogram.maxValue:
            self.spinLowerThreshold = self.histogram.maxValue

        if self.spinLowerThreshold >= self.spinUpperThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold

        self.generateGraph()

    def changeUpperSpin(self):
        self.percentil = 0

        if self.spinUpperThreshold < self.histogram.minValue:
            self.spinUpperThreshold = self.histogram.minValue
        elif self.spinUpperThreshold > self.histogram.maxValue:
            self.spinUpperThreshold = self.histogram.maxValue

        if self.spinUpperThreshold <= self.spinLowerThreshold:
            self.spinLowerThreshold = self.spinUpperThreshold

        self.generateGraph()

    def generateGraph(self, N_changed=False):
        self.searchStringTimer.stop()
        self.attributeCombo.box.setEnabled(False)
        self.error()
        matrix = None
        self.warning('')

        if N_changed:
            self.netOption = 1

        if self.matrix == None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            return

        #print len(self.histogram.yData), len(self.histogram.xData)
        nEdgesEstimate = 2 * sum([self.histogram.yData[i] for i, e in enumerate(self.histogram.xData) if self.spinLowerThreshold <= e <= self.spinUpperThreshold])

        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = Orange.network.Graph()
            graph.add_nodes_from(range(self.matrix.dim))
            matrix = self.matrix

            if hasattr(self.matrix, "items"):
                if type(self.matrix.items) == Orange.data.Table:
                    graph.set_items(self.matrix.items)
                else:
                    data = [[str(x)] for x in self.matrix.items]
                    items = Orange.data.Table(Orange.data.Domain(Orange.data.variable.String('label'), 0), data)
                    graph.set_items(items)

            # set the threshold
            # set edges where distance is lower than threshold
            self.warning(0)
            if self.kNN >= self.matrix.dim:
                self.warning(0, "kNN larger then supplied distance matrix dimension. Using k = %i" % (self.matrix.dim - 1))
            #nedges = graph.fromDistanceMatrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.dim - 1), self.andor)
            edge_list = Orange.network.GraphLayout().edges_from_distance_matrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.dim - 1))
            if self.dstWeight == 1:
                graph.add_edges_from(((u, v, {'weight':1 - d}) for u, v, d in edge_list))
            else:
                graph.add_edges_from(((u, v, {'weight':d}) for u, v, d in edge_list))

            # exclude unconnected
            if str(self.netOption) == '1':
                components = [x for x in Orange.network.nx.algorithms.components.connected_components(graph) if len(x) > self.excludeLimit]
                if len(components) > 0:
                    include = reduce(lambda x, y: x + y, components)
                    if len(include) > 1:
                        self.graph = graph.subgraph(include)
                        matrix = self.matrix.getitems(include)
                    else:
                        self.graph = None
                        matrix = None
                else:
                    self.graph = None
                    matrix = None
            # largest connected component only
            elif str(self.netOption) == '2':
                component = Orange.network.nx.algorithms.components.connected_components(graph)[0]
                if len(component) > 1:
                    self.graph = graph.subgraph(include)
                    matrix = self.matrix.getitems(component)
                else:
                    self.graph = None
                    matrix = None
            # connected component with vertex by label
            elif str(self.netOption) == '3':
                self.attributeCombo.box.setEnabled(True)
                self.graph = None
                matrix = None
                #print self.attributeCombo.currentText()
                if self.attributeCombo.currentText() != '' and self.label != '':
                    components = Orange.network.nx.algorithms.components.connected_components(graph)

                    txt = self.label.lower()
                    #print 'txt:',txt
                    nodes = [i for i, values in enumerate(self.matrix.items) if txt in str(values[str(self.attributeCombo.currentText())]).lower()]
                    #print "nodes:",nodes
                    if len(nodes) > 0:
                        vertices = []
                        for component in components:
                            for node in nodes:
                                if node in component:
                                    if len(component) > 0:
                                        vertices.extend(component)

                        if len(vertices) > 0:
                            #print "n vertices:", len(vertices), "n set vertices:", len(set(vertices))
                            vertices = list(set(vertices))
                            self.graph = graph.subgraph(include)
                            matrix = self.matrix.getitems(vertices)
            else:
                self.graph = graph

        if matrix != None:
            matrix.items = self.graph.items()
            self.graph_matrix = matrix

        self.pconnected = self.graph.number_of_nodes()
        self.nedges = self.graph.number_of_edges()
        if hasattr(self, "infoa"):
            self.infoa.setText("Matrix size: %d" % self.matrix.dim)
        if hasattr(self, "infob"):
            self.infob.setText("Graph nodes: %d (%3.1f%%)" % (self.pconnected, self.pconnected / float(self.matrix.dim) * 100))
        if hasattr(self, "infoc"):
            self.infoc.setText("Graph edges: %d (%.2f edges/node)" % (self.nedges, self.nedges / float(self.pconnected)))

        #print 'self.graph:',self.graph+
        if hasattr(self, "sendSignals"):
            self.sendSignals()

        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)


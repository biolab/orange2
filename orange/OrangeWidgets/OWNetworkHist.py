#
# OWHist.py
#
# the base for network histograms

import OWGUI
from OWWidget import *
from OWGraph import *
import orngNetwork

import numpy, math
from OWHist import *

class OWNetworkHist():
    
    def __init__(self, parent=None, type=0):
        self.parent = parent
        
    def addHistogramControls(self):
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
        
        boxGeneral = OWGUI.widgetBox(self.controlArea, box = "Distance boundaries")
        
        OWGUI.lineEdit(boxGeneral, self, "spinLowerThreshold", "Lower:", orientation='horizontal', callback=self.changeLowerSpin, valueType=float)
        OWGUI.lineEdit(boxGeneral, self, "spinUpperThreshold", "Upper:", orientation='horizontal', callback=self.changeUpperSpin, valueType=float)
        ribg = OWGUI.radioButtonsInBox(boxGeneral, self, "andor", [], orientation='horizontal', callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "andor", "OR", callback = self.generateGraph)
        b = OWGUI.appendRadioButton(ribg, self, "andor", "AND", callback = self.generateGraph)
        b.setEnabled(False)
        OWGUI.spin(boxGeneral, self, "kNN", 0, 1000, 1, label="kNN:", orientation='horizontal', callback=self.generateGraph)
        OWGUI.doubleSpin(boxGeneral, self, "percentil", 0, 100, 0.1, label="Percentil:", orientation='horizontal', callback=self.setPercentil, callbackOnReturn=1)
        
        # Options
        self.attrColor = ""
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "netOption", [], "Options", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "All vertices", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Exclude small components", callback = self.generateGraph)
        OWGUI.spin(OWGUI.indentedBox(ribg), self, "excludeLimit", 1, 100, 1, label="Less vertices than: ", callback = (lambda h=True: self.generateGraph(h)))
        OWGUI.appendRadioButton(ribg, self, "netOption", "Largest connected component only", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Connected component with vertex")
        self.attribute = None
        self.attributeCombo = OWGUI.comboBox(ribg, self, "attribute", box = "Filter attribute")#, callback=self.setVertexColor)
        
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "dstWeight", [], "Distance -> Weight", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "dstWeight", "Weight := distance", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "dstWeight", "Weight := 1 - distance", callback = self.generateGraph)
        
        self.label = ''
        self.searchString = OWGUI.lineEdit(self.attributeCombo.box, self, "label", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.generateGraph)
        
        if str(self.netOption) != '3':
            self.attributeCombo.box.setEnabled(False)
            
    def setPercentil(self):
        self.spinLowerThreshold = self.histogram.minValue
        net = orngNetwork.Network(self.matrix.dim, 0)
        lower, upper = net.getDistanceMatrixThreshold(self.matrix, self.percentil/100)
        self.spinUpperThreshold = upper
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
        self.spinLowerThreshold = self.spinUpperThreshold = low - (0.03 * (upp - low))
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
        
    def generateGraph(self, N_changed = False):
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
        nEdgesEstimate = 2 * sum([self.histogram.yData[i] for i,e in enumerate(self.histogram.xData) if self.spinLowerThreshold <= e <= self.spinUpperThreshold])
        
        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = orngNetwork.Network(self.matrix.dim, 0)
            matrix = self.matrix
            
            if hasattr(self.matrix, "items"):               
                if type(self.matrix.items) == type(orange.ExampleTable(orange.Domain(orange.StringVariable('tmp')))):
                    #graph.setattr("items", self.data.items)
                    graph.items = self.matrix.items
                else:
                    data = [[str(x)] for x in self.matrix.items]
                    items = orange.ExampleTable(orange.Domain(orange.StringVariable('label'), 0), data)
                    #graph.setattr("items", list(items))
                    graph.items = list(items)
                
            # set the threshold
            # set edges where distance is lower than threshold
                  
            self.warning(0)
            if self.kNN >= self.matrix.dim:
                self.warning(0, "kNN larger then supplied distance matrix dimension. Using k = %i" % (self.matrix.dim - 1))
            nedges = graph.fromDistanceMatrix(self.matrix, self.spinLowerThreshold, self.spinUpperThreshold, min(self.kNN, self.matrix.dim - 1), self.andor)
            edges = graph.getEdges()
            
            #print graph.nVertices, self.matrix.dim
            
            if self.dstWeight == 1:
                if graph.directed:
                    for u,v in edges:
                        foo = 1
                        if str(graph[u,v]) != "0":
                            foo = 1.0 - float(graph[u,v])
                        
                        graph[u,v] = foo
                else:
                    for u,v in edges:
                        if u <= v:
                            foo = 1
                            if str(graph[u,v]) != "0":
                                foo = 1.0 - float(graph[u,v])
                            
                            graph[u,v] = foo
                    
            n = len(edges)
            #print 'self.netOption',self.netOption
            # exclude unconnected
            if str(self.netOption) == '1':
                components = [x for x in graph.getConnectedComponents() if len(x) > self.excludeLimit]
                if len(components) > 0:
                    include = reduce(lambda x,y: x+y, components)
                    if len(include) > 1:
                        self.graph = orngNetwork.Network(graph.getSubGraph(include))
                        matrix = self.matrix.getitems(include)
                    else:
                        self.graph = None
                        matrix = None
                else:
                    self.graph = None
                    matrix = None
            # largest connected component only        
            elif str(self.netOption) == '2':
                component = graph.getConnectedComponents()[0]
                if len(component) > 1:
                    self.graph = orngNetwork.Network(graph.getSubGraph(component))
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
                    components = graph.getConnectedComponents()
                        
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
                            self.graph = orngNetwork.Network(graph.getSubGraph(vertices))
                            matrix = self.matrix.getitems(vertices)
            else:
                self.graph = graph
        
        if matrix != None:
            self.matrix = matrix
            
        self.pconnected = nedges
        self.nedges = n
        if hasattr(self, "infoa"):
            self.infoa.setText("%d vertices" % self.matrix.dim)
        if hasattr(self, "infob"):
            self.infob.setText("%d connected (%3.1f%%)" % (nedges, nedges / float(self.matrix.dim) * 100))
        if hasattr(self, "infoc"):
            self.infoc.setText("%d edges (%d average)" % (n, n / float(self.matrix.dim)))
        
        #print 'self.graph:',self.graph+
        if hasattr(self, "sendSignals"):
            self.sendSignals()
        
        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)
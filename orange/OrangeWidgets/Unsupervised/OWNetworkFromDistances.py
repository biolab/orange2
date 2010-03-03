"""
<name>Network from Distances</name>
<description>Costructs Graph object by connecting nodes from ExampleTable where distance between them is between given threshold.</description>
<icon>icons/NetworkFromDistances.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3300</priority>
"""

#
# OWNetworkFromDistances.py
#

import OWGUI
import orange
import orngNetwork
import copy, random

from OWWidget import *
from OWGraph import *
from OWHist import *

class OWNetworkFromDistances(OWWidget):
    settingsList=["spinLowerThreshold", "spinUpperThreshold", "netOption", "dstWeight", "kNN", "andor", "excludeLimit"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix)]
        self.outputs = [("Network", orngNetwork.Network), ("Examples", ExampleTable), ("Distance Matrix", orange.SymMatrix)]

        # set default settings
        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
        self.netOption = 0
        self.dstWeight = 0
        self.kNN = 0
        self.andor = 0
        self.data = None
        self.excludeLimit = 1
        # get settings from the ini file, if they exist
        self.loadSettings()
        
        # GUI
        # general settings
        boxHistogram = OWGUI.widgetBox(self.mainArea, box = "Distance histogram")
        self.histogram = OWHist(self, boxHistogram)
        boxHistogram.layout().addWidget(self.histogram)

        boxHistogram.setMinimumWidth(500)
        boxHistogram.setMinimumHeight(300)
        
        boxGeneral = OWGUI.widgetBox(self.controlArea, box = "Distance boundaries")
        
        OWGUI.lineEdit(boxGeneral, self, "spinLowerThreshold", "Lower:", orientation='horizontal', callback=self.changeLowerSpin, valueType=float)
        OWGUI.lineEdit(boxGeneral, self, "spinUpperThreshold", "Upper:", orientation='horizontal', callback=self.changeUpperSpin, valueType=float)
        ribg = OWGUI.radioButtonsInBox(boxGeneral, self, "andor", [], orientation='horizontal', callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "andor", "OR", callback = self.generateGraph)
        b = OWGUI.appendRadioButton(ribg, self, "andor", "AND", callback = self.generateGraph)
        b.setEnabled(False)
        OWGUI.spin(boxGeneral, self, "kNN", 0, 1000, 1, label="kNN:", orientation='horizontal', callback=self.generateGraph)
        
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
            
        # info
        boxInfo = OWGUI.widgetBox(self.controlArea, box = "Network info")
        self.infoa = OWGUI.widgetLabel(boxInfo, "No data loaded.")
        self.infob = OWGUI.widgetLabel(boxInfo, '')
        self.infoc = OWGUI.widgetLabel(boxInfo, '')
        
        self.resize(700, 100)
        
    def enableAttributeSelection(self):
        self.attributeCombo.box.setEnabled(True)
        
    def setSearchStringTimer(self):
        self.searchStringTimer.stop()
        self.searchStringTimer.start(750)

    def setMatrix(self, data):
        if data == None: return
        
        self.data = data
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
        if (self.data != None):
            if hasattr(self.data, "items"):
                 
                if isinstance(self.data.items, orange.ExampleTable):
                    vars = list(self.data.items.domain.variables)
                
                    metas = self.data.items.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
                        
        self.icons = self.createAttributeIconDict()
                     
        for var in vars:
            try:
                self.attributeCombo.addItem(self.icons[var.varType], unicode(var.name))
            except:
                print "error adding ", var, " to the attribute combo"

    def changeLowerSpin(self):
        if self.spinLowerThreshold < self.histogram.minValue:
            self.spinLowerThreshold = self.histogram.minValue
        elif self.spinLowerThreshold > self.histogram.maxValue:
            self.spinLowerThreshold = self.histogram.maxValue
            
        if self.spinLowerThreshold >= self.spinUpperThreshold:
            self.spinUpperThreshold = self.spinLowerThreshold
            
        self.generateGraph()
        
    def changeUpperSpin(self):
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
            
        if self.data == None:
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
            graph = orngNetwork.Network(self.data.dim, 0)
            matrix = self.data
            
            if hasattr(self.data, "items"):               
                if type(self.data.items) == type(orange.ExampleTable(orange.Domain(orange.StringVariable('tmp')))):
                    #graph.setattr("items", self.data.items)
                    graph.items = self.data.items
                else:
                    data = [[str(x)] for x in self.data.items]
                    items = orange.ExampleTable(orange.Domain(orange.StringVariable('label'), 0), data)
                    #graph.setattr("items", list(items))
                    graph.items = list(items)
                
            # set the threshold
            # set edges where distance is lower than threshold
                  
            nedges = graph.fromDistanceMatrix(self.data, self.spinLowerThreshold, self.spinUpperThreshold, self.kNN, self.andor)
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
                        matrix = self.data.getitems(include)
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
                    matrix = self.data.getitems(component)
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
                    nodes = [i for i, values in enumerate(self.data.items) if txt in str(values[str(self.attributeCombo.currentText())]).lower()]
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
                            matrix = self.data.getitems(vertices)
            else:
                self.graph = graph
                
                
        self.pconnected = nedges
        self.nedges = n
        self.infoa.setText("%d vertices" % self.data.dim)
        self.infob.setText("%d connected (%3.1f%%)" % (nedges, nedges / float(self.data.dim) * 100))
        self.infoc.setText("%d edges (%d average)" % (n, n / float(self.data.dim)))
        
        #print 'self.graph:',self.graph+
        if self.graph != None:
            #setattr(matrix, "items", self.graph.items)
            matrix.items = self.graph.items
        
        self.send("Network", self.graph)
        
        if matrix:
            self.send("Distance Matrix", matrix)
            
        if self.graph == None:
             self.send("Examples", None)
        else:
            self.send("Examples", self.graph.items)
        
        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Edge thresholds", "%.5f - %.5f" % (self.spinLowerThreshold, self.spinUpperThreshold)),
                             ("Selected vertices", ["All", "Without isolated vertices", "Largest component", "Connected with vertex"][self.netOption]),
                             ("Weight", ["Distance", "1 - Distance"][self.dstWeight])])
        self.reportSection("Histogram")
        self.reportImage(self.histogram.saveToFileDirect, QSize(400,300))
        self.reportSettings("Output graph",
                            [("Vertices", self.data.dim),
                             ("Edges", self.nedges),
                             ("Connected vertices", "%i (%.1f%%)" % (self.pconnected, self.pconnected / max(1, float(self.data.dim))*100)),
                             ])
                                                                     
if __name__ == "__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetworkFromDistances()
    ow.show()
    appl.exec_()

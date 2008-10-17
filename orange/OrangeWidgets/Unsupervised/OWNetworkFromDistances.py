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
from OWWidget import *
from OWGraph import *
from orngNetwork import * 
from orangeom import Network
from OWHist import *

            
class OWNetworkFromDistances(OWWidget):
    settingsList=["spinLowerThreshold", "spinUpperThreshold", "netOption"]
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "Network from Distances")
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix)]
        self.outputs = [("Network", Network), ("Examples", ExampleTable), ("Distance Matrix", orange.SymMatrix)]

        # set default settings
        self.spinLowerThreshold = 0
        self.spinLowerChecked = False
        self.spinUpperThreshold = 0
        self.spinUpperChecked = False
        self.netOption = 0
        self.data = None
        
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
        
        # Options
        self.attrColor = ""
        ribg = OWGUI.radioButtonsInBox(self.controlArea, self, "netOption", [], "Options", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "All vertices", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Exclude unconnected vertices", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Largest connected component only", callback = self.generateGraph)
        OWGUI.appendRadioButton(ribg, self, "netOption", "Connected component with vertex")
        self.attribute = None
        self.attributeCombo = OWGUI.comboBox(ribg, self, "attribute", box = "Filter attribute")#, callback=self.setVertexColor)
        
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
        data.matrixType = data.Lower
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
                    vars[:0] = self.data.items.domain.variables
                
                    metas = self.data.items.domain.getmetas(0)
                    for i, var in metas.iteritems():
                        vars.append(var)
                        
        self.icons = self.createAttributeIconDict()
                        
        for var in vars:
            self.attributeCombo.addItem(self.icons[var.varType], unicode(var.name))

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
        
    def generateGraph(self):
        self.searchStringTimer.stop()
        self.attributeCombo.box.setEnabled(False)
        self.error()
        
        if self.data == None:
            self.infoa.setText("No data loaded.")
            self.infob.setText("")
            return
        
        nEdgesEstimate = 2 * sum([self.histogram.yData[i] for i,e in enumerate(self.histogram.xData) if self.spinLowerThreshold <= e <= self.spinUpperThreshold])
        
        if nEdgesEstimate > 200000:
            self.graph = None
            nedges = 0
            n = 0
            self.error('Estimated number of edges is too high (%d).' % nEdgesEstimate)
        else:
            graph = Network(self.data.dim, 0)
            matrix = self.data
            
            if hasattr(self.data, "items"):               
                if type(self.data.items) == type(orange.ExampleTable(orange.Domain(orange.StringVariable('tmp')))):
                    graph.setattr("items", self.data.items)
                else:
                    data = [[str(x)] for x in self.data.items]
                    items = orange.ExampleTable(orange.Domain(orange.StringVariable('label'), 0), data)
                    graph.setattr("items", list(items))
                
            # set the threshold
            # set edges where distance is lower than threshold
            nedges = graph.fromDistanceMatrix(self.data, self.spinLowerThreshold, self.spinUpperThreshold)
            n = len(graph.getEdges())
            #print 'self.netOption',self.netOption
            
            if str(self.netOption) == '1':
                components = [x for x in graph.getConnectedComponents() if len(x) > 1]
                if len(components) > 0:
                    include = reduce(lambda x,y: x+y, components)
                    if len(include) > 1:
                        self.graph = Network(graph.getSubGraph(include))
                        matrix = self.data.getitems(include)
                    else:
                        self.graph = None
                        matrix = None
                else:
                    self.graph = None
                    matrix = None
                    
            elif str(self.netOption) == '2':
                component = graph.getConnectedComponents()[0]
                if len(component) > 1:
                    self.graph = Network(graph.getSubGraph(component))
                    matrix = self.data.getitems(component)
                else:
                    self.graph = None
                    matrix = None

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
                                    if len(component) > 1:
                                        vertices.extend(component)
                                        
                        if len(vertices) > 0:
                            #print vertices
                            self.graph = Network(graph.getSubGraph(vertices))
                            matrix = self.data.getitems(vertices)
            else:
                self.graph = graph
                
        self.infoa.setText("%d vertices" % self.data.dim)
        self.infob.setText("%d connected (%3.1f%%)" % (nedges, nedges / float(self.data.dim) * 100))
        self.infoc.setText("%d edges (%d average)" % (n, n / float(self.data.dim)))
        
        #print 'self.graph:',self.graph
        setattr(matrix, "items", self.graph.items)
        
        self.send("Network", self.graph)
        self.send("Distance Matrix", matrix)
        if self.graph == None:
             self.send("Examples", None)
        else:
            self.send("Examples", self.graph.items)
        
        self.histogram.setBoundary(self.spinLowerThreshold, self.spinUpperThreshold)
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFromDistances()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()
"""
<name>Network</name>
<description>Network Widget visualizes graphs.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2040</priority>
"""
import OWGUI

from OWWidget import *
from qwt import *
from qt import *
from OWGraphDrawerCanvas import *
from orngNetwork import * 
from time import *
import OWToolbars
from statc import mean

#ta razred je potreben, da lahko narisemo pixmap z painterjem
class pixmaps(QWidget):
    def __init__(self):
        apply(QWidget.__init__, (self,))

        #risanje lupe
        self.LOOKG=QPixmap(20,20)
        self.LOOKG.fill()
        painter=QPainter(self.LOOKG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(Qt.red))
        pa=QPointArray([2,5,2,10,5,13,10,13,13,10,13,5,10,2,5,2,2,5])
        painter.drawPolygon(pa)
        painter.drawLine(12,11,19,18)
        painter.drawLine(11,12,18,19)
        painter.drawLine(12,12,18,18)
        painter.end()

        #risanje roke
        self.HAND=QPixmap(20,20)
        self.HAND.fill()
        painter=QPainter(self.HAND)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(Qt.red))
        pa=QPointArray([1,4,1,6,4,6,4,8,6,8,6,10,8,10,8,12,11,12,15,8,15,7,12,4,1,4])
        painter.drawPolygon(pa)
        painter.drawLine(5,6,7,6)
        painter.drawLine(7,8,9,8)
        painter.drawPoint(9,10)
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([11,12,14,15,18,11,15,8,11,12])
        painter.drawPolygon(pa)
        painter.end()

        #risanje pravokotnega izbiralnika
        self.RECTG=QPixmap(20,20)
        self.RECTG.fill()
        painter=QPainter(self.RECTG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([2,2,2,17,17,17,17,2,2,2])
        painter.drawPolygon(pa)
        painter.setPen(QPen(Qt.white,1))
        pa=QPointArray([2,5,2,7,2,9,2,11,2,13,2,15,4,17,6,17,8,17,10,17,12,17,14,17,17,14,17,12,17,10,17,8,17,6,17,4,15,2,13,2,11,2,9,2,7,2,5,2])
        painter.drawPoints(pa,0,-1)
        painter.setPen(QPen(Qt.black,1))
        pa=QPointArray([6,6,6,13,13,13,13,6,6,6])
        painter.drawPolygon(pa)
        painter.drawLine(6,6,13,13)
        painter.drawLine(6,13,13,6)
        painter.setPen(QPen(QColor(192,192,255),1))
        pa=QPointArray([6,9,6,10,9,13,10,13,13,10,13,9,10,6,9,6])
        painter.drawPoints(pa,0,-1)
        painter.end()

        #risanje poligonskega izbiralnika
        self.POLYG=QPixmap(20,20)
        self.POLYG.fill()
        painter=QPainter(self.POLYG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([1,5,3,15,9,13,14,18,18,12,18,11,13,2,13,1,1,5])
        painter.drawPolygon(pa)
        painter.end()


class OWNetwork(OWWidget):
    settingsList=["autoSendSelection", "spinExplicit", "spinPercentage"]
    contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="markerAttributes"),
                                                     ContextField("attributes", selected="tooltipAttributes"),
                                                     "color"])}


    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Network')

        self.inputs = [("Graph with ExampleTable", orange.Graph, self.setGraph), ("Example Subset", orange.ExampleTable, self.setExampleSubset)]
        self.outputs=[("Selected Examples", ExampleTable), ("Selected Graph", orange.Graph)]
        
        self.markerAttributes = []
        self.tooltipAttributes = []
        self.attributes = []
        self.autoSendSelection = False
        self.graphShowGrid = 1  # show gridlines in the graph
        
        self.markNConnections = 2
        self.markNumber = 0
        self.markProportion = 0
        self.markSearchString = ""
        self.markDistance = 2
        self.frSteps = 1
        self.hubs = 0
        self.color = 0
        self.nVertices = self.nMarked = self.nSelected = self.nHidden = self.nShown = 0
        self.optimizeWhat = 1
        
        self.loadSettings()

        self.visualize = None

        self.graph = OWGraphDrawerCanvas(self, self.mainArea, "Network")
        #self.optimize = OWGraphDrawingOptimize(parent=self);
        #start of content (right) area
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)

        self.tabs = QTabWidget(self.controlArea)
#        self.mainTab = QVGroupBox(self)
        self.displayTab = QVGroupBox(self)
        self.mainTab = self.displayTab
        self.markTab = QVGroupBox(self)
#        self.tabs.insertTab(self.mainTab, "Main")
        self.tabs.insertTab(self.displayTab, "Display")
        self.tabs.insertTab(self.markTab, "Mark")
        OWGUI.separator(self.controlArea)


        self.optimizeBox = OWGUI.radioButtonsInBox(self.mainTab, self, "optimizeWhat", [], "Optimize", addSpace=True)
        OWGUI.button(self.optimizeBox, self, "Random", callback=self.random)
        OWGUI.button(self.optimizeBox, self, "Fruchterman Reingold", callback=self.fr)
        OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 10000, 1, label="Steps ")
        OWGUI.button(self.optimizeBox, self, "F-R Radial", callback=self.frRadial)
        OWGUI.button(self.optimizeBox, self, "Circular Original", callback=self.circularOriginal)
        OWGUI.button(self.optimizeBox, self, "Circular Random", callback=self.circularRandom)
        OWGUI.button(self.optimizeBox, self, "Circular Crossing Reduction", callback=self.circularCrossingReduction)
        OWGUI.separator(self.optimizeBox)
        OWGUI.widgetLabel("Optimize")
        ib = OWGUI.indentedBox(self.optimizeBox)
        for wh in ("All points", "Shown points", "Selected points"):
            OWGUI.appendRadioButton(self.optimizeBox, self, "optimizeWhat", wh, insertInto=ib)
        
        self.colorCombo = OWGUI.comboBox(self.displayTab, self, "color", box = "Color attribute", callback=self.setVertexColor)
        self.colorCombo.insertItem("(none)")
        
        self.attListBox = OWGUI.listBox(self.displayTab, self, "markerAttributes", "attributes", box = "Labels", selectionMode=QListBox.Multi, callback=self.clickedAttLstBox)
        
        self.tooltipBox = OWGUI.widgetBox(self.mainTab, "Tooltips", addSpace = True)
        self.tooltipListBox = OWGUI.listBox(self.tooltipBox, self, "tooltipAttributes", "attributes", selectionMode=QListBox.Multi, callback=self.clickedTooltipLstBox)

        OWGUI.button(self.mainTab, self, "Show degree distribution", callback=self.showDegreeDistribution)
        
        ib = OWGUI.widgetBox(self.markTab, "Info", addSpace = True)
        OWGUI.label(ib, self, "Number of vertices: %(nVertices)i")
        OWGUI.label(ib, self, "Shown/Hidden vertices: %(nShown)i/%(nHidden)i")
        OWGUI.label(ib, self, "Selected vertices: %(nSelected)i")
        OWGUI.label(ib, self, "Marked vertices: %(nMarked)i")
        
        ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Method", callback = self.setHubs, addSpace = True)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark vertices given in the input signal")
        OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Find vertices which label contain")
        self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.setHubs)
        
        OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of focused vertex")
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of selected vertices")
        ib = OWGUI.indentedBox(ribg, orientation = 0)
        self.ctrlMarkDistance = OWGUI.spin(ib, self, "markDistance", 0, 100, 1, label="Distance ", callback=(lambda h=2: self.setHubs(h)))
        OWGUI.separator(ib, 4, 4)
        self.ctrlMarkFreeze = OWGUI.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
        OWGUI.separator(ribg)
        OWGUI.widgetLabel(ribg, "Mark  vertices with ...")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at least N connections")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at most N connections")
        self.ctrlMarkNConnections = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N ", callback=(lambda h=4: self.setHubs(h)))
        OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than any neighbour")
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than avg neighbour")
        OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "most connections")
        ib = OWGUI.indentedBox(ribg)
        self.ctrlMarkNumber = OWGUI.spin(ib, self, "markNumber", 0, 1000000, 1, label="Number of vertices" + ": ", callback=(lambda h=8: self.setHubs(h)))
        OWGUI.widgetLabel(ib, "(More vertices are marked in case of ties)")
#        self.ctrlMarkProportion = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markProportion", 0, 100, 1, label="Percentage" + ": ", callback=self.setHubs)
        
        ib = OWGUI.widgetBox(self.markTab, "Selection", addSpace = True)
        OWGUI.button(ib, self, "Add Marked to Selection", callback = self.markedToSelection)
        OWGUI.button(ib, self, "Remove Marked from Selection",callback = self.markedFromSelection)
        OWGUI.button(ib, self, "Set Selection to Marked", callback = self.setSelectionToMarked)

        self.hideBox = QVGroupBox("Hide", self.markTab)
        OWGUI.button(self.hideBox, self, "Hide selected", callback=self.hideSelected)
        OWGUI.button(self.hideBox, self, "Hide unselected", callback=self.hideAllButSelected)
        OWGUI.button(self.hideBox, self, "Show all", callback=self.showAllNodes)
        
        pics=pixmaps()
        
        T = OWToolbars.ZoomSelectToolbar
        self.zoomSelectToolbar = OWToolbars.ZoomSelectToolbar(self, self.controlArea, self.graph, self.autoSendSelection,
                                                              buttons = (T.IconZoom, T.IconPan, None,
                                                                         ("Move selection", "buttonMoveSelection", "activateMoveSelection", QPixmap(OWToolbars.dlg_zoom), Qt.sizeAllCursor, 1),
                                                                         T.IconRectangle, T.IconPolygon, 
                                                                         T.IconSendSelection))
        
        OWGUI.button(self.controlArea, self, "Save network", callback=self.saveNetwork)
        OWGUI.button(self.controlArea, self, "test replot", callback=self.testRefresh)

        self.icons = self.createAttributeIconDict()
        self.setHubs()
        
        self.resize(850, 700)    
    
    def setSearchStringTimer(self):
        self.hubs = 1
        self.searchStringTimer.stop()
        self.searchStringTimer.start(750, True)
         
    def setHubs(self, i = None):
        if not i is None:
            self.hubs = i
            
        self.graph.tooltipNeighbours = self.hubs == 2 and self.markDistance or 0
        self.graph.markWithRed = False

        if not self.visualize or not self.visualize.graph:
            return
        
        hubs = self.hubs
        vgraph = self.visualize.graph

        if hubs == 0:
            return
        
        elif hubs == 1:
            txt = self.markSearchString
            labelText = self.graph.labelText
            self.graph.markWithRed = self.graph.nVertices > 200
            self.graph.setMarkedNodes([i for i, values in enumerate(vgraph.items) if txt in " ".join([str(values[ndx]) for ndx in labelText])])
            print [i for i, values in enumerate(vgraph.items) if txt in " ".join([str(values[ndx]) for ndx in labelText])]
            return
        
        elif hubs == 2:
            self.graph.setMarkedNodes([])
            self.graph.tooltipNeighbours = self.markDistance
            return

        elif hubs == 3:
            self.graph.setMarkedNodes([])
            self.graph.selectionNeighbours = self.markDistance
            self.graph.markSelectionNeighbours()
            return
        
        self.graph.tooltipNeighbours = self.graph.selectionNeighbours = 0
        powers = vgraph.getDegrees()
        
        if hubs == 4: # at least N connections
            N = self.markNConnections
            self.graph.setMarkedNodes([i for i, power in enumerate(powers) if power >= N])
        elif hubs == 5:
            N = self.markNConnections
            self.graph.setMarkedNodes([i for i, power in enumerate(powers) if power <= N])
        elif hubs == 6:
            self.graph.setMarkedNodes([i for i, power in enumerate(powers) if power > max([0]+[powers[nn] for nn in vgraph.getNeighbours(i)])])
        elif hubs == 7:
            self.graph.setMarkedNodes([i for i, power in enumerate(powers) if power > mean([0]+[powers[nn] for nn in vgraph.getNeighbours(i)])])
        elif hubs == 8:
            sortedIdx = range(len(powers))
            sortedIdx.sort(lambda x,y: -cmp(powers[x], powers[y]))
            cutP = self.markNumber
            cutPower = powers[sortedIdx[cutP]]
            while cutP < len(powers) and powers[sortedIdx[cutP]] == cutPower:
                cutP += 1
            self.graph.setMarkedNodes(sortedIdx[:cutP-1])
            
        
    def clickedAttLstBox(self):
        self.graph.setLabelText([self.attributes[i][0] for i in self.markerAttributes])
        self.updateCanvas()
    
        
    def clickedTooltipLstBox(self):
        self.graph.setTooltipText([self.attributes[i][0] for i in self.tooltipAttributes])
        self.updateCanvas()
    

    def testRefresh(self):
        start = time()
        self.graph.replot()
        stop = time()    
        print "replot in " + str(stop - start)
        
    def saveNetwork(self):
        filename = QFileDialog.getSaveFileName(QString.null,'PAJEK networks (*.net)')
        if filename:
            fn = ""
            head, tail = os.path.splitext(str(filename))
            if not tail:
                fn = head + ".net"
            else:
                fn = str(filename)
            
            self.graph.visualizer.saveNetwork(fn)
    
    def selectConnectedNodes(self):
        self.graph.selectConnectedNodes(self.connectDistance)
        
    def selectAllConnectedNodes(self):
        self.graph.selectConnectedNodes(1000000)
            
    
    def sendData(self):
        graph = self.graph.getSelectedGraph()
        
        if graph != None:
            if graph.items != None:
                self.send("Selected Examples", graph.items)
            else:
                self.send("Selected Examples", self.graph.getSelectedExamples())
                
            self.send("Selected Graph", graph)
        else:
            items = self.graph.getSelectedExamples()
            if items != None:
                self.send("Selected Examples", items)
            self.send("Selected Graph", None)
   
    
    
    def setGraph(self, graph):
        if graph == None:
            return
        #print "OWNetwork/setGraph: new visualizer..."
        self.visualize = NetworkVisualizer(graph, self)
        self.nVertices = len(graph)
        #print "done."
        vars = self.visualize.getVars()
        self.attributes = [(var.name, var.varType) for var in vars]

        self.colorCombo.clear()
        self.colorCombo.insertItem("(one color)")
        for var in vars:
            if var.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                self.colorCombo.insertItem(self.icons[var.varType], unicode(var.name))

        #print "OWNetwork/setGraph: add visualizer..."
        self.graph.addVisualizer(self.visualize)
        #print "done."
        #print "OWNetwork/setGraph: display random..."
        k = 1.13850193174e-008
        nodes = self.visualize.nVertices()
        t = k * nodes * nodes
        self.frSteps = int(5.0 / t)
        if self.frSteps <   1: self.frSteps = 1;
        if self.frSteps > 1500: self.frSteps = 1500;
        
        self.random()
        #print "done."
    
    def setExampleSubset(self, subset):
        if self.graph == None:
            return
        
        hiddenNodes = []
        
        if subset != None:
            try:
                expected = 1
                for row in subset:
                    index = int(row['index'].value)
                    if expected != index:
                        hiddenNodes += range(expected-1, index-1)
                        expected = index + 1
                    else:
                        expected += 1
                        
                hiddenNodes += range(expected-1, self.graph.nVertices)
                
                self.graph.setHiddenNodes(hiddenNodes)
            except:
                print "Error. Index column does not exists."
        
        #print "hiddenNodes:"
        #print hiddenNodes
        
    def hideSelected(self):
        #print self.graph.selection
        toHide = self.graph.selection + self.graph.hiddenNodes
        self.graph.setHiddenNodes(toHide)
        self.graph.removeSelection()
        
    def hideAllButSelected(self):
        allNodes = set(range(self.graph.nVertices))
        allButSelected = list(allNodes - set(self.graph.selection))
        toHide = allButSelected + self.graph.hiddenNodes
        self.graph.setHiddenNodes(toHide)
    
    def showAllNodes(self):
        self.graph.setHiddenNodes([])
        
    def random(self):
        #print "OWNetwork/random.."
        if self.visualize == None:   #grafa se ni
            return    
            
        self.visualize.random()
        
        #print "OWNetwork/random: updating canvas..."
        self.updateCanvas();
        #print "done."
        
        
    def fr(self):
        #print "OWNetwork/ff..."
        if self.visualize == None:   #grafa se ni
            return
        
#        k = 1.13850193174e-008
#        #k = 1.61735442033e-008
#        nodes = self.visualize.nVertices()
#        t = k * nodes * nodes
#        refreshRate = int(5.0 / t)
#        if refreshRate <   1: refreshRate = 1;
#        if refreshRate > 1500: refreshRate = 1500;
        print "refreshRate: " + str(self.frSteps)        
        tolerance = 5
        initTemp = 1000
        initTemp = self.visualize.fruchtermanReingold(self.frSteps, initTemp, self.graph.hiddenNodes)
        self.updateCanvas()
        #print "done."
        
    def frRadial(self):
        #print "F-R Radial"
        
        k = 1.13850193174e-008
        nodes = self.visualize.nVertices()
        t = k * nodes * nodes
        refreshRate = int(5.0 / t)
        if refreshRate <   1: refreshRate = 1;
        if refreshRate > 1500: refreshRate = 1500;
        print "refreshRate: " + str(refreshRate)
        
        tolerance = 5
        initTemp = 100
        centerNdx = 0
        if len(self.graph.selection) > 0:
            centerNdx = self.graph.selection[0]
            
        #print "center ndx: " + str(centerNdx)
        initTemp = self.visualize.radialFruchtermanReingold(centerNdx, refreshRate, initTemp)
        self.graph.circles = [10000 / 12, 10000/12*2, 10000/12*3]#, 10000/12*4, 10000/12*5]
        #self.graph.circles = [100, 200, 300]
        self.updateCanvas()
        self.graph.circles = []
        
    def circularOriginal(self):
        #print "Circular Original"
        self.visualize.circularOriginal()
        self.updateCanvas()
        
    def circularRandom(self):
        #print "Circular Random"
        self.visualize.circularRandom()
        self.updateCanvas()


    def circularCrossingReduction(self):
        #print "Circular Crossing Reduction"
        self.visualize.circularCrossingReduction()
        self.updateCanvas()
        
    def setVertexColor(self):
        self.graph.setVertexColor(self.colorCombo.currentText())
        self.updateCanvas()
        
    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)
                    
    def updateCanvas(self):
        #ce imamo graf
        if self.visualize != None:
            self.graph.updateCanvas()#self.visualize.xCoors, self.visualize.yCoors)
        
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.graph.controlPressed = True
            #print "cp"
        elif e.key() == Qt.Key_Alt:
            self.graph.altPressed = True
        QWidget.keyPressEvent(self, e)
        
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.graph.controlPressed = False
        elif e.key() == Qt.Key_Alt:
            self.graph.altPressed = False
        QWidget.keyReleaseEvent(self, e)
        

#    def keyPressEvent(self, e):
#        if e.text() == "f":
#            self.graph.freezeNeighbours = not self.graph.freezeNeighbours
#        else:
#            OWWidget.keyPressEvent(self, e)


    def markedToSelection(self):
        self.graph.addSelection(self.graph.markedNodes)
    
    def markedFromSelection(self):
        self.graph.removeSelection(self.graph.markedNodes)
    
    def setSelectionToMarked(self):
        self.graph.removeSelection(None, False)
        self.graph.addSelection(self.graph.markedNodes)
        
    def showDegreeDistribution(self):
        from matplotlib import rcParams
        rcParams['text.fontname'] = 'cmr10'
        import pylab as p
        
        x = self.visualize.graph.getDegrees()
        #print len(x)
        #print x
        # the histogram of the data
        n, bins, patches = p.hist(x, 500)
        
        p.xlabel('No. of nodes')
        p.ylabel('Degree')
        p.title(r'Degree distribution')
        
        p.show()


if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetwork()
    appl.setMainWidget(ow)
    ow.show()
    appl.exec_loop()

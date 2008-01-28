"""
<name>Network</name>
<description>Network Widget visualizes graphs.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2040</priority>
"""
from OWWidget import *

import OWGUI
from qwt import *
from qt import *
from OWNetworkCanvas import *
from orngNetwork import * 
from time import *
import OWToolbars
from statc import mean

dir = os.path.dirname(__file__) + "/../icons/"
dlg_mark2sel = dir + "Dlg_Mark2Sel.png"
dlg_sel2mark = dir + "Dlg_Sel2Mark.png"
dlg_selIsmark = dir + "Dlg_SelisMark.png"
dlg_selected = dir + "Dlg_SelectedNodes.png"
dlg_unselected = dir + "Dlg_UnselectedNodes.png"

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
        self.nVertices = self.nMarked = self.nSelected = self.nHidden = self.nShown = self.nEdges = self.verticesPerEdge = self.edgesPerVertex = self.diameter = 0
        self.optimizeWhat = 1
        self.stopOptimization = 0
        
        self.loadSettings()

        self.visualize = None

        self.graph = OWNetworkCanvas(self, self.mainArea, "Network")
        #self.optimize = OWGraphDrawingOptimize(parent=self);
        #start of content (right) area
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)

        self.tabs = QTabWidget(self.controlArea)
#        self.mainTab = QVGroupBox(self)
        self.displayTab = QVGroupBox(self)
        self.mainTab = self.displayTab
        self.markTab = QVGroupBox(self)
        self.infoTab = QVGroupBox(self)
        self.protoTab = QVGroupBox(self)
#        self.tabs.insertTab(self.mainTab, "Main")
        self.tabs.insertTab(self.displayTab, "Display")
        self.tabs.insertTab(self.markTab, "Mark")
        self.tabs.insertTab(self.infoTab, "Info")
        self.tabs.insertTab(self.protoTab, "Prototypes")
        OWGUI.separator(self.controlArea)


        self.optimizeBox = OWGUI.radioButtonsInBox(self.mainTab, self, "optimizeWhat", [], "Optimize", addSpace=False)
        OWGUI.button(self.optimizeBox, self, "Random", callback=self.random)
        self.frButton = OWGUI.button(self.optimizeBox, self, "Fruchterman Reingold", callback=self.fr, toggleButton=1)
        OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 10000, 1, label="Iterations: ")
        #OWGUI.button(self.optimizeBox, self, "F-R Special", callback=self.frSpecial)
        OWGUI.button(self.optimizeBox, self, "F-R Radial", callback=self.frRadial)
        OWGUI.button(self.optimizeBox, self, "Circular Original", callback=self.circularOriginal)
        OWGUI.button(self.optimizeBox, self, "Circular Random", callback=self.circularRandom)
        OWGUI.button(self.optimizeBox, self, "Circular Crossing Reduction", callback=self.circularCrossingReduction)
        #OWGUI.separator(self.optimizeBox)
        #OWGUI.widgetLabel("Optimize")
        #ib = OWGUI.indentedBox(self.optimizeBox)
        #for wh in ("All points", "Shown points", "Selected points"):
        #    OWGUI.appendRadioButton(self.optimizeBox, self, "optimizeWhat", wh, insertInto=ib)
        
        self.colorCombo = OWGUI.comboBox(self.displayTab, self, "color", box = "Color attribute", callback=self.setVertexColor)
        self.colorCombo.insertItem("(none)")
        
        self.attBox = OWGUI.widgetBox(self.mainTab, "Labels", addSpace = False)
        self.attListBox = OWGUI.listBox(self.attBox, self, "markerAttributes", "attributes", selectionMode=QListBox.Multi, callback=self.clickedAttLstBox)
        
        self.tooltipBox = OWGUI.widgetBox(self.mainTab, "Tooltips", addSpace = False)  
        self.tooltipListBox = OWGUI.listBox(self.tooltipBox, self, "tooltipAttributes", "attributes", selectionMode=QListBox.Multi, callback=self.clickedTooltipLstBox)
        
        self.labelsOnMarkedOnly = 0
        OWGUI.checkBox(self.mainTab, self, 'labelsOnMarkedOnly', 'Show labels on marked nodes only', callback = self.labelsOnMarked)
        
        OWGUI.separator(self.mainTab)

        OWGUI.button(self.mainTab, self, "Show degree distribution", callback=self.showDegreeDistribution)
        OWGUI.button(self.mainTab, self, "Save network", callback=self.saveNetwork)
        
        ib = OWGUI.widgetBox(self.markTab, "Info", addSpace = True)
        OWGUI.label(ib, self, "Vertices (shown/hidden): %(nVertices)i (%(nShown)i/%(nHidden)i)")
        OWGUI.label(ib, self, "Selected and marked vertices: %(nSelected)i - %(nMarked)i")
        
        ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Method", callback = self.setHubs, addSpace = True)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark vertices given in the input signal")
        #OWGUI.separator(ribg) 
        OWGUI.appendRadioButton(ribg, self, "hubs", "Find vertices which label contain")
        self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.setHubs)
        
        #OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of focused vertex")
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of selected vertices")
        ib = OWGUI.indentedBox(ribg, orientation = 0)
        self.ctrlMarkDistance = OWGUI.spin(ib, self, "markDistance", 0, 100, 1, label="Distance ", callback=(lambda h=2: self.setHubs(h)))
        #OWGUI.separator(ib, 4, 4)
        self.ctrlMarkFreeze = OWGUI.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
        #OWGUI.separator(ribg)
        OWGUI.widgetLabel(ribg, "Mark  vertices with ...")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at least N connections")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at most N connections")
        self.ctrlMarkNConnections = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N ", callback=(lambda h=4: self.setHubs(h)))
        #OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than any neighbour")
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than avg neighbour")
        #OWGUI.separator(ribg)
        OWGUI.appendRadioButton(ribg, self, "hubs", "most connections")
        ib = OWGUI.indentedBox(ribg)
        self.ctrlMarkNumber = OWGUI.spin(ib, self, "markNumber", 0, 1000000, 1, label="Number of vertices" + ": ", callback=(lambda h=8: self.setHubs(h)))
        OWGUI.widgetLabel(ib, "(More vertices are marked in case of ties)")
#        self.ctrlMarkProportion = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markProportion", 0, 100, 1, label="Percentage" + ": ", callback=self.setHubs)
        
        #ib = OWGUI.widgetBox(self.markTab, "Selection", addSpace = True)
        ib = QHGroupBox("Selection", self.markTab)
        btnM2S = OWGUI.button(ib, self, "", callback = self.markedToSelection)
        btnM2S.setPixmap(QPixmap(dlg_mark2sel))
        QToolTip.add(btnM2S, "Add Marked to Selection")
        btnS2M = OWGUI.button(ib, self, "",callback = self.markedFromSelection)
        btnS2M.setPixmap(QPixmap(dlg_sel2mark))
        QToolTip.add(btnS2M, "Remove Marked from Selection")
        btnSIM = OWGUI.button(ib, self, "", callback = self.setSelectionToMarked)
        btnSIM.setPixmap(QPixmap(dlg_selIsmark))
        QToolTip.add(btnSIM, "Set Selection to Marked")
        
        self.hideBox = QHGroupBox("Hide vertices", self.markTab)
        btnSEL = OWGUI.button(self.hideBox, self, "", callback=self.hideSelected)
        btnSEL.setPixmap(QPixmap(dlg_selected))
        QToolTip.add(btnSEL, "Selected")
        btnUN = OWGUI.button(self.hideBox, self, "", callback=self.hideAllButSelected)
        btnUN.setPixmap(QPixmap(dlg_unselected))
        QToolTip.add(btnUN, "Unselected")
        OWGUI.button(self.hideBox, self, "Show", callback=self.showAllNodes)
                
        T = OWToolbars.NavigateSelectToolbar
        self.zoomSelectToolbar = OWToolbars.NavigateSelectToolbar(self, self.controlArea, self.graph, self.autoSendSelection,
                                                              buttons = (T.IconZoom, T.IconZoomExtent, T.IconZoomSelection, ("", "", "", None, None, 0, "navigate"), T.IconPan, 
                                                                         ("Move selection", "buttonMoveSelection", "activateMoveSelection", QPixmap(OWToolbars.dlg_select), Qt.arrowCursor, 1, "select"),
                                                                         T.IconRectangle, T.IconPolygon, ("", "", "", None, None, 0, "select"), T.IconSendSelection))
        
        #OWGUI.button(self.controlArea, self, "test replot", callback=self.testRefresh)
        
        ib = OWGUI.widgetBox(self.infoTab, "General", addSpace = True)
        OWGUI.label(ib, self, "Number of vertices: %(nVertices)i")
        OWGUI.label(ib, self, "Number of edges: %(nEdges)i")
        OWGUI.label(ib, self, "Vertices per edge: %(verticesPerEdge).2f")
        OWGUI.label(ib, self, "Edges per vertex: %(edgesPerVertex).2f")
        OWGUI.label(ib, self, "Diameter: %(diameter)i")
        
        self.insideView = 0
        self.insideViewNeighbours = 2
        OWGUI.spin(self.protoTab, self, "insideViewNeighbours", 1, 6, 1, label="Inside view (neighbours): ", checked = "insideView", checkCallback = self.insideview, callback = self.insideviewneighbours)
        #OWGUI.button(self.protoTab, self, "Clustering", callback=self.clustering)
        OWGUI.button(self.protoTab, self, "Collapse", callback=self.collapse)
        
        self.icons = self.createAttributeIconDict()
        self.setHubs()
        
        self.resize(850, 700)    

    def collapse(self):
        print "collapse"
        self.visualize.collapse()
        self.graph.addVisualizer(self.visualize)
        #if not nodes is None:
        #    self.graph.updateData()
        #    self.graph.addSelection(nodes, False)
        self.updateCanvas()
        
    def clustering(self):
        print "clustering"
        self.visualize.graph.getClusters()
        
    def insideviewneighbours(self):
        if self.graph.insideview == 1:
            self.graph.insideviewNeighbours = self.insideViewNeighbours
            self.frButton.setOn(True)
            self.fr()
        
    def insideview(self):
        if len(self.graph.selection) == 1:
            if self.graph.insideview == 1:
                self.graph.insideview = 0
                self.graph.hiddenNodes = []
                self.updateCanvas()
            else:
                self.graph.insideview = 1
                self.graph.insideviewNeighbors = self.insideViewNeighbours
                self.frButton.setOn(True)
                self.fr()
    
        else:
            print "One node must be selected!"
            
        
    def labelsOnMarked(self):
        self.graph.labelsOnMarkedOnly = self.labelsOnMarkedOnly
        self.graph.updateData()
        self.graph.replot()
    
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
        self.graph.updateData()
        self.graph.replot()
            
        
    def clickedTooltipLstBox(self):
        self.graph.setTooltipText([self.attributes[i][0] for i in self.tooltipAttributes])
        self.graph.updateData()
        self.graph.replot()


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
        self.visualize = NetworkOptimization(graph, self)
        self.nVertices = len(graph)
        self.nShown = len(graph)
        self.nEdges = len(graph.getEdges())
        if self.nEdges > 0:
            self.verticesPerEdge = float(self.nVertices) / float(self.nEdges)
        else:
            self.verticesPerEdge = 0
            
        if self.nVertices > 0:
            self.edgesPerVertex = float(self.nEdges) / float(self.nVertices)
        else:
            self.edgesPerVertex = 0
        self.diameter = graph.getDiameter()
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
        self.nHidden = len(toHide)
        self.nShown = self.nVertices - self.nHidden 
        self.graph.setHiddenNodes(toHide)
        self.graph.removeSelection()
        
        
    def hideAllButSelected(self):
        allNodes = set(range(self.graph.nVertices))
        allButSelected = list(allNodes - set(self.graph.selection))
        toHide = allButSelected + self.graph.hiddenNodes
        self.nHidden = len(toHide)
        self.nShown = self.nVertices - self.nHidden 
        self.graph.setHiddenNodes(toHide)
    
    
    def showAllNodes(self):
        self.graph.setHiddenNodes([])
        self.nHidden = 0
        self.nShown = self.nVertices
        
        
    def random(self):
        #print "OWNetwork/random.."
        if self.visualize == None:   #grafa se ni
            return    
            
        self.visualize.random()
        
        #print "OWNetwork/random: updating canvas..."
        self.updateCanvas();
        #print "done."
        
    def fr(self):
        from qt import qApp
        if self.visualize == None:   #grafa se ni
            return
              
        if not self.frButton.isOn():
            self.stopOptimization = 1
            self.frButton.setOn(0)
            self.frButton.setText("Fruchterman Reingold")
            return
        
        self.frButton.setText("Stop")
        qApp.processEvents()
        self.stopOptimization = 0
        tolerance = 5
        initTemp = 1000
        breakpoints = 6
        k = int(self.frSteps / breakpoints)
        o = self.frSteps % breakpoints
        iteration = 0
        coolFactor = exp(log(10.0/10000.0) / self.frSteps)

        if k > 0:
            while iteration < breakpoints:
                #print "iteration, initTemp: " + str(initTemp)
                if self.stopOptimization:
                    return
                initTemp = self.visualize.fruchtermanReingold(k, initTemp, coolFactor, self.graph.hiddenNodes)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
            
            #print "ostanek: " + str(o) + ", initTemp: " + str(initTemp)
            if self.stopOptimization:
                    return
            initTemp = self.visualize.fruchtermanReingold(o, initTemp, coolFactor, self.graph.hiddenNodes)
            qApp.processEvents()
            self.updateCanvas()
        else:
            while iteration < o:
                #print "iteration ostanek, initTemp: " + str(initTemp)
                if self.stopOptimization:
                    return
                initTemp = self.visualize.fruchtermanReingold(1, initTemp, coolFactor, self.graph.hiddenNodes)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
                
        self.frButton.setOn(0)
        self.frButton.setText("Fruchterman Reingold")
        
    def frSpecial(self):
        steps = 100
        initTemp = 1000
        coolFactor = exp(log(10.0/10000.0) / steps)
        oldXY = []
        for rec in self.visualize.coors:
            oldXY.append([rec[0], rec[1]])
        #print oldXY
        initTemp = self.visualize.fruchtermanReingold(steps, initTemp, coolFactor, self.graph.hiddenNodes)
        #print oldXY
        self.graph.updateDataSpecial(oldXY)
        self.graph.replot()
        
        
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
        self.graph.updateData()
        self.graph.replot()
        
        
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

"""
<name>Net Explorer</name>
<description>Orange widget for network exploration.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3200</priority>
"""
import orange
import OWGUI
import OWColorPalette
import orngNetwork
import OWToolbars
import math
import operator
import orngMDS

from OWWidget import *
from OWNetworkCanvas import *
from time import *
from statc import mean
from operator import itemgetter

dir = os.path.dirname(__file__) + "/../icons/"
dlg_mark2sel = dir + "Dlg_Mark2Sel.png"
dlg_sel2mark = dir + "Dlg_Sel2Mark.png"
dlg_selIsmark = dir + "Dlg_SelisMark.png"
dlg_selected = dir + "Dlg_SelectedNodes.png"
dlg_unselected = dir + "Dlg_UnselectedNodes.png"
dlg_showall = dir + "Dlg_clear.png"

class OWNetExplorer(OWWidget):
    settingsList = ["autoSendSelection", "spinExplicit", "spinPercentage",
    "maxLinkSize", "maxVertexSize", "renderAntialiased", "labelsOnMarkedOnly",
    "invertSize", "optMethod", "lastVertexSizeColumn", "lastColorColumn",
    "lastNameComponentAttribute", "lastLabelColumns", "lastTooltipColumns",
    "showWeights", "showIndexes",  "showEdgeLabels", "colorSettings", 
    "selectedSchemaIndex", "edgeColorSettings", "selectedEdgeSchemaIndex",
    "showMissingValues", "fontSize", "mdsTorgerson", "mdsAvgLinkage",
    "mdsSteps", "mdsRefresh", "mdsStressDelta", "organism","showTextMiningInfo", 
    "toolbarSelection"] 
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Net Explorer')
        #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="markerAttributes"), ContextField("attributes", selected="tooltipAttributes"), "color"])}
        self.inputs = [("Network", orngNetwork.Network, self.setGraph, Default), 
                       ("Items", orange.ExampleTable, self.setItems),
                       ("Items to Mark", orange.ExampleTable, self.markItems), 
                       ("Items Subset", orange.ExampleTable, self.setExampleSubset), 
                       ("Vertex Distance", orange.SymMatrix, self.setVertexDistance)]
        
        self.outputs = [("Selected Network", orngNetwork.Network),
                        ("Selected Distance Matrix", orange.SymMatrix),
                        ("Selected Examples", ExampleTable), 
                        ("Unselected Examples", ExampleTable), 
                        ("Marked Examples", ExampleTable),
                        ("Attribute Selection List", AttributeList)]
        
        self.markerAttributes = []
        self.tooltipAttributes = []
        self.edgeLabelAttributes = []
        self.attributes = []
        self.edgeAttributes = []
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
        self.edgeColor = 0
        self.vertexSize = 0
        self.nVertices = self.nShown = self.nHidden = self.nMarked = self.nSelected = self.nEdges = self.verticesPerEdge = self.edgesPerVertex = self.diameter = self.clustering_coefficient = 0
        self.optimizeWhat = 1
        self.stopOptimization = 0
        self.maxLinkSize = 3
        self.maxVertexSize = 5
        self.renderAntialiased = 1
        self.labelsOnMarkedOnly = 0
        self.invertSize = 0
        self.optMethod = 0
        self.lastVertexSizeColumn = ''
        self.lastColorColumn = ''
        self.lastNameComponentAttribute = ''
        self.lastLabelColumns = set()
        self.lastTooltipColumns = set()
        self.showWeights = 0
        self.showIndexes = 0
        self.showEdgeLabels = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.edgeColorSettings = [('net_edges', [[], [('contPalette', (4294967295L, 4278190080L, 0))], [('discPalette', [(204, 204, 204), (179, 226, 205), (253, 205, 172), (203, 213, 232), (244, 202, 228), (230, 245, 201), (255, 242, 174), (241, 226, 204)])]]), ('Default', [[], [('contPalette', (4294967295L, 4278190080L, 0))], [('discPalette', [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255), (255, 223, 128), (127, 111, 64), (92, 46, 0), (0, 84, 0), (192, 192, 0), (0, 127, 127), (128, 0, 0), (127, 0, 127)])]])]
        self.selectedEdgeSchemaIndex = 0
        self.vertexDistance = None
        self.showDistances = 0
        self.showMissingValues = 0
        self.fontSize = 12
        self.mdsTorgerson = 0
        self.mdsAvgLinkage = 0
        self.mdsSteps = 120
        self.mdsRefresh = 30
        self.mdsStressDelta = 0.00001
        self.organism = 'goa_human'
        self.showTextMiningInfo = 0
        self.toolbarSelection = 0
        
        self.loadSettings()
        
        self.visualize = None
        self.markInputItems = None
        
        self.mainArea.layout().setContentsMargins(0,4,4,4)
        self.controlArea.layout().setContentsMargins(4,4,0,4)
        
        self.graph = OWNetworkCanvas(self, self.mainArea, "Net Explorer")
        self.graph.showMissingValues = self.showMissingValues
        self.mainArea.layout().addWidget(self.graph)
        
        self.graph.maxLinkSize = self.maxLinkSize
        self.graph.maxVertexSize = self.maxVertexSize
        
        self.hcontroArea = OWGUI.widgetBox(self.controlArea, orientation='horizontal')
        
        self.tabs = OWGUI.tabWidget(self.hcontroArea)
        self.verticesTab = OWGUI.createTabPage(self.tabs, "Vertices")
        self.edgesTab = OWGUI.createTabPage(self.tabs, "Edges")
        self.markTab = OWGUI.createTabPage(self.tabs, "Mark")
        self.infoTab = OWGUI.createTabPage(self.tabs, "Info")
        self.editTab = OWGUI.createTabPage(self.tabs, "Edit")
        

        self.optimizeBox = OWGUI.radioButtonsInBox(self.verticesTab, self, "optimizeWhat", [], "Optimize", addSpace=False)
        
        self.optCombo = OWGUI.comboBox(self.optimizeBox, self, "optMethod", label='Method:     ', orientation='horizontal', callback=self.setOptMethod)
        self.optCombo.addItem("No optimization")
        self.optCombo.addItem("Random")
        self.optCombo.addItem("Fruchterman Reingold")
        self.optCombo.addItem("Fruchterman Reingold Weighted")
        self.optCombo.addItem("Fruchterman Reingold Radial")
        self.optCombo.addItem("Circular Crossing Reduction")
        self.optCombo.addItem("Circular Original")
        self.optCombo.addItem("Circular Random")
        self.optCombo.addItem("Pivot MDS")
        self.optCombo.setCurrentIndex(self.optMethod)
        self.stepsSpin = OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 20000, 1, label="Iterations: ")
        self.stepsSpin.setEnabled(False)
        
        self.optButton = OWGUI.button(self.optimizeBox, self, "Optimize layout", callback=self.optLayout, toggleButton=1)
        
        colorBox = OWGUI.widgetBox(self.verticesTab, "Vertex color attribute", orientation="horizontal", addSpace = False)
        self.colorCombo = OWGUI.comboBox(colorBox, self, "color", callback=self.setVertexColor)
        self.colorCombo.addItem("(same color)")
        OWGUI.button(colorBox, self, "Set vertex color palette", self.setColors, tooltip = "Set vertex color palette", debuggingEnabled = 0)
        
        self.vertexSizeCombo = OWGUI.comboBox(self.verticesTab, self, "vertexSize", box = "Vertex size attribute", callback=self.setVertexSize)
        self.vertexSizeCombo.addItem("(none)")
        
        OWGUI.spin(self.vertexSizeCombo.box, self, "maxVertexSize", 5, 200, 1, label="Max vertex size:", callback = self.setVertexSize)
        OWGUI.checkBox(self.vertexSizeCombo.box, self, "invertSize", "Invert vertex size", callback = self.setVertexSize)
        
        colorBox = OWGUI.widgetBox(self.edgesTab, "Edge color attribute", orientation="horizontal", addSpace = False)
        self.edgeColorCombo = OWGUI.comboBox(colorBox, self, "edgeColor", callback=self.setEdgeColor)
        self.edgeColorCombo.addItem("(same color)")
        OWGUI.button(colorBox, self, "Set edge color palette", self.setEdgeColorPalette, tooltip = "Set edge color palette", debuggingEnabled = 0)
        
        self.attBox = OWGUI.widgetBox(self.verticesTab, "Vertex labels | tooltips", orientation="vertical", addSpace = False)
        OWGUI.spin(self.attBox, self, "fontSize", 4, 30, 1, label="Set font size:", callback = self.setFontSize)
        
        self.attBox = OWGUI.widgetBox(self.attBox, orientation="horizontal", addSpace = False)
        self.attListBox = OWGUI.listBox(self.attBox, self, "markerAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedAttLstBox)
        self.tooltipListBox = OWGUI.listBox(self.attBox, self, "tooltipAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedTooltipLstBox)
        
        self.edgeLabelBox = OWGUI.widgetBox(self.edgesTab, "Edge labels", addSpace = False)
        self.edgeLabelListBox = OWGUI.listBox(self.edgeLabelBox, self, "edgeLabelAttributes", "edgeAttributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedEdgeLabelListBox)
        self.edgeLabelBox.setEnabled(False)
        
        ib = OWGUI.widgetBox(self.edgesTab, "General", orientation="vertical")
        OWGUI.checkBox(ib, self, 'showWeights', 'Show weights', callback = self.showWeightLabels)
        OWGUI.checkBox(ib, self, 'showEdgeLabels', 'Show labels on edges', callback = self.showEdgeLabelsClick)
        OWGUI.spin(ib, self, "maxLinkSize", 1, 50, 1, label="Max edge width:", callback = self.setMaxLinkSize)
        OWGUI.checkBox(ib, self, 'showDistances', 'Explore vertex distances', callback = self.showDistancesClick)
        
        ib = OWGUI.widgetBox(self.verticesTab, "General", orientation="vertical")
        OWGUI.checkBox(ib, self, 'showIndexes', 'Show indexes', callback = self.showIndexLabels)
        OWGUI.checkBox(ib, self, 'labelsOnMarkedOnly', 'Show labels on marked vertices only', callback = self.labelsOnMarked)
        OWGUI.checkBox(ib, self, 'renderAntialiased', 'Render antialiased', callback = self.setRenderAntialiased)
        self.insideView = 0
        self.insideViewNeighbours = 2
        OWGUI.spin(ib, self, "insideViewNeighbours", 1, 6, 1, label="Inside view (neighbours): ", checked = "insideView", checkCallback = self.insideview, callback = self.insideviewneighbours)
        OWGUI.checkBox(ib, self, 'showMissingValues', 'Show missing values', callback = self.setShowMissingValues)
        
        ib = OWGUI.widgetBox(self.markTab, "Info", orientation="vertical")
        OWGUI.label(ib, self, "Vertices (shown/hidden): %(nVertices)i (%(nShown)i/%(nHidden)i)")
        OWGUI.label(ib, self, "Selected and marked vertices: %(nSelected)i - %(nMarked)i")
        
        ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Method", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "None", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Find vertices", callback = self.setMarkMode)
        self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.setMarkMode)
        
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of focused vertices", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of selected vertices", callback = self.setMarkMode)
        ib = OWGUI.indentedBox(ribg, orientation = 0)
        self.ctrlMarkDistance = OWGUI.spin(ib, self, "markDistance", 0, 100, 1, label="Distance ", callback=(lambda h=2: self.setMarkMode(h)))
        #self.ctrlMarkFreeze = OWGUI.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
        OWGUI.widgetLabel(ribg, "Mark  vertices with ...")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at least N connections", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "at most N connections", callback = self.setMarkMode)
        self.ctrlMarkNConnections = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N ", callback=(lambda h=4: self.setMarkMode(h)))
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than any neighbour", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than avg neighbour", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "most connections", callback = self.setMarkMode)
        ib = OWGUI.indentedBox(ribg)
        self.ctrlMarkNumber = OWGUI.spin(ib, self, "markNumber", 0, 1000000, 1, label="Number of vertices" + ": ", callback=(lambda h=8: self.setMarkMode(h)))
        OWGUI.widgetLabel(ib, "(More vertices are marked in case of ties)")
        self.markInputRadioButton = OWGUI.appendRadioButton(ribg, self, "hubs", "Mark vertices given in the input signal", callback = self.setMarkMode)
        ib = OWGUI.indentedBox(ribg)
        self.markInput = 0
        self.markInputCombo = OWGUI.comboBox(ib, self, "markInput", callback=(lambda h=9: self.setMarkMode(h)))
        self.markInputRadioButton.setEnabled(False)
        
        ib = OWGUI.widgetBox(self.markTab, "General", orientation="vertical")
        self.checkSendMarkedNodes = 0
        OWGUI.checkBox(ib, self, 'checkSendMarkedNodes', 'Send marked vertices', callback = self.setSendMarkedNodes, disabled=0)
        
        
        T = OWToolbars.NavigateSelectToolbar
        self.zoomSelectToolbar = T(self, self.hcontroArea, self.graph, self.autoSendSelection,
                                  buttons = (T.IconZoom, 
                                             T.IconZoomExtent, 
                                             T.IconZoomSelection, 
                                             T.IconPan, 
                                             ("", "", "", None, None, 0),
                                             #("Move selection", "buttonMoveSelection", "activateMoveSelection", QIcon(OWToolbars.dlg_select), Qt.ArrowCursor, 1),
                                             T.IconRectangle, 
                                             #T.IconPolygon,  
                                             T.IconSendSelection,
                                             ("", "", "", None, None, 0, "select"),
                                             ("Add marked to selection", "buttonM2S", "markedToSelection", QIcon(dlg_mark2sel), Qt.ArrowCursor, 0),
                                             ("Add selection to marked", "buttonS2M", "selectionToMarked", QIcon(dlg_sel2mark), Qt.ArrowCursor, 0),
                                             ("Remove selection", "buttonRMS", "removeSelection", QIcon(dlg_selIsmark), Qt.ArrowCursor, 0),
                                             ("", "", "", None, None, 0, "select"),
                                             ("Hide selected", "buttonSEL", "hideSelectedVertices", QIcon(dlg_selected), Qt.ArrowCursor, 0),
                                             ("Hide unselected", "buttonUN", "hideUnSelectedVertices", QIcon(dlg_unselected), Qt.ArrowCursor, 0),
                                             ("Show all nodes", "buttonSW", "showAllVertices", QIcon(dlg_showall), Qt.ArrowCursor, 0)))
                        
        OWGUI.rubber(self.zoomSelectToolbar)
        
        ib = OWGUI.widgetBox(self.infoTab, "General")
        OWGUI.label(ib, self, "Number of vertices: %(nVertices)i")
        OWGUI.label(ib, self, "Number of edges: %(nEdges)i")
        OWGUI.label(ib, self, "Vertices per edge: %(verticesPerEdge).2f")
        OWGUI.label(ib, self, "Edges per vertex: %(edgesPerVertex).2f")
        OWGUI.label(ib, self, "Diameter: %(diameter)i")
        OWGUI.label(ib, self, "Clustering Coefficient: %(clustering_coefficient).1f%%")
        
        ib = OWGUI.widgetBox(self.infoTab, orientation="horizontal")
        
        OWGUI.button(ib, self, "Degree distribution", callback=self.showDegreeDistribution)
        OWGUI.button(ib, self, "Save network", callback=self.saveNetwork)
        OWGUI.button(ib, self, "Save image", callback=self.graph.saveToFile)
        
        #OWGUI.button(self.edgesTab, self, "Clustering", callback=self.clustering)
        
        ib = OWGUI.widgetBox(self.infoTab, "Prototype")
        #OWGUI.button(ib, self, "Collapse", callback=self.collapse)
        
        #ib = OWGUI.widgetBox(ibProto, "Name components")
        OWGUI.lineEdit(ib, self, "organism", "Organism:", orientation='horizontal')
        
        self.nameComponentAttribute = 0
        self.nameComponentCombo = OWGUI.comboBox(ib, self, "nameComponentAttribute", callback=self.nameComponents, label="Name components:", orientation="horizontal")
        self.nameComponentCombo.addItem("Select attribute")
        
        self.showComponentAttribute = 0
        self.showComponentCombo = OWGUI.comboBox(ib, self, "showComponentAttribute", callback=self.showComponents, label="Labels on components:", orientation="horizontal")
        self.showComponentCombo.addItem("Select attribute")
        OWGUI.checkBox(ib, self, 'showTextMiningInfo', "Show text mining info")
        
        #ib = OWGUI.widgetBox(ibProto, "Distance Matrix")
        ibs = OWGUI.widgetBox(ib, orientation="horizontal")
        self.btnMDS = OWGUI.button(ibs, self, "Fragviz", callback=self.mdsComponents, toggleButton=1)
        self.btnESIM = OWGUI.button(ibs, self, "eSim", callback=self.exactSimulation, toggleButton=1)
        self.btnMDSv = OWGUI.button(ibs, self, "MDS", callback=self.mdsVertices, toggleButton=1)
        ibs = OWGUI.widgetBox(ib, orientation="horizontal")
        self.btnRotate = OWGUI.button(ibs, self, "Rotate", callback=self.rotateComponents, toggleButton=1)
        self.btnRotateMDS = OWGUI.button(ibs, self, "Rotate (MDS)", callback=self.rotateComponentsMDS, toggleButton=1)
        self.btnForce = OWGUI.button(ibs, self, "Draw Force", callback=self.drawForce, toggleButton=1)
        self.scalingRatio = 0
        OWGUI.spin(ib, self, "scalingRatio", 0, 9, 1, label="Set scalingRatio: ")
        OWGUI.doubleSpin(ib, self, "mdsStressDelta", 0, 10, 0.0000000000000001, label="Min stress change: ")
        OWGUI.spin(ib, self, "mdsSteps", 1, 100000, 1, label="MDS steps: ")
        OWGUI.spin(ib, self, "mdsRefresh", 1, 100000, 1, label="MDS refresh steps: ")
        ibs = OWGUI.widgetBox(ib, orientation="horizontal")
        OWGUI.checkBox(ibs, self, 'mdsTorgerson', "Torgerson's approximation")
        OWGUI.checkBox(ibs, self, 'mdsAvgLinkage', "Use average linkage")
        self.mdsInfoA=OWGUI.widgetLabel(ib, "Avg. stress:")
        self.mdsInfoB=OWGUI.widgetLabel(ib, "Num. steps:")
        self.rotateSteps = 100
        
        OWGUI.spin(ib, self, "rotateSteps", 1, 10000, 1, label="Rotate max steps: ")
        
        self.attSelectionAttribute = 0
        self.comboAttSelection = OWGUI.comboBox(ib, self, "attSelectionAttribute", label='Send attribute selection list:', orientation='horizontal', callback=self.sendAttSelectionList)
        self.comboAttSelection.addItem("Select attribute")
        
        self.icons = self.createAttributeIconDict()
        self.setMarkMode()
        
        self.editAttribute = 0
        self.editCombo = OWGUI.comboBox(self.editTab, self, "editAttribute", label="Edit attribute:", orientation="horizontal")
        self.editCombo.addItem("Select attribute")
        self.editValue = ''
        OWGUI.lineEdit(self.editTab, self, "editValue", "Value:", orientation='horizontal')
        OWGUI.button(self.editTab, self, "Edit", callback=self.edit)
        
        self.verticesTab.layout().addStretch(1)
        self.edgesTab.layout().addStretch(1)
        self.markTab.layout().addStretch(1)
        self.infoTab.layout().addStretch(1)
        self.editTab.layout().addStretch(1)
        
        dlg = self.createColorDialog(self.colorSettings, self.selectedSchemaIndex)
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        
        dlg = self.createColorDialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
        self.graph.contEdgePalette = dlg.getContinuousPalette("contPalette")
        self.graph.discEdgePalette = dlg.getDiscretePalette("discPalette")
        
        self.setOptMethod()
        self.setFontSize()
        
        self.resize(1000, 600)
        self.setGraph(None)
        #self.controlArea.setEnabled(False)
        
    def sendAttSelectionList(self):
        vars = [x.name for x in self.visualize.getVars()]
        if not self.comboAttSelection.currentText() in vars:
            return
        att = str(self.comboAttSelection.currentText())
        vertices = self.graph.networkCurve.getSelectedVertices()
        
        if len(vertices) != 1:
            return
        
        attributes = str(self.visualize.graph.items[vertices[0]][att]).split(', ')
        self.send("Attribute Selection List", attributes)
        
    def edit(self):
        vars = [x.name for x in self.visualize.getVars()]
        if not self.editCombo.currentText() in vars:
            return
        att = str(self.editCombo.currentText())
        vertices = self.graph.networkCurve.getSelectedVertices()
        
        if len(vertices) == 0:
            return
        
        if self.visualize.graph.items.domain[att].varType == orange.VarTypes.Continuous:
            for v in vertices:
                self.visualize.graph.items[v][att] = float(self.editValue)
        else:
            for v in vertices:
                self.visualize.graph.items[v][att] = str(self.editValue)
        
        self.setItems(self.visualize.graph.items)
        
    def drawForce(self):
        if self.btnForce.isChecked():
            self.graph.forceVectors = self.visualize.computeForces() 
        else:
            self.graph.forceVectors = None
            
        self.updateCanvas()
        
    def rotateProgress(self, curr, max):
        self.progressBarSet(int(curr * 100 / max))
        qApp.processEvents()
        
    def rotateComponentsMDS(self):
        print "rotate"
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            self.btnRotateMDS.setChecked(False)
            return
        
        if self.visualize == None:
            self.information('No network found')
            self.btnRotateMDS.setChecked(False)
            return
        
        if self.vertexDistance.dim != self.visualize.graph.nVertices:
            self.error('Distance matrix dimensionality must equal number of vertices')
            self.btnRotateMDS.setChecked(False)
            return
        
        if not self.btnRotateMDS.isChecked():
          self.visualize.stopMDS = 1
          #self.btnMDS.setChecked(False)
          #self.btnMDS.setText("MDS on graph components")
          return
        
        self.btnRotateMDS.setText("Stop")
        qApp.processEvents()
        
        self.visualize.vertexDistance = self.vertexDistance
        self.progressBarInit()
        
        self.visualize.mdsComponents(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, rotationOnly=True)            
            
        self.btnRotateMDS.setChecked(False)
        self.btnRotateMDS.setText("Rotate graph components (MDS)")
        self.progressBarFinished()
    
    def rotateComponents(self):
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            self.btnRotate.setChecked(False)
            return
        
        if self.visualize == None:
            self.information('No network found')
            self.btnRotate.setChecked(False)
            return
        
        if self.vertexDistance.dim != self.visualize.graph.nVertices:
            self.error('Distance matrix dimensionality must equal number of vertices')
            self.btnRotate.setChecked(False)
            return
        
        if not self.btnRotate.isChecked():
          self.visualize.stopRotate = 1
          return
      
        self.btnRotate.setText("Stop")
        qApp.processEvents()
        
        self.visualize.vertexDistance = self.vertexDistance
        self.progressBarInit()
        self.visualize.rotateComponents(self.rotateSteps, 0.0001, self.rotateProgress, self.updateCanvas)
        self.btnRotate.setChecked(False)
        self.btnRotate.setText("Rotate graph components")
        self.progressBarFinished()
        
    def mdsProgress(self, avgStress, stepCount):    
        self.drawForce()

        self.mdsInfoA.setText("Avg. Stress: %.20f" % avgStress)
        self.mdsInfoB.setText("Num. steps: %i" % stepCount)
        self.progressBarSet(int(stepCount * 100 / self.mdsSteps))
        qApp.processEvents()
        
    def mdsComponents(self, mdsType=orngNetwork.MdsType.componentMDS):
        if mdsType == orngNetwork.MdsType.componentMDS:
            btn = self.btnMDS
        elif mdsType == orngNetwork.MdsType.exactSimulation:
            btn = self.btnESIM
        elif mdsType == orngNetwork.MdsType.MDS:
            btn = self.btnMDSv
        
        btnCaption = btn.text()
        
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            btn.setChecked(False)
            return
        
        if self.visualize == None:
            self.information('No network found')
            btn.setChecked(False)
            return
        
        if self.vertexDistance.dim != self.visualize.graph.nVertices:
            self.error('Distance matrix dimensionality must equal number of vertices')
            btn.setChecked(False)
            return
        
        if not btn.isChecked():
            self.visualize.stopMDS = 1
            btn.setChecked(False)
            btn.setText(btnCaption)
            return
        
        btn.setText("Stop")
        qApp.processEvents()
        
        self.visualize.vertexDistance = self.vertexDistance
        self.progressBarInit()
        
        if self.mdsAvgLinkage:
            self.visualize.mdsComponentsAvgLinkage(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, scalingRatio = self.scalingRatio)
        else:
            self.visualize.mdsComponents(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, mdsType=mdsType, scalingRatio=self.scalingRatio)            
        
        btn.setChecked(False)
        btn.setText(btnCaption)
        self.progressBarFinished()
        
    def exactSimulation(self):
        self.mdsComponents(orngNetwork.MdsType.exactSimulation)
        
    def mdsVertices(self):
        self.mdsComponents(orngNetwork.MdsType.MDS)
        
    def setVertexDistance(self, matrix):
        self.error('')
        self.information('')
        
        if matrix == None or self.visualize == None or self.visualize.graph == None:
            self.vertexDistance = None
            if self.visualize: self.visualize.vertexDistance = None
            return
        
        if matrix.dim != self.visualize.graph.nVertices:
            self.error('Distance matrix dimensionality must equal number of vertices')
            self.vertexDistance = None
            if self.visualize: self.visualize.vertexDistance = None
            return
        
        self.vertexDistance = matrix
        if self.visualize: self.visualize.vertexDistance = matrix
            
    def setSendMarkedNodes(self):
        if self.checkSendMarkedNodes:
            self.graph.sendMarkedNodes = self.sendMarkedNodes
            self.sendMarkedNodes(self.graph.getMarkedVertices())
        else:
            self.send("Marked Examples", None)
            self.graph.sendMarkedNodes = None
        
    def sendMarkedNodes(self, markedNodes):        
        if len(markedNodes) == 0:
            self.send("Marked Examples", None)
            return
        
        if self.visualize != None and self.visualize.graph != None and self.visualize.graph.items != None:
            items = self.visualize.graph.items.getitems(markedNodes)
            self.send("Marked Examples", items)
            return
        
        self.send("Marked Examples", None)

    def collapse(self):
        #print "collapse"
        self.visualize.collapse()
        self.graph.addVisualizer(self.visualize)
        #if not nodes is None:
        #    self.graph.updateData()
        #    self.graph.addSelection(nodes, False)
        self.updateCanvas()
        
    def clustering(self):
        #print "clustering"
        self.visualize.graph.getClusters()
        
    def insideviewneighbours(self):
        if self.graph.insideview == 1:
            self.graph.insideviewNeighbours = self.insideViewNeighbours
            self.optButton.setChecked(True)
            self.fr(False)
        
    def insideview(self):
        print self.graph.getSelectedVertices()
        if len(self.graph.getSelectedVertices()) == 1:
            if self.graph.insideview == 1:
                print "insideview: 1"
                self.graph.insideview = 0
                self.graph.showAllVertices()
                self.updateCanvas()
            else:
                print "insideview: 0"
                self.graph.insideview = 1
                self.graph.insideviewNeighbors = self.insideViewNeighbours
                self.optButton.setChecked(True)
                self.fr(False)
    
        else:
            print "One node must be selected!"
        
    def showComponents(self):
        if self.visualize == None or self.visualize.graph == None or self.visualize.graph.items == None:
            return
        
        vars = [x.name for x in self.visualize.getVars()]
        
        if not self.showComponentCombo.currentText() in vars:
            self.graph.showComponentAttribute = None
            self.lastNameComponentAttribute = ''
        else:
            self.graph.showComponentAttribute = self.showComponentCombo.currentText()     
            
        self.graph.drawPlotItems()
        
    def nameComponents(self):
        """Names connected components of genes according to GO terms."""
        self.progressBarFinished()
        self.lastNameComponentAttribute = None
        
        if self.visualize == None or self.visualize.graph == None or self.visualize.graph.items == None:
            return
        
        vars = [x.name for x in self.visualize.getVars()]
        if not self.nameComponentCombo.currentText() in vars:
            return
        
        self.progressBarInit()
        components = [c for c in self.visualize.graph.getConnectedComponents() if len(c) > 1]
        if 'component name' in self.visualize.graph.items.domain:
            keyword_table = self.visualize.graph.items
        else:
            keyword_table = orange.ExampleTable(orange.Domain(orange.StringVariable('component name')), [[''] for i in range(len(self.visualize.graph.items))]) 
        
        import obiGO 
        ontology = obiGO.Ontology.Load(progressCallback=self.progressBarSet) 
        annotations = obiGO.Annotations.Load(self.organism, ontology=ontology, progressCallback=self.progressBarSet)

        allGenes = set([e[str(self.nameComponentCombo.currentText())].value for e in self.visualize.graph.items])
        foundGenesets = False
        if len(annotations.geneNames & allGenes) < 1:
            allGenes = set(reduce(operator.add, [e[str(self.nameComponentCombo.currentText())].value.split(', ') for e in self.visualize.graph.items]))
            if len(annotations.geneNames & allGenes) < 1:            
                self.warning('no genes found')
                return
            else:
                foundGenesets = True
            
        def rank(a, j, reverse=False):                    
            if len(a) <= 0: return
            
            if reverse:                
                a.sort(lambda x, y: 1 if x[j] > y[j] else -1 if x[j] < y[j] else 0)
                top_value = a[0][j]
                top_rank = len(a)
                max_rank = float(len(a))
                int_ndx = 0
                for k in range(len(a)):
                    if top_value < a[k][j]:
                        top_value = a[k][j] 
                        if k - int_ndx > 1:
                            avg_rank = (a[int_ndx][j] + a[k-1][j]) / 2
                            for l in range(int_ndx, k):
                                a[l][j] = avg_rank
                        
                        int_ndx = k

                    a[k][j] = top_rank / max_rank
                    top_rank -= 1
                
                k += 1
                if k - int_ndx > 1:
                    avg_rank = (a[int_ndx][j] + a[k-1][j]) / 2
                    for l in range(int_ndx, k):
                        a[l][j] = avg_rank    
                
            else:
                a.sort(lambda x, y: 1 if x[j] < y[j] else -1 if x[j] > y[j] else 0)
                top_value = a[0][j]
                top_rank = len(a)
                max_rank = float(len(a))
                int_ndx = 0
                for k in range(len(a)):
                    if top_value > a[k][j]:
                        top_value = a[k][j] 
                        if k - int_ndx > 1:
                            avg_rank = (a[int_ndx][j] + a[k-1][j]) / 2
                            for l in range(int_ndx, k):
                                a[l][j] = avg_rank
                        
                        int_ndx = k

                    a[k][j] = top_rank / max_rank
                    top_rank -= 1
                
                k += 1
                if k - int_ndx > 1:
                    avg_rank = (a[int_ndx][j] + a[k-1][j]) / 2
                    for l in range(int_ndx, k):
                        a[l][j] = avg_rank
        
        for i in range(len(components)):
            component = components[i]
            if len(component) <= 1:
                continue
            
            if foundGenesets:
                genes = reduce(operator.add, [self.visualize.graph.items[v][str(self.nameComponentCombo.currentText())].value.split(', ') for v in component])
            else:
                genes = [self.visualize.graph.items[v][str(self.nameComponentCombo.currentText())].value for v in component]
                    
            res1 = annotations.GetEnrichedTerms(genes, aspect="P")
            res2 = annotations.GetEnrichedTerms(genes, aspect="F")
            res = res1.items() + res2.items()
            #namingScore = [[(1-p_value) * (float(len(g)) / len(genes)) / (float(ref) / len(annotations.geneNames)), ontology.terms[GOId].name, len(g), ref, p_value] for GOId, (g, p_value, ref) in res.items() if p_value < 0.1]
            #namingScore = [[(1-p_value) * len(g) / ref, ontology.terms[GOId].name, len(g), ref, p_value] for GOId, (g, p_value, ref) in res.items() if p_value < 0.1]
            
            namingScore = [[len(g), ref, p_value, ontology[GOId].name, len(g), ref, p_value] for GOId, (g, p_value, ref) in res if p_value < 0.1]
            if len(namingScore) == 0:
                continue
            
            annotated_genes = max([a[0] for a in namingScore])
            
            rank(namingScore, 1, reverse=True)
            rank(namingScore, 2, reverse=True)
            rank(namingScore, 0)
            
            namingScore = [[10 * rank_genes + 0.5 * rank_ref + rank_p_value, name, g, ref, p_value] for rank_genes, rank_ref, rank_p_value, name, g, ref, p_value in namingScore]
            namingScore.sort(reverse=True)
            
            if len(namingScore) < 1:
                print "warning. no annotations found for group of genes: " + ", ".join(genes)
                continue
            elif len(namingScore[0]) < 2:
                print "warning. error computing score for group of genes: " + ", ".join(genes)
                continue
            
            for v in component:
                name = str(namingScore[0][1])
                attrs = "%d/%d, %d, %lf" % (namingScore[0][2], annotated_genes, namingScore[0][3], namingScore[0][4])
                info = ''
                if self.showTextMiningInfo:
                    info = "\n" + attrs + "\n" + str(namingScore[0][0])
                keyword_table[v]['component name'] = name + info
            
            self.progressBarSet(i*100.0/len(components))
                
        self.lastNameComponentAttribute = self.nameComponentCombo.currentText()
        self.setItems(orange.ExampleTable([self.visualize.graph.items, keyword_table]))
        self.progressBarFinished()   
        
    def nameComponents_old(self):
        if self.visualize == None or self.visualize.graph == None or self.visualize.graph.items == None:
            return
        
        vars = [x.name for x in self.visualize.getVars()]
        
        if not self.nameComponentCombo.currentText() in vars:
            return
        
        components = self.visualize.graph.getConnectedComponents()
        keyword_table = orange.ExampleTable(orange.Domain(orange.StringVariable('component name')), [[''] for i in range(len(self.visualize.graph.items))]) 
        
        excludeWord = ["AND", "OF", "KEGG", "ST", "IN", "SIG"]
        excludePart = ["HSA"]
        keywords = set()
        sameKeywords = set()
        
        for component in components:
            words = []
            all_values = []
            for vertex in component:
                values = []
                value =  str(self.visualize.graph.items[vertex][str(self.nameComponentCombo.currentText())])
                
                value = value.replace(" ", ",")
                value_top = value.split(",")
                
                for value in value_top:
                    if len(value) > 0:
                        tmp = value.split("_")
                        tmp = [value.strip() for value in tmp if len(value) > 0]
                        all_values.append(tmp)
                        values.extend(tmp)
                                
                values = [value.strip() for value in values if len(value) > 0]
                words.extend(values)
                
                
                #value =  str(self.visualize.graph.items[vertex][str(self.nameComponentCombo.currentText())])
                #value = value.replace(" ", "_")
                #value = value.replace(",", "_")
                #values = value.split("_")
                #values = [value.strip() for value in values if len(value) > 0]
                #print "values:", values
                #all_values.append(values)
                
                #words.extend(values)
            #print "all_values:", all_values
            toExclude = []
            
            words = [word for word in words if not word.upper() in excludeWord]
            toExclude = [word for word in words for part in excludePart if word.find(part) != -1]
            
            for word in toExclude:
                try:
                    while True:
                        words.remove(word)
                except:
                    pass
            
            counted_words = {}
            for word in words:
                if word in counted_words:
                    count = counted_words[word]
                    counted_words[word] = count + 1
                else:
                    counted_words[word] = 1
            
            words = sorted(counted_words.items(), key=itemgetter(1), reverse=True)
            keyword = ""
            keyword_words = []
            max_count = 0
            i = 0
            
            while i < len(words) and words[i][1] >= max_count:
                max_count = words[i][1]
                keyword += words[i][0] + " "
                keyword_words.append(words[i][0])
                i += 1
            
            if len(keyword_words) > 1:
                new_all_values = []
                for values in all_values:
                    new_values = [value for value in values if value in keyword_words]
                    new_all_values.append(new_values) 
                     
                #print new_all_values
                word_position = []
                
                for word in keyword_words:
                    sum = 0
                    for v in new_all_values:
                        if word in v:
                            sum += v.index(word)
                            
                    word_position.append((word, sum))
                 
                words = sorted(word_position, key=itemgetter(1))
                #print "words:", words
                #print all_values
                #print new_all_values
                
                keyword = ""
                for word in words:
                    keyword += word[0] + " "
                    
            keyword = keyword.strip()
            
            for vertex in component:
                keyword_table[vertex]['component name'] = keyword
                
            if keyword in keywords:
                sameKeywords.add(keyword)
            else:
                keywords.add(keyword)
        #print "sameKeywords:", sameKeywords       
        sameComponents = [component for component in components if str(keyword_table[component[0]]['component name']) in sameKeywords]
        #print "same components:", sameComponents
        
        for component in sameComponents:
            words = []
            all_values = []
            for vertex in component:
                values = []
                value =  str(self.visualize.graph.items[vertex][str(self.nameComponentCombo.currentText())])
                
                value = value.replace(" ", ",")
                value_top = value.split(",")
                
                for value in value_top:
                    if len(value) > 0:
                        tmp = value.split("_")
                        tmp = [value.strip() for value in tmp if len(value) > 0]
                        all_values.append(tmp)
                        values.extend(tmp)
                                
                values = [value.strip() for value in values if len(value) > 0]
                words.extend(values)
            
            toExclude = []
            
            words = [word for word in words if not word.upper() in excludeWord]
            toExclude = [word for word in words for part in excludePart if word.find(part) != -1]
            
            for word in toExclude:
                try:
                    while True:
                        words.remove(word)
                except:
                    pass
            
            counted_words = {}
            for word in words:
                if word in counted_words:
                    count = counted_words[word]
                    counted_words[word] = count + 1
                else:
                    counted_words[word] = 1
            
            words = sorted(counted_words.items(), key=itemgetter(1), reverse=True)
            keyword = ""
            counts = [int(word[1]) for word in words] 
            max_count = max(counts)
            
            try:
                while True and len(counts) > 1:
                    counts.remove(max_count)
            except:
                pass
            max_count = max(counts)
            i = 0
            keyword_words = []
            while i < len(words) and words[i][1] >= max_count:
                keyword += words[i][0] + " "
                keyword_words.append(words[i][0])
                i += 1
                
            if len(keyword_words) > 1:
                new_all_values = []
                for values in all_values:
                    new_values = [value for value in values if value in keyword_words]
                    new_all_values.append(new_values) 
                     
                #print new_all_values
                word_position = []
                
                for word in keyword_words:
                    sum = 0
                    for v in new_all_values:
                        if word in v:
                            sum += v.index(word)
                            
                    word_position.append((word, sum))
                 
                words = sorted(word_position, key=itemgetter(1))
                #print "words:", words
                #print all_values
                #print new_all_values
                
                keyword = ""
                for word in words:
                    keyword += word[0] + " "
                 
            keyword = keyword.strip()
            for vertex in component:
                keyword_table[vertex]['component name'] = keyword
        
        self.lastNameComponentAttribute = self.nameComponentCombo.currentText()
        #print "self.lastNameComponentAttribute:", self.lastNameComponentAttribute
        items = orange.ExampleTable([self.visualize.graph.items, keyword_table])
        self.setItems(items)
        
        #for item in items:
        #    print item
                        
    def showIndexLabels(self):
        self.graph.showIndexes = self.showIndexes
        self.graph.updateData()
        self.graph.replot()
        
    def showWeightLabels(self):
        self.graph.showWeights = self.showWeights
        self.graph.updateData()
        self.graph.replot()
        
    def showDistancesClick(self):
        if self.visualize.vertexDistance == None:
            self.warning("Vertex distance signal is not set. Distances are not known.")
        self.graph.showDistances = self.showDistances
        
    def showEdgeLabelsClick(self):
        self.graph.showEdgeLabels = self.showEdgeLabels
        self.graph.updateData()
        self.graph.replot()
        
    def labelsOnMarked(self):
        self.graph.labelsOnMarkedOnly = self.labelsOnMarkedOnly
        self.graph.updateData()
        self.graph.replot()
    
    def setSearchStringTimer(self):
        self.hubs = 1
        self.searchStringTimer.stop()
        self.searchStringTimer.start(1000)
         
    def setMarkMode(self, i = None):
        self.searchStringTimer.stop()
        if not i is None:
            self.hubs = i
        
        #print self.hubs
        self.graph.tooltipNeighbours = self.hubs == 2 and self.markDistance or 0
        self.graph.markWithRed = False

        if not self.visualize or not self.visualize.graph:
            return
        
        hubs = self.hubs
        vgraph = self.visualize.graph

        if hubs == 0:
            self.graph.setMarkedVertices([])
            self.graph.replot()
            return
        
        elif hubs == 1:
            #print "mark on given label"
            txt = self.markSearchString
            labelText = self.graph.labelText
            self.graph.markWithRed = self.graph.nVertices > 200
            
            toMark = [i for i, values in enumerate(vgraph.items) if txt.lower() in " ".join([str(values[ndx]).decode("ascii", "ignore").lower() for ndx in range(len(vgraph.items.domain)) + vgraph.items.domain.getmetas().keys()])]
            self.graph.setMarkedVertices(toMark)
            self.graph.replot()
            return
        
        elif hubs == 2:
            #print "mark on focus"
            self.graph.unMark()
            self.graph.tooltipNeighbours = self.markDistance
            return

        elif hubs == 3:
            #print "mark selected"
            self.graph.unMark()
            self.graph.selectionNeighbours = self.markDistance
            self.graph.markSelectionNeighbours()
            return
        
        self.graph.tooltipNeighbours = self.graph.selectionNeighbours = 0
        powers = vgraph.getDegrees()
        
        if hubs == 4: # at least N connections
            #print "mark at least N connections"
            N = self.markNConnections
            self.graph.setMarkedVertices([i for i, power in enumerate(powers) if power >= N])
            self.graph.replot()
        elif hubs == 5:
            #print "mark at most N connections"
            N = self.markNConnections
            self.graph.setMarkedVertices([i for i, power in enumerate(powers) if power <= N])
            self.graph.replot()
        elif hubs == 6:
            #print "mark more than any"
            self.graph.setMarkedVertices([i for i, power in enumerate(powers) if power > max([0]+[powers[nn] for nn in vgraph.getNeighbours(i)])])
            self.graph.replot()
        elif hubs == 7:
            #print "mark more than avg"
            self.graph.setMarkedVertices([i for i, power in enumerate(powers) if power > mean([0]+[powers[nn] for nn in vgraph.getNeighbours(i)])])
            self.graph.replot()
        elif hubs == 8:
            #print "mark most"
            sortedIdx = range(len(powers))
            sortedIdx.sort(lambda x,y: -cmp(powers[x], powers[y]))
            cutP = self.markNumber - 1
            cutPower = powers[sortedIdx[cutP]]
            while cutP < len(powers) and powers[sortedIdx[cutP]] == cutPower:
                cutP += 1
            self.graph.setMarkedVertices(sortedIdx[:cutP])
            self.graph.replot()
        elif hubs == 9:
            self.setMarkInput()
       
    def testRefresh(self):
        start = time()
        self.graph.replot()
        stop = time()    
        print "replot in " + str(stop - start)
        
    def saveNetwork(self):
        if self.graph == None or self.graph.visualizer == None:
            return
        
        filename = QFileDialog.getSaveFileName(self, 'Save Network File', '', 'PAJEK networks (*.net)')
        if filename:
            fn = ""
            head, tail = os.path.splitext(str(filename))
            if not tail:
                fn = head + ".net"
            else:
                fn = str(filename)
            
            self.visualize.graph.saveNetwork(fn)
                    
    def sendData(self):
        graph = self.graph.getSelectedGraph()
        vertices = self.graph.getSelectedVertices()
        
        if graph != None:
            if graph.items != None:
                self.send("Selected Examples", graph.items)
            else:
                self.send("Selected Examples", self.graph.getSelectedExamples())
            
            #print "sendData:", self.visualize.graph.items.domain
            self.send("Unselected Examples", self.graph.getUnselectedExamples())    
            self.send("Selected Network", graph)
        else:
            items = self.graph.getSelectedExamples()
            self.send("Selected Examples", items)
                
            items = self.graph.getUnselectedExamples()
            self.send("Unselected Examples", items)
        
        matrix = None
        if self.vertexDistance != None:
            matrix = self.vertexDistance.getitems(vertices)

        self.send("Selected Distance Matrix", matrix)
                
    def setCombos(self):
        vars = self.visualize.getVars()
        edgeVars = self.visualize.getEdgeVars()
        lastLabelColumns = self.lastLabelColumns
        lastTooltipColumns = self.lastTooltipColumns
        
        self.clearCombos()
        
        self.attributes = [(var.name, var.varType) for var in vars]
        self.edgeAttributes = [(var.name, var.varType) for var in edgeVars]
        
        for var in vars:
            if var.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                self.colorCombo.addItem(self.icons[var.varType], unicode(var.name))
                
            if var.varType in [orange.VarTypes.String] and hasattr(self.visualize.graph, 'items') and self.visualize.graph.items != None and len(self.visualize.graph.items) > 0:
                
                value = self.visualize.graph.items[0][var].value
                
                # can value be a list?
                try:
                    if type(eval(value)) == type([]):
                        self.vertexSizeCombo.addItem(self.icons[var.varType], unicode(var.name))
                        continue
                except:
                    pass
                
                if len(value.split(',')) > 1:
                    self.vertexSizeCombo.addItem(self.icons[var.varType], "num of " + unicode(var.name))
                
            elif var.varType in [orange.VarTypes.Continuous]:
                self.vertexSizeCombo.addItem(self.icons[var.varType], unicode(var.name))
                
            self.nameComponentCombo.addItem(self.icons[var.varType], unicode(var.name))
            self.showComponentCombo.addItem(self.icons[var.varType], unicode(var.name))
            self.editCombo.addItem(self.icons[var.varType], unicode(var.name))
            self.comboAttSelection.addItem(self.icons[var.varType], unicode(var.name))
        
        for var in edgeVars:
            if var.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                self.edgeColorCombo.addItem(self.icons[var.varType], unicode(var.name))
                
        for i in range(self.vertexSizeCombo.count()):
            if self.lastVertexSizeColumn == self.vertexSizeCombo.itemText(i):
                self.vertexSize = i
                break
            
        for i in range(self.colorCombo.count()):
            if self.lastColorColumn == self.colorCombo.itemText(i):
                self.color = i
                break

        for i in range(self.attListBox.count()):
            if str(self.attListBox.item(i).text()) in lastLabelColumns:
                self.attListBox.item(i).setSelected(1)
                
        for i in range(self.tooltipListBox.count()):
            if str(self.tooltipListBox.item(i).text()) in lastTooltipColumns:
                self.tooltipListBox.item(i).setSelected(1)
            
        self.lastLabelColumns = lastLabelColumns
        self.lastTooltipColumns = lastTooltipColumns
        
    def clearCombos(self):
        self.attributes = []
        self.edgeAttributes = []
        
        self.colorCombo.clear()
        self.vertexSizeCombo.clear()
        self.nameComponentCombo.clear()
        self.showComponentCombo.clear()
        self.edgeColorCombo.clear()
        self.editCombo.clear()
        self.comboAttSelection.clear()
        
        self.colorCombo.addItem("(same color)")
        self.edgeColorCombo.addItem("(same color)")
        self.vertexSizeCombo.addItem("(same size)")
        self.nameComponentCombo.addItem("Select attribute")
        self.showComponentCombo.addItem("Select attribute")
        self.editCombo.addItem("Select attribute")
        self.comboAttSelection.addItem("Select attribute")
      
    def setGraph(self, graph):      
        if graph == None:
            self.visualize = None
            self.graph.addVisualizer(self.visualize)
            self.clearCombos()
            return
        
        #print "OWNetwork/setGraph: new visualizer..."
        self.visualize = orngNetwork.NetworkOptimization(graph)
        self.graph.addVisualizer(self.visualize)

        #for i in range(graph.nVertices):
        #    print "x:", graph.coors[0][i], " y:", graph.coors[1][i]

        self.nVertices = graph.nVertices
        self.nShown = graph.nVertices
        
        if graph.directed:
            self.nEdges = len(graph.getEdges())
        else:
            self.nEdges = len(graph.getEdges()) / 2
        
        if self.nEdges > 0:
            self.verticesPerEdge = float(self.nVertices) / float(self.nEdges)
        else:
            self.verticesPerEdge = 0
            
        if self.nVertices > 0:
            self.edgesPerVertex = float(self.nEdges) / float(self.nVertices)
        else:
            self.edgesPerVertex = 0
            
        self.diameter = graph.getDiameter()
        self.clustering_coefficient = graph.getClusteringCoefficient() * 100
        
        self.setCombos()
        
        lastNameComponentAttributeFound = False
        for i in range(self.nameComponentCombo.count()):
            if self.lastNameComponentAttribute == self.nameComponentCombo.itemText(i):
                lastNameComponentAttributeFound = True
                self.nameComponentAttribute = i
                self.nameComponents()
                self.showComponentAttribute = self.showComponentCombo.count() - 1
                self.showComponents()
                break
            
        if not lastNameComponentAttributeFound:
            self.lastNameComponentAttribute = ''
        
        self.showComponentAttribute = None

        k = 1.13850193174e-008
        nodes = self.visualize.nVertices()
        t = k * nodes * nodes
        self.frSteps = int(5.0 / t)
        if self.frSteps <   1: self.frSteps = 1;
        if self.frSteps > 1500: self.frSteps = 1500;
        
        self.graph.labelsOnMarkedOnly = self.labelsOnMarkedOnly
        self.graph.showWeights = self.showWeights
        self.graph.showIndexes = self.showIndexes
        # if graph is large, set random layout, min vertex size, min edge size
        if self.frSteps < 10:
            self.renderAntialiased = 0
            self.maxVertexSize = 5
            self.maxLinkSize = 1
            self.optMethod = 0
            self.setOptMethod()            
            
        self.graph.renderAntialiased = self.renderAntialiased
        self.graph.showEdgeLabels = self.showEdgeLabels
        self.graph.maxVertexSize = self.maxVertexSize
        self.graph.maxEdgeSize = self.maxLinkSize
        self.lastVertexSizeColumn = self.vertexSizeCombo.currentText()
        self.lastColorColumn = self.colorCombo.currentText()

        if self.vertexSize > 0:
            self.graph.setVerticesSize(self.vertexSizeCombo.currentText(), self.invertSize)
        else:
            self.graph.setVerticesSize()
            
        self.setVertexColor()
        self.setEdgeColor()
            
        self.graph.setEdgesSize()
        self.clickedAttLstBox()
        self.clickedTooltipLstBox()
        self.clickedEdgeLabelListBox()
        
        self.optButton.setChecked(1)
        self.optLayout()        
        self.information(0)
        #self.controlArea.setEnabled(True)
        self.updateCanvas()
        
    def setItems(self, items=None):
        self.error('')
        
        if self.visualize == None or self.visualize.graph == None or items == None:
            return
        
        if len(items) != self.visualize.graph.nVertices:
            self.error('ExampleTable items must have one example for each vertex.')
            return
        
        self.visualize.graph.setattr("items", items)
        
        self.setVertexSize()
        self.showIndexLabels()
        self.showWeightLabels()
        self.showEdgeLabelsClick()
        
        self.setCombos()
        self.graph.updateData()
        
    def setMarkInput(self):
        var = str(self.markInputCombo.currentText())
        #print 'combo:',self.markInputCombo.currentText()
        if self.markInputItems != None and len(self.markInputItems) > 0:
            values = [str(x[var]).strip().upper() for x in self.markInputItems]
            toMark = [i for (i,x) in enumerate(self.visualize.graph.items) if str(x[var]).strip().upper() in values]
            #print "mark:", toMark
            self.graph.setMarkedVertices(list(toMark))
            self.graph.replot()
            
        else:
            self.graph.setMarkedVertices([])
            self.graph.replot()            
    
    def markItems(self, items):
        self.markInputCombo.clear()
        self.markInputRadioButton.setEnabled(False)
        self.markInputItems = items
        
        if self.visualize == None or self.visualize.graph == None or self.visualize.graph.items == None or items == None:
            return
        
        if len(items) > 0:
            lstOrgDomain = [x.name for x in self.visualize.graph.items.domain] + [self.visualize.graph.items.domain[x].name for x in self.visualize.graph.items.domain.getmetas()]
            lstNewDomain = [x.name for x in items.domain] + [items.domain[x].name for x in items.domain.getmetas()]
            commonVars = set(lstNewDomain) & set(lstOrgDomain)

            if len(commonVars) > 0:
                for var in commonVars:
                    orgVar = self.visualize.graph.items.domain[var]
                    mrkVar = items.domain[var]

                    if orgVar.varType == mrkVar.varType and orgVar.varType == orange.VarTypes.String:
                        self.markInputCombo.addItem(self.icons[orgVar.varType], unicode(orgVar.name))
                        self.markInputRadioButton.setEnabled(True)
                
                        self.setMarkMode(9)
              
    def setExampleSubset(self, subset):
        if self.graph == None:
            return
        
        self.warning('')
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
                self.warning('"index" attribute does not exist in "items" table.')
                            
    def updateCanvas(self):
        # if network exists
        if self.visualize != None:
            self.graph.updateCanvas()
              
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.graph.controlPressed = True
            #print "cp"
        elif e.key() == Qt.Key_Alt:
            self.graph.altPressed = True
        OWWidget.keyPressEvent(self, e)
               
    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Control:
            self.graph.controlPressed = False
        elif e.key() == Qt.Key_Alt:
            self.graph.altPressed = False
        OWWidget.keyReleaseEvent(self, e)
        
#    def keyPressEvent(self, e):
#        if e.text() == "f":
#            self.graph.freezeNeighbours = not self.graph.freezeNeighbours
#        else:
#            OWWidget.keyPressEvent(self, e)

    def showDegreeDistribution(self):
        if self.visualize == None:
            return
        
        from matplotlib import rcParams
        import pylab as p
        
        x = self.visualize.graph.getDegrees()
        nbins = len(set(x))
        if nbins > 500:
          bbins = 500
        #print len(x)
        print x
        # the histogram of the data
        n, bins, patches = p.hist(x, nbins)
        
        p.xlabel('Degree')
        p.ylabel('No. of nodes')
        p.title(r'Degree distribution')
        
        p.show()
        
    def setColors(self):
        dlg = self.createColorDialog(self.colorSettings, self.selectedSchemaIndex)
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contPalette = dlg.getContinuousPalette("contPalette")
            self.graph.discPalette = dlg.getDiscretePalette("discPalette")
            
            self.setVertexColor()
            
    def setEdgeColorPalette(self):
        dlg = self.createColorDialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
        if dlg.exec_():
            self.edgeColorSettings = dlg.getColorSchemas()
            self.selectedEdgeSchemaIndex = dlg.selectedSchemaIndex
            self.graph.contEdgePalette = dlg.getContinuousPalette("contPalette")
            self.graph.discEdgePalette = dlg.getDiscretePalette("discPalette")
            
            self.setEdgeColor()
    
    def createColorDialog(self, colorSettings, selectedSchemaIndex):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        c.setColorSchemas(colorSettings, selectedSchemaIndex)
        return c
    """
    Layout Optimization
    """
    def optLayout(self):
        if self.visualize == None:   #grafa se ni
            self.optButton.setChecked(False)
            return
        
        if not self.optButton.isChecked():
            self.optButton.setChecked(False)
            return
        
        qApp.processEvents()
            
        if self.optMethod == 1:
            self.random()
        elif self.optMethod == 2:
            self.fr(False)
        elif self.optMethod == 3:
            self.fr(True)
        elif self.optMethod == 4:
            self.frRadial()
        elif self.optMethod == 5:
            self.circularCrossingReduction()
        elif self.optMethod == 6:
            self.circularOriginal()
        elif self.optMethod == 7:
            self.circularRandom()
        elif self.optMethod == 8:
            self.pivotMDS()
            
        self.optButton.setChecked(False)
        qApp.processEvents()
        
    def setOptMethod(self, method=None):
        self.stepsSpin.label.setText('Iterations: ')
        
        if method != None:
            self.optMethod = method
            
        if str(self.optMethod) == '0':
            self.optButton.setEnabled(False)
        else:
            self.optButton.setEnabled(True)
            
        if str(self.optMethod) in ['2', '3', '4']:
            self.stepsSpin.setEnabled(True)
        elif str(self.optMethod) == '8':
            self.stepsSpin.label.setText('Pivots: ')
            self.stepsSpin.setEnabled(True)
        else:
            self.stepsSpin.setEnabled(False)
            self.optButton.setChecked(True)
            self.optLayout()

    def random(self):
        #print "OWNetwork/random.."
        if self.visualize == None:   #grafa se ni
            return    
            
        self.visualize.random()
        #print self.visualize.coors
        #print "OWNetwork/random: updating canvas..."
        self.updateCanvas();
        
    def fr(self, weighted):
        if self.visualize == None:   #grafa se ni
            return
              
        if not self.optButton.isChecked():
          #print "not"
          self.stopOptimization = 1
          self.optButton.setChecked(False)
          self.optButton.setText("Optimize layout")
          return
        
        self.optButton.setText("Stop")
        qApp.processEvents()
        self.stopOptimization = 0
        tolerance = 5
        initTemp = 1000
        breakpoints = 6
        k = int(self.frSteps / breakpoints)
        o = self.frSteps % breakpoints
        iteration = 0
        coolFactor = math.exp(math.log(10.0/10000.0) / self.frSteps)

        if k > 0:
            while iteration < breakpoints:
                #print "iteration, initTemp: " + str(initTemp)
                if self.stopOptimization:
                    return
                initTemp = self.visualize.fruchtermanReingold(k, initTemp, coolFactor, self.graph.hiddenNodes, weighted)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
            
            #print "ostanek: " + str(o) + ", initTemp: " + str(initTemp)
            if self.stopOptimization:
                    return
            initTemp = self.visualize.fruchtermanReingold(o, initTemp, coolFactor, self.graph.hiddenNodes, weighted)
            qApp.processEvents()
            self.updateCanvas()
        else:
            while iteration < o:
                #print "iteration ostanek, initTemp: " + str(initTemp)
                if self.stopOptimization:
                    return
                initTemp = self.visualize.fruchtermanReingold(1, initTemp, coolFactor, self.graph.hiddenNodes, weighted)
                iteration += 1
                qApp.processEvents()
                self.updateCanvas()
                
        self.optButton.setChecked(False)
        self.optButton.setText("Optimize layout")
        
    def frSpecial(self):
        if self.visualize == None:   #grafa se ni
            return
        
        steps = 100
        initTemp = 1000
        coolFactor = math.exp(math.log(10.0/10000.0) / steps)
        oldXY = []
        for rec in self.visualize.network.coors:
            oldXY.append([rec[0], rec[1]])
        #print oldXY
        initTemp = self.visualize.fruchtermanReingold(steps, initTemp, coolFactor, self.graph.hiddenNodes)
        #print oldXY
        self.graph.updateDataSpecial(oldXY)
        self.graph.replot()
                
    def frRadial(self):
        if self.visualize == None:   #grafa se ni
            return
        
        #print "F-R Radial"
        k = 1.13850193174e-008
        nodes = self.visualize.nVertices()
        t = k * nodes * nodes
        refreshRate = int(5.0 / t)
        if refreshRate <    1: refreshRate = 1;
        if refreshRate > 1500: refreshRate = 1500;
        #print "refreshRate: " + str(refreshRate)
        
        tolerance = 5
        initTemp = 100
        centerNdx = 0
        
        selection = self.graph.getSelection()
        if len(selection) > 0:
            centerNdx = selection[0]
            
        #print "center ndx: " + str(centerNdx)
        initTemp = self.visualize.radialFruchtermanReingold(centerNdx, refreshRate, initTemp)
        self.graph.circles = [10000 / 12, 10000/12*2, 10000/12*3]#, 10000/12*4, 10000/12*5]
        #self.graph.circles = [100, 200, 300]
        self.updateCanvas()
        self.graph.circles = []
        
    def circularOriginal(self):
        #print "Circular Original"
        if self.visualize != None:
            self.visualize.circularOriginal()
            self.updateCanvas()
           
    def circularRandom(self):
        #print "Circular Random"
        if self.visualize != None:
            self.visualize.circularRandom()
            self.updateCanvas()

    def circularCrossingReduction(self):
        #print "Circular Crossing Reduction"
        if self.visualize != None:
            self.visualize.circularCrossingReduction()
            self.updateCanvas()
            
    def pivotMDS(self):
        if self.vertexDistance == None:
            self.information('Set distance matrix to input signal')
            return
        
        if self.visualize == None:
            self.information('No network found')
            return
        
        if self.vertexDistance.dim != self.visualize.graph.nVertices:
            self.error('Distance matrix dimensionality must equal number of vertices')
            return
        
        self.frSteps = min(self.frSteps, self.vertexDistance.dim)
        qApp.processEvents()
        mds = orngMDS.PivotMDS(self.vertexDistance, self.frSteps)
        x,y = mds.optimize()
        self.visualize.graph.coors[0] = x
        self.visualize.graph.coors[1] = y
        self.updateCanvas()
    
      
    """
    Network Visualization
    """
       
    def clickedAttLstBox(self):
        if self.visualize == None:
            return
        
        self.lastLabelColumns = set([self.attributes[i][0] for i in self.markerAttributes])
        self.graph.setLabelText(self.lastLabelColumns)
        self.graph.updateData()
        self.graph.replot()
  
    def clickedTooltipLstBox(self):
        if self.visualize == None:
            return
        
        self.lastTooltipColumns = set([self.attributes[i][0] for i in self.tooltipAttributes])
        self.graph.setTooltipText(self.lastTooltipColumns)
        self.graph.updateData()
        self.graph.replot()
        
    def clickedEdgeLabelListBox(self):
        if self.visualize == None:
            return
        
        self.lastEdgeLabelAttributes = set([self.edgeAttributes[i][0] for i in self.edgeLabelAttributes])
        self.graph.setEdgeLabelText(self.lastEdgeLabelAttributes)
        self.graph.updateData()
        self.graph.replot()

    def setVertexColor(self):
        if self.visualize == None:
            return
        
        self.graph.setVertexColor(self.colorCombo.currentText())
        self.lastColorColumn = self.colorCombo.currentText()
        self.graph.updateData()
        self.graph.replot()
        
    def setEdgeColor(self):
        if self.visualize == None:
            return
        
        self.graph.setEdgeColor(self.edgeColorCombo.currentText())
        self.lastEdgeColorColumn = self.edgeColorCombo.currentText()
        self.graph.updateData()
        self.graph.replot()
                  
    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)
    
    def selectAllConnectedNodes(self):
        self.graph.selectConnectedNodes(1000000)
        
    def setShowMissingValues(self):
        self.graph.showMissingValues = self.showMissingValues
        self.graph.updateData()
        self.graph.replot()
        
    def setMaxLinkSize(self):
        if self.visualize == None:
            return
        
        self.graph.maxEdgeSize = self.maxLinkSize
        self.graph.setEdgesSize()
        self.graph.replot()
    
    def setVertexSize(self):
        if self.visualize == None:
            return
        
        self.graph.maxVertexSize = self.maxVertexSize
        self.lastVertexSizeColumn = self.vertexSizeCombo.currentText()
        
        if self.vertexSize > 0:
            self.graph.setVerticesSize(self.lastVertexSizeColumn, self.invertSize)
        else:
            self.graph.setVerticesSize()
            
        self.graph.replot()
        
    def setFontSize(self):
        if self.graph == None:
            return
        
        self.graph.fontSize = self.fontSize
        self.graph.drawPlotItems()
        
    def setRenderAntialiased(self):
        self.graph.renderAntialiased = self.renderAntialiased
        self.graph.updateData()
        self.graph.replot()
        
    def sendReport(self):
        self.reportSettings("Graph data",
                            [("Number of vertices", self.nVertices),
                             ("Number of edges", self.nEdges),
                             ("Vertices per edge", "%.3f" % self.verticesPerEdge),
                             ("Edges per vertex", "%.3f" % self.edgesPerVertex),
                             ("Diameter", self.diameter),
                             ("Clustering Coefficient", "%.1f%%" % self.clustering_coefficient)
                             ])
        if self.color or self.vertexSize or self.markerAttributes or self.edgeColor:
            self.reportSettings("Visual settings",
                                [self.color and ("Vertex color", self.colorCombo.currentText()),
                                 self.vertexSize and ("Vertex size", str(self.vertexSizeCombo.currentText()) + " (inverted)" if self.invertSize else ""),
                                 self.markerAttributes and ("Labels", ", ".join(self.attributes[i][0] for i in self.markerAttributes)),
                                 self.edgeColor and ("Edge colors", self.edgeColorCombo.currentText()),
                                ])
        self.reportSettings("Optimization",
                            [("Method", self.optCombo.currentText()),
                             ("Iterations", self.frSteps)])
        self.reportSection("Graph")
        self.reportImage(self.graph.saveToFileDirect)        
        
        
if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetExplorer()
    ow.show()
    appl.exec_()
    

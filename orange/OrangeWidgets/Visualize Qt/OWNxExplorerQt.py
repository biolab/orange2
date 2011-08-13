"""
<name>Nx Explorer (Qt)</name>
<description>Orange widget for network exploration.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>310</priority>
"""
import math
import operator
import statc
import time

import Orange 
import OWGUI
import OWColorPalette
import OWToolbars
import orngMDS

from OWWidget import *
from OWNxCanvasQt import *
from operator import itemgetter

dir = os.path.dirname(__file__) + "/../icons/"
dlg_mark2sel = dir + "Dlg_Mark2Sel.png"
dlg_sel2mark = dir + "Dlg_Sel2Mark.png"
dlg_selIsmark = dir + "Dlg_SelisMark.png"
dlg_selected = dir + "Dlg_SelectedNodes.png"
dlg_unselected = dir + "Dlg_UnselectedNodes.png"
dlg_showall = dir + "Dlg_clear.png"

class OWNxExplorerQt(OWWidget):
    settingsList = ["autoSendSelection", "spinExplicit", "spinPercentage",
    "maxLinkSize", "minVertexSize", "maxVertexSize", "networkCanvas.use_antialiasing", "networkCanvas.use_animations",
    "invertSize", "optMethod", "lastVertexSizeColumn", "lastColorColumn", "networkCanvas.show_indices",
    "lastNameComponentAttribute", "lastLabelColumns", "lastTooltipColumns",
    "showWeights", "showEdgeLabels", "colorSettings", 
    "selectedSchemaIndex", "edgeColorSettings", "selectedEdgeSchemaIndex",
    "showMissingValues", "fontSize", "mdsTorgerson", "mdsAvgLinkage",
    "mdsSteps", "mdsRefresh", "mdsStressDelta", "organism","showTextMiningInfo", 
    "toolbarSelection", "minComponentEdgeWidth", "maxComponentEdgeWidth",
    "mdsFromCurrentPos", "labelsOnMarkedOnly", "tabIndex"] 
    
    def __init__(self, parent=None, signalManager=None, name = 'Nx Explorer (qt)', 
                 NetworkCanvas=OWNxCanvas):
        OWWidget.__init__(self, parent, signalManager, name)
        #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="markerAttributes"), ContextField("attributes", selected="tooltipAttributes"), "color"])}
        self.inputs = [("Nx View", Orange.network.NxView, self.set_network_view),
                       ("Network", Orange.network.Graph, self.set_graph, Default),
                       ("Items", Orange.data.Table, self.setItems),
                       ("Items to Mark", Orange.data.Table, self.markItems), 
                       ("Items Subset", Orange.data.Table, self.setExampleSubset), 
                       ("Items Distance Matrix", Orange.core.SymMatrix, self.set_items_distance_matrix)]
        
        self.outputs = [("Selected Network", Orange.network.Graph),
                        ("Selected Items Distance Matrix", Orange.core.SymMatrix),
                        ("Selected Items", Orange.data.Table), 
                        ("Unselected Items", Orange.data.Table), 
                        ("Marked Items", Orange.data.Table),
                        ("Attribute Selection List", AttributeList)]
        
        self.networkCanvas = NetworkCanvas(self, self.mainArea, "Net Explorer")
        
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
        self.nShown = self.nHidden = self.nMarked = self.nSelected = self.verticesPerEdge = self.edgesPerVertex = self.diameter = self.clustering_coefficient = 0
        self.optimizeWhat = 1
        self.maxLinkSize = 3
        self.maxVertexSize = 5
        self.minVertexSize = 5
        self.labelsOnMarkedOnly = 0
        self.invertSize = 0
        self.optMethod = 0
        self.lastVertexSizeColumn = ''
        self.lastColorColumn = ''
        self.lastNameComponentAttribute = ''
        self.lastLabelColumns = set()
        self.lastTooltipColumns = set()
        self.showWeights = 0
        self.showEdgeLabels = 0
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.edgeColorSettings = [('net_edges', [[], [('contPalette', (4294967295L, 4278190080L, 0))], [('discPalette', [(204, 204, 204), (179, 226, 205), (253, 205, 172), (203, 213, 232), (244, 202, 228), (230, 245, 201), (255, 242, 174), (241, 226, 204)])]]), ('Default', [[], [('contPalette', (4294967295L, 4278190080L, 0))], [('discPalette', [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255), (255, 223, 128), (127, 111, 64), (92, 46, 0), (0, 84, 0), (192, 192, 0), (0, 127, 127), (128, 0, 0), (127, 0, 127)])]])]
        self.selectedEdgeSchemaIndex = 0
        self.items_matrix = None
        self.showDistances = 0
        self.showMissingValues = 0
        self.fontSize = 12
        self.mdsTorgerson = 0
        self.mdsAvgLinkage = 1
        self.mdsSteps = 10000
        self.mdsRefresh = 50
        self.mdsStressDelta = 0.0000001
        self.organism = 'goa_human'
        self.showTextMiningInfo = 0
        self.toolbarSelection = 0
        self.minComponentEdgeWidth = 10
        self.maxComponentEdgeWidth = 70
        self.mdsFromCurrentPos = 0
        self.tabIndex = 0
        self.number_of_nodes_label = -1
        self.number_of_edges_label = -1
        
        self.loadSettings()
        
        self._network_view = None
#        self.layout = Orange.network.GraphLayout()
        self.graph = None
        self.graph_base = None
        self.markInputItems = None
        
        self.mainArea.layout().setContentsMargins(0,4,4,4)
        self.controlArea.layout().setContentsMargins(4,4,0,4)
        
        self.networkCanvas.showMissingValues = self.showMissingValues
        self.mainArea.layout().addWidget(self.networkCanvas)
        
        self.networkCanvas.maxLinkSize = self.maxLinkSize
        
        self.hcontroArea = OWGUI.widgetBox(self.controlArea, orientation='horizontal')
        
        self.tabs = OWGUI.tabWidget(self.hcontroArea)
        
        self.verticesTab = OWGUI.createTabPage(self.tabs, "Nodes")
        self.edgesTab = OWGUI.createTabPage(self.tabs, "Edges")
        self.markTab = OWGUI.createTabPage(self.tabs, "Mark")
        self.infoTab = OWGUI.createTabPage(self.tabs, "Info")
        #self.editTab = OWGUI.createTabPage(self.tabs, "Edit")
        
        self.tabs.setCurrentIndex(self.tabIndex)
        self.connect(self.tabs, SIGNAL("currentChanged(int)"), lambda index: setattr(self, 'tabIndex', index))
        
        self.optimizeBox = OWGUI.radioButtonsInBox(self.verticesTab, self, "optimizeWhat", [], "Optimize", addSpace=False)
        
        self.optCombo = OWGUI.comboBox(self.optimizeBox, self, "optMethod", label='Method:     ', orientation='horizontal', callback=self.graph_layout_method)
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
        self.stepsSpin = OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 100000, 1, label="Iterations: ")
        self.stepsSpin.setEnabled(False)
        
        self.optButton = OWGUI.button(self.optimizeBox, self, "Optimize layout", callback=self.graph_layout, toggleButton=1)
        
        colorBox = OWGUI.widgetBox(self.verticesTab, "Node color attribute", orientation="horizontal", addSpace = False)
        self.colorCombo = OWGUI.comboBox(colorBox, self, "color", callback=self.setVertexColor)
        self.colorCombo.addItem("(same color)")
        OWGUI.button(colorBox, self, "Set node color palette", self.setColors, tooltip = "Set node color palette", debuggingEnabled = 0)
        
        ib = OWGUI.widgetBox(self.verticesTab, "Node size attribute", orientation="vertical", addSpace = False)
        hb = OWGUI.widgetBox(ib, orientation="horizontal", addSpace = False)
        OWGUI.checkBox(hb, self, "invertSize", "Invert size", callback = self.set_vertex_size)
        OWGUI.spin(hb, self, "minVertexSize", 5, 200, 1, label="Min:", callback=self.set_vertex_size)
        OWGUI.spin(hb, self, "maxVertexSize", 5, 200, 1, label="Max:", callback=self.set_vertex_size)
        self.vertexSizeCombo = OWGUI.comboBox(ib, self, "vertexSize", callback=self.set_vertex_size)
        self.vertexSizeCombo.addItem("(none)")
        
        ib = OWGUI.widgetBox(self.verticesTab, "Node labels | tooltips", orientation="vertical", addSpace = False)
        OWGUI.spin(ib, self, "fontSize", 4, 30, 1, label="Set font size:", callback = self.setFontSize)
        hb = OWGUI.widgetBox(ib, orientation="horizontal", addSpace = False)
        self.attListBox = OWGUI.listBox(hb, self, "markerAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedAttLstBox)
        self.tooltipListBox = OWGUI.listBox(hb, self, "tooltipAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedTooltipLstBox)
        
        ib = OWGUI.widgetBox(self.edgesTab, "General", orientation="vertical")
        OWGUI.checkBox(ib, self, 'showWeights', 'Show weights', callback=(lambda: self._set_canvas_attr('showWeights', self.showWeights)))
        OWGUI.checkBox(ib, self, 'showEdgeLabels', 'Show labels on edges', callback=(lambda: self._set_canvas_attr('showEdgeLabels', self.showEdgeLabels)))
        OWGUI.spin(ib, self, "maxLinkSize", 1, 50, 1, label="Max edge width:", callback = self.setMaxLinkSize)
        self.showDistancesCheckBox = OWGUI.checkBox(ib, self, 'showDistances', 'Explore node distances', callback=(lambda: self._set_canvas_attr('showDistances', self.showDistances)), disabled=1)
        
        colorBox = OWGUI.widgetBox(self.edgesTab, "Edge color attribute", orientation="horizontal", addSpace = False)
        self.edgeColorCombo = OWGUI.comboBox(colorBox, self, "edgeColor", callback=self.setEdgeColor)
        self.edgeColorCombo.addItem("(same color)")
        OWGUI.button(colorBox, self, "Set edge color palette", self.setEdgeColorPalette, tooltip = "Set edge color palette", debuggingEnabled = 0)
        
        self.edgeLabelBox = OWGUI.widgetBox(self.edgesTab, "Edge labels", addSpace = False)
        self.edgeLabelListBox = OWGUI.listBox(self.edgeLabelBox, self, "edgeLabelAttributes", "edgeAttributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedEdgeLabelListBox)
        self.edgeLabelBox.setEnabled(False)
        
        ib = OWGUI.widgetBox(self.verticesTab, "General", orientation="vertical")
        OWGUI.checkBox(ib, self, 'networkCanvas.show_indices', 'Show indices', callback=self.networkCanvas.set_label_attributes)
        OWGUI.checkBox(ib, self, 'labelsOnMarkedOnly', 'Show labels on marked nodes only', callback=(lambda: self.networkCanvas.set_labels_on_marked_only(self.labelsOnMarkedOnly)))
        self.networkCanvas.gui.antialiasing_check_box(ib)
        self.networkCanvas.gui.animations_check_box(ib)
        
        ib = OWGUI.widgetBox(self.markTab, "Info", orientation="vertical")
        OWGUI.label(ib, self, "Nodes (shown/hidden): %(number_of_nodes_label)i (%(nShown)i/%(nHidden)i)")
        OWGUI.label(ib, self, "Selected: %(nSelected)i, marked: %(nMarked)i")
        
        ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Mark", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "None", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Search", callback = self.set_mark_mode)
        self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.set_mark_mode)
        
        OWGUI.appendRadioButton(ribg, self, "hubs", "Neighbors of focused", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Neighbours of selected", callback = self.set_mark_mode)
        ib = OWGUI.indentedBox(ribg, orientation = 0)
        self.ctrlMarkDistance = OWGUI.spin(ib, self, "markDistance", 0, 100, 1, label="Distance ", callback=(lambda h=2: self.set_mark_mode(h)))
        #self.ctrlMarkFreeze = OWGUI.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
        OWGUI.widgetLabel(ribg, "Mark nodes with ...")
        OWGUI.appendRadioButton(ribg, self, "hubs", "at least N connections", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "at most N connections", callback = self.set_mark_mode)
        self.ctrlMarkNConnections = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N ", callback=(lambda h=4: self.set_mark_mode(h)))
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than any neighbour", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than avg neighbour", callback = self.set_mark_mode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "most connections", callback = self.set_mark_mode)
        ib = OWGUI.indentedBox(ribg)
        self.ctrlMarkNumber = OWGUI.spin(ib, self, "markNumber", 0, 1000000, 1, label="Number of nodes:", callback=(lambda h=8: self.set_mark_mode(h)))
        OWGUI.widgetLabel(ib, "(More nodes are marked in case of ties)")
        self.markInputRadioButton = OWGUI.appendRadioButton(ribg, self, "hubs", "Mark nodes given in the input signal", callback = self.set_mark_mode)
        ib = OWGUI.indentedBox(ribg)
        self.markInput = 0
        self.markInputCombo = OWGUI.comboBox(ib, self, "markInput", callback=(lambda h=9: self.set_mark_mode(h)))
        self.markInputRadioButton.setEnabled(False)
        
        ib = OWGUI.widgetBox(self.markTab, "General", orientation="vertical")
        self.checkSendMarkedNodes = 0
        OWGUI.checkBox(ib, self, 'checkSendMarkedNodes', 'Send marked nodes', callback = self.send_marked_nodes, disabled=0)
        
        G = self.networkCanvas.gui
        
        self.zoomSelectToolbar = G.zoom_select_toolbar(self.hcontroArea, orientation = Qt.Vertical,
            buttons = 
                [G.Spacing] * 2 + 
                G.default_zoom_select_buttons + 
                [
                    G.Spacing,
                    ("buttonM2S", "Add marked to selection", None, None, "marked_to_selected", 'Dlg_Mark2Sel'),
                    ("buttonS2M", "Add selection to marked", None, None, "selected_to_marked", 'Dlg_Sel2Mark'),
                    G.Spacing,
                    ("buttonSEL", "Hide selected", None, None, "hideSelectedVertices", 'Dlg_UnselectedNodes'),
                    ("buttonUN", "Hide unselected", None, None, "hideUnSelectedVertices", 'Dlg_SelectedNodes'),
                    ("buttonSW", "Show all nodes", None, None, "showAllVertices", 'Dlg_clear'),
                ])
        self.zoomSelectToolbar.buttons[G.SendSelection].clicked.connect(self.send_data)
        OWGUI.rubber(self.zoomSelectToolbar)
        
        ib = OWGUI.widgetBox(self.infoTab, "General")
        OWGUI.label(ib, self, "Number of nodes: %(number_of_nodes_label)i")
        OWGUI.label(ib, self, "Number of edges: %(number_of_edges_label)i")
        OWGUI.label(ib, self, "Nodes per edge: %(verticesPerEdge).2f")
        OWGUI.label(ib, self, "Edges per node: %(edgesPerVertex).2f")
        OWGUI.label(ib, self, "Diameter: %(diameter)i")
        OWGUI.label(ib, self, "Clustering Coefficient: %(clustering_coefficient).1f%%")
        
        ib = OWGUI.widgetBox(self.infoTab, orientation="horizontal")
        
        OWGUI.button(ib, self, "Degree distribution", callback=self.showDegreeDistribution, debuggingEnabled=False)
        OWGUI.button(ib, self, "Save network", callback=self.save_network, debuggingEnabled=False)
        OWGUI.button(ib, self, "Save image", callback=self.networkCanvas.saveToFile, debuggingEnabled=False)
        
        #OWGUI.button(self.edgesTab, self, "Clustering", callback=self.clustering)
        
        ib = OWGUI.widgetBox(self.infoTab, "Prototype")
        ib.setVisible(False)
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
        self.btnMDS = OWGUI.button(ibs, self, "Fragviz", callback=self.mds_components, toggleButton=1)
        self.btnESIM = OWGUI.button(ibs, self, "eSim", callback=(lambda: self.mds_components(Orange.network.MdsType.exactSimulation)), toggleButton=1)
        self.btnMDSv = OWGUI.button(ibs, self, "MDS", callback=(lambda: self.mds_components(Orange.network.MdsType.MDS)), toggleButton=1)
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
        OWGUI.checkBox(ib, self, 'mdsFromCurrentPos', "MDS from current positions")
        self.mdsInfoA=OWGUI.widgetLabel(ib, "Avg. stress:")
        self.mdsInfoB=OWGUI.widgetLabel(ib, "Num. steps:")
        self.rotateSteps = 100
        
        OWGUI.spin(ib, self, "rotateSteps", 1, 10000, 1, label="Rotate max steps: ")
        OWGUI.spin(ib, self, "minComponentEdgeWidth", 0, 100, 1, label="Min component edge width: ", callback=(lambda changedMin=1: self.setComponentEdgeWidth(changedMin)))
        OWGUI.spin(ib, self, "maxComponentEdgeWidth", 0, 200, 1, label="Max component edge width: ", callback=(lambda changedMin=0: self.setComponentEdgeWidth(changedMin)))
        
        self.attSelectionAttribute = 0
        self.comboAttSelection = OWGUI.comboBox(ib, self, "attSelectionAttribute", label='Send attribute selection list:', orientation='horizontal', callback=self.sendAttSelectionList)
        self.comboAttSelection.addItem("Select attribute")
        self.autoSendAttributes = 0
        OWGUI.checkBox(ib, self, 'autoSendAttributes', "auto send attributes", callback=self.setAutoSendAttributes)
        
        self.icons = self.createAttributeIconDict()
        self.set_mark_mode()
        
        self.editAttribute = 0
        self.editCombo = OWGUI.comboBox(self.infoTab, self, "editAttribute", label="Edit attribute:", orientation="horizontal")
        self.editCombo.addItem("Select attribute")
        self.editValue = ''
        OWGUI.lineEdit(self.infoTab, self, "editValue", "Value:", orientation='horizontal')
        OWGUI.button(self.infoTab, self, "Edit", callback=self.edit)
        
        self.verticesTab.layout().addStretch(1)
        self.edgesTab.layout().addStretch(1)
        self.markTab.layout().addStretch(1)
        self.infoTab.layout().addStretch(1)
        
        dlg = self.createColorDialog(self.colorSettings, self.selectedSchemaIndex)
        self.networkCanvas.contPalette = dlg.getContinuousPalette("contPalette")
        self.networkCanvas.discPalette = dlg.getDiscretePalette("discPalette")
        
        dlg = self.createColorDialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
        self.networkCanvas.contEdgePalette = dlg.getContinuousPalette("contPalette")
        self.networkCanvas.discEdgePalette = dlg.getDiscretePalette("discPalette")
        
        self.graph_layout_method()
        self.setFontSize()
        self.set_graph(None)
        self.setMinimumWidth(900)
        
        #self.resize(1000, 600)
        #self.controlArea.setEnabled(False)
        
    def setComponentEdgeWidth(self, changedMin=True):
        if self.networkCanvas is None:
            return
        
        canvas = self.networkCanvas
        if changedMin:
            if self.maxComponentEdgeWidth < self.minComponentEdgeWidth:
                self.maxComponentEdgeWidth = self.minComponentEdgeWidth
        else:
            if self.minComponentEdgeWidth > self.maxComponentEdgeWidth:
                self.minComponentEdgeWidth = self.maxComponentEdgeWidth
        
        canvas.minComponentEdgeWidth = self.minComponentEdgeWidth
        canvas.maxComponentEdgeWidth = self.maxComponentEdgeWidth
        self.networkCanvas.updateCanvas()
    
    def setAutoSendAttributes(self):
        print 'TODO setAutoSendAttributes'
        #if self.autoSendAttributes:
        #    self.networkCanvas.callbackSelectVertex = self.sendAttSelectionList
        #else:
        #    self.networkCanvas.callbackSelectVertex = None

    def sendAttSelectionList(self):
        if not self.graph is None:
            vars = [x.name for x in self.graph_base.links_vars()]
            if not self.comboAttSelection.currentText() in vars:
                return
            att = str(self.comboAttSelection.currentText())
            vertices = self.networkCanvas.networkCurve.get_selected_vertices()
            
            if len(vertices) != 1:
                return
            
            attributes = str(self.graph_base.items()[vertices[0]][att]).split(', ')
        else:
            attributes = None
        self.send("Attribute Selection List", attributes)
        
    def edit(self):
        if self.graph is None:
            return
        
        vars = [x.name for x in self.graph_base.items_vars()]
        if not self.editCombo.currentText() in vars:
            return
        att = str(self.editCombo.currentText())
        vertices = self.networkCanvas.networkCurve.get_selected_vertices()
        
        if len(vertices) == 0:
            return
        
        if self.graph_base.items().domain[att].var_type == Orange.data.Type.Continuous:
            for v in vertices:
                self.graph_base.items()[v][att] = float(self.editValue)
        else:
            for v in vertices:
                self.graph_base.items()[v][att] = str(self.editValue)
        
        self.setItems(self.graph_base.items())
        
    def drawForce(self):
#        if self.btnForce.isChecked() and self.graph is not None:
#            self.networkCanvas.forceVectors = self.layout._computeForces() 
#        else:
#            self.networkCanvas.forceVectors = None
#            
#        self.networkCanvas.updateCanvas()
        pass
        
    def rotateProgress(self, curr, max):
        self.progressBarSet(int(curr * 100 / max))
        qApp.processEvents()
        
    def rotateComponentsMDS(self):
#        print "rotate"
#        if self.items_matrix is None:
#            self.information('Set distance matrix to input signal')
#            self.btnRotateMDS.setChecked(False)
#            return
#        
#        if self.graph is None:
#            self.information('No network found')
#            self.btnRotateMDS.setChecked(False)
#            return
#        if self.items_matrix.dim != self.graph.number_of_nodes():
#            self.error('Distance matrix dimensionality must equal number of vertices')
#            self.btnRotateMDS.setChecked(False)
#            return
#        
#        if not self.btnRotateMDS.isChecked():
#          self.layout.stopMDS = 1
#          #self.btnMDS.setChecked(False)
#          #self.btnMDS.setText("MDS on graph components")
#          return
#        
#        self.btnRotateMDS.setText("Stop")
#        qApp.processEvents()
#        
#        self.layout.items_matrix = self.items_matrix
#        self.progressBarInit()
#        
#        self.layout.mds_components(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, rotationOnly=True, mdsFromCurrentPos=self.mdsFromCurrentPos)            
#            
#        self.btnRotateMDS.setChecked(False)
#        self.btnRotateMDS.setText("Rotate graph components (MDS)")
#        self.progressBarFinished()
        pass
    
    def rotateComponents(self):
#        if self.items_matrix is None:
#            self.information('Set distance matrix to input signal')
#            self.btnRotate.setChecked(False)
#            return
#        
#        if self.graph is None:
#            self.information('No network found')
#            self.btnRotate.setChecked(False)
#            return
#        
#        if self.items_matrix.dim != self.graph.number_of_nodes():
#            self.error('Distance matrix dimensionality must equal number of vertices')
#            self.btnRotate.setChecked(False)
#            return
#        
#        if not self.btnRotate.isChecked():
#          self.layout.stopRotate = 1
#          return
#      
#        self.btnRotate.setText("Stop")
#        qApp.processEvents()
#        
#        self.layout.items_matrix = self.items_matrix
#        self.progressBarInit()
#        self.layout.rotateComponents(self.rotateSteps, 0.0001, self.rotateProgress, self.updateCanvas)
#        self.btnRotate.setChecked(False)
#        self.btnRotate.setText("Rotate graph components")
#        self.progressBarFinished()
        pass
        
    def mdsProgress(self, avgStress, stepCount):    
        self.drawForce()

        self.mdsInfoA.setText("Avg. Stress: %.20f" % avgStress)
        self.mdsInfoB.setText("Num. steps: %i" % stepCount)
        self.progressBarSet(int(stepCount * 100 / self.mdsSteps))
        qApp.processEvents()
        
    def mds_components(self, mdsType=Orange.network.MdsType.componentMDS):
#        if mdsType == Orange.network.MdsType.componentMDS:
#            btn = self.btnMDS
#        elif mdsType == Orange.network.MdsType.exactSimulation:
#            btn = self.btnESIM
#        elif mdsType == Orange.network.MdsType.MDS:
#            btn = self.btnMDSv
#        
#        btnCaption = btn.text()
#        
#        if self.items_matrix is None:
#            self.information('Set distance matrix to input signal')
#            btn.setChecked(False)
#            return
#        
#        if self.layout is None:
#            self.information('No network found')
#            btn.setChecked(False)
#            return
#        
#        if self.items_matrix.dim != self.graph.number_of_nodes():
#            self.error('Distance matrix dimensionality must equal number of vertices')
#            btn.setChecked(False)
#            return
#        
#        if not btn.isChecked():
#            self.layout.stopMDS = 1
#            btn.setChecked(False)
#            btn.setText(btnCaption)
#            return
#        
#        btn.setText("Stop")
#        qApp.processEvents()
#        
#        self.layout.items_matrix = self.items_matrix
#        self.progressBarInit()
#        
#        if self.mdsAvgLinkage:
#            self.layout.mds_components_avg_linkage(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.networkCanvas.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, scalingRatio = self.scalingRatio, mdsFromCurrentPos=self.mdsFromCurrentPos)
#        else:
#            self.layout.mds_components(self.mdsSteps, self.mdsRefresh, self.mdsProgress, self.networkCanvas.updateCanvas, self.mdsTorgerson, self.mdsStressDelta, mdsType=mdsType, scalingRatio=self.scalingRatio, mdsFromCurrentPos=self.mdsFromCurrentPos)            
#        
#        btn.setChecked(False)
#        btn.setText(btnCaption)
#        self.progressBarFinished()
        pass
    
    def set_items_distance_matrix(self, matrix):
        self.error('')
        self.information('')
        self.showDistancesCheckBox.setEnabled(0)
        
        if matrix is None or self.graph_base is None:
            self.items_matrix = None
            self.networkCanvas.items_matrix = None
            return

        if matrix.dim != self.graph_base.number_of_nodes():
            self.error('The number of vertices does not match matrix size.')
            self.items_matrix = None
            self.networkCanvas.items_matrix = None
            return
        
        self.items_matrix = matrix
        self.networkCanvas.items_matrix = matrix
        self.showDistancesCheckBox.setEnabled(1)
        
        if self.optMethod == 8:
            self.optButton.setChecked(True)
            self.graph_layout()
    
    def send_marked_nodes(self):
        if self.checkSendMarkedNodes:
            markedNodes = self.networkCanvas.marked_nodes()
            
            if len(markedNodes) == 0:
                self.send("Marked Items", None)
                return
            
            if self.graph is not None and self.graph_base.items() is not None:
                items = self.graph_base.items().getitems(markedNodes)
                self.send("Marked Items", items)
                return
            
            self.send("Marked Items", None)
        else:
            self.networkCanvas.send_marked_nodes = None
            self.send("Marked Items", None)
            
    def showComponents(self):
        if self.graph is None or self.graph_base.items() is None:
            return
        
        vars = [x.name for x in self.graph_base.items_vars()]
        
        if not self.showComponentCombo.currentText() in vars:
            self.networkCanvas.showComponentAttribute = None
            self.lastNameComponentAttribute = ''
        else:
            self.networkCanvas.showComponentAttribute = self.showComponentCombo.currentText()     
            
#        self.networkCanvas.drawPlotItems()
        
    def nameComponents(self):
        """Names connected components of genes according to GO terms."""
        self.progressBarFinished()
        self.lastNameComponentAttribute = None
        
        if self.graph is None or self.graph_base.items() is None:
            return
        
        vars = [x.name for x in self.graph_base.items_vars()]
        if not self.nameComponentCombo.currentText() in vars:
            return
        
        self.progressBarInit()
        components = [c for c in Orange.network.nx.algorithms.components.connected_components(self.graph) if len(c) > 1]
        if 'component name' in self.graph_base.items().domain:
            keyword_table = self.graph_base.items()
        else:
            keyword_table = Orange.data.Table(Orange.data.Domain(Orange.data.variable.String('component name')), [[''] for i in range(len(self.graph_base.items()))]) 
        
        import obiGO 
        ontology = obiGO.Ontology.Load(progressCallback=self.progressBarSet) 
        annotations = obiGO.Annotations.Load(self.organism, ontology=ontology, progressCallback=self.progressBarSet)

        allGenes = set([e[str(self.nameComponentCombo.currentText())].value for e in self.graph_base.items()])
        foundGenesets = False
        if len(annotations.geneNames & allGenes) < 1:
            allGenes = set(reduce(operator.add, [e[str(self.nameComponentCombo.currentText())].value.split(', ') for e in self.graph_base.items()]))
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
                genes = reduce(operator.add, [self.graph_base.items()[v][str(self.nameComponentCombo.currentText())].value.split(', ') for v in component])
            else:
                genes = [self.graph_base.items()[v][str(self.nameComponentCombo.currentText())].value for v in component]
                    
            res1 = annotations.GetEnrichedTerms(genes, aspect="P")
            res2 = annotations.GetEnrichedTerms(genes, aspect="F")
            res = res1_base.items() + res2.items()
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
        self.setItems(Orange.data.Table([self.graph_base.items(), keyword_table]))
        self.progressBarFinished()   
        
    def nameComponents_old(self):
        if self.graph is None or self.graph_base.items() is None:
            return
        
        vars = [x.name for x in self.graph_base.items_vars()]
        
        if not self.nameComponentCombo.currentText() in vars:
            return
        
        components = Orange.network.nx.algorithms.components.connected_components(self.graph)
        keyword_table = Orange.data.Table(Orange.data.Domain(Orange.data.variables.String('component name')), [[''] for i in range(len(self.graph_base.items()))]) 
        
        excludeWord = ["AND", "OF", "KEGG", "ST", "IN", "SIG"]
        excludePart = ["HSA"]
        keywords = set()
        sameKeywords = set()
        
        for component in components:
            words = []
            all_values = []
            for vertex in component:
                values = []
                value =  str(self.graph_base.items()[vertex][str(self.nameComponentCombo.currentText())])
                
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
                
                
                #value =  str(self.graph.items()[vertex][str(self.nameComponentCombo.currentText())])
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
                value =  str(self.graph_base.items()[vertex][str(self.nameComponentCombo.currentText())])
                
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
        items = Orange.data.Table([self.graph_base.items(), keyword_table])
        self.setItems(items)
        
        #for item in items:
        #    print item
                        
    def _set_canvas_attr(self, attr, value):
        setattr(self.networkCanvas, attr, value)
        self.networkCanvas.updateCanvas()
    
    def _set_curve_attr(self, attr, value):
        setattr(self.networkCanvas.networkCurve, attr, value)
        self.networkCanvas.updateCanvas()
                
    def setSearchStringTimer(self):
        self.hubs = 1
        self.searchStringTimer.stop()
        self.searchStringTimer.start(1000)
         
    def set_mark_mode(self, i=None):
        self.searchStringTimer.stop()
        if not i is None:
            self.hubs = i
        
        QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self.networkCanvas.mark_on_selection_changed)
        
        if self.graph is None:
            return
        
        hubs = self.hubs
        
        if hubs in [0,1,2,3]:
            if hubs == 0:
                self.networkCanvas.networkCurve.clear_node_marks()
            elif hubs == 1:
                #print "mark on given label"
                txt = self.markSearchString
                
                toMark = set(i for i, values in enumerate(self.graph_base.items()) if txt.lower() in " ".join([str(values[ndx]).decode("ascii", "ignore").lower() for ndx in range(len(self.graph_base.items().domain)) + self.graph_base.items().domain.getmetas().keys()]))
                toMark = toMark.intersection(self.graph.nodes())
                self.networkCanvas.networkCurve.clear_node_marks()
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) for i in toMark))
            elif hubs == 2:
                #print "mark on focus"
                self.networkCanvas.mark_neighbors = self.markDistance
            elif hubs == 3:
                #print "mark selected"
                self.networkCanvas.mark_neighbors = self.markDistance
                QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self.networkCanvas.mark_on_selection_changed)
                self.networkCanvas.mark_on_selection_changed()
                
        elif hubs in [4,5,6,7,8,9]:
            
            powers = sorted(self.graph.degree_iter(), key=itemgetter(1), reverse=True)
            
            if hubs == 4:
                #print "mark at least N connections"
                N = self.markNConnections
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) if \
                    d >= N else (i, False) for i, d in powers))
            elif hubs == 5:
                #print "mark at most N connections"
                N = self.markNConnections
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) if \
                    d <= N else (i, False) for i, d in powers))
            elif hubs == 6:
                #print "mark more than any"
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) if \
                    d > max([0]+self.graph.degree(self.graph.neighbors(i)).values()) \
                    else (i, False) for i,d in powers ))
            elif hubs == 7:
                #print "mark more than avg"
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) if \
                    d > statc.mean([0]+self.graph.degree(self.graph.neighbors(i)).values()) \
                    else (i, False) for i,d in powers ))
                self.networkCanvas.replot()
            elif hubs == 8:
                #print "mark most"
                self.networkCanvas.networkCurve.clear_node_marks()
                
                if self.markNumber < 1:
                    return
                
                cut = self.markNumber
                cutPower = powers[cut-1][1]
                while cut < len(powers) and powers[cut][1] == cutPower:
                    cut += 1
    
                self.networkCanvas.networkCurve.clear_node_marks()
                self.networkCanvas.networkCurve.set_node_marks(dict((i, True) for \
                    i,d in powers[:cut]))
                
            elif hubs == 9:
                var = str(self.markInputCombo.currentText())
                if self.markInputItems is not None and len(self.markInputItems) > 0:
                    values = [str(x[var]).strip().upper() for x in self.markInputItems]
                    tomark = dict((x, True) for x in self.graph.nodes() if str(self.graph_base.items()[x][var]).strip().upper() in values)
                    self.networkCanvas.networkCurve.clear_node_marks()
                    self.networkCanvas.networkCurve.set_node_marks(tomark)
                    
                else:
                    self.networkCanvas.networkCurve.clear_node_marks()      
        
        self.nMarked = len(self.networkCanvas.marked_nodes())
        self.send_marked_nodes()
       
    def testRefresh(self):
        start = time.time()
        self.networkCanvas.replot()
        stop = time.time()    
        print "replot in " + str(stop - start)
        
    def save_network(self):
#        if self.networkCanvas is None or self.graph is None:
#            return
#        
#        filename = QFileDialog.getSaveFileName(self, 'Save Network File', '', 'NetworkX graph as Python pickle (*.gpickle)\nPajek network (*.net)\nGML network (*.gml)')
#        if filename:
#            fn = ""
#            head, tail = os.path.splitext(str(filename))
#            if not tail:
#                fn = head + ".net"
#            else:
#                fn = str(filename)
#            
#            for i in range(self.graph.number_of_nodes()):
#                node = self.graph.node[i]
#                node['x'] = self.layout.coors[0][i]
#                node['y'] = self.layout.coors[1][i]
#
#            Orange.network.readwrite.write(self.graph, fn)
        pass
            
    def send_data(self):
        selected_nodes = self.networkCanvas.selected_nodes()
        graph = self.graph_base.subgraph(selected_nodes)
        
        if graph is not None:
            self.send("Selected Items", graph.items())
            
            nodes = self.networkCanvas.not_selected_nodes()
            if len(nodes) > 0 and self.graph_base.items() is not None:
                self.send("Unselected Items", self.graph_base.items().getitems(nodes))
            else:
                self.send("Unselected Items", None)
                
            self.send("Selected Network", graph)
        else:
            self.send("Selected Items", None)
            self.send("Unselected Items", None)
            self.send("Selected Network", None)
            
        matrix = None if self.items_matrix is None else self.items_matrix.getitems(selected_nodes)
        self.send("Selected Items Distance Matrix", matrix)
                
    def setCombos(self):
        vars = self.graph_base.items_vars()
        edgeVars = self.graph_base.links_vars()
        lastLabelColumns = self.lastLabelColumns
        lastTooltipColumns = self.lastTooltipColumns
        
        self.clearCombos()
        
        self.attributes = [(var.name, var.varType) for var in vars]
        self.edgeAttributes = [(var.name, var.varType) for var in edgeVars]
        
        for var in vars:
            if var.varType in [Orange.data.Type.Discrete, Orange.data.Type.Continuous]:
                self.colorCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
                
            if var.varType in [Orange.data.Type.String] and hasattr(self.graph, 'items') and self.graph_base.items() is not None and len(self.graph_base.items()) > 0:
                
                value = self.graph_base.items()[0][var].value
                
                # can value be a list?
                try:
                    if type(eval(value)) == type([]):
                        self.vertexSizeCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
                        continue
                except:
                    pass
                
                if len(value.split(',')) > 1:
                    self.vertexSizeCombo.addItem(self.icons.get(var.varType, self.icons[-1]), "num of " + unicode(var.name))
                
            elif var.varType in [Orange.data.Type.Continuous]:
                self.vertexSizeCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))

            self.nameComponentCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
            self.showComponentCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
            self.editCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
            self.comboAttSelection.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
        
        for var in edgeVars:
            if var.varType in [Orange.data.Type.Discrete, Orange.data.Type.Continuous]:
                self.edgeColorCombo.addItem(self.icons.get(var.varType, self.icons[-1]), unicode(var.name))
                
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
      
    def compute_network_info(self):
        self.nShown = self.graph.number_of_nodes()
        
        if self.graph.number_of_edges() > 0:
            self.verticesPerEdge = float(self.graph.number_of_nodes()) / float(self.graph.number_of_edges())
        else:
            self.verticesPerEdge = 0
            
        if self.graph.number_of_nodes() > 0:
            self.edgesPerVertex = float(self.graph.number_of_edges()) / float(self.graph.number_of_nodes())
        else:
            self.edgesPerVertex = 0
        
        undirected_graph = self.graph.to_undirected() if self.graph.is_directed() else self.graph
        components = Orange.network.nx.algorithms.components.connected_components(undirected_graph)
        if len(components) > 1:
            self.diameter = -1
        else:
            self.diameter = Orange.network.nx.algorithms.distance_measures.diameter(self.graph)
            
        if self.graph.is_multigraph():
            self.clustering_coefficient = -1
        else:
            self.clustering_coefficient = Orange.network.nx.algorithms.cluster.average_clustering(undirected_graph) * 100
      
    def change_graph(self, newgraph):
        self.information()
        
        # if graph has more nodes and edges than pixels in 1600x1200 display, 
        # it is too big to visualize!
        if newgraph.number_of_nodes() + newgraph.number_of_edges() > 50000:
            self.information('New graph is too big to visualize. Keeping the old graph.')
            return
        
        self.graph = newgraph
        
        self.number_of_nodes_label = self.graph.number_of_nodes()
        self.number_of_edges_label = self.graph.number_of_edges()
        
        if not self.networkCanvas.change_graph(self.graph):
            return
        
        self.compute_network_info()
        
        t = 1.13850193174e-008 * (self.graph.number_of_nodes()**2 + self.graph.number_of_edges())
        self.frSteps = int(2.0 / t)
        if self.frSteps <   1: self.frSteps = 1
        if self.frSteps > 100: self.frSteps = 100
#        
#        if self.frSteps < 10:
#            self.networkCanvas.use_antialiasing = 0
#            self.networkCanvas.use_animations = 0
#            self.minVertexSize = 5
#            self.maxVertexSize = 5
#            self.maxLinkSize = 1
#            self.optMethod = 0
#            self.graph_layout_method()
        
        self.set_vertex_size()
        self.setVertexColor()
        self.setEdgeColor()
            
        self.networkCanvas.setEdgesSize()
        self.clickedAttLstBox()
        self.clickedTooltipLstBox()
        self.clickedEdgeLabelListBox()
        
        self.networkCanvas.replot()
        qApp.processEvents()
        self.networkCanvas.networkCurve.fr(100, weighted=False, smooth_cooling=True)
        self.networkCanvas.networkCurve.update_properties()
        self.networkCanvas.replot()
          
        
        
    def set_graph_none(self):
        self.graph = None
        self.graph_base = None
        self.clearCombos()
        self.number_of_nodes_label = -1
        self.number_of_edges_label = -1
        self._items = None
        self._links = None
        self.set_items_distance_matrix(None)
        
    def set_graph(self, graph):
        self.information()
        self.error()
        
        if graph is None:
            self.set_graph_none()
            return
        
        self.graph_base = graph
        
        if self._network_view is not None:
            graph = self._network_view.init_network(graph)
        
        self.graph = graph
        
        # if graph has more nodes and edges than pixels in 1600x1200 display, 
        # it is too big to visualize!
        if self.graph.number_of_nodes() + self.graph.number_of_edges() > 50000:
            self.set_graph_none()
            self.error('Graph is too big to visualize. Try using one of the network views.')
            return
        
        if self.items_matrix is not None and self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.set_items_distance_matrix(None)
        
#        if self._items is not None and 'x' in self._items.domain and 'y' in self._items.domain:
#            positions = [(self._items[node]['x'].value, self._items[node]['y'].value) \
#                         for node in sorted(self.graph) if self._items[node]['x'].value != '?' \
#                         and self._items[node]['y'].value != '?']
#            self.layout.set_graph(self.graph, positions)
#        else:
#            self.layout.set_graph(self.graph)
        
        self.number_of_nodes_label = self.graph.number_of_nodes()
        self.number_of_edges_label = self.graph.number_of_edges()
        
        #self.networkCanvas.set_graph_layout(self.graph, self.layout, items=self.graph_base.items(), links=self.graph_base.links())
        self.networkCanvas.set_graph(self.graph, items=self.graph_base.items(), links=self.graph_base.links())
        
        self.networkCanvas.showEdgeLabels = self.showEdgeLabels
        self.networkCanvas.maxEdgeSize = self.maxLinkSize
        self.networkCanvas.minComponentEdgeWidth = self.minComponentEdgeWidth
        self.networkCanvas.maxComponentEdgeWidth = self.maxComponentEdgeWidth
        self.networkCanvas.set_labels_on_marked_only(self.labelsOnMarkedOnly)
        
        self.compute_network_info()
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

        t = 1.13850193174e-008 * (self.graph.number_of_nodes()**2 + self.graph.number_of_edges())
        self.frSteps = int(2.0 / t)
        if self.frSteps <   1: self.frSteps = 1
        if self.frSteps > 100: self.frSteps = 100
        
        # if graph is large, set random layout, min vertex size, min edge size
        if self.frSteps < 10:
            self.networkCanvas.update_antialiasing(False)
            self.networkCanvas.update_animations(False)
            self.minVertexSize = 5
            self.maxVertexSize = 5
            self.maxLinkSize = 1
            self.optMethod = 0
            self.graph_layout_method()            
        
        self.networkCanvas.labelsOnMarkedOnly = self.labelsOnMarkedOnly
        self.networkCanvas.showWeights = self.showWeights
            
        self.set_vertex_size()
        self.setVertexColor()
        self.setEdgeColor()
            
        self.networkCanvas.setEdgesSize()
        self.clickedAttLstBox()
        self.clickedTooltipLstBox()
        self.clickedEdgeLabelListBox()
        
        self.optButton.setChecked(1)
        self.graph_layout()        
        
    def set_network_view(self, nxView):
        if self._network_view is not None:
            QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)
            
        self._network_view = nxView
        
        if self._network_view is not None:
            self._network_view.set_nx_explorer(self)
        
        self.set_graph(self.graph_base)
        
        if self._network_view is not None:
            QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)
        
    def setItems(self, items=None):
        self.error('')
        
        if self.graph is None or items is None:
            return
        
        if len(items) != self.graph_base.number_of_nodes():
            self.error('ExampleTable items must have one example for each vertex.')
            return
        
        self.graph_base.set_items(items)
        
        self.set_vertex_size()
        self.networkCanvas.items = items
        self.networkCanvas.showWeights = self.showWeights
        self.networkCanvas.showEdgeLabels = self.showEdgeLabels
        self.setCombos()
        #self.networkCanvas.updateData()
        
    def markItems(self, items):
        self.markInputCombo.clear()
        self.markInputRadioButton.setEnabled(False)
        self.markInputItems = items
        
        if self.graph is None or self.graph_base.items() is None or items is None:
            return
        
        if len(items) > 0:
            lstOrgDomain = [x.name for x in self.graph_base.items().domain] + [self.graph_base.items().domain[x].name for x in self.graph_base.items().domain.getmetas()]
            lstNewDomain = [x.name for x in items.domain] + [items.domain[x].name for x in items.domain.getmetas()]
            commonVars = set(lstNewDomain) & set(lstOrgDomain)

            if len(commonVars) > 0:
                for var in commonVars:
                    orgVar = self.graph_base.items().domain[var]
                    mrkVar = items.domain[var]

                    if orgVar.varType == mrkVar.varType and orgVar.varType == Orange.data.Type.String:
                        self.markInputCombo.addItem(self.icons[orgVar.varType], unicode(orgVar.name))
                        self.markInputRadioButton.setEnabled(True)
                
                        self.set_mark_mode(9)
              
    def setExampleSubset(self, subset):
        if self.networkCanvas is None:
            return
        
        self.warning('')
        hiddenNodes = []
        
        if subset is not None:
            try:
                expected = 1
                for row in subset:
                    index = int(row['index'].value)
                    if expected != index:
                        hiddenNodes += range(expected-1, index-1)
                        expected = index + 1
                    else:
                        expected += 1
                        
                hiddenNodes += range(expected-1, self.graph.number_of_nodes())
                
                self.networkCanvas.setHiddenNodes(hiddenNodes)
            except:
                self.warning('"index" attribute does not exist in "items" table.')
                    
    def showDegreeDistribution(self):
        if self.graph is None:
            return
        
        from matplotlib import rcParams
        import pylab as p
        
        x = self.graph.degree().values()
        nbins = len(set(x))
        if nbins > 500:
          bbins = 500
        
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
            self.networkCanvas.contPalette = dlg.getContinuousPalette("contPalette")
            self.networkCanvas.discPalette = dlg.getDiscretePalette("discPalette")
            
            self.setVertexColor()
            
    def setEdgeColorPalette(self):
        dlg = self.createColorDialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
        if dlg.exec_():
            self.edgeColorSettings = dlg.getColorSchemas()
            self.selectedEdgeSchemaIndex = dlg.selectedSchemaIndex
            self.networkCanvas.contEdgePalette = dlg.getContinuousPalette("contPalette")
            self.networkCanvas.discEdgePalette = dlg.getDiscretePalette("discPalette")
            
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
    
    def graph_layout(self):
        if self.graph is None:   #grafa se ni
            self.optButton.setChecked(False)
            return
        
        if not self.optButton.isChecked() and not self.optMethod == 2:
            self.optButton.setChecked(False)
            return
        
        qApp.processEvents()
            
        if self.optMethod == 1:
            self.networkCanvas.networkCurve.random()
        elif self.optMethod == 2:
            self.graph_layout_fr(False)
        elif self.optMethod == 3:
            self.graph_layout_fr(True)
        elif self.optMethod == 4:
            self.graph_layout_fr_radial()
        elif self.optMethod == 5:
            self.networkCanvas.networkCurve.circular(NetworkCurve.circular_crossing)
        elif self.optMethod == 6:
            self.networkCanvas.networkCurve.circular(NetworkCurve.circular_original)
        elif self.optMethod == 7:
            self.networkCanvas.networkCurve.circular(NetworkCurve.circular_random)
        elif self.optMethod == 8:
            self.graph_layout_pivot_mds()
            
        self.optButton.setChecked(False)
#        self.networkCanvas.networkCurve.coors = self.layout.map_to_graph(self.graph) 
        #self.networkCanvas.updateCanvas()
        self.networkCanvas.networkCurve.update_properties()
        self.networkCanvas.replot()
        qApp.processEvents()
        
    def graph_layout_method(self, method=None):
        self.information()
        self.stepsSpin.label.setText('Iterations: ')
        
        if method is not None:
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
            
            if self.items_matrix is None:
                self.information('Set distance matrix to input signal')
                return
            if self.graph is None:
                self.information('No network found')
                return
            if self.items_matrix.dim != self.graph.number_of_nodes():
                self.error('Distance matrix dimensionality must equal number of vertices')
                return
        else:
            self.stepsSpin.setEnabled(False)
            self.optButton.setChecked(True)
            self.graph_layout()
        
    def graph_layout_fr(self, weighted):
        if self.graph is None:
            return
              
        if not self.optButton.isChecked():
            print 'stop opt'
            self.networkCanvas.networkCurve.stop_optimization()
            self.optButton.setChecked(False)
            self.optButton.setText("Optimize layout")
            return
        
        self.optButton.setText("Stop")
        qApp.processEvents()
        self.networkCanvas.networkCurve.fr(self.frSteps, False)
        self.networkCanvas.update_canvas()
        self.optButton.setChecked(False)
        self.optButton.setText("Optimize layout")
                
    def graph_layout_fr_radial(self):
        if self.graph is None:   #grafa se ni
            return
        
#        #print "F-R Radial"
#        k = 1.13850193174e-008
#        nodes = self.graph.number_of_nodes()
#        t = k * nodes * nodes
#        refreshRate = int(5.0 / t)
#        if refreshRate <    1: refreshRate = 1
#        if refreshRate > 1500: refreshRate = 1500
#        #print "refreshRate: " + str(refreshRate)
#        
#        tolerance = 5
#        initTemp = 100
#        centerNdx = 0
#        
#        selection = self.networkCanvas.getSelection()
#        if len(selection) > 0:
#            centerNdx = selection[0]
#            
#        #print "center ndx: " + str(centerNdx)
#        initTemp = self.layout.fr_radial(centerNdx, refreshRate, initTemp)
#        self.networkCanvas.circles = [10000 / 12, 10000/12*2, 10000/12*3]#, 10000/12*4, 10000/12*5]
#        #self.networkCanvas.circles = [100, 200, 300]
#        self.networkCanvas.updateCanvas()
#        self.networkCanvas.circles = []
            
    def graph_layout_pivot_mds(self):
        self.information()
        
        if self.items_matrix is None:
            self.information('Set distance matrix to input signal')
            return
        
        if self.graph_base is None:
            self.information('No network found')
            return
        
        if self.items_matrix.dim != self.graph_base.number_of_nodes():
            self.error('The number of vertices does not match matrix size.')
            return
        
        self.frSteps = min(self.frSteps, self.graph.number_of_nodes())
        qApp.processEvents()
        
        if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
            matrix = self.items_matrix
        else:
            matrix = self.items_matrix.get_items(sorted(self.graph.nodes()))
        
        mds = orngMDS.PivotMDS(matrix, self.frSteps)
        x,y = mds.optimize()
        xy = zip(list(x), list(y))
        coors = dict(zip(sorted(self.graph.nodes()), xy))
        self.networkCanvas.networkCurve.set_node_coordinates(coors)
        self.networkCanvas.update_canvas()

    """
    Network Visualization
    """
       
    def clickedAttLstBox(self):
        if self.graph is None:
            return
        
        self.lastLabelColumns = [self.attributes[i][0] for i in self.markerAttributes]
        self.networkCanvas.set_label_attributes(self.lastLabelColumns)
        self.networkCanvas.replot()
  
    def clickedTooltipLstBox(self):
        if self.graph is None:
            return
        
        self.lastTooltipColumns = [self.attributes[i][0] for i in self.tooltipAttributes]
        self.networkCanvas.set_tooltip_attributes(self.lastTooltipColumns)
        self.networkCanvas.replot()
        
    def clickedEdgeLabelListBox(self):
        if self.graph is None:
            return
        
        self.lastEdgeLabelAttributes = set([self.edgeAttributes[i][0] for i in self.edgeLabelAttributes])
        self.networkCanvas.setEdgeLabelText(self.lastEdgeLabelAttributes)
        #self.networkCanvas.updateData()
        self.networkCanvas.replot()

    def setVertexColor(self):
        if self.graph is None:
            return
        
        self.networkCanvas.set_node_colors(self.colorCombo.currentText())
        self.lastColorColumn = self.colorCombo.currentText()
        
    def setEdgeColor(self):
        if self.graph is None:
            return
        
        self.networkCanvas.set_edge_color(self.edgeColorCombo.currentText())
        self.lastEdgeColorColumn = self.edgeColorCombo.currentText()
                  
    def setGraphGrid(self):
        self.networkCanvas.enableGridY(self.networkCanvasShowGrid)
        self.networkCanvas.enableGridX(self.networkCanvasShowGrid)
    
    def selectAllConnectedNodes(self):
        self.networkCanvas.selectConnectedNodes(1000000)
                
    def setMaxLinkSize(self):
        if self.graph is None:
            return
        
        self.networkCanvas.maxEdgeSize = self.maxLinkSize
        self.networkCanvas.setEdgesSize()
        self.networkCanvas.replot()
    
    def set_vertex_size(self):
        if self.graph is None or self.networkCanvas is None:
            return
        
        if self.minVertexSize > self.maxVertexSize:
            self.maxVertexSize = self.minVertexSize
        
        items = self.graph_base.items()
        
        if items is None:
            self.networkCanvas.networkCurve.set_node_sizes({}, min_size=self.minVertexSize, max_size=self.maxVertexSize)
            return
        
        self.lastVertexSizeColumn = self.vertexSizeCombo.currentText()
        column = str(self.vertexSizeCombo.currentText())
        
        values = {}
        if column in items.domain or (column.startswith("num of ") and column.replace("num of ", "") in items.domain):
            if column in items.domain:
                values = dict((x, items[x][column].value) for x in self.graph if not items[x][column].isSpecial())
            else:
                values = dict((x, len(items[x][column.replace("num of ", "")].value.split(','))) for x in self.graph)
        
        if len(values) == 0:
            values = dict((node, 1.) for node in self.graph)
            
        if self.invertSize:
            maxval = max(values.itervalues())
            values.update((key, maxval-val) for key, val in values.iteritems())
            self.networkCanvas.networkCurve.set_node_sizes(values, min_size=self.minVertexSize, max_size=self.maxVertexSize)
        else:
            self.networkCanvas.networkCurve.set_node_sizes(values, min_size=self.minVertexSize, max_size=self.maxVertexSize)
        
        self.networkCanvas.replot()
        
    def setFontSize(self):
        if self.networkCanvas is None:
            return
        
        font = self.networkCanvas.font()
        font.setPointSize(self.fontSize)
        self.networkCanvas.setFont(font) 
        
                
    def sendReport(self):
        self.reportSettings("Graph data",
                            [("Number of vertices", self.graph.number_of_nodes()),
                             ("Number of edges", self.graph.number_of_edges()),
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
        self.reportImage(self.networkCanvas.saveToFileDirect)        
        
        
if __name__=="__main__":    
    a=QApplication(sys.argv)
    ow=OWNxExplorerQt()
    ow.show()
    root = 'c:\\Users\\miha\\Projects\\res\\Orange\\test\\'
    #net = Orange.network.readwrite.read(root + 'airlines_4.net')
    #net.set_items(Orange.data.Table(root + 'airlines_4.tab'))
    #net = Orange.network.readwrite.read(root + 'K4K2.net')
    net = Orange.network.readwrite.read(root + 'K4K4K5_.net')
    ow.set_graph(net)
    #ow.handleNewSignals()
    #import OWNetExplorer
    #ow1=OWNetExplorer.OWNetExplorer()
    #ow1.show()
    #net1 = Orange.network.Network.read(root + 'K4K4K5_.net')
    #ow1.setGraph(net1)
    a.exec_()
    #save settings
    ow.saveSettings()
    #ow1.saveSettings()
    
    
    

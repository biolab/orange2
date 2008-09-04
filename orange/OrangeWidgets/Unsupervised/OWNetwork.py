"""
<name>Network</name>
<description>Network Widget visualizes graphs.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3200</priority>
"""
from OWWidget import *

import OWGUI, OWColorPalette
from OWNetworkCanvas import *
from orngNetwork import * 
from time import *
import OWToolbars
from statc import mean
from orangeom import Network
from operator import itemgetter

dir = os.path.dirname(__file__) + "/../icons/"
dlg_mark2sel = dir + "Dlg_Mark2Sel.png"
dlg_sel2mark = dir + "Dlg_Sel2Mark.png"
dlg_selIsmark = dir + "Dlg_SelisMark.png"
dlg_selected = dir + "Dlg_SelectedNodes.png"
dlg_unselected = dir + "Dlg_UnselectedNodes.png"
dlg_showall = dir + "Dlg_clear.png"

class OWNetwork(OWWidget):
    settingsList = ["autoSendSelection", 
                    "spinExplicit", 
                    "spinPercentage",
                    "maxLinkSize",
                    "maxVertexSize",
                    "renderAntialiased",
                    "labelsOnMarkedOnly",
                    "invertSize",
                    "optMethod",
                    "lastVertexSizeColumn",
                    "lastColorColumn",
                    "lastNameComponentAttribute",
                    "lastLabelColumns",
                    "lastTooltipColumns",
                    "showWeights",
                    "showIndexes", 
                    "showEdgeLabels", 
                    "colorSettings", 
                    "selectedSchemaIndex",
                    "edgeColorSettings",
                    "selectedEdgeSchemaIndex"] 
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Network')
        
        #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="markerAttributes"), ContextField("attributes", selected="tooltipAttributes"), "color"])}
        self.inputs = [("Network", Network, self.setGraph, Default), ("Example Subset", orange.ExampleTable, self.setExampleSubset), ("Mark Items", orange.ExampleTable, self.markItems), ("Add Items", orange.ExampleTable, self.setItems)]
        self.outputs = [("Selected Network", Network), ("Selected Examples", ExampleTable), ("Marked Examples", ExampleTable)]
        
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
        self.edgeColorSettings = None
        self.selectedEdgeSchemaIndex = 0
        self.loadSettings()
        
        self.visualize = None
        self.markInputItems = None
        
        self.graph = OWNetworkCanvas(self, self.mainArea, "Network")
        self.mainArea.layout().addWidget(self.graph)
        
        self.graph.maxLinkSize = self.maxLinkSize
        self.graph.maxVertexSize = self.maxVertexSize
        
        self.hcontroArea = OWGUI.widgetBox(self.controlArea, orientation='horizontal')
        
        self.tabs = OWGUI.tabWidget(self.hcontroArea)
        self.displayTab = OWGUI.createTabPage(self.tabs, "Display")
        self.markTab = OWGUI.createTabPage(self.tabs, "Mark")
        self.infoTab = OWGUI.createTabPage(self.tabs, "Info")
        self.settingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        self.optimizeBox = OWGUI.radioButtonsInBox(self.displayTab, self, "optimizeWhat", [], "Optimize", addSpace=False)
        
        OWGUI.label(self.optimizeBox, self, "Select layout optimization method:")
        
        self.optCombo = OWGUI.comboBox(self.optimizeBox, self, "optMethod", callback=self.setOptMethod)
        self.optCombo.addItem("No optimization")
        self.optCombo.addItem("Random")
        self.optCombo.addItem("Fruchterman Reingold")
        self.optCombo.addItem("Fruchterman Reingold Weighted")
        self.optCombo.addItem("Fruchterman Reingold Radial")
        self.optCombo.addItem("Circular Crossing Reduction")
        self.optCombo.addItem("Circular Original")
        self.optCombo.addItem("Circular Random")
        self.optCombo.setCurrentIndex(self.optMethod)
        self.stepsSpin = OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 10000, 1, label="Iterations: ")
        self.stepsSpin.setEnabled(False)
        
        self.optButton = OWGUI.button(self.optimizeBox, self, "Optimize layout", callback=self.optLayout, toggleButton=1)
        
        colorBox = OWGUI.widgetBox(self.displayTab, "Color attribute", addSpace = False)
        self.colorCombo = OWGUI.comboBox(colorBox, self, "color", callback=self.setVertexColor)
        self.colorCombo.addItem("(vertex color attribute)")
        
        self.edgeColorCombo = OWGUI.comboBox(colorBox, self, "edgeColor", callback=self.setEdgeColor)
        self.edgeColorCombo.addItem("(edge color attribute)")
        
        self.attBox = OWGUI.widgetBox(self.displayTab, "Labels", addSpace = False)
        self.attListBox = OWGUI.listBox(self.attBox, self, "markerAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedAttLstBox)
        
        self.tooltipBox = OWGUI.widgetBox(self.displayTab, "Tooltips", addSpace = False)  
        self.tooltipListBox = OWGUI.listBox(self.tooltipBox, self, "tooltipAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedTooltipLstBox)
        
        self.edgeLabelBox = OWGUI.widgetBox(self.displayTab, "Labels on edges", addSpace = False)
        self.edgeLabelListBox = OWGUI.listBox(self.edgeLabelBox, self, "edgeLabelAttributes", "edgeAttributes", selectionMode=QListWidget.MultiSelection, callback=self.clickedEdgeLabelListBox)
        self.edgeLabelBox.setEnabled(False)
        
        ib = OWGUI.widgetBox(self.settingsTab, "General", orientation="vertical")
        OWGUI.checkBox(ib, self, 'showIndexes', 'Show indexes', callback = self.showIndexLabels)
        OWGUI.checkBox(ib, self, 'showWeights', 'Show weights', callback = self.showWeightLabels)
        OWGUI.checkBox(ib, self, 'showEdgeLabels', 'Show labels on edges', callback = self.showEdgeLabelsClick)
        OWGUI.checkBox(ib, self, 'labelsOnMarkedOnly', 'Show labels on marked nodes only', callback = self.labelsOnMarked)
        OWGUI.spin(ib, self, "maxLinkSize", 1, 50, 1, label="Max link size:", callback = self.setMaxLinkSize)
        OWGUI.checkBox(ib, self, 'renderAntialiased', 'Render antialiased', callback = self.setRenderAntialiased)
        self.insideView = 0
        self.insideViewNeighbours = 2
        OWGUI.spin(ib, self, "insideViewNeighbours", 1, 6, 1, label="Inside view (neighbours): ", checked = "insideView", checkCallback = self.insideview, callback = self.insideviewneighbours)
        
        self.vertexSizeCombo = OWGUI.comboBox(self.settingsTab, self, "vertexSize", box = "Vertex size attribute", callback=self.setVertexSize)
        self.vertexSizeCombo.addItem("(none)")
        
        OWGUI.spin(self.vertexSizeCombo.box, self, "maxVertexSize", 5, 50, 1, label="Max vertex size:", callback = self.setVertexSize)
        
        OWGUI.checkBox(self.vertexSizeCombo.box, self, "invertSize", "Invert vertex size", callback = self.setVertexSize)
         
        self.checkSendMarkedNodes = 0
        OWGUI.checkBox(self.displayTab, self, 'checkSendMarkedNodes', 'Send marked nodes', callback = self.setSendMarkedNodes)
        
        OWGUI.separator(self.displayTab)

        OWGUI.button(self.displayTab, self, "Save network", callback=self.saveNetwork)
        
        ib = OWGUI.widgetBox(self.markTab, "Info", orientation="vertical")
        OWGUI.label(ib, self, "Vertices (shown/hidden): %(nVertices)i (%(nShown)i/%(nHidden)i)")
        OWGUI.label(ib, self, "Selected and marked vertices: %(nSelected)i - %(nMarked)i")
        
        ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Method", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "None", callback = self.setMarkMode)
        OWGUI.appendRadioButton(ribg, self, "hubs", "Find vertices which label contain", callback = self.setMarkMode)
        self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self.setSearchStringTimer, callbackOnType=True)
        self.searchStringTimer = QTimer(self)
        self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.setMarkMode)
        
        OWGUI.appendRadioButton(ribg, self, "hubs", "Mark neighbours of focused vertex", callback = self.setMarkMode)
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
#        self.ctrlMarkProportion = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markProportion", 0, 100, 1, label="Percentage" + ": ", callback=self.setHubs)
        
        #self.markInputBox = OWGUI.widgetBox(self.markTab, "Mark by input signal", orientation="vertical")
        #self.markInputBox.setEnabled(False)
        self.markInputRadioButton = OWGUI.appendRadioButton(ribg, self, "hubs", "Mark vertices given in the input signal", callback = self.setMarkMode)
        ib = OWGUI.indentedBox(ribg)
        self.markInput = 0
        self.markInputCombo = OWGUI.comboBox(ib, self, "markInput", callback=(lambda h=9: self.setMarkMode(h)))
        self.markInputRadioButton.setEnabled(False)
        
        T = OWToolbars.NavigateSelectToolbar
        self.zoomSelectToolbar = T(self, self.hcontroArea, self.graph, self.autoSendSelection,
                                  buttons = (T.IconZoom, T.IconZoomExtent, T.IconZoomSelection, ("", "", "", None, None, 0, "navigate"), T.IconPan, 
                                             ("Move selection", "buttonMoveSelection", "activateMoveSelection", QIcon(OWToolbars.dlg_select), Qt.ArrowCursor, 1, "select"),
                                             T.IconRectangle, T.IconPolygon, ("", "", "", None, None, 0, "select"), T.IconSendSelection))
        
        ib = OWGUI.widgetBox(self.zoomSelectToolbar, "Inv", orientation="vertical")
        btnM2S = OWGUI.button(ib, self, "", callback = self.markedToSelection)
        btnM2S.setIcon(QIcon(dlg_mark2sel))
        btnM2S.setToolTip("Add Marked to Selection")
        btnS2M = OWGUI.button(ib, self, "",callback = self.markedFromSelection)
        btnS2M.setIcon(QIcon(dlg_sel2mark))
        btnS2M.setToolTip("Remove Marked from Selection")
        btnSIM = OWGUI.button(ib, self, "", callback = self.setSelectionToMarked)
        btnSIM.setIcon(QIcon(dlg_selIsmark))
        btnSIM.setToolTip("Set Selection to Marked")
        
        self.hideBox = OWGUI.widgetBox(self.zoomSelectToolbar, "Hide", orientation="vertical")
        btnSEL = OWGUI.button(self.hideBox, self, "", callback=self.hideSelected)
        btnSEL.setIcon(QIcon(dlg_selected))
        btnSEL.setToolTip("Hide selected")
        btnUN = OWGUI.button(self.hideBox, self, "", callback=self.hideAllButSelected)
        btnUN.setIcon(QIcon(dlg_unselected))
        btnUN.setToolTip("Hide unselected")
        btnSW = OWGUI.button(self.hideBox, self, "", callback=self.showAllNodes)
        btnSW.setIcon(QIcon(dlg_showall))
        btnSW.setToolTip("Show all nodes")
        
        OWGUI.rubber(self.zoomSelectToolbar)
        
        ib = OWGUI.widgetBox(self.infoTab, "General")
        OWGUI.label(ib, self, "Number of vertices: %(nVertices)i")
        OWGUI.label(ib, self, "Number of edges: %(nEdges)i")
        OWGUI.label(ib, self, "Vertices per edge: %(verticesPerEdge).2f")
        OWGUI.label(ib, self, "Edges per vertex: %(edgesPerVertex).2f")
        OWGUI.label(ib, self, "Diameter: %(diameter)i")
        OWGUI.label(ib, self, "Clustering Coefficient: %(clustering_coefficient).1f%%")
        
        OWGUI.button(self.infoTab, self, "Show degree distribution", callback=self.showDegreeDistribution)
        
        
        #OWGUI.button(self.settingsTab, self, "Clustering", callback=self.clustering)
        
        self.colorButtonsBox = OWGUI.widgetBox(self.settingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(self.colorButtonsBox, self, "Set vertex colors", self.setColors, tooltip = "Set the canvas background color, grid color and color palette for coloring continuous variables", debuggingEnabled = 0)
        OWGUI.button(self.colorButtonsBox, self, "Set edge colors", self.setEdgeColorPalette, tooltip = "Set edge color and color palette for coloring continuous variables", debuggingEnabled = 0)

        ib = OWGUI.widgetBox(self.settingsTab, "Prototype")
        OWGUI.button(ib, self, "Collapse", callback=self.collapse)
        OWGUI.label(ib, self, "Name components:")
        self.nameComponentAttribute = 0
        self.nameComponentCombo = OWGUI.comboBox(ib, self, "nameComponentAttribute", callback=self.nameComponents)
        self.nameComponentCombo.addItem("Select attribute")
        
        OWGUI.label(ib, self, "Show labels on components:")
        self.showComponentAttribute = 0
        self.showComponentCombo = OWGUI.comboBox(ib, self, "showComponentAttribute", callback=self.showComponents)
        self.showComponentCombo.addItem("Select attribute")
        
        self.icons = self.createAttributeIconDict()
        self.setMarkMode()
        
        self.displayTab.layout().addStretch(1)
        self.markTab.layout().addStretch(1)
        self.infoTab.layout().addStretch(1)
        self.settingsTab.layout().addStretch(1)
        
        dlg = self.createColorDialog(self.colorSettings, self.selectedSchemaIndex)
        self.graph.contPalette = dlg.getContinuousPalette("contPalette")
        self.graph.discPalette = dlg.getDiscretePalette("discPalette")
        
        dlg = self.createColorDialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
        self.graph.contEdgePalette = dlg.getContinuousPalette("contPalette")
        self.graph.discEdgePalette = dlg.getDiscretePalette("discPalette")
        
        self.setOptMethod()
         
        self.resize(1000, 600)
        self.controlArea.setEnabled(False)
        self.information('No network loaded.')

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
            self.fr()
        
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
                self.fr()
    
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
        if self.visualize == None or self.visualize.graph == None or self.visualize.graph.items == None:
            return
        
        vars = [x.name for x in self.visualize.getVars()]
        
        if not self.nameComponentCombo.currentText() in vars:
            return
        
        components = self.visualize.graph.getConnectedComponents()
        keyword_table = orange.ExampleTable(orange.Domain(orange.StringVariable('keyword')), [[''] for i in range(len(self.visualize.graph.items))]) 
        
        excludeWord = ["AND", "OF"]
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
                keyword_table[vertex]['keyword'] = keyword
                
            if keyword in keywords:
                sameKeywords.add(keyword)
            else:
                keywords.add(keyword)
        #print "sameKeywords:", sameKeywords       
        sameComponents = [component for component in components if str(keyword_table[component[0]]['keyword']) in sameKeywords]
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
                while True:
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
                keyword_table[vertex]['keyword'] = keyword
        
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
        self.searchStringTimer.start(750)
         
    def setMarkMode(self, i = None):
        if not i is None:
            self.hubs = i
        
        print self.hubs
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
            toMark = [i for i, values in enumerate(vgraph.items) if txt in " ".join([str(values[ndx]) for ndx in labelText])]
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
            
            self.graph.visualizer.saveNetwork(fn)
                    
    def sendData(self):
        graph = self.graph.getSelectedGraph()
        
        if graph != None:
            if graph.items != None:
                self.send("Selected Examples", graph.items)
            else:
                self.send("Selected Examples", self.graph.getSelectedExamples())
                
            self.send("Selected Network", graph)
        else:
            items = self.graph.getSelectedExamples()
            if items != None:
                self.send("Selected Examples", items)
                
    def setCombos(self):
        vars = self.visualize.getVars()
        edgeVars = self.visualize.getEdgeVars()
        lastLabelColumns = self.lastLabelColumns
        lastTooltipColumns = self.lastTooltipColumns
        self.attributes = [(var.name, var.varType) for var in vars]
        self.edgeAttributes = [(var.name, var.varType) for var in edgeVars]
        
        self.colorCombo.clear()
        self.vertexSizeCombo.clear()
        self.nameComponentCombo.clear()
        self.showComponentCombo.clear()
        self.edgeColorCombo.clear()
        
        self.colorCombo.addItem("(vertex color attribute)")
        self.edgeColorCombo.addItem("(edge color attribute)")
        self.vertexSizeCombo.addItem("(same size)")
        self.nameComponentCombo.addItem("Select attribute")
        self.showComponentCombo.addItem("Select attribute")
        
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
      
    def setGraph(self, graph):        
        if graph == None:
            self.visualize = None
            self.graph.addVisualizer(self.visualize)
            self.information('No network loaded.')
            self.controlArea.setEnabled(False)
            return
        
        #print "OWNetwork/setGraph: new visualizer..."
        self.visualize = NetworkOptimization(graph)
        
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
        self.controlArea.setEnabled(True)
        self.updateCanvas()
        
    def setItems(self, items=None):
        self.error('')
        
        if self.visualize == None or self.visualize.graph == None or items == None:
            return
        
        if len(items) != self.visualize.graph.nVertices:
            self.error('ExampleTable items must have one example for each vertex.')
            return
        
        self.visualize.graph.setattr("items", items)
        #self.setGraph(self.visualize.graph)
        self.graph.updateData()
        self.setVertexSize()
        self.showIndexLabels()
        self.showWeightLabels()
        self.showEdgeLabelsClick()
        
        self.setCombos()
        
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
            lstOrgDomain = [x.name for x in self.visualize.graph.items.domain]
            lstNewDomain = [x.name for x in items.domain]
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
        
    def hideSelected(self):
        self.graph.hideSelectedVertices()
                
    def hideAllButSelected(self):
        self.graph.hideUnSelectedVertices()
      
    def showAllNodes(self):
        self.graph.showAllVertices()
                               
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
        if not self.optButton.isChecked():
            return
            
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
            
        self.optButton.setChecked(False)
    
    def setOptMethod(self):
        if str(self.optMethod) == '0':
            self.optButton.setEnabled(False)
        else:
            self.optButton.setEnabled(True)
            
        if str(self.optMethod) == '2' or str(self.optMethod) == '3':
            self.stepsSpin.setEnabled(True)
        else:
            self.stepsSpin.setEnabled(False)

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
        coolFactor = exp(log(10.0/10000.0) / self.frSteps)

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
        steps = 100
        initTemp = 1000
        coolFactor = exp(log(10.0/10000.0) / steps)
        oldXY = []
        for rec in self.visualize.network.coors:
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
      
    """
    Network Visualization
    """
       
    def clickedAttLstBox(self):
        self.lastLabelColumns = set([self.attributes[i][0] for i in self.markerAttributes])
        self.graph.setLabelText(self.lastLabelColumns)
        self.graph.updateData()
        self.graph.replot()
  
    def clickedTooltipLstBox(self):
        self.lastTooltipColumns = set([self.attributes[i][0] for i in self.tooltipAttributes])
        self.graph.setTooltipText(self.lastTooltipColumns)
        self.graph.updateData()
        self.graph.replot()
        
    def clickedEdgeLabelListBox(self):
        self.lastEdgeLabelAttributes = set([self.edgeAttributes[i][0] for i in self.edgeLabelAttributes])
        self.graph.setEdgeLabelText(self.lastEdgeLabelAttributes)
        self.graph.updateData()
        self.graph.replot()

    def setVertexColor(self):
        self.graph.setVertexColor(self.colorCombo.currentText())
        self.lastColorColumn = self.colorCombo.currentText()
        self.graph.updateData()
        self.graph.replot()
        
    def setEdgeColor(self):
        self.graph.setEdgeColor(self.edgeColorCombo.currentText())
        self.lastEdgeColorColumn = self.edgeColorCombo.currentText()
        self.graph.updateData()
        self.graph.replot()
                  
    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)
    
    def markedToSelection(self):
        self.graph.markedToSelection()
      
    def markedFromSelection(self):
        self.graph.selectionToMarked()
    
    def setSelectionToMarked(self):
        self.graph.removeSelection(False)
        self.graph.markedToSelection()
    
    def selectAllConnectedNodes(self):
        self.graph.selectConnectedNodes(1000000)
        
    def setMaxLinkSize(self):
        self.graph.maxEdgeSize = self.maxLinkSize
        self.graph.setEdgesSize()
        self.graph.replot()
    
    def setVertexSize(self):
        self.graph.maxVertexSize = self.maxVertexSize
        self.lastVertexSizeColumn = self.vertexSizeCombo.currentText()
        
        if self.vertexSize > 0:
            self.graph.setVerticesSize(self.lastVertexSizeColumn, self.invertSize)
        else:
            self.graph.setVerticesSize()
            
        self.graph.replot()
        
    def setRenderAntialiased(self):
        self.graph.renderAntialiased = self.renderAntialiased
        self.graph.updateData()
        self.graph.replot()
        
if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetwork()
    ow.show()
    appl.exec_()
    

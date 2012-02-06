"""
<name>Net Explorer</name>
<description>Orange widget for network exploration.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>6420</priority>
"""
import math
import operator
import statc
import time

import Orange
import OWGUI
import OWColorPalette
import orngMDS

from OWWidget import *
from operator import itemgetter

try:
    from OWNxCanvasQt import *

    class OWNxExplorer(OWWidget):

        settingsList = ["autoSendSelection", "spinExplicit", "spinPercentage",
        "maxLinkSize", "minVertexSize", "maxVertexSize", "networkCanvas.animate_plot",
        "networkCanvas.animate_points", "networkCanvas.antialias_plot",
        "networkCanvas.antialias_points", "networkCanvas.antialias_lines",
        "networkCanvas.auto_adjust_performance", "invertSize", "optMethod",
        "lastVertexSizeColumn", "lastColorColumn", "networkCanvas.show_indices", "networkCanvas.show_weights",
        "lastNameComponentAttribute", "lastLabelColumns", "lastTooltipColumns",
        "showWeights", "showEdgeLabels", "colorSettings",
        "selectedSchemaIndex", "edgeColorSettings", "selectedEdgeSchemaIndex",
        "showMissingValues", "fontSize", "mdsTorgerson", "mdsAvgLinkage",
        "mdsSteps", "mdsRefresh", "mdsStressDelta", "organism", "showTextMiningInfo",
        "toolbarSelection", "minComponentEdgeWidth", "maxComponentEdgeWidth",
        "mdsFromCurrentPos", "labelsOnMarkedOnly", "tabIndex",
        "networkCanvas.trim_label_words", "opt_from_curr", "networkCanvas.explore_distances",
        "networkCanvas.show_component_distances", "fontWeight", "networkCanvas.state",
        "networkCanvas.selection_behavior"]

        def __init__(self, parent=None, signalManager=None, name='Net Explorer',
                     NetworkCanvas=OWNxCanvas):
            OWWidget.__init__(self, parent, signalManager, name, noReport=True)
            #self.contextHandlers = {"": DomainContextHandler("", [ContextField("attributes", selected="markerAttributes"), ContextField("attributes", selected="tooltipAttributes"), "color"])}
            self.inputs = [("Network", Orange.network.Graph, self.set_graph, Default),
                           ("Items", Orange.data.Table, self.set_items),
                           ("Item Subset", Orange.data.Table, self.mark_items),
                           ("Distances", Orange.core.SymMatrix, self.set_items_distance_matrix),
                           ("Net View", Orange.network.NxView, self.set_network_view)]

            self.outputs = [("Selected Network", Orange.network.Graph),
                            ("Distance Matrix", Orange.core.SymMatrix),
                            ("Marked Items", Orange.data.Table),
                            ("Selected Items", Orange.data.Table),
                            ("Other Items", Orange.data.Table)]
                            #("Attribute Selection List", AttributeList)]

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
            self.nShown = self.nHidden = self.nMarked = self.nSelected = self.verticesPerEdge = self.edgesPerVertex = 0
            self.optimizeWhat = 1
            self.maxLinkSize = 3
            self.maxVertexSize = 7
            self.minVertexSize = 12
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
            self.fontWeight = 1
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
            self.opt_from_curr = False

            self.checkSendMarkedNodes = True
            self.checkSendSelectedNodes = True
            self.explore_distances = False

            self.loadSettings()

            self._network_view = None
            self.graph = None
            self.graph_base = None
            self.markInputItems = None

            # if optimization method is set to FragViz, set it to FR
            if self.optMethod == 9:
                self.optMethod = 3

            self.networkCanvas.showMissingValues = self.showMissingValues
            self.mainArea.layout().addWidget(self.networkCanvas)

            self.networkCanvas.maxLinkSize = self.maxLinkSize

            self.tabs = OWGUI.tabWidget(self.controlArea)

            self.verticesTab = OWGUI.createTabPage(self.tabs, "Nodes")
            self.edgesTab = OWGUI.createTabPage(self.tabs, "Edges")
            self.markTab = OWGUI.createTabPage(self.tabs, "Mark")
            self.infoTab = OWGUI.createTabPage(self.tabs, "Info")
            self.performanceTab = OWGUI.createTabPage(self.tabs, "Performance")

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
            self.optCombo.addItem("FragViz")
            self.optCombo.addItem("MDS")
            self.optCombo.addItem("Pivot MDS")
            self.optCombo.setCurrentIndex(self.optMethod)
            self.stepsSpin = OWGUI.spin(self.optimizeBox, self, "frSteps", 1, 100000, 1, label="Iterations: ")
            self.cb_opt_from_curr = OWGUI.checkBox(self.optimizeBox, self, "opt_from_curr", label="Optimize from current positions")
            self.cb_opt_from_curr.setEnabled(False)
            self.stepsSpin.setEnabled(False)

            self.optButton = OWGUI.button(self.optimizeBox, self, "Optimize layout", callback=self.graph_layout, toggleButton=1)

            colorBox = OWGUI.widgetBox(self.verticesTab, "Node color attribute", orientation="horizontal", addSpace=False)
            self.colorCombo = OWGUI.comboBox(colorBox, self, "color", callback=self.set_node_colors)
            self.colorCombo.addItem("(same color)")
            OWGUI.button(colorBox, self, "palette", self._set_colors, tooltip="Set node color palette", width=60, debuggingEnabled=0)

            ib = OWGUI.widgetBox(self.verticesTab, "Node size attribute", orientation="vertical", addSpace=False)
            hb = OWGUI.widgetBox(ib, orientation="horizontal", addSpace=False)
            OWGUI.checkBox(hb, self, "invertSize", "Invert size", callback=self.set_node_sizes)
            OWGUI.spin(hb, self, "minVertexSize", 5, 200, 1, label="Min:", callback=self.set_node_sizes)
            OWGUI.spin(hb, self, "maxVertexSize", 5, 200, 1, label="Max:", callback=self.set_node_sizes)
            self.vertexSizeCombo = OWGUI.comboBox(ib, self, "vertexSize", callback=self.set_node_sizes)
            self.vertexSizeCombo.addItem("(none)")

            self.attBox = OWGUI.widgetBox(self.verticesTab, "Node labels | tooltips", orientation="vertical", addSpace=False)
            hb = OWGUI.widgetBox(self.attBox, orientation="horizontal", addSpace=False)
            self.attListBox = OWGUI.listBox(hb, self, "markerAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self._clicked_att_lstbox)
            self.tooltipListBox = OWGUI.listBox(hb, self, "tooltipAttributes", "attributes", selectionMode=QListWidget.MultiSelection, callback=self._clicked_tooltip_lstbox)
            OWGUI.spin(self.attBox, self, "networkCanvas.trim_label_words", 0, 5, 1, label='Trim label words to', callback=self._clicked_att_lstbox)

            ib = OWGUI.widgetBox(self.edgesTab, "General", orientation="vertical")
            OWGUI.checkBox(ib, self, 'networkCanvas.show_weights', 'Show weights', callback=self.networkCanvas.set_edge_labels)
            #OWGUI.checkBox(ib, self, 'showEdgeLabels', 'Show labels on edges', callback=(lambda: self._set_canvas_attr('showEdgeLabels', self.showEdgeLabels)))
            OWGUI.spin(ib, self, "maxLinkSize", 1, 50, 1, label="Max edge width:", callback=self.set_edge_sizes)
            self.cb_show_distances = OWGUI.checkBox(ib, self, 'explore_distances', 'Explore node distances', callback=self.set_explore_distances, disabled=1)
            self.cb_show_component_distances = OWGUI.checkBox(ib, self, 'networkCanvas.show_component_distances', 'Show component distances', callback=self.networkCanvas.set_show_component_distances, disabled=1)

            colorBox = OWGUI.widgetBox(self.edgesTab, "Edge color attribute", orientation="horizontal", addSpace=False)
            self.edgeColorCombo = OWGUI.comboBox(colorBox, self, "edgeColor", callback=self.set_edge_colors)
            self.edgeColorCombo.addItem("(same color)")
            OWGUI.button(colorBox, self, "palette", self._set_edge_color_palette, tooltip="Set edge color palette", width=60, debuggingEnabled=0)

            self.edgeLabelBox = OWGUI.widgetBox(self.edgesTab, "Edge labels", addSpace=False)
            self.edgeLabelListBox = OWGUI.listBox(self.edgeLabelBox, self, "edgeLabelAttributes", "edgeAttributes", selectionMode=QListWidget.MultiSelection, callback=self._clicked_edge_label_listbox)
            #self.edgeLabelBox.setEnabled(False)

            ib = OWGUI.widgetBox(self.verticesTab, "General", orientation="vertical")
            OWGUI.checkBox(ib, self, 'networkCanvas.show_indices', 'Show indices', callback=self.networkCanvas.set_node_labels)
            OWGUI.checkBox(ib, self, 'labelsOnMarkedOnly', 'Show labels on marked nodes only', callback=(lambda: self.networkCanvas.set_labels_on_marked(self.labelsOnMarkedOnly)))
            OWGUI.spin(ib, self, "fontSize", 4, 30, 1, label="Font size:", callback=self.set_font)
            self.comboFontWeight = OWGUI.comboBox(ib, self, "fontWeight", label='Font weight:', orientation='horizontal', callback=self.set_font)
            self.comboFontWeight.addItem("Normal")
            self.comboFontWeight.addItem("Bold")
            self.comboFontWeight.setCurrentIndex(self.fontWeight)

            ib = OWGUI.widgetBox(self.markTab, "Info", orientation="vertical")
            OWGUI.label(ib, self, "Nodes (shown/hidden): %(number_of_nodes_label)i (%(nShown)i/%(nHidden)i)")
            OWGUI.label(ib, self, "Selected: %(nSelected)i, marked: %(nMarked)i")

            ribg = OWGUI.radioButtonsInBox(self.markTab, self, "hubs", [], "Mark", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "None", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "Search", callback=self.set_mark_mode)
            self.ctrlMarkSearchString = OWGUI.lineEdit(OWGUI.indentedBox(ribg), self, "markSearchString", callback=self._set_search_string_timer, callbackOnType=True)
            self.searchStringTimer = QTimer(self)
            self.connect(self.searchStringTimer, SIGNAL("timeout()"), self.set_mark_mode)

            OWGUI.appendRadioButton(ribg, self, "hubs", "Neighbors of focused", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "Neighbours of selected", callback=self.set_mark_mode)
            ib = OWGUI.indentedBox(ribg, orientation=0)
            self.ctrlMarkDistance = OWGUI.spin(ib, self, "markDistance", 0, 100, 1, label="Distance ", callback=(lambda h=2: self.set_mark_mode(h)))
            #self.ctrlMarkFreeze = OWGUI.button(ib, self, "&Freeze", value="graph.freezeNeighbours", toggleButton = True)
            OWGUI.widgetLabel(ribg, "Mark nodes with ...")
            OWGUI.appendRadioButton(ribg, self, "hubs", "at least N connections", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "at most N connections", callback=self.set_mark_mode)
            self.ctrlMarkNConnections = OWGUI.spin(OWGUI.indentedBox(ribg), self, "markNConnections", 0, 1000000, 1, label="N ", callback=(lambda h=4: self.set_mark_mode(h)))
            OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than any neighbour", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "more connections than avg neighbour", callback=self.set_mark_mode)
            OWGUI.appendRadioButton(ribg, self, "hubs", "most connections", callback=self.set_mark_mode)
            ib = OWGUI.indentedBox(ribg)
            self.ctrlMarkNumber = OWGUI.spin(ib, self, "markNumber", 0, 1000000, 1, label="Number of nodes:", callback=(lambda h=8: self.set_mark_mode(h)))
            OWGUI.widgetLabel(ib, "(More nodes are marked in case of ties)")
            self.markInputRadioButton = OWGUI.appendRadioButton(ribg, self, "hubs", "Mark nodes given in the input signal", callback=self.set_mark_mode)
            ib = OWGUI.indentedBox(ribg)
            self.markInput = 0
            self.markInputCombo = OWGUI.comboBox(ib, self, "markInput", callback=(lambda h=9: self.set_mark_mode(h)))
            self.markInputRadioButton.setEnabled(False)

            #ib = OWGUI.widgetBox(self.markTab, "General", orientation="vertical")
            #self.checkSendMarkedNodes = True
            #OWGUI.checkBox(ib, self, 'checkSendMarkedNodes', 'Send marked nodes', callback = self.send_marked_nodes, disabled=0)

            self.toolbar = OWGUI.widgetBox(self.controlArea, orientation='horizontal')
            prevstate = self.networkCanvas.state
            prevselbeh = self.networkCanvas.selection_behavior
            G = self.networkCanvas.gui
            self.zoomSelectToolbar = G.zoom_select_toolbar(self.toolbar, nomargin=True, buttons=
                G.default_zoom_select_buttons +
                [
                    G.Spacing,
                    ("buttonM2S", "Marked to selection", None, None, "marked_to_selected", 'Dlg_Mark2Sel'),
                    ("buttonS2M", "Selection to marked", None, None, "selected_to_marked", 'Dlg_Sel2Mark'),
                    ("buttonHSEL", "Hide selection", None, None, "hide_selection", 'Dlg_HideSelection'),
                    ("buttonSSEL", "Show all nodes", None, None, "show_selection", 'Dlg_ShowSelection'),
                    #("buttonUN", "Hide unselected", None, None, "hideUnSelectedVertices", 'Dlg_SelectedNodes'),
                    #("buttonSW", "Show all nodes", None, None, "showAllVertices", 'Dlg_clear'),
                ])
            self.zoomSelectToolbar.buttons[G.SendSelection].clicked.connect(self.send_data)
            self.zoomSelectToolbar.buttons[G.SendSelection].clicked.connect(self.send_marked_nodes)
            self.zoomSelectToolbar.buttons[("buttonHSEL", "Hide selection", None, None, "hide_selection", 'Dlg_HideSelection')].clicked.connect(self.hide_selection)
            self.zoomSelectToolbar.buttons[("buttonSSEL", "Show all nodes", None, None, "show_selection", 'Dlg_ShowSelection')].clicked.connect(self.show_selection)
            #self.zoomSelectToolbar.buttons[G.SendSelection].hide()
            self.zoomSelectToolbar.select_selection_behaviour(prevselbeh)
            self.zoomSelectToolbar.select_state(prevstate)

            ib = OWGUI.widgetBox(self.infoTab, "General")
            OWGUI.label(ib, self, "Number of nodes: %(number_of_nodes_label)i")
            OWGUI.label(ib, self, "Number of edges: %(number_of_edges_label)i")
            OWGUI.label(ib, self, "Nodes per edge: %(verticesPerEdge).2f")
            OWGUI.label(ib, self, "Edges per node: %(edgesPerVertex).2f")

            ib = OWGUI.widgetBox(self.infoTab, orientation="horizontal")

            OWGUI.button(ib, self, "Save net", callback=self.save_network, debuggingEnabled=False)
            OWGUI.button(ib, self, "Save img", callback=self.networkCanvas.saveToFile, debuggingEnabled=False)
            self.reportButton = OWGUI.button(ib, self, "&Report", self.reportAndFinish, debuggingEnabled=0)
            self.reportButton.setAutoDefault(0)

            #OWGUI.button(self.edgesTab, self, "Clustering", callback=self.clustering)
            ib = OWGUI.widgetBox(self.infoTab, "Edit")
            self.editAttribute = 0
            self.editCombo = OWGUI.comboBox(ib, self, "editAttribute", label="Edit attribute:", orientation="horizontal")
            self.editCombo.addItem("Select attribute")
            self.editValue = ''
            hb = OWGUI.widgetBox(ib, orientation="horizontal")
            OWGUI.lineEdit(hb, self, "editValue", "Value:", orientation='horizontal')
            OWGUI.button(hb, self, "Set", callback=self.edit)

            ib = OWGUI.widgetBox(self.infoTab, "Prototype")
            ib.setVisible(True)

            OWGUI.lineEdit(ib, self, "organism", "Organism:", orientation='horizontal')

            self.nameComponentAttribute = 0
            self.nameComponentCombo = OWGUI.comboBox(ib, self, "nameComponentAttribute", callback=self.nameComponents, label="Name components:", orientation="horizontal")
            self.nameComponentCombo.addItem("Select attribute")

            self.showComponentAttribute = 0
            self.showComponentCombo = OWGUI.comboBox(ib, self, "showComponentAttribute", callback=self.showComponents, label="Labels on components:", orientation="horizontal")
            self.showComponentCombo.addItem("Select attribute")
            OWGUI.checkBox(ib, self, 'showTextMiningInfo', "Show text mining info")

            #OWGUI.spin(ib, self, "rotateSteps", 1, 10000, 1, label="Rotate max steps: ")
            OWGUI.spin(ib, self, "minComponentEdgeWidth", 0, 100, 1, label="Min component edge width: ", callback=(lambda changedMin=1: self.set_component_edge_width(changedMin)))
            OWGUI.spin(ib, self, "maxComponentEdgeWidth", 0, 200, 1, label="Max component edge width: ", callback=(lambda changedMin=0: self.set_component_edge_width(changedMin)))

            self.attSelectionAttribute = 0
            self.comboAttSelection = OWGUI.comboBox(ib, self, "attSelectionAttribute", label='Send attribute selection list:', orientation='horizontal', callback=self.sendAttSelectionList)
            self.comboAttSelection.addItem("Select attribute")
            self.autoSendAttributes = 0
            OWGUI.checkBox(ib, self, 'autoSendAttributes', "auto send attributes", callback=self.setAutoSendAttributes)

            self.icons = self.createAttributeIconDict()
            self.set_mark_mode()

            self.networkCanvas.gui.effects_box(self.performanceTab)

            self.verticesTab.layout().addStretch(1)
            self.edgesTab.layout().addStretch(1)
            self.markTab.layout().addStretch(1)
            self.infoTab.layout().addStretch(1)
            self.performanceTab.layout().addStretch(1)

            dlg = self._create_color_dialog(self.colorSettings, self.selectedSchemaIndex)
            self.networkCanvas.contPalette = dlg.getContinuousPalette("contPalette")
            self.networkCanvas.discPalette = dlg.getDiscretePalette("discPalette")

            dlg = self._create_color_dialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
            self.networkCanvas.contEdgePalette = dlg.getContinuousPalette("contPalette")
            self.networkCanvas.discEdgePalette = dlg.getDiscretePalette("discPalette")

            self.graph_layout_method()
            self.set_font()
            self.set_graph(None)

            self.setMinimumWidth(900)

            self.connect(self.networkCanvas, SIGNAL("marked_points_changed()"), self.send_marked_nodes)
            self.connect(self.networkCanvas, SIGNAL("selection_changed()"), self.send_data)

        def hide_selection(self):
            nodes = set(self.graph.nodes()).difference(self.networkCanvas.selected_nodes())
            self.change_graph(Orange.network.nx.Graph.subgraph(self.graph, nodes))

        def show_selection(self):
            self.change_graph(self.graph_base)

        def edit(self):
            if self.graph is None:
                return

            vars = [x.name for x in self.graph_base.items_vars()]
            if not self.editCombo.currentText() in vars:
                return
            att = str(self.editCombo.currentText())
            vertices = self.networkCanvas.selected_nodes()

            if len(vertices) == 0:
                return

            items = self.graph_base.items()
            if items.domain[att].var_type == Orange.data.Type.Continuous:
                for v in vertices:
                    items[v][att] = float(self.editValue)
            else:
                for v in vertices:
                    items[v][att] = str(self.editValue)

        def set_items_distance_matrix(self, matrix):
            self.error()
            self.warning()
            self.information()

            self.cb_show_distances.setEnabled(0)
            self.cb_show_component_distances.setEnabled(0)

            if matrix is None or self.graph_base is None:
                self.items_matrix = None
                self.networkCanvas.items_matrix = None
                self.information('No graph found!')
                return

            if matrix.dim != self.graph_base.number_of_nodes():
                self.error('The number of vertices does not match matrix size.')
                self.items_matrix = None
                self.networkCanvas.items_matrix = None
                return

            self.items_matrix = matrix
            self.networkCanvas.items_matrix = matrix
            self.cb_show_distances.setEnabled(1)
            self.cb_show_component_distances.setEnabled(1)

            if str(self.optMethod) in ['8', '9', '10']:
                if self.items_matrix is not None and self.graph is not None and \
                self.items_matrix.dim == self.graph.number_of_nodes():
                    self.optButton.setEnabled(True)
                    self.optButton.setChecked(True)
                    self.graph_layout()

        def _set_canvas_attr(self, attr, value):
            setattr(self.networkCanvas, attr, value)
            self.networkCanvas.updateCanvas()

        def _set_curve_attr(self, attr, value):
            setattr(self.networkCanvas.networkCurve, attr, value)
            self.networkCanvas.updateCanvas()

        def _set_search_string_timer(self):
            self.hubs = 1
            self.searchStringTimer.stop()
            self.searchStringTimer.start(1000)

        def set_mark_mode(self, i=None):
            self.searchStringTimer.stop()
            if not i is None:
                self.hubs = i

            QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self.networkCanvas.mark_on_selection_changed)
            QObject.disconnect(self.networkCanvas, SIGNAL('point_hovered(Point*)'), self.networkCanvas.mark_on_focus_changed)

            if self.graph is None:
                return

            hubs = self.hubs

            if hubs in [0, 1, 2, 3]:
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
                    QObject.connect(self.networkCanvas, SIGNAL('point_hovered(Point*)'), self.networkCanvas.mark_on_focus_changed)
                elif hubs == 3:
                    #print "mark selected"
                    self.networkCanvas.mark_neighbors = self.markDistance
                    QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self.networkCanvas.mark_on_selection_changed)
                    self.networkCanvas.mark_on_selection_changed()

            elif hubs in [4, 5, 6, 7, 8, 9]:

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
                        d > max([0] + self.graph.degree(self.graph.neighbors(i)).values()) \
                        else (i, False) for i, d in powers))
                elif hubs == 7:
                    #print "mark more than avg"
                    self.networkCanvas.networkCurve.set_node_marks(dict((i, True) if \
                        d > statc.mean([0] + self.graph.degree(self.graph.neighbors(i)).values()) \
                        else (i, False) for i, d in powers))
                    self.networkCanvas.replot()
                elif hubs == 8:
                    #print "mark most"
                    self.networkCanvas.networkCurve.clear_node_marks()

                    if self.markNumber < 1:
                        return

                    cut = self.markNumber
                    cutPower = powers[cut - 1][1]
                    while cut < len(powers) and powers[cut][1] == cutPower:
                        cut += 1

                    self.networkCanvas.networkCurve.clear_node_marks()
                    self.networkCanvas.networkCurve.set_node_marks(dict((i, True) for \
                        i, d in powers[:cut]))

                elif hubs == 9:
                    var = str(self.markInputCombo.currentText())
                    if self.markInputItems is not None and len(self.markInputItems) > 0:
                        if var == 'ID':
                            values = [x.id for x in self.markInputItems]
                            tomark = dict((x, True) for x in self.graph.nodes() if self.graph_base.items()[x].id in values)
                        else:
                            values = [str(x[var]).strip().upper() for x in self.markInputItems]
                            tomark = dict((x, True) for x in self.graph.nodes() if str(self.graph_base.items()[x][var]).strip().upper() in values)
                        self.networkCanvas.networkCurve.clear_node_marks()
                        self.networkCanvas.networkCurve.set_node_marks(tomark)

                    else:
                        self.networkCanvas.networkCurve.clear_node_marks()

            self.nMarked = len(self.networkCanvas.marked_nodes())

        def save_network(self):
            if self.networkCanvas is None or self.graph is None:
                return

            filename = QFileDialog.getSaveFileName(self, 'Save Network File', \
                '', 'NetworkX graph as Python pickle (*.gpickle)\nPajek ' + \
                'network (*.net)\nGML network (*.gml)')
            if filename:
                fn = ""
                head, tail = os.path.splitext(str(filename))
                if not tail:
                    fn = head + ".net"
                else:
                    fn = str(filename)

                items = self.graph.items()
                for i in range(self.graph.number_of_nodes()):
                    graph_node = self.graph.node[i]
                    plot_node = self.networkCanvas.networkCurve.nodes()[i]

                    if items is not None:
                        ex = items[i]
                        if 'x' in ex.domain:
                            ex['x'] = plot_node.x()
                        if 'y' in ex.domain:
                            ex['y'] = plot_node.y()

                    graph_node['x'] = plot_node.x()
                    graph_node['y'] = plot_node.y()

                Orange.network.readwrite.write(self.graph, fn)

        def send_data(self):
            if len(self.signalManager.getLinks(self, None, \
                "Selected Items", None)) > 0 or \
                    len(self.signalManager.getLinks(self, None, \
                        "Unselected Items", None)) > 0 or \
                            len(self.signalManager.getLinks(self, None, \
                                "Selected Network", None)) > 0:

                # signal connected
                selected_nodes = self.networkCanvas.selected_nodes()
                graph = self.graph_base.subgraph(selected_nodes)

                if graph is not None:
                    self.send("Selected Items", graph.items())

                    if len(self.signalManager.getLinks(self, None, \
                                                "Unselected Items", None)) > 0:
                        nodes = self.networkCanvas.not_selected_nodes()
                        if len(nodes) > 0 and self.graph_base.items() is not None:
                            self.send("Other Items", self.graph_base.items().getitems(nodes))
                        else:
                            self.send("Other Items", None)

                    self.send("Selected Network", graph)
                else:
                    self.send("Selected Items", None)
                    self.send("Other Items", None)
                    self.send("Selected Network", None)

            if len(self.signalManager.getLinks(self, None, \
                                "Selected Items Distance Matrix", None)) > 0:
                # signal connected
                matrix = None if self.items_matrix is None else self.items_matrix.getitems(selected_nodes)
                self.send("Distance Matrix", matrix)

        def send_marked_nodes(self):
            if self.checkSendMarkedNodes and \
                len(self.signalManager.getLinks(self, None, \
                                                "Marked Items", None)) > 0:
                # signal connected
                markedNodes = self.networkCanvas.marked_nodes()

                if len(markedNodes) > 0 and self.graph is not None and\
                                         self.graph_base.items() is not None:

                    items = self.graph_base.items().getitems(markedNodes)
                    self.send("Marked Items", items)
                else:
                    self.send("Marked Items", None)

        def _set_combos(self):
            vars = self.graph_base.items_vars()
            edgeVars = self.graph_base.links_vars()
            lastLabelColumns = self.lastLabelColumns
            lastTooltipColumns = self.lastTooltipColumns

            self._clear_combos()

            self.attributes = [(var.name, var.varType) for var in vars]
            self.edgeAttributes = [(var.name, var.varType) for var in edgeVars]

            for var in vars:
                if var.varType in [Orange.data.Type.Discrete, \
                                   Orange.data.Type.Continuous]:
                    self.colorCombo.addItem(self.icons.get(var.varType, \
                                            self.icons[-1]), unicode(var.name))

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
                if self.lastVertexSizeColumn == \
                        self.vertexSizeCombo.itemText(i):
                    self.vertexSize = i
                    self.set_node_sizes()
                    break

            for i in range(self.colorCombo.count()):
                if self.lastColorColumn == self.colorCombo.itemText(i):
                    self.color = i
                    self.set_node_colors()
                    break

            for i in range(self.attListBox.count()):
                if str(self.attListBox.item(i).text()) in lastLabelColumns:
                    self.attListBox.item(i).setSelected(1)
                self._clicked_att_lstbox()

            for i in range(self.tooltipListBox.count()):
                if str(self.tooltipListBox.item(i).text()) \
                                                    in lastTooltipColumns:
                    self.tooltipListBox.item(i).setSelected(1)
                self._clicked_tooltip_lstbox()

            self.lastLabelColumns = lastLabelColumns
            self.lastTooltipColumns = lastTooltipColumns

        def _clear_combos(self):
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

            if self.graph.number_of_nodes() > 0:
                t = 1.13850193174e-008 * (self.graph.number_of_nodes() ** 2 + self.graph.number_of_edges())
                self.frSteps = int(2.0 / t)
                if self.frSteps < 1: self.frSteps = 1
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

            animation_enabled = self.networkCanvas.animate_points;
            self.networkCanvas.animate_points = False;

            self.set_node_sizes()
            self.set_node_colors()
            self.set_edge_sizes()
            self.set_edge_colors()

            self._clicked_att_lstbox()
            self._clicked_tooltip_lstbox()
            self._clicked_edge_label_listbox()

            self.networkCanvas.replot()

            self.networkCanvas.animate_points = animation_enabled
            qApp.processEvents()
            self.networkCanvas.networkCurve.layout_fr(100, weighted=False, smooth_cooling=True)
            self.networkCanvas.update_canvas()

        def set_graph_none(self):
            self.graph = None
            #self.graph_base = None
            self._clear_combos()
            self.number_of_nodes_label = -1
            self.number_of_edges_label = -1
            self._items = None
            self._links = None
            self.set_items_distance_matrix(None)
            self.networkCanvas.set_graph(None)

        def set_graph(self, graph, curve=None):
            self.information()
            self.warning()
            self.error()

            if graph is None:
                self.set_graph_none()
                return

            if graph.number_of_nodes() < 2:
                self.set_graph_none()
                self.information('I\'m not really in a mood to visualize just one node. Try again tomorrow.')
                return

            if graph == self.graph_base and self.graph is not None and \
                                                    self._network_view is None:
                self.set_items(graph.items())
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

            self.number_of_nodes_label = self.graph.number_of_nodes()
            self.number_of_edges_label = self.graph.number_of_edges()

            self.networkCanvas.set_graph(self.graph, curve, items=self.graph_base.items(), links=self.graph_base.links())

            if self._items is not None and 'x' in self._items.domain and 'y' in self._items.domain:
                positions = dict((node, (self._items[node]['x'].value, self._items[node]['y'].value)) \
                             for node in self.graph if self._items[node]['x'].value != '?' \
                             and self._items[node]['y'].value != '?')

                self.networkCanvas.networkCurve.set_node_coordinates(positions)


            self.networkCanvas.showEdgeLabels = self.showEdgeLabels
            self.networkCanvas.maxEdgeSize = self.maxLinkSize
            self.networkCanvas.minComponentEdgeWidth = self.minComponentEdgeWidth
            self.networkCanvas.maxComponentEdgeWidth = self.maxComponentEdgeWidth
            self.networkCanvas.set_labels_on_marked(self.labelsOnMarkedOnly)

            self.compute_network_info()
            self._set_combos()

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

            t = 1.13850193174e-008 * (self.graph.number_of_nodes() ** 2 + self.graph.number_of_edges())
            self.frSteps = int(2.0 / t)
            if self.frSteps < 1: self.frSteps = 1
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

            self.set_node_sizes()
            self.set_node_colors()
            self.set_edge_sizes()
            self.set_edge_colors()

            self._clicked_att_lstbox()
            self._clicked_tooltip_lstbox()
            self._clicked_edge_label_listbox()

            self.optButton.setChecked(1)
            self.graph_layout()

        def set_network_view(self, nxView):
            self.error()
            self.warning()
            self.information()

            if self.graph is None:
                self.information('Do not forget to add a graph!')

            if self._network_view is not None:
                QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)

            self._network_view = nxView

            if self._network_view is not None:
                self._network_view.set_nx_explorer(self)

            self.set_graph(self.graph_base)

            if self._network_view is not None:
                QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self._network_view.node_selection_changed)

        def set_items(self, items=None):
            self.error()
            self.warning()
            self.information()

            if items is None:
                return

            if self.graph is None:
                self.warning('No graph found!')
                return

            if len(items) != self.graph_base.number_of_nodes():
                self.error('Table items must have one example for each node.')
                return

            self.graph_base.set_items(items)

            self.set_node_sizes()
            self.networkCanvas.items = items
            self.networkCanvas.showWeights = self.showWeights
            self.networkCanvas.showEdgeLabels = self.showEdgeLabels
            self._set_combos()
            #self.networkCanvas.updateData()

        def mark_items(self, items):
            self.markInputCombo.clear()
            self.markInputRadioButton.setEnabled(False)
            self.markInputItems = items

            self.error()
            self.warning()
            self.information()

            if items is None:
                return

            if self.graph is None or self.graph_base.items() is None or items is None:
                self.warning('No graph found or no items attached to the graph.')
                return

            if len(items) > 0:
                lstOrgDomain = [x.name for x in self.graph_base.items().domain] + [self.graph_base.items().domain[x].name for x in self.graph_base.items().domain.getmetas()]
                lstNewDomain = [x.name for x in items.domain] + [items.domain[x].name for x in items.domain.getmetas()]
                commonVars = set(lstNewDomain) & set(lstOrgDomain)

                self.markInputCombo.addItem(self.icons[Orange.data.Type.Discrete], unicode("ID"))

                if len(commonVars) > 0:
                    for var in commonVars:
                        orgVar = self.graph_base.items().domain[var]
                        mrkVar = items.domain[var]

                        if orgVar.varType == mrkVar.varType and orgVar.varType == Orange.data.Type.String:
                            self.markInputCombo.addItem(self.icons[orgVar.varType], unicode(orgVar.name))

                self.markInputRadioButton.setEnabled(True)
                self.set_mark_mode(9)

        def set_explore_distances(self):
            QObject.disconnect(self.networkCanvas, SIGNAL('selection_changed()'), self.explore_focused)

            if self.explore_distances:
                QObject.connect(self.networkCanvas, SIGNAL('selection_changed()'), self.explore_focused)

        def explore_focused(self):
            sel = self.networkCanvas.selected_nodes()
            if len(sel) == 1:
                ndx_1 = sel[0]
                self.networkCanvas.label_distances = [['%.2f' % \
                                self.items_matrix[ndx_1][ndx_2]] \
                                for ndx_2 in self.networkCanvas.graph.nodes()]
            else:
                self.networkCanvas.label_distances = None

            self.networkCanvas.set_node_labels(self.lastLabelColumns)
            self.networkCanvas.replot()

        #######################################################################
        ### Layout Optimization                                             ###
        #######################################################################

        def graph_layout(self):
            if self.graph is None or self.graph.number_of_nodes <= 0:   #grafa se ni
                self.optButton.setChecked(False)
                return

            if not self.optButton.isChecked() and not self.optMethod in [2, 3, 9, 10]:
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
                self.networkCanvas.networkCurve.circular(\
                                            NetworkCurve.circular_crossing)
            elif self.optMethod == 6:
                self.networkCanvas.networkCurve.circular(\
                                            NetworkCurve.circular_original)
            elif self.optMethod == 7:
                self.networkCanvas.networkCurve.circular(\
                                            NetworkCurve.circular_random)
            elif self.optMethod == 8:
                self.graph_layout_fragviz()
            elif self.optMethod == 9:
                self.graph_layout_mds()
            elif self.optMethod == 10:
                self.graph_layout_pivot_mds()

            self.optButton.setChecked(False)
            self.networkCanvas.update_canvas()

        def graph_layout_method(self, method=None):
            self.information()
            self.stepsSpin.label.setText('Iterations: ')
            self.optButton.setEnabled(True)
            self.cb_opt_from_curr.setEnabled(False)

            if method is not None:
                self.optMethod = method

            if str(self.optMethod) == '0':
                self.optButton.setEnabled(False)
            else:
                self.optButton.setEnabled(True)

            if str(self.optMethod) in ['2', '3', '4']:
                self.stepsSpin.setEnabled(True)

            elif str(self.optMethod) in ['8', '9', '10']:
                if str(self.optMethod) == '8':
                    self.stepsSpin.label.setText('Pivots: ')

                if str(self.optMethod) in ['9', '10']:
                    self.cb_opt_from_curr.setEnabled(True)

                self.stepsSpin.setEnabled(True)

                if self.items_matrix is None:
                    self.information('Set distance matrix to input signal')
                    self.optButton.setEnabled(False)
                    return
                if self.graph is None:
                    self.information('No network found')
                    self.optButton.setEnabled(False)
                    return
                if self.items_matrix.dim != self.graph.number_of_nodes():
                    self.error('Distance matrix dimensionality must equal number of vertices')
                    self.optButton.setEnabled(False)
                    return
            else:
                self.stepsSpin.setEnabled(False)
                self.optButton.setChecked(True)
                self.graph_layout()

        def mds_progress(self, avgStress, stepCount):
            #self.drawForce()

            #self.mdsInfoA.setText("Avg. Stress: %.20f" % avgStress)
            #self.mdsInfoB.setText("Num. steps: %i" % stepCount)
            self.progressBarSet(int(stepCount * 100 / self.frSteps))
            qApp.processEvents()

        def graph_layout_fragviz(self):
            if self.items_matrix is None:
                self.information('Set distance matrix to input signal')
                self.optButton.setChecked(False)
                return

            if self.layout is None:
                self.information('No network found')
                self.optButton.setChecked(False)
                return

            if self.items_matrix.dim != self.graph.number_of_nodes():
                self.error('Distance matrix dimensionality must equal number of vertices')
                self.optButton.setChecked(False)
                return

            if not self.optButton.isChecked():
                self.networkCanvas.networkCurve.stopMDS = True
                self.optButton.setChecked(False)
                self.optButton.setText("Optimize layout")
                return

            self.optButton.setText("Stop")
            self.progressBarInit()
            qApp.processEvents()

            if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
                matrix = self.items_matrix
            else:
                matrix = self.items_matrix.get_items(sorted(self.graph.nodes()))

            self.networkCanvas.networkCurve.layout_fragviz(self.frSteps, matrix, self.graph, self.mds_progress, self.opt_from_curr)

            self.optButton.setChecked(False)
            self.optButton.setText("Optimize layout")
            self.progressBarFinished()

        def graph_layout_mds(self):
            if self.items_matrix is None:
                self.information('Set distance matrix to input signal')
                self.optButton.setChecked(False)
                return

            if self.layout is None:
                self.information('No network found')
                self.optButton.setChecked(False)
                return

            if self.items_matrix.dim != self.graph.number_of_nodes():
                self.error('Distance matrix dimensionality must equal number of vertices')
                self.optButton.setChecked(False)
                return

            if not self.optButton.isChecked():
                self.networkCanvas.networkCurve.stopMDS = True
                self.optButton.setChecked(False)
                self.optButton.setText("Optimize layout")
                return

            self.optButton.setText("Stop")
            self.progressBarInit()
            qApp.processEvents()

            if self.graph.number_of_nodes() == self.graph_base.number_of_nodes():
                matrix = self.items_matrix
            else:
                matrix = self.items_matrix.get_items(sorted(self.graph.nodes()))

            self.networkCanvas.networkCurve.layout_mds(self.frSteps, matrix, self.mds_progress, self.opt_from_curr)

            self.optButton.setChecked(False)
            self.optButton.setText("Optimize layout")
            self.progressBarFinished()

        def graph_layout_fr(self, weighted):
            if self.graph is None:
                return

            if not self.optButton.isChecked():
                self.networkCanvas.networkCurve.stop_optimization()
                self.optButton.setChecked(False)
                self.optButton.setText("Optimize layout")
                return

            self.optButton.setText("Stop")
            qApp.processEvents()
            self.networkCanvas.networkCurve.layout_fr(self.frSteps, False)
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
            x, y = mds.optimize()
            xy = zip(list(x), list(y))
            coors = dict(zip(sorted(self.graph.nodes()), xy))
            self.networkCanvas.networkCurve.set_node_coordinates(coors)
            self.networkCanvas.update_canvas()

        #######################################################################
        ### Network Visualization                                           ###
        #######################################################################

        def _set_colors(self):
            dlg = self._create_color_dialog(self.colorSettings, self.selectedSchemaIndex)
            if dlg.exec_():
                self.colorSettings = dlg.getColorSchemas()
                self.selectedSchemaIndex = dlg.selectedSchemaIndex
                self.networkCanvas.contPalette = dlg.getContinuousPalette("contPalette")
                self.networkCanvas.discPalette = dlg.getDiscretePalette("discPalette")

                self.set_node_colors()

        def _set_edge_color_palette(self):
            dlg = self._create_color_dialog(self.edgeColorSettings, self.selectedEdgeSchemaIndex)
            if dlg.exec_():
                self.edgeColorSettings = dlg.getColorSchemas()
                self.selectedEdgeSchemaIndex = dlg.selectedSchemaIndex
                self.networkCanvas.contEdgePalette = dlg.getContinuousPalette("contPalette")
                self.networkCanvas.discEdgePalette = dlg.getDiscretePalette("discPalette")

                self.set_edge_colors()

        def _create_color_dialog(self, colorSettings, selectedSchemaIndex):
            c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
            c.createDiscretePalette("discPalette", "Discrete Palette")
            c.createContinuousPalette("contPalette", "Continuous Palette")
            c.setColorSchemas(colorSettings, selectedSchemaIndex)
            return c

        def _clicked_att_lstbox(self):
            if self.graph is None:
                return

            self.lastLabelColumns = [self.attributes[i][0] for i in self.markerAttributes]
            self.networkCanvas.set_node_labels(self.lastLabelColumns)
            self.networkCanvas.replot()

        def _clicked_tooltip_lstbox(self):
            if self.graph is None:
                return

            self.lastTooltipColumns = [self.attributes[i][0] for i in self.tooltipAttributes]
            self.networkCanvas.set_tooltip_attributes(self.lastTooltipColumns)
            self.networkCanvas.replot()

        def _clicked_edge_label_listbox(self):
            if self.graph is None:
                return

            self.lastEdgeLabelAttributes = set([self.edgeAttributes[i][0] for i in self.edgeLabelAttributes])
            self.networkCanvas.set_edge_labels(self.lastEdgeLabelAttributes)
            self.networkCanvas.replot()

        def set_node_colors(self):
            if self.graph is None:
                return

            self.networkCanvas.set_node_colors(self.colorCombo.currentText())
            self.lastColorColumn = self.colorCombo.currentText()

        def set_edge_colors(self):
            if self.graph is None:
                return

            self.networkCanvas.set_edge_colors(self.edgeColorCombo.currentText())
            self.lastEdgeColorColumn = self.edgeColorCombo.currentText()

        def set_edge_sizes(self):
            if self.graph is None:
                return

            self.networkCanvas.networkCurve.set_edge_sizes(self.maxLinkSize)
            self.networkCanvas.replot()

        def set_node_sizes(self):
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
                values.update((key, maxval - val) for key, val in values.iteritems())
                self.networkCanvas.networkCurve.set_node_sizes(values, min_size=self.minVertexSize, max_size=self.maxVertexSize)
            else:
                self.networkCanvas.networkCurve.set_node_sizes(values, min_size=self.minVertexSize, max_size=self.maxVertexSize)

            self.networkCanvas.replot()

        def set_font(self):
            if self.networkCanvas is None:
                return

            weights = {0: 50, 1: 80}

            font = self.networkCanvas.font()
            font.setPointSize(self.fontSize)
            font.setWeight(weights[self.fontWeight])
            self.networkCanvas.setFont(font)
            self.networkCanvas.fontSize = font
            self.networkCanvas.set_node_labels()

        def sendReport(self):
            self.reportSettings("Graph data",
                                [("Number of vertices", self.graph.number_of_nodes()),
                                 ("Number of edges", self.graph.number_of_edges()),
                                 ("Vertices per edge", "%.3f" % self.verticesPerEdge),
                                 ("Edges per vertex", "%.3f" % self.edgesPerVertex),
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

        #######################################################################
        ### PROTOTYPE                                                       ### 
        #######################################################################

        def set_component_edge_width(self, changedMin=True):
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

        def showComponents(self):
            if self.graph is None or self.graph_base.items() is None:
                return

            vars = [x.name for x in self.graph_base.items_vars()]

            if not self.showComponentCombo.currentText() in vars:
                self.networkCanvas.showComponentAttribute = None
                self.lastNameComponentAttribute = ''
            else:
                self.networkCanvas.showComponentAttribute = self.showComponentCombo.currentText()

            self.networkCanvas.drawComponentKeywords()

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
                                avg_rank = (a[int_ndx][j] + a[k - 1][j]) / 2
                                for l in range(int_ndx, k):
                                    a[l][j] = avg_rank

                            int_ndx = k

                        a[k][j] = top_rank / max_rank
                        top_rank -= 1

                    k += 1
                    if k - int_ndx > 1:
                        avg_rank = (a[int_ndx][j] + a[k - 1][j]) / 2
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
                                avg_rank = (a[int_ndx][j] + a[k - 1][j]) / 2
                                for l in range(int_ndx, k):
                                    a[l][j] = avg_rank

                            int_ndx = k

                        a[k][j] = top_rank / max_rank
                        top_rank -= 1

                    k += 1
                    if k - int_ndx > 1:
                        avg_rank = (a[int_ndx][j] + a[k - 1][j]) / 2
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

                self.progressBarSet(i * 100.0 / len(components))

            self.lastNameComponentAttribute = self.nameComponentCombo.currentText()
            self.set_items(Orange.data.Table([self.graph_base.items(), keyword_table]))
            self.progressBarFinished()


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
                vertices = self.networkCanvas.selected_nodes()

                if len(vertices) != 1:
                    return

                attributes = str(self.graph_base.items()[vertices[0]][att]).split(', ')
            else:
                attributes = None
            self.send("Features", attributes)


except ImportError as err:
    try:
        from OWNxCanvas import *
    except:
        # if Qwt is also not installed throw could not import orangeqt error
        raise err

    from OWNxExplorerQwt import OWNxExplorerQwt as OWNxExplorer

if __name__ == "__main__":
    a = QApplication(sys.argv)
    ow = OWNxExplorer()
    ow.show()

    def setNetwork(signal, data, id=None):
        if signal == 'Network':
            ow.set_graph(data)
        #if signal == 'Items':
        #    ow.set_items(data)

    import OWNxFile
    owFile = OWNxFile.OWNxFile()
    owFile.send = setNetwork
    owFile.show()
    owFile.selectNetFile(0)

    a.exec_()
    ow.saveSettings()
    owFile.saveSettings()

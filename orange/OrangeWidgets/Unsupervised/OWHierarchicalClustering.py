"""
<name>Hierarchical Clustering</name>
<description>Hierarchical clustering based on distance matrix, and a dendrogram viewer.</description>
<icon>icons/HierarchicalClustering.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>2100</priority>
"""
from __future__ import with_statement

from OWWidget import *
from OWQCanvasFuncts import *
import OWClustering
import OWGUI
import OWColorPalette
import math
import numpy
import os

import orange
from Orange.clustering import hierarchical 

from OWDlgs import OWChooseImageSizeDlg

from PyQt4.QtCore import *
from PyQt4.QtGui import *

class OWHierarchicalClustering(OWWidget):
    settingsList = ["Linkage", "Annotation", "PrintDepthCheck",
                    "PrintDepth", "HDSize", "VDSize", "ManualHorSize","AutoResize",
                    "TextSize", "LineSpacing", "SelectionMode",
                    "AppendClusters", "CommitOnChange", "ClassifyName", "addIdAs"]
    
    contextHandlers={"":DomainContextHandler("", [ContextField("Annotation", DomainContextHandler.Required)])}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Hierarchical Clustering', wantGraph=True)
        
        self.inputs = [("Distance matrix", orange.SymMatrix, self.set_matrix)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable), ("Centroids", ExampleTable)]
        self.linkage = [("Single linkage", orange.HierarchicalClustering.Single),
                        ("Average linkage", orange.HierarchicalClustering.Average),
                        ("Ward's linkage", orange.HierarchicalClustering.Ward),
                        ("Complete linkage", orange.HierarchicalClustering.Complete),
                       ]
        self.Linkage = 3
        self.Annotation = 0
        self.PrintDepthCheck = 0
        self.PrintDepth = 10
        self.HDSize = 500         #initial horizontal and vertical dendrogram size
        self.VDSize = 800
        self.ManualHorSize = 0
        self.AutoResize = 0
        self.TextSize = 8
        self.LineSpacing = 4
        self.SelectionMode = 0
        self.AppendClusters = 0
        self.CommitOnChange = 0
        self.ClassifyName = "HC_class"
        self.addIdAs = 0
        
        self.loadSettings()
        
        self.inputMatrix = None
        self.matrixSource = "Unknown"
        self.root_cluster = None
        self.selectedExamples = None
        
        self.selectionChanged = False

        self.linkageMethods = [a[0] for a in self.linkage]

        #################################
        ##GUI
        #################################
        
        #HC Settings
        OWGUI.comboBox(self.controlArea, self, "Linkage", box="Linkage",
                items=self.linkageMethods, tooltip="Choose linkage method",
                callback=self.run_clustering, addSpace = True)
        #Label
        box = OWGUI.widgetBox(self.controlArea, "Annotation", addSpace = True)
        self.labelCombo = OWGUI.comboBox(box, self, "Annotation",
                items=["None"],tooltip="Choose label attribute",
                callback=self.update_labels)

        OWGUI.spin(box, self, "TextSize", label="Text size",
                        min=5, max=15, step=1, 
                        callback=self.update_font,
                        controlWidth=40,
                        keyboardTracking=False)
#        OWGUI.spin(box,self, "LineSpacing", label="Line spacing",
#                        min=2,max=8,step=1, 
#                        callback=self.update_spacing, 
#                        controlWidth=40,
#                        keyboardTracking=False)

        # Dendrogram graphics settings
        dendrogramBox = OWGUI.widgetBox(self.controlArea, "Limits", 
                                        addSpace=True)
        
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        
        # Depth settings 
        sw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal",
                             addToLayout=False)
        cw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", 
                             addToLayout=False)
        
        sllp = OWGUI.hSlider(sw, self, "PrintDepth", minValue=1, maxValue=50, 
                             callback=self.on_depth_change)
        cblp = OWGUI.checkBox(cw, self, "PrintDepthCheck", "Show to depth", 
                              callback = self.on_depth_change, 
                              disables=[sw])
        form.addRow(cw, sw)
        
        checkWidth = OWGUI.checkButtonOffsetHint(cblp)
        
        # Width settings
        sw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", 
                             addToLayout=False)
        cw = OWGUI.widgetBox(dendrogramBox, orientation="horizontal", 
                             addToLayout=False)
        
        hsb = OWGUI.spin(sw, self, "HDSize", min=200, max=10000, step=10, 
                         callback=self.on_width_changed, 
                         callbackOnReturn = False,
                         keyboardTracking=False)
        cbhs = OWGUI.checkBox(cw, self, "ManualHorSize", "Horizontal size", 
                              callback=self.on_width_changed, 
                              disables=[sw])
        
        self.hSizeBox = hsb
        form.addRow(cw, sw)
        dendrogramBox.layout().addLayout(form)

        # Selection settings 
        box = OWGUI.widgetBox(self.controlArea, "Selection")
        OWGUI.checkBox(box, self, "SelectionMode", "Show cutoff line",
                       callback=self.update_cutoff_line)
        cb = OWGUI.checkBox(box, self, "AppendClusters", "Append cluster IDs",
                            callback=self.commit_data_if)
        
        self.classificationBox = ib = OWGUI.widgetBox(box, margin=0)
        
        form = QWidget()
        le = OWGUI.lineEdit(form, self, "ClassifyName", None, callback=None,
                            orientation="horizontal")
        self.connect(le, SIGNAL("editingFinished()"), self.commit_data_if)

        aa = OWGUI.comboBox(form, self, "addIdAs", label=None,
                            orientation="horizontal",
                            items=["Class attribute",
                                   "Attribute",
                                   "Meta attribute"],
                            callback=self.commit_data_if)
        
        layout = QFormLayout()
        layout.setSpacing(8)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignLeft)
        layout.addRow("Name  ", le)
        layout.addRow("Place  ", aa)
        
        form.setLayout(layout)
        
        ib.layout().addWidget(form)
        ib.layout().setContentsMargins(checkWidth, 5, 5, 5)
        
        cb.disables.append(ib)
        cb.makeConsistent()
        
        OWGUI.separator(box)
        cbAuto = OWGUI.checkBox(box, self, "CommitOnChange", "Commit on change")
        btCommit = OWGUI.button(box, self, "&Commit", self.commit_data, default=True)
        OWGUI.setStopper(self, btCommit, cbAuto, "selectionChanged", self.commit_data)
        
        
        OWGUI.rubber(self.controlArea)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveGraph)

        self.scale_scene = scale = ScaleScene(self, self)
        self.headerView = ScaleView(scale, self)
        self.footerView = ScaleView(scale, self)
        
        self.dendrogram = DendrogramScene(self)
        self.dendrogramView = DendrogramView(self.dendrogram, self.mainArea)
        
        self.connect(self.dendrogram, SIGNAL("clusterSelectionChanged()"), self.on_selection_change)
        self.connect(self.dendrogram, SIGNAL("sceneRectChanged(QRectF)"), scale.scene_rect_update)
        self.connect(self.dendrogram, SIGNAL("dendrogramGeometryChanged(QRectF)"), self.on_dendrogram_geometry_change)
        self.connect(self.dendrogram, SIGNAL("cutoffValueChanged(float)"), self.on_cuttof_value_changed)
        
        self.connect(self.dendrogramView, SIGNAL("viewportResized(QSize)"), self.on_width_changed)
        self.connect(self.dendrogramView, SIGNAL("transformChanged(QTransform)"), self.headerView.setTransform)
        self.connect(self.dendrogramView, SIGNAL("transformChanged(QTransform)"), self.footerView.setTransform)
        
        self.mainArea.layout().addWidget(self.headerView)
        self.mainArea.layout().addWidget(self.dendrogramView)
        self.mainArea.layout().addWidget(self.footerView)
        
        self.dendrogram.header = self.headerView
        self.dendrogram.footer = self.footerView

        self.connect(self.dendrogramView.horizontalScrollBar(),SIGNAL("valueChanged(int)"),
                self.footerView.horizontalScrollBar().setValue)
        self.connect(self.dendrogramView.horizontalScrollBar(),SIGNAL("valueChanged(int)"),
                self.headerView.horizontalScrollBar().setValue)
        self.dendrogram.setSceneRect(0, 0, self.HDSize,self.VDSize)
        self.dendrogram.update()
        self.resize(800, 500)
        
        self.natural_dendrogram_width = 800 
        self.dendrogramView.set_fit_to_width(not self.ManualHorSize)
        
        self.matrix = None
        self.selectionList = []
        self.selected_clusters = [] 

    def sendReport(self):
        self.reportSettings("Settings",
                            [("Linkage", self.linkageMethods[self.Linkage]),
                             ("Annotation", self.labelCombo.currentText()),
                             self.PrintDepthCheck and ("Shown depth limited to", self.PrintDepth),
                             self.SelectionMode and hasattr(self, "cutoff_height") and ("Cutoff line at", self.cutoff_height)])
        
        self.reportSection("Dendrogram")
        canvases = header, graph, footer = self.headerView.scene(), self.dendrogramView.scene(), self.footerView.scene()
        buffer = QPixmap(max(c.width() for c in canvases), sum(c.height() for c in canvases))
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        header.render(painter, QRectF(0, 0, header.width(), header.height()), QRectF(0, 0, header.width(), header.height()))
        graph.render(painter, QRectF(0, header.height(), graph.width(), graph.height()), QRectF(0, 0, graph.width(), graph.height()))
        footer.render(painter, QRectF(0, header.height()+graph.height(), footer.width(), footer.height()), QRectF(0, 0, footer.width(), footer.height()))
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))
        
    def clear(self):
        self.matrix = None
        self.root_cluster = None
        self.selected_clusters = None
        self.dendrogram.clear()
        self.labelCombo.clear()
        
    def set_matrix(self, data):
        self.clear()
        self.matrix = data
        self.closeContext()
        if not self.matrix:
            self.root_cluster = None
            self.selectedExamples = None
            self.dendrogram.clear()
            self.labelCombo.clear()
            self.send("Selected Examples", None)
            self.send("Unselected Examples", None)
            self.classificationBox.setDisabled(True)
            return

        self.matrixSource = "Unknown"
        items = getattr(self.matrix, "items")
        if isinstance(items, orange.ExampleTable): #Example Table from Example Distance

            self.labels = ["None", "Default"]+ \
                          [a.name for a in items.domain.attributes]
            if items.domain.classVar:
                self.labels.append(items.domain.classVar.name)

            self.labelInd = range(len(self.labels) - 2)
            self.labels.extend([m.name for m in items.domain.getmetas().values()])
            self.labelInd.extend(items.domain.getmetas().keys())
            self.numMeta = len(items.domain.getmetas())
            self.metaLabels = items.domain.getmetas().values()
            self.matrixSource = "Example Distance"
        elif isinstance(items, list):   # a list of item (most probably strings)
            self.labels = ["None", "Default", "Name", "Strain"]
            self.Annotation = 0
            self.matrixSource = "Data Distance"
        else:   #From Attribute Distance
            self.labels = ["None", "Attribute Name"]
            self.Annotation = 1
            self.matrixSource="Attribute Distance"
            
        self.labelCombo.clear()
        self.labelCombo.addItems(self.labels)
        if len(self.labels) < self.Annotation - 1:
                self.Annotation = 0
        self.labelCombo.setCurrentIndex(self.Annotation)
        if self.matrixSource == "Example Distance":
            self.classificationBox.setDisabled(False)
        else:
            self.classificationBox.setDisabled(True)
        if self.matrixSource == "Example Distance":
            self.openContext("", items)
        self.error(0)
        try:
            self.run_clustering()
        except orange.KernelException, ex:
            self.error(0, "Could not cluster data! %s" % ex.message)
            self.setMatrix(None)
        
    def update_labels(self):
        """ Change the labels in the scene.
        """
        if self.matrix is None:
            return
        
        items = getattr(self.matrix, "items", range(self.matrix.dim))
        
        if self.Annotation == 0:
            labels = [""] * len(items)
        elif self.Annotation == 1:
            if isinstance(items, ExampleTable):
                labels = [str(i) for i in range(len(items))]
            else:
                try:
                    labels = [item.name for item in items]
                except AttributeError:
                    labels = [str(item) for item in items]
        elif self.Annotation > 1 and isinstance(items, ExampleTable):
            attr = self.labelInd[min(self.Annotation - 2, len(self.labelInd) - 1)]
            labels = [str(ex[attr]) for ex in items]
        else:
            labels = [str(item) for item in items]
            
        self.dendrogram.set_labels(labels)
        self.dendrogram.set_tool_tips(labels)

    def run_clustering(self):
        if self.matrix:
            self.progressBarInit()
            self.root_cluster = orange.HierarchicalClustering(self.matrix,
                linkage=self.linkage[self.Linkage][1],
                progressCallback=lambda value, a: self.progressBarSet(value*100))
            self.progressBarFinished()
            self.display_tree()

    def display_tree(self):
        root = self.root_cluster
        if self.PrintDepthCheck:
            root = hierarchical.pruned(root, level=self.PrintDepth)
        self.display_tree1(root)
        
    def display_tree1(self, tree):
        self.dendrogram.clear()
        self.update_font()
        self.cutoff_height = tree.height * 0.95
        self.dendrogram.set_cluster(tree)
        self.update_labels()
        self.update_cutoff_line()
        
    def update_font(self):
        font = self.font()
        font.setPointSize(self.TextSize)
        self.dendrogram.setFont(font)
        if self.dendrogram.widget:
            self.update_labels()
            
    def update_spacing(self):
        if self.dendrogram.labels_widget:
            layout = self.dendrogram.labels_widget.layout()
            layout.setSpacing(self.LineSpacing)
        
    def on_width_changed(self, size=None):
        if size is not None:
            auto_width = size.width() - 20
        else:
            auto_width = self.natural_dendrogram_width
            
        self.dendrogramView.set_fit_to_width(not self.ManualHorSize)
        if self.ManualHorSize:
            self.dendrogram.set_scene_width_hint(self.HDSize)
            self.update_labels()
        else:
            self.dendrogram.set_scene_width_hint(auto_width)
            self.update_labels()
            
    def on_dendrogram_geometry_change(self, geometry):
        if self.root_cluster and self.dendrogram.widget:
            widget = self.dendrogram.widget
            left, top, right, bottom = widget.layout().getContentsMargins()
            geometry = geometry.adjusted(left, top, -right, -bottom)
            self.scale_scene.set_scale_bounds(geometry.left(), geometry.right())
            self.scale_scene.set_scale(self.root_cluster.height, 0.0)
            pos = widget.pos_at_height(self.cutoff_height)
            self.scale_scene.set_marker(pos)
            
    def on_depth_change(self):
        if self.root_cluster and self.dendrogram.widget:
            selected = self.dendrogram.widget.selected_clusters()
            selected = set([(c.first, c.last) for c in selected])
            root = self.root_cluster
            if self.PrintDepthCheck:
                root = hierarchical.pruned(root, level=self.PrintDepth)
            self.display_tree1(root)
            
            selected = [c for c in hierarchical.preorder(root) if (c.first, c.last) in selected]
            self.dendrogram.widget.set_selected_clusters(selected) 

    def set_cuttof_position_from_scale(self, pos):
        """ Cuttof position changed due to a mouse event in the scale scene.
        """
        if self.root_cluster and self.dendrogram.cutoff_line:
            height = self.dendrogram.widget.height_at(pos)
            self.cutoff_height = height
            line = self.dendrogram.cutoff_line.set_cutoff_at_height(height)
            
    def on_cuttof_value_changed(self, height):
        """ Cuttof value changed due to line drag in the dendrogram.
        """
        if self.root_cluster and self.dendrogram.widget:
            self.cutoff_height = height
            widget = self.dendrogram.widget
            pos = widget.pos_at_height(height)
            self.scale_scene.set_marker(pos)
            
    def update_cutoff_line(self):
        if self.matrix:
            if self.SelectionMode:
                self.dendrogram.cutoff_line.show()
                self.scale_scene.marker.show()
            else:
                self.dendrogram.cutoff_line.hide()
                self.scale_scene.marker.hide()
            
    def on_selection_change(self):
        if self.matrix:
            try:
                items = self.dendrogram.widget.selected_items
                self.selected_clusters = [item.cluster for item in items]
                self.commit_data_if()
            except RuntimeError: # underlying C/C++ object has been deleted
                pass
        
    def commit_data_if(self):
        if self.CommitOnChange:
            self.commit_data()
        else:
            self.selectionChanged = True

    def commit_data(self):
        items = getattr(self.matrix, "items", None)
        if not items:
            return # nothing to commit
        
        self.selectionChanged = False
        self.selectedExamples = None
        selection = self.selected_clusters
        selection = sorted(selection, key=lambda c: c.first)
        maps = [list(self.root_cluster.mapping[c.first: c.last]) for c in selection]
        
        from operator import add
        selected_indices = reduce(add, maps, [])
        unselected_indices = sorted(set(self.root_cluster.mapping) - set(selected_indices))
        
        self.selection = selected = [items[k] for k in selected_indices]
        unselected = [items[k] for k in unselected_indices]
        
        if not selected:
            self.send("Selected Examples", None)
            self.send("Unselected Examples", None)
            self.send("Centroids", None)
            return
        
        if isinstance(items, ExampleTable): 
            c = [i for i in range(len(maps)) for j in maps[i]]
            aid = clustVar = None
            if self.AppendClusters:
                clustVar = orange.EnumVariable(str(self.ClassifyName) ,
                            values=["Cluster " + str(i) for i in range(len(maps))] + ["Other"])
                origDomain = items.domain
                if self.addIdAs == 0:
                    domain = orange.Domain(origDomain.attributes, clustVar)
                    if origDomain.classVar:
                        domain.addmeta(orange.newmetaid(), origDomain.classVar)
                    aid = -1
                elif self.addIdAs == 1:
                    domain = orange.Domain(origDomain.attributes + [clustVar], origDomain.classVar)
                    aid = len(origDomain.attributes)
                else:
                    domain = orange.Domain(origDomain.attributes, origDomain.classVar)
                    aid = orange.newmetaid()
                    domain.addmeta(aid, clustVar)

                domain.addmetas(origDomain.getmetas())
                table1 = table2 = None
                if selected:
                    table1 = orange.ExampleTable(domain, selected)
                    for i in range(len(selected)):
                        table1[i][clustVar] = clustVar("Cluster " + str(c[i]))
                
                if unselected:
                    table2 = orange.ExampleTable(domain, unselected)
                    for ex in table2:
                        ex[clustVar] = clustVar("Other")

                self.selectedExamples = table1
                self.unselectedExamples = table2
            else:
                self.selectedExamples = orange.ExampleTable(selected) if selected else None
                self.unselectedExamples = orange.ExampleTable(unselected) if unselected else None
                
            self.send("Selected Examples", self.selectedExamples)
            self.send("Unselected Examples", self.unselectedExamples)

            self.centroids = None
            if self.selectedExamples:
                self.centroids = orange.ExampleTable(self.selectedExamples.domain)
                for i in range(len(maps)):
                    clusterEx = [ex for cluster, ex in zip(c, self.selectedExamples) if cluster == i]
                    clusterEx = orange.ExampleTable(clusterEx)
                    contstat = orange.DomainBasicAttrStat(clusterEx)
                    discstat = orange.DomainDistributions(clusterEx, 0, 0, 1)
                    ex = [cs.avg if cs else (ds.modus() if ds else "?") for cs, ds in zip(contstat, discstat)]
                    example = orange.Example(self.centroids.domain, ex)
                    if clustVar is not None:
                        example[clustVar] = clustVar(i)
                    self.centroids.append(ex)
            self.send("Centroids", self.centroids)
            
        elif self.matrixSource=="Data Distance":
            names=list(set([d.strain for d in self.selection]))
            data=[(name, [d for d in filter(lambda a:a.strain==name, self.selection)]) for name in names]
            self.send("Structured Data Files",data)
            
    def saveGraph(self):
       sizeDlg = OWChooseImageSizeDlg(self.dendrogram, parent=self)
       sizeDlg.exec_()
       

class DendrogramView(QGraphicsView):
    def __init__(self, *args):
        QGraphicsView.__init__(self, *args)
        self.viewport().setMouseTracking(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        self.setFocusPolicy(Qt.WheelFocus)
#        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.setRenderHints(QPainter.TextAntialiasing)
        self.fit_contents = True
        self.connect(self, SIGNAL("viewportResized(QSize)"), self.updateTransform)
        self.connect(self.scene(), SIGNAL("sceneRectChanged(QRectF)"), lambda: self.updateTransform(self.viewport().size()))

    def resizeEvent(self, e):
        QGraphicsView.resizeEvent(self, e)
        self.emit(SIGNAL("viewportResized(QSize)"), e.size())
        
    def updateTransform(self, size):
        return
#        if self.fit_contents:
#            scene_rect = self.scene().sceneRect()
#            trans = QTransform()
#            scale = size.width() / scene_rect.width()
#            trans.scale(scale, scale)
#            self.setTransform(trans)
#        else:
#            self.setTransform(QTransform())
#        self.emit(SIGNAL("transformChanged(QTransform)"), self.transform())
        
    def set_fit_to_width(self, state):
        self.fit_contents = state
        self.updateTransform(self.viewport().size())

from OWGraphics import GraphicsSimpleTextList

class DendrogramScene(QGraphicsScene):
    def __init__(self, *args):
        QGraphicsScene.__init__(self, *args)
        self.root_cluster = None
        self.header = None
        self.footer = None
        
        self.grid_widget = None
        self.widget = None
        self.labels_widget = None
        self.scene_width_hint = 800
        self.leaf_labels = {}

    def set_cluster(self, root):
        """ Set the cluster to display 
        """
        self.clear()
        self.root_cluster = root
        self.cutoff_line = None
        
        if not self.root_cluster:
            return
        
        # the main widget containing the dendrogram and labels
        self.grid_widget = QGraphicsWidget()
        self.addItem(self.grid_widget)
        layout = QGraphicsGridLayout()
        self.grid_widget.setLayout(layout)
        
        # dendrogram widget
        self.widget = widget = OWClustering.DendrogramWidget(root=root, orientation=Qt.Vertical, parent=self.grid_widget)
        self.connect(widget, SIGNAL("dendrogramGeometryChanged(QRectF)"), 
                     lambda rect: self.emit(SIGNAL("dendrogramGeometryChanged(QRectF)"),
                                            rect))
        self.connect(widget, SIGNAL("selectionChanged()"),
                     lambda: self.emit(SIGNAL("clusterSelectionChanged()"))
                     )
        
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addItem(self.widget, 0, 0)
        self.grid_widget.setMinimumWidth(self.scene_width_hint)
        self.grid_widget.setMaximumWidth(self.scene_width_hint)
        
        spacing = QFontMetrics(self.font()).lineSpacing()
        left, top, right, bottom = widget.layout().getContentsMargins()
        widget.layout().setContentsMargins(0.0, spacing / 2.0, 0.0, spacing / 2.0)
        
        labels = [self.cluster_text(leaf.cluster) for leaf in widget.leaf_items()]
        
        # Labels widget
        labels = GraphicsSimpleTextList(labels, orientation=Qt.Vertical, parent=self.grid_widget)
        labels.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        labels.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        labels.setFont(self.font())
        layout.addItem(labels, 0, 1)
        
        # Cutoff line    
        self.cutoff_line = OWClustering.CutoffLine(widget)
        self.connect(self.cutoff_line.emiter, SIGNAL("cutoffValueChanged(float)"),
                     lambda val: self.emit(SIGNAL("cutoffValueChanged(float)"), val))
        self.cutoff_line.set_cutoff_at_height(self.root_cluster.height * 0.95)
            
        self.labels_widget = labels
        
        layout.activate() 
        self._update_scene_rect()
        
    def cluster_text(self, cluster):
        """ Return the text to display next to the cluster.
        """
        if cluster in self.leaf_labels:
            return self.leaf_labels[cluster]
        elif cluster.first == cluster.last - 1:
            value = str(cluster.mapping[cluster.first])
            return value
        else:
            values = [str(cluster.mapping[i]) \
                      for i in range(cluster.first, cluster.last)]
            return str(values[0]) + "..."

    def set_labels(self, labels):
        """ Set the item labels.
        """
        assert(len(labels) == len(self.root_cluster.mapping))
        if self.labels_widget:
            label_items = []
            for leaf in self.widget.leaf_items():
                cluster = leaf.cluster
                indices = cluster.mapping[cluster.first: cluster.last]
                text = [labels[i] for i in indices]
                if len(text) > 1:
                    text = text[0] + "..."
                else:
                    text = text[0]
                label_items.append(text)
                
            self.labels_widget.set_labels(label_items)
        self._update_scene_rect()

    def set_tool_tips(self, tool_tips):
        """ Set the item tool tips.
        """
        assert(len(tool_tips) == len(self.root_cluster.mapping))
        if self.labels_widget:
            for leaf, label in zip(self.widget.leaf_items(), self.labels_widget):
                cluster = leaf.cluster
                indices = cluster.mapping[cluster.first: cluster.last]
                text = [tool_tips[i] for i in indices]
                text = "<br>".join(text)
                label.setToolTip(text)
                
    def set_scene_width_hint(self, width):
        self.scene_width_hint = width
        if self.grid_widget:
            self.grid_widget.setMinimumWidth(self.scene_width_hint)
            self.grid_widget.setMaximumWidth(self.scene_width_hint)
            
    def clear(self):
        self.widget = None
        self.grid_widget = None
        self.labels_widget = None
        self.root_cluster = None
        self.cutoff_line = None
        QGraphicsScene.clear(self)
        
    def setFont(self, font):
        QGraphicsScene.setFont(self, font)
        if self.labels_widget:
            self.labels_widget.setFont(self.font())
        if self.widget:
            # Fix widget top and bottom margins.
            spacing = QFontMetrics(self.font()).lineSpacing()
            left, top, right, bottom = self.widget.layout().getContentsMargins()
            self.widget.layout().setContentsMargins(left, spacing / 2.0, right, spacing / 2.0)
            self.grid_widget.resize(self.grid_widget.sizeHint(Qt.PreferredSize))
            
    def _update_scene_rect(self):
        self.setSceneRect(reduce(QRectF.united, [item.sceneBoundingRect() for item in self.items()], QRectF()).adjusted(-10, -10, 10, 10))
        
    
class AxisScale(QGraphicsWidget):
    """ A graphics widget for an axis scale
    """
    # Defaults
    orientation = Qt.Horizontal
    tick_count=5
    tick_align = Qt.AlignTop
    text_align = Qt.AlignHCenter | Qt.AlignBottom
    axis_scale = (0.0, 1.0)
    
    def __init__(self, parent=None, orientation=Qt.Horizontal, tick_count=5,
                 tick_align = Qt.AlignBottom,
                 text_align = Qt.AlignHCenter | Qt.AlignBottom,
                 axis_scale = (0.0, 1.0)):
        QGraphicsWidget.__init__(self, parent) 
        self.orientation = orientation
        self.tick_count = tick_count
        self.tick_align = tick_align
        self.text_align = text_align
        self.axis_scale = axis_scale
        
    def set_orientation(self, orientation):
        self.orientation = orientation
        if self.orientation == Qt.Horizontal:
            self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        self.updateGeometry()
        
    def ticks(self):
        minval, maxval = self.axis_scale
        ticks = ["%.2f" % val for val in numpy.linspace(minval, maxval, self.tick_count)]
        return ticks
        
    def paint(self, painter, option, widget=0):
        painter.setFont(self.font())
        size = self.geometry().size()
        metrics = QFontMetrics(painter.font())
        minval, maxval = self.axis_scale
        tick_count = self.tick_count
        
        if self.orientation == Qt.Horizontal:
            spanx, spany = size.width(), 0.0
            xadv, yadv =  spanx / tick_count, 0.0
            tick_w, tick_h = 0.0, -5.0
            tick_offset = QPointF(0.0, tick_h)
        else:
            spanx, spany = 0.0, size.height()
            xadv, yadv = 0.0, spany / tick_count
            tick_w, tick_h = 5.0, 0.0
            tick_func = lambda : (y / spany)
            tick_offset = QPointF(tick_w + 1.0, metrics.ascent()/2)
            
        ticks = self.ticks()
            
        xstart, ystart = 0.0, 0.0
        
        if self.orientation == Qt.Horizontal:
            painter.translate(0.0, size.height())
            
        painter.drawLine(xstart, ystart, xstart + tick_count*xadv, ystart + tick_count*yadv)
        
        linspacex = numpy.linspace(0.0, spanx, tick_count)
        linspacey = numpy.linspace(0.0, spany, tick_count)
        
        metrics = painter.fontMetrics()
        for x, y, tick in zip(linspacex, linspacey, ticks):
            painter.drawLine(x, y, x + tick_w, y + tick_h)
            if self.orientation == Qt.Horizontal:
                rect = QRectF(metrics.boundingRect(tick))
                rect.moveCenter(QPointF(x, y) + tick_offset + QPointF(0.0, -rect.height()/2.0))
                painter.drawText(rect, tick)
            else:
                painter.drawText(QPointF(x, y) + tick_offset, tick)
                
#        rect =  self.geometry()
#        rect.translate(-self.pos())
#        painter.drawRect(rect)
        
    def setGeometry(self, rect):
        self.prepareGeometryChange()
        return QGraphicsWidget.setGeometry(self, rect)
        
    def sizeHint(self, which, *args):
        if which == Qt.PreferredSize:
            minval, maxval = self.axis_scale
            ticks = self.ticks()
            metrics = QFontMetrics(self.font())
            if self.orientation == Qt.Horizontal:
                h = metrics.height() + 5
                w = 100 
            else:
                h = 100
                w = max([metrics.width(t) for t in ticks]) + 5
            return QSizeF(w, h)
        else:
            return QSizeF()

    def boundingRect(self):
        metrics = QFontMetrics(self.font())
        geometry = self.geometry()
        ticks = self.ticks()
        if self.orientation == Qt.Horizontal:
            h = 5 + metrics.height()
            left = - metrics.boundingRect(ticks[0]).width()/2.0 #+ geometry.left() 
            right = geometry.width() + metrics.boundingRect(ticks[-1]).width()/2.0
            rect = QRectF(left, 0.0, right - left, h)
        else:
            h = geometry.height()
            w = max([metrics.width(t) for t in ticks]) + 5
            rect = QRectF(0.0, 0.0, w, h) 
        return rect
    
    def set_axis_scale(self, min, max):
        self.axis_scale = (min, max)
        self.updateGeometry()
        
    def set_axis_ticks(self, ticks):
        if isinstance(ticks, dict):
            self.ticks = ticks
        self.updateGeometry()
            
    def tick_layout(self):
        """ Return the tick layout
        """
        minval, maxval = self.axis_scale
        ticks = numpy.linspace(minval, maxval, self.tick_count)
        return zip(ticks, self.ticks())
    
#        min, max = self.axis_scale
#        ticks = self.ticks()
#        span = max - min
#        span_log = math.log10(span)
#        log_sign = -1 if log_sign < 0.0 else 1
#        span_log = math.floor(span_log)
#        major_ticks = [(x, 5.0, tick(i, span_log)) for i in range(self.tick_count)]
#        minor_ticks = [(x, 3.0, tick(i, span_log + log_sign))  for i in range(self.tick_count * 2)]
#        return [(i, major, label) for i, tick, label in major_ticks]
    
    
class ScaleScene(QGraphicsScene):
    def __init__(self, widget, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.widget = widget
        self.scale_widget = AxisScale(orientation=Qt.Horizontal)
        font = self.font()
        font.setPointSize(10)
        self.scale_widget.setFont(font)
        self.marker = QGraphicsLineItem()
        pen = QPen(Qt.black, 2)
        pen.setCosmetic(True)
        self.marker.setPen(pen)
        self.marker.setLine(0.0, 0.0, 0.0, 25.0)
        self.marker.setCursor(Qt.SizeHorCursor)
        self.addItem(self.scale_widget)
        self.addItem(self.marker)
        self.setSceneRect(QRectF(0, 0, self.scale_widget.size().height(), 25.0))
        
    def set_scale(self, min, max):
        self.scale_widget.set_axis_scale(min, max)
        
    def set_scale_bounds(self, start, end):
        self.scale_widget.setPos(start, 0)
        size_hint = self.scale_widget.sizeHint(Qt.PreferredSize)
        self.scale_widget.resize(end - start, self.scale_widget.size().height())
        
    def scene_rect_update(self, rect):
        scale_rect = self.scale_widget.sceneBoundingRect()
        rect = QRectF(rect.x(), scale_rect.y(), rect.width(), scale_rect.height())
        rect = QRectF(rect.x(), 0.0, rect.width(), scale_rect.height())
        self.marker.setLine(0, 0, 0, scale_rect.height())
        self.setSceneRect(rect)
        
    def set_marker(self, pos):
        pos = self.scale_widget.mapToScene(pos)
        self.marker.setPos(pos.x(), 0.0)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.scale_widget.mapFromScene(event.scenePos())
            self.widget.set_cuttof_position_from_scale(pos)
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            pos = self.scale_widget.mapFromScene(event.scenePos())
            self.widget.set_cuttof_position_from_scale(pos)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.scale_widget.mapFromScene(event.scenePos())
            self.widget.set_cuttof_position_from_scale(pos)
            
                
class ScaleView(QGraphicsView):
    def __init__(self, scene=None, parent=None):
        QGraphicsView.__init__(self, scene, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setAlignment(Qt.AlignLeft | Qt.AlignCenter)
        self.setFixedHeight(25.0)


def test():
    app=QApplication(sys.argv)
    w=OWHierarchicalClustering()
    w.show()
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
    id=orange.newmetaid()
    data.domain.addmeta(id, orange.FloatVariable("a"))
    data.addMetaAttribute(id)
    matrix = orange.SymMatrix(len(data))
    dist = orange.ExamplesDistanceConstructor_Euclidean(data)
    matrix = orange.SymMatrix(len(data))
    matrix.setattr('items', data)
    for i in range(len(data)):
        for j in range(i+1):
            matrix[i, j] = dist(data[i], data[j])

    w.set_matrix(matrix)
    app.exec_()
    
if __name__=="__main__":
    test()

"""
<name>Decision Models Visualization</name>
<description>Orange widget for visualization of decision models.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>0</priority>
"""

import orange
import orngVizRank
import scipy.stats
import orngNetwork

import OWToolbars

from orngScaleLinProjData import *
from OWWidget import *
from OWNetworkCanvas import *
from OWkNNOptimization import OWVizRank
from OWNetworkHist import *

dir = os.path.dirname(__file__) + "/../"

ICON_PATHS = [("TREE",              "Classify/icons/ClassificationTree"),
              ("SCATTERPLOT",       "Visualize/icons/ScatterPlot"),
              ("SCATTTERPLOT",      "Visualize/icons/ScatterPlot"),
              ("LINEAR_PROJECTION", "Visualize/icons/LinearProjection"),
              ("SPCA",              "Visualize/icons/LinearProjection"),
              ("RADVIZ",            "Visualize/icons/Radviz"),
              ("POLYVIZ",           "Visualize/icons/Polyviz"),
              ("BAYES",             "Classify/icons/NaiveBayes"),
              ("KNN",               "Classify/icons/kNearestNeighbours")]

ICON_SIZES = ["16", "32", "40", "48", "60"]

MODEL_IMAGES = {"MISSING": "%sicons/Unknown.png" % dir}

for size in ICON_SIZES:
    for model, path in ICON_PATHS:
        MODEL_IMAGES[model + size] = "%s%s_%s.png" % (dir, path, size)

dir = os.path.dirname(__file__) + "/../icons/"
dlg_mark2sel   = dir + "Dlg_Mark2Sel.png"
dlg_sel2mark   = dir + "Dlg_Sel2Mark.png"
dlg_selIsmark  = dir + "Dlg_SelisMark.png"
dlg_selected   = dir + "Dlg_SelectedNodes.png"
dlg_unselected = dir + "Dlg_UnselectedNodes.png"
dlg_showall    = dir + "Dlg_clear.png"

class ProjCurve(NetworkCurve):
    def __init__(self, parent, pen=QPen(Qt.black), xData=None, yData=None):
        NetworkCurve.__init__(self, parent, pen=QPen(Qt.black), xData=None, yData=None)
        
    def draw(self, painter, xMap, yMap, rect):
        for edge in self.edges:
            if edge.u.show and edge.v.show:
                painter.setPen(edge.pen)
    
                px1 = xMap.transform(self.coors[0][edge.u.index])   #ali pa tudi self.x1, itd
                py1 = yMap.transform(self.coors[1][edge.u.index])
                px2 = xMap.transform(self.coors[0][edge.v.index])
                py2 = yMap.transform(self.coors[1][edge.v.index])
                
                painter.drawLine(px1, py1, px2, py2)
                
                d = 12
                #painter.setPen(QPen(Qt.lightGray, 1))
                painter.setBrush(Qt.lightGray)
                if edge.arrowu:
                    x = px1 - px2
                    y = py1 - py2
                    
                    fi = math.atan2(y, x) * 180 * 16 / math.pi 
        
                    if not fi is None:
                        # (180*16) - fi - (20*16), (40*16)
                        painter.drawPie(px1 - d, py1 - d, 2 * d, 2 * d, 2560 - fi, 640)
                        
                if edge.arrowv:
                    x = px1 - px2
                    y = py1 - py2
                    
                    fi = math.atan2(y, x) * 180 * 16 / math.pi 
                    if not fi is None:
                        # (180*16) - fi - (20*16), (40*16)
                        painter.drawPie(px1 - d, py1 - d, 2 * d, 2 * d, 2560 - fi, 640)
                        
                if self.showEdgeLabels and len(edge.label) > 0:
                    lbl = ', '.join(edge.label)
                    x = (px1 + px2) / 2
                    y = (py1 + py2) / 2
                    
                    th = painter.fontMetrics().height()
                    tw = painter.fontMetrics().width(lbl)
                    r = QRect(x - tw / 2, y - th / 2, tw, th)
                    painter.fillRect(r, QBrush(Qt.white))
                    painter.drawText(r, Qt.AlignHCenter + Qt.AlignVCenter, lbl)
    
        for vertex in self.vertices:
            if vertex.show:
                pX = xMap.transform(self.coors[0][vertex.index])   #dobimo koordinati v pikslih (tipa integer)
                pY = yMap.transform(self.coors[1][vertex.index])   #ki se stejeta od zgornjega levega kota canvasa
                style = (1 - vertex.style) * 2 * 100
                #print style
                #style=-50
                if vertex.selected:
                    size = int(vertex.size) + 5
                    brushColor = QColor(Qt.yellow)
                    brushColor.setAlpha(150)
                    #painter.setPen(QPen(brushColor, 0))
                    painter.setPen(QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap))
                    gradient = QRadialGradient(QPointF(pX, pY), size)
                    gradient.setColorAt(0., brushColor)
                    gradient.setColorAt(1., QColor(255, 255, 255, 0))
                    painter.setBrush(QBrush(gradient))
                    painter.drawRoundedRect(pX - size/2, pY - size/2, size, size, style, style, Qt.RelativeSize)
                    #painter.drawEllipse(pX - size/2, pY - size/2, size, size)
                elif vertex.marked:
                    size = int(vertex.size) + 5
                    brushColor = QColor(Qt.cyan)
                    brushColor.setAlpha(80)
                    #painter.setPen(QPen(brushColor, 0))
                    painter.setPen(QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap))
                    gradient = QRadialGradient(QPointF(pX, pY), size)
                    gradient.setColorAt(0., brushColor)
                    gradient.setColorAt(1., QColor(255, 255, 255, 0))
                    painter.setBrush(QBrush(gradient))
                    painter.drawRoundedRect(pX - size/2, pY - size/2, size, size, style, style, Qt.RelativeSize)
                    #painter.drawEllipse(pX - size/2, pY - size/2, size, size)
                else:
                    painter.setPen(QPen(QBrush(QColor(125, 162, 206, 192)), 1, Qt.SolidLine, Qt.RoundCap))
                    size = int(vertex.size) + 5
                    gradient = QRadialGradient(QPointF(pX, pY), size)
                    gradient.setColorAt(0., QColor(217, 232, 252, 192))
                    gradient.setColorAt(1., QColor(255, 255, 255, 0))
                    painter.setBrush(QBrush(gradient))
                    painter.drawRoundedRect(pX - size/2, pY - size/2, size, size, style, style, Qt.RelativeSize)
                    #painter.drawEllipse(pX - size/2, pY - size/2, size, size)
    
        for vertex in self.vertices:
            if vertex.show:                
                pX = xMap.transform(self.coors[0][vertex.index])   #dobimo koordinati v pikslih (tipa integer)
                pY = yMap.transform(self.coors[1][vertex.index])   #ki se stejeta od zgornjega levega kota canvasa
#               
                if vertex.image:
                    size = vertex.image.size().width()
                    painter.drawImage(QRect(pX - size/2, pY - size/2, size, size), vertex.image)
                    
class OWMetaMinerCanvas(OWNetworkCanvas):
    
    def __init__(self, master, parent=None, name="None"):
        OWNetworkCanvas.__init__(self, master, parent, name)
        self.networkCurve = ProjCurve(self)
        self.selectionNeighbours = 1
        self.tooltipNeighbours = 1
        
    def drawToolTips(self):
        # add ToolTips
        self.tooltipData = []
        self.tooltipKeys = {}
        self.tips.removeAll()
        
        for vertex in self.vertices:
            if not vertex.show:
                continue
            
            x1 = self.visualizer.network.coors[0][vertex.index]
            y1 = self.visualizer.network.coors[1][vertex.index]
            lbl  = "CA: %.4g\n" % self.visualizer.graph.items[vertex.index]["CA"].value
            lbl += "AUC: %.4g\n" % self.visualizer.graph.items[vertex.index]["AUC"].value
            lbl += "CA best: %.4g\n" % self.visualizer.graph.items[vertex.index]["cluster CA"].value
            lbl += "Attributes: %d\n" % len(self.visualizer.graph.items[vertex.index]["label"].value.split(", "))
            lbl += ", ".join(sorted(self.visualizer.graph.items[vertex.index]["label"].value.split(", ")))
            self.tips.addToolTip(x1, y1, lbl)
            self.tooltipKeys[vertex.index] = len(self.tips.texts) - 1
        
    def drawLabels(self):
        pass
    
    def drawIndexes(self):
        pass
    
    def drawWeights(self):
        pass
    
    def loadIcons(self):
        items = self.visualizer.graph.items
        maxsize = str(max(map(int, ICON_SIZES)))
        for v in self.vertices:
            size = maxsize
            for i in range(len(ICON_SIZES) - 1):
                if int(ICON_SIZES[i]) < v.size <= int(ICON_SIZES[i+1]):
                    size = ICON_SIZES[i]
            imageKey = items[v.index]['model'].value + size
            if imageKey not in MODEL_IMAGES:
                imageKey = "MISSING"
            v.image = QImage(MODEL_IMAGES[imageKey])

    def addVisualizer(self, networkOptimization):
        OWNetworkCanvas.addVisualizer(self, networkOptimization, ProjCurve(self))
           
        
class OWMetaMiner(OWWidget, OWNetworkHist):
    settingsList = ["vertexSize", "lastSizeAttribute", "maxVertexSize", "minVertexSize", "tabIndex"]
    
    def __init__(self, parent=None, signalManager=None, name = "Meta Miner"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix, Default)]
        self.outputs = [("Model", orange.Example),
                        ("Classifier", orange.Classifier)]
        
        self.vertexSize = 32
        self.autoSendSelection = False
        self.minVertexSize = 16
        self.maxVertexSize = 16
        self.vertexSizeAttribute = 0
        self.optimization = None
        self.tabIndex = 0
        self.lastSizeAttribute = ''
        self.graph = None
        
        self.loadSettings()
        
        self.netCanvas = OWMetaMinerCanvas(self, self.mainArea, "Meta Miner")
        self.netCanvas.appendToSelection = 0
        self.netCanvas.minVertexSize = self.minVertexSize
        self.netCanvas.maxVertexSize = self.maxVertexSize
        self.netCanvas.invertEdgeSize = 1
        
        self.mainArea.layout().setContentsMargins(0,4,4,4)
        self.controlArea.layout().setContentsMargins(4,4,0,4)
        
        self.mainArea.layout().addWidget(self.netCanvas)
        
        self.hcontroArea = OWGUI.widgetBox(self.controlArea, orientation='horizontal')
        
        self.tabs = OWGUI.tabWidget(self.hcontroArea)
        self.generalTab = OWGUI.createTabPage(self.tabs, "General")
        self.matrixTab = OWGUI.createTabPage(self.tabs, "Matrix")
        self.networkTab = OWGUI.createTabPage(self.tabs, "Network")
        self.tabs.setCurrentIndex(self.tabIndex)
        QObject.connect(self.tabs, SIGNAL("currentChanged(int)"), self.currentTabChanged)
        
        T = OWToolbars.NavigateSelectToolbar
        self.zoomSelectToolbar = T(self, self.hcontroArea, self.netCanvas, self.autoSendSelection,
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
        
        # GENERAL CONTROLS
        self.optimizationButtons = OWGUI.widgetBox(self.generalTab, "Optimization Dialogs", orientation = "vertical")
        self.btnFR = OWGUI.button(self.optimizationButtons, self, "Fruchterman-Reingold", toggleButton=1)
        self.btnMDS = OWGUI.button(self.optimizationButtons, self, "MDS", toggleButton=1)
        
        QObject.connect(self.btnFR,  SIGNAL("clicked()"), (lambda m=0,btn=self.btnFR:  self.optimize(m, btn)))
        QObject.connect(self.btnMDS, SIGNAL("clicked()"), (lambda m=1,btn=self.btnMDS: self.optimize(m, btn)))
        
        # MARTIX CONTROLS
        self.addHistogramControls(self.matrixTab)
        self.kNN = 1
        boxHistogram = OWGUI.widgetBox(self.matrixTab, box = "Distance histogram")
        self.histogram = OWHist(self, boxHistogram)
        boxHistogram.layout().addWidget(self.histogram)
        
        
        # NETWORK CONTROLS
        self.sizeAttributeCombo = OWGUI.comboBox(self.networkTab, self, "vertexSizeAttribute", box = "Vertex size attribute", callback=self.setMaxVertexSize)
        self.sizeAttributeCombo.addItem("(none)")
        OWGUI.spin(self.sizeAttributeCombo.box, self, "minVertexSize", 16, 80, 2, label="Min vertex size:", callback = self.setMaxVertexSize)
        OWGUI.spin(self.sizeAttributeCombo.box, self, "maxVertexSize", 16, 80, 2, label="Max vertex size:", callback = self.setMaxVertexSize)
        
        
        self.generalTab.layout().addStretch(1)
        self.matrixTab.layout().addStretch(1)
        self.networkTab.layout().addStretch(1)
        
        self.icons = self.createAttributeIconDict()
        self.resize(900, 600)
        
        self.netCanvas.callbackSelectVertex = self.sendNetworkSignals

    def currentTabChanged(self, index): 
            self.tabIndex = index
    
    def setMatrix(self, matrix):
        self.warning()
        
        if not matrix:
            self.warning("Data matrix is None.")
            return
        
        if not hasattr(matrix, "items") or not hasattr(matrix, "results"):
            self.warning("Data matrix does not have required data for items and results.")
            return
        
        requiredAttrs = set(["CA", "AUC"])
        attrs = [attr.name for attr in matrix.items.domain] 
        if len(requiredAttrs.intersection(attrs)) != len(requiredAttrs):
            self.warning("Items ExampleTable does not contain required attributes CA and AUC.")
            return
        
        OWNetworkHist.setMatrix(self, matrix)
                
    def setMaxVertexSize(self):
        if self.netCanvas == None:
            return
        
        if self.minVertexSize > self.maxVertexSize:
            self.maxVertexSize = self.minVertexSize
        
        self.netCanvas.minVertexSize = self.minVertexSize
        self.netCanvas.maxVertexSize = self.maxVertexSize
        
        print self.vertexSizeAttribute
        if self.vertexSizeAttribute > 0:
            self.lastSizeAttribute = self.sizeAttributeCombo.currentText()
            self.netCanvas.setVerticesSize(self.lastSizeAttribute)
        else:
            self.netCanvas.setVerticesSize()
        
        self.netCanvas.loadIcons()
        self.netCanvas.replot()
        
    def setVertexStyle(self):
        for v in self.netCanvas.vertices:
            auc = self.matrix.items[v.index]["AUC"].value
            v.style = auc            
        
    def sendNetworkSignals(self):
        if self.graph != None and self.graph.items != None:
            selection = self.netCanvas.getSelectedVertices()
            if len(selection) > 0: 
                model = self.graph.items[selection[0]]
                uuid = model["uuid"].value
                method, vizr_result, projection_points, classifier, attrs = self.matrix.results[uuid]
                
                if len(vizr_result) > 5 and type(vizr_result[5]) == type({}) and "YAnchors" in vizr_result[5] and "XAnchors" in vizr_result[5]:
                    model.domain.addmeta(orange.newmetaid(), orange.PythonVariable("anchors"))
                    model["anchors"] = (vizr_result[5]["XAnchors"], vizr_result[5]["YAnchors"])
                    
                if classifier:
                    self.send("Classifier", classifier)
                    model.domain.addmeta(orange.newmetaid(), orange.PythonVariable("classifier"))
                    model["classifier"] = classifier
                    
                self.send("Model", model)
            else:
                self.send("Model", None)
        else:
            self.send("Model", None)
            
    def mdsProgress(self, avgStress, stepCount):
        self.progressBarSet(int(stepCount * 100 / 10000))
        qApp.processEvents()
            
    def optimize(self, method=0, btn=None):
        if self.graph == None:
            return 
        
        if btn != None:
            btnCaption = btn.text()
            btn.setText("Stop")
            qApp.processEvents()
            
        if btn != None and not btn.isChecked():
            self.optimization.stopMDS = 1
            btn.setChecked(False)
            btn.setText(btnCaption)
            return
        
        self.progressBarInit()
        
        if method == 0:
            tolerance = 5
            initTemp = 1000
            breakpoints = 6
            frSteps = 1500
            k = int(frSteps / breakpoints)
            o = frSteps % breakpoints
            iteration = 0
            coolFactor = math.exp(math.log(10.0/10000.0) / frSteps)
            while iteration < breakpoints:
                initTemp = self.optimization.fruchtermanReingold(k, initTemp, coolFactor)
                iteration += 1
                qApp.processEvents()
                self.netCanvas.updateCanvas()
            initTemp = self.optimization.fruchtermanReingold(o, initTemp, coolFactor)
            qApp.processEvents()
            self.netCanvas.updateCanvas()
            
        if method == 1:
            self.optimization.mdsComponents(10000, 10, callbackProgress=self.mdsProgress, \
                                            callbackUpdateCanvas=self.netCanvas.updateCanvas, \
                                            torgerson=0, minStressDelta=0, avgLinkage=0, rotationOnly=0, \
                                            mdsType=orngNetwork.MdsType.MDS, scalingRatio=1, mdsFromCurrentPos=1)
            self.netCanvas.updateCanvas()
            
        if btn != None:
            btn.setText(btnCaption)
            btn.setChecked(False)
        
        self.progressBarFinished()
        
    def sendSignals(self):
        if self.graph != None:
            self.optimization = orngNetwork.NetworkOptimization(self.graph)
            self.optimization.vertexDistance = self.matrix
            self.netCanvas.addVisualizer(self.optimization)
            self.netCanvas.setLabelText(["label"])    
            self.sizeAttributeCombo.clear()
            self.sizeAttributeCombo.addItem("(same size)")
            vars = self.optimization.getVars()
            for var in vars:
                if var.varType in [orange.VarTypes.Continuous]:
                    self.sizeAttributeCombo.addItem(self.icons[var.varType], unicode(var.name))
            
            index = self.sizeAttributeCombo.findText(self.lastSizeAttribute)
            if index > -1:
                self.sizeAttributeCombo.setCurrentIndex(index)
                self.vertexSizeAttribute = index
                
            self.setMaxVertexSize()
            self.setVertexStyle()
            self.optimize(0)
                
if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWProjViz()
    ow.show()
    appl.exec_()
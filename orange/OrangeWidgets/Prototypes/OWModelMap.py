"""
<name>Model Map (Qwt)</name>
<description>Visualization and analysis of prediction models.</description>
<icon>icons/Network.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>1110</priority>
"""

import scipy.stats

import Orange
import orange
import orngVizRank
import orngStat

import OWToolbars
import OWColorPalette

from orngScaleLinProjData import *
from OWNxExplorer import *
from OWNxCanvas import *
from OWkNNOptimization import OWVizRank
from OWNxHist import *
from OWDistributions import OWDistributionGraph

dir = os.path.dirname(__file__) + "/../"

ICON_PATHS = [("TREE",              "Classify/icons/ClassificationTree"),
              ("SCATTERPLOT",       "Visualize/icons/ScatterPlot"),
              ("SCATTTERPLOT",      "Visualize/icons/ScatterPlot"),
              ("LINEAR_PROJECTION", "Visualize/icons/LinearProjection"),
              ("SPCA",              "Visualize/icons/LinearProjection"),
              ("RADVIZ",            "Visualize/icons/Radviz"),
              ("POLYVIZ",           "Visualize/icons/Polyviz"),
              ("BAYES",             "Classify/icons/NaiveBayes"),
              ("KNN",               "Classify/icons/kNearestNeighbours"),
              ("SVM",               "Classify/icons/BasicSVM")]

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

class ModelCurve(NetworkCurve):
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
                #if vertex.highlight:
                #    painter.setPen(QPen(QBrush(QColor(255,   0,   0, 192)), 1, Qt.SolidLine, Qt.RoundCap))
                #else:
                #    #oldcolor = QColor(125, 162, 206, 192)
                painter.setPen(QPen(QBrush(vertex.color), 1, Qt.SolidLine, Qt.RoundCap))
                
                if vertex.selected:
                    size = int(vertex.size) + 5
                    brushColor = QColor(Qt.yellow)
                    brushColor.setAlpha(150)
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
                    gradient = QRadialGradient(QPointF(pX, pY), size)
                    gradient.setColorAt(0., brushColor)
                    gradient.setColorAt(1., QColor(255, 255, 255, 0))
                    painter.setBrush(QBrush(gradient))
                    painter.drawRoundedRect(pX - size/2, pY - size/2, size, size, style, style, Qt.RelativeSize)
                    #painter.drawEllipse(pX - size/2, pY - size/2, size, size)
                else:
                    size = int(vertex.size) + 5
                    gradient = QRadialGradient(QPointF(pX, pY), size)
                    #gradient.setColorAt(0., QColor(217, 232, 252, 192))
                    gradient.setColorAt(0., vertex.color)
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
                    
class OWModelMapCanvas(OWNxCanvas):
    
    def __init__(self, master, parent=None, name="None"):
        OWNxCanvas.__init__(self, master, parent, name)
        self.networkCurve = ModelCurve(self)
        self.selectionNeighbours = 1
        self.tooltipNeighbours = 1
        self.plotAccuracy = None
        self.vizAttributes = None
        self.radius = 100
        
    def mouseMoveEvent(self, event):
        if self.graph is None or self.layout is None:
          return
        
        if self.plotAccuracy or self.vizAttributes:
            px = self.invTransform(QwtPlot.xBottom, event.x())
            py = self.invTransform(QwtPlot.yLeft, event.y())
            ndx, mind = self.layout.closest_vertex(px, py)
            
            dX = self.transform(QwtPlot.xBottom, self.layout.coors[0][ndx]) - event.x()
            dY = self.transform(QwtPlot.yLeft,   self.layout.coors[1][ndx]) - event.y()
            # transform to pixel distance
            distance = math.sqrt(dX**2 + dY**2) 
            if ndx != -1 and distance <= (self.vertices[ndx].size + 5) / 2:
                toMark = set(self.getNeighboursUpTo(ndx, self.tooltipNeighbours))
                toMark = list(toMark)
                self.networkCurve.setMarkedVertices(toMark)
                if self.plotAccuracy:
                    self.plotAccuracy(toMark)
                if self.vizAttributes:
                    self.vizAttributes(toMark)
                self.drawPlotItems()
            else:
                vd = sorted(self.layout.vertex_distances(px, py))[:10]
                vd = [(math.sqrt((self.transform(QwtPlot.xBottom, self.layout.coors[0][v]) - event.x())**2 + \
                                 (self.transform(QwtPlot.yLeft,   self.layout.coors[1][v]) - event.y())**2), v) for d,v in vd]
                vd = [v for d,v in vd if d < self.radius]
                self.networkCurve.setMarkedVertices(vd)
                if self.plotAccuracy:
                    self.plotAccuracy(vd)
                if self.vizAttributes:
                    self.vizAttributes(vd)
                self.drawPlotItems()
        else:
            OWNxCanvas.mouseMoveEvent(self, event)
        
    def drawToolTips(self):
        # add ToolTips
        self.tooltipData = []
        self.tooltipKeys = {}
        self.tips.removeAll()

        for vertex in self.vertices:
            if not vertex.show:
                continue
            
            clusterCA = self.graph.items()[vertex.index]["cluster CA"].value
            if type(clusterCA) == type(1.): clusterCA = "%.4g" % clusterCA 
            
            x1 = self.layout.coors[0][vertex.index]
            y1 = self.layout.coors[1][vertex.index]
            lbl  = "%s\n" % self.graph.items()[vertex.index]["label"].value
            lbl += "CA: %.4g\n" % self.graph.items()[vertex.index]["CA"].value
            #lbl += "AUC: %.4g\n" % self.graph.items()[vertex.index]["AUC"].value
            #lbl += "CA best: %s\n" % clusterCA 
            lbl += "Attributes: %d\n" % len(self.graph.items()[vertex.index]["attributes"].value.split(", "))
            lbl += ", ".join(sorted(self.graph.items()[vertex.index]["attributes"].value.split(", ")))
            self.tips.addToolTip(x1, y1, lbl)
            self.tooltipKeys[vertex.index] = len(self.tips.texts) - 1
        
    def drawLabels(self):
        pass
    
    def drawIndexes(self):
        pass
    
    def drawWeights(self):
        pass
    
    def loadIcons(self):
        items = self.graph.items()
        maxsize = str(max(map(int, ICON_SIZES)))
        minsize = min(map(int, ICON_SIZES))
        for v in self.vertices:
            size = str(minsize) if v.size <= minsize else maxsize
            
            for i in range(len(ICON_SIZES) - 1):
                if int(ICON_SIZES[i]) < v.size <= int(ICON_SIZES[i+1]):
                    size = ICON_SIZES[i]
            imageKey = items[v.index]['model'].value + size
            if imageKey not in MODEL_IMAGES:
                imageKey = "MISSING"
            v.image = QImage(MODEL_IMAGES[imageKey])

    def set_graph_layout(self, graph, layout):
        OWNxCanvas.set_graph_layout(self, graph, layout, ModelCurve(self))

class OWModelMap(OWNxExplorer, OWNxHist):
    settingsList = ["vertexSize", "lastSizeAttribute", "lastColorAttribute", 
                    "maxVertexSize", "minVertexSize", "tabIndex", 
                    "colorSettings", "selectedSchemaIndex", "iterations", 
                    "radius", "vizAccurancy", "vizAttributes", 
                    "autoSendSelection", "spinExplicit", "spinPercentage",
                    "maxLinkSize", "renderAntialiased", "labelsOnMarkedOnly",
                    "invertSize", "optMethod", "lastVertexSizeColumn", 
                    "lastColorColumn", "lastNameComponentAttribute", 
                    "lastLabelColumns", "lastTooltipColumns", "showWeights",
                    "showIndexes",  "showEdgeLabels", "edgeColorSettings", 
                    "selectedEdgeSchemaIndex", "showMissingValues", "fontSize", 
                    "mdsTorgerson", "mdsAvgLinkage", "mdsSteps", "mdsRefresh", 
                    "mdsStressDelta", "organism","showTextMiningInfo", 
                    "toolbarSelection", "minComponentEdgeWidth", 
                    "maxComponentEdgeWidth", "mdsFromCurrentPos"]
    
    def __init__(self, parent=None, signalManager=None, name="Model Map (Qwt)"):
        OWNxExplorer.__init__(self, parent, signalManager, name, 
                               NetworkCanvas=OWModelMapCanvas)
        
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix, Default),
                       ("Model Subset", orange.ExampleTable, self.setSubsetModels)]
        self.outputs = [("Model", orange.Example),
                        ("Classifier", orange.Classifier),
                        ("Selected Models", orange.ExampleTable)]
        
        self.vertexSize = 32
        self.autoSendSelection = False
        self.minVertexSize = 16
        self.maxVertexSize = 16
        self.lastColorAttribute = ""
        self.lastSizeAttribute = ""
        self.vizAccurancy = False
        self.vizAttributes = False
        self.radius = 100
        self.attrIntersection = []
        self.attrIntersectionList = []
        self.attrDifference = []
        self.attrDifferenceList = []
        
        self.loadSettings()
        
        self.matrixTab = OWGUI.widgetBox(self.tabs, addToLayout = 0, margin = 4)
        self.modelTab = OWGUI.widgetBox(self.tabs, addToLayout = 0, margin = 4)
        self.tabs.insertTab(0, self.matrixTab, "Matrix")
        self.tabs.insertTab(1, self.modelTab, "Model Info")

        self.networkCanvas.appendToSelection = 0
        self.networkCanvas.minVertexSize = self.minVertexSize
        self.networkCanvas.maxVertexSize = self.maxVertexSize
        self.networkCanvas.invertEdgeSize = 1
        self.networkCanvas.callbackSelectVertex = self.sendNetworkSignals
        
        # MARTIX CONTROLS
        self.addHistogramControls(self.matrixTab)
        self.kNN = 1
        boxHistogram = OWGUI.widgetBox(self.matrixTab, box = "Distance histogram")
        self.histogram = OWHist(self, boxHistogram)
        boxHistogram.layout().addWidget(self.histogram)
        
        # VISUALIZATION CONTROLS
        vizPredAcc = OWGUI.widgetBox(self.modelTab, "Prediction Accuracy", orientation = "vertical")
        OWGUI.checkBox(vizPredAcc, self, "vizAccurancy", "Visualize prediction accurancy", callback=self.visualizeInfo)
        OWGUI.spin(vizPredAcc, self, "radius", 10, 300, 1, label="Radius: ", callback = self.visualizeInfo)
        self.predGraph = OWDistributionGraph(self, vizPredAcc)
        self.predGraph.setMaximumSize(QSize(300, 300))
        self.predGraph.setYRlabels(None)
        self.predGraph.setAxisScale(QwtPlot.xBottom, 0.0, 1.0, 0.1)
        self.predGraph.numberOfBars = 2
        self.predGraph.barSize = 200 / (self.predGraph.numberOfBars + 1)
        vizPredAcc.layout().addWidget(self.predGraph)
        
        vizPredAcc = OWGUI.widgetBox(self.modelTab, "Attribute lists", orientation = "vertical")
        OWGUI.checkBox(vizPredAcc, self, "vizAttributes", "Display attribute lists", callback = self.visualizeInfo)
        self.attrIntersectionBox = OWGUI.listBox(vizPredAcc, self, "attrIntersection", "attrIntersectionList", "Attribute intersection", selectionMode=QListWidget.NoSelection)
        self.attrDifferenceBox = OWGUI.listBox(vizPredAcc, self, "attrDifference", "attrDifferenceList", "Attribute difference", selectionMode=QListWidget.NoSelection)
        
        self.attBox.setVisible(0)
        self.visualizeInfo()
        
        self.matrixTab.layout().addStretch(1)
        self.modelTab.layout().addStretch(1)
        self.setMinimumWidth(900)
        self.controlArea.setMinimumWidth(378)
        
    def plotAccuracy(self, vertices=None):
        self.predGraph.tips.removeAll()
        self.predGraph.clear()
        #self.predGraph.setAxisScale(QwtPlot.yRight, 0.0, 1.0, 0.2)
        self.predGraph.setAxisScale(QwtPlot.xBottom,  0.0, 1.0, 0.2)
        
        if not vertices:
            self.predGraph.replot()
            return
        
        self.predGraph.setAxisScale(QwtPlot.yLeft, -0.5, len(self.matrix.originalData.domain.classVar.values) - 0.5, 1)
        
        scores = [[float(ca) for ca in ex["CA by class"].value.split(", ")] for ex in self.graph.items().getitems(vertices)]
        scores = [sum(score) / len(score) for score in zip(*scores)]
        
        currentBarsHeight = [0] * len(scores)
        for cn, score in enumerate(scores):
            subBarHeight = score
            ckey = PolygonCurve(pen = QPen(self.predGraph.discPalette[cn]), brush = QBrush(self.predGraph.discPalette[cn]))
            ckey.attach(self.predGraph)
            ckey.setRenderHint(QwtPlotItem.RenderAntialiased, self.predGraph.useAntialiasing)
        
            tmpx = cn - (self.predGraph.barSize/2.0)/100.0
            tmpx2 = cn + (self.predGraph.barSize/2.0)/100.0
            ckey.setData([currentBarsHeight[cn], currentBarsHeight[cn] + subBarHeight, currentBarsHeight[cn] + subBarHeight, currentBarsHeight[cn]], [tmpx, tmpx, tmpx2, tmpx2])
            currentBarsHeight[cn] += subBarHeight
            
        self.predGraph.replot()
        
    def displayAttributeInfo(self, vertices=None):
        self.attrIntersectionList = []
        self.attrDifferenceList = []
        
        if vertices is None or len(vertices) == 0:
            return
        
        attrList = [self.graph.items()[v]["attributes"].value.split(", ") for v in vertices]
        
        attrIntersection = set(attrList[0])
        attrUnion = set()
        for attrs in attrList:
            attrIntersection = attrIntersection.intersection(attrs)
            attrUnion = attrUnion.union(attrs)
            
        self.attrIntersectionList = attrIntersection
        self.attrDifferenceList = attrUnion.difference(attrIntersection)
            
    def visualizeInfo(self):
        self.networkCanvas.radius = self.radius
        
        if self.vizAccurancy:
            self.networkCanvas.plotAccuracy = self.plotAccuracy
        else:
            self.networkCanvas.plotAccuracy = None
            self.plotAccuracy(None)
            
        if self.vizAttributes:
            self.networkCanvas.vizAttributes = self.displayAttributeInfo
        else:
            self.networkCanvas.vizAttributes = None
            self.displayAttributeInfo(None)
            
    def setSubsetModels(self, subsetData):
        self.info()
        
        if "uuid" not in subsetData.domain:
            self.info("Invalid subset data. Data domain must contain 'uuid' attribute.")
            return
        
        uuids = set([ex["uuid"].value for ex in subsetData])
        for v in self.vertices:
            v.highlight = 1 if v.uuid in uuids else 0
                
    def setMatrix(self, matrix):
        self.warning()
        
        if matrix is None:
            self.warning("Data matrix is None.")
            return
        
        if not hasattr(matrix, "items") or not hasattr(matrix, "results") or not hasattr(matrix, "originalData"):
            self.warning("Data matrix does not have required data for items, results and originalData.")
            return
        
        requiredAttrs = set(["CA", "AUC", "attributes", "uuid"])
        attrs = [attr.name for attr in matrix.items.domain] 
        if len(requiredAttrs.intersection(attrs)) != len(requiredAttrs):
            self.warning("Items ExampleTable does not contain required attributes: %s." % ", ".join(requiredAttrs))
            return
            
        for ex in matrix.items:
            ex["attributes"] = ", ".join(sorted(ex["attributes"].value.split(", ")))
            
        OWNxHist.setMatrix(self, matrix)
        
    def setVertexSize(self):
        OWNxExplorer.setVertexSize(self)
        self.networkCanvas.loadIcons()
        self.networkCanvas.replot()
        
    def setVertexStyle(self):
        for v in self.networkCanvas.vertices:
            auc = self.graph.items()[v.index]
            v.style = 1 #auc            
        
    def sendNetworkSignals(self):
        self.warning()
        
        if self.graph is None or self.graph.items() is None or self.graph_matrix is None:
            self.send("Model", None)
            self.send("Selected Models", None)
            return
        
        if self.graph.number_of_nodes() != self.graph_matrix.dim:
            self.warning('Network items and matrix results not of equal length.')
            self.send("Model", None)
            self.send("Selected Models", None)
            return
                    
        selection = self.networkCanvas.getSelectedVertices()
        
        if len(selection) == 1:
            modelInstance = self.graph.items()[selection[0]]
            # modelInfo - Python Dict; keys: method, classifier, probabilities,
            # results, XAnchors, YAnchors, attributes
            modelInfo = self.graph_matrix.results[modelInstance['uuid'].value]
            
            #uuid = modelInstance["uuid"].value
            #method, vizr_result, projection_points, classifier, attrs = self.matrix.results[uuid]
            
            if 'YAnchors' in modelInfo and 'XAnchors' in modelInfo:
                if not modelInstance.domain.hasmeta('anchors'):
                    modelInstance.domain.addmeta(orange.newmetaid(), orange.PythonVariable('anchors'))
                modelInstance['anchors'] = (modelInfo['XAnchors'], modelInfo['YAnchors'])
                
            if 'classifier' in modelInfo and modelInfo['classifier'] is not None:
                if not modelInstance.domain.hasmeta('classifier'):
                    modelInstance.domain.addmeta(orange.newmetaid(), orange.PythonVariable('classifier'))
                modelInstance['classifier'] = modelInfo['classifier']
                self.send('Classifier', modelInfo['classifier'])
                
            self.send('Model', modelInstance)
            self.send('Selected Models', self.graph.items().getitems(selection))
        elif len(selection) > 1: 
            self.send('Model', None)
            self.send('Selected Models', self.graph.items().getitems(selection))
        else:
            self.send('Model', None)
            self.send('Selected Models', None)
            
    def setColors(self):
        dlg = self.createColorDialog(self.colorSettings, self.selectedSchemaIndex)
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.networkCanvas.contPalette = dlg.getContinuousPalette("contPalette")
            self.networkCanvas.discPalette = dlg.getDiscretePalette("discPalette")
            
            self.setVertexColor()
            
    def setVertexColor(self):
        if self.graph is None:
            return
        
        self.networkCanvas.setVertexColor(self.colorCombo.currentText())
        self.lastColorAttribute = self.colorCombo.currentText()
        self.networkCanvas.updateData()
        self.networkCanvas.replot()
            
    def sendSignals(self):
        if self.graph is None or self.graph_matrix is None:
            return
        
        self.set_graph(self.graph)
        self.set_items_distance_matrix(self.graph_matrix)
        # TODO clickedAttLstBox -> setLabelText(["attributes"]
        
        for i, ex in enumerate(self.graph.items()):
            self.networkCanvas.vertices[i].uuid = ex["uuid"].value
        
        self.setVertexSize()
        self.setVertexStyle()
        self.setVertexColor()
        
        labels = self.matrix.originalData.domain.classVar.values.native()
        self.predGraph.numberOfBars = len(labels)
        self.predGraph.barSize = 200 / (self.predGraph.numberOfBars + 1)
        self.predGraph.setYLlabels(labels)
        #self.predGraph.setShowMainTitle(self.showMainTitle)
        #self.predGraph.setYLaxisTitle(self.matrix.originalData.domain.classVar.name)
        #self.predGraph.setShowYLaxisTitle(True)
        self.predGraph.setAxisScale(QwtPlot.xBottom,  0.0, 1.0, 0.2)
        self.predGraph.setAxisScale(QwtPlot.yLeft, -0.5, len(self.matrix.originalData.domain.classVar.values) - 0.5, 1)
    
        self.predGraph.enableYRaxis(0)
        self.predGraph.setYRaxisTitle("")
        self.predGraph.setXaxisTitle("CA")
        self.predGraph.setShowXaxisTitle(True)
        self.predGraph.replot()
        
        self.visualizeInfo()

if __name__=="__main__":    
    import OWModelFile
    import pickle
    modelName = 'zoo-168'
    root = 'c:\\Users\\miha\\Projects\\res\\metamining\\'
    
    appl = QApplication(sys.argv)
    ow = OWModelMap()
    ow.show()
    mroot = '%snew\\' % root
    matrix,labels,data = OWModelFile.readMatrix('%s%s.npy' % (mroot,modelName))
    if os.path.exists('%s%s.tab' % (mroot, modelName)):
        matrix.items = orange.ExampleTable('%s%s.tab' % (mroot, modelName))
    else:
        print 'ExampleTable %s not found!\n' % ('%s%s.tab' % (mroot,modelName))
    if os.path.exists('%s%s.res' % (mroot, modelName)):
        matrix.results = pickle.load(open('%s%s.res' % \
                                               (mroot, modelName), 'rb'))
    else:
        print 'Results pickle %s not found!\n' % \
              ('%s%s.res' % (mroot, modelName))
    
    matrix.originalData = Orange.data.Table('%stab\\zoo.tab' % root)
    ow.setMatrix(matrix)
    appl.exec_()
    
    
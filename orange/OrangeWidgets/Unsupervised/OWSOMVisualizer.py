"""
<name>SOM Visualizer</name>
<description>Visualizes a trained self organising maps.</description>
<icon>SOMVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>5020</priority>
"""
import orange, orngSOM
import math, numpy
import OWGUI, OWColorPalette
from OWWidget import *

from OWDlgs import OWChooseImageSizeDlg

DefColor=QColor(200, 200, 0)
BaseColor=QColor(0, 0, 200)

class GraphicsSOMItem(QGraphicsPolygonItem):
    startAngle=0
    segment=6
    def __init__(self, *args):
        QGraphicsPolygonItem.__init__(self, *args)
        self.node=None
        self.hasNode=False
        self.isSelected=False
        self.labelText=""
        self.textColor=Qt.black
        self.defaultPen = QPen(Qt.black)
        self.outlinePoints=None
        self.histObj=[]
        self.setSize(10)
        self.setZValue(0)

    def areaPoints(self):
        return self.outlinePoints
    
    def setSize(self, size):
        self.outlinePoints = [QPointF(math.cos(2*math.pi/self.segments*i + self.startAngle)*size,
                                      math.sin(2*math.pi/self.segments*i + self.startAngle)*size)
                              for i in range(self.segments)]
        self.setPolygon(QPolygonF(self.outlinePoints))

    def setPie(self, attrIndex, size):
        for e in self.histObj:
            self.scene().removeItem(e)
        self.histObj = []
        if len(self.node.examples) < 0:
            return
        dist = orange.Distribution(attrIndex, self.node.examples)
        colors = OWColorPalette.ColorPaletteHSV(len(dist))
        distSum = max(sum(dist), 1)
        startAngle = 0
        for i in range(len(dist)):
            angle = 360/distSum*dist[i]*16
            c = QGraphicsEllipseItem(-size/2, -size/2, size, size, self, self.scene())
            c.setStartAngle(startAngle)
            c.setSpanAngle(angle)
            c.setBrush(QBrush(colors[i]))
            c.show()
            startAngle += angle
            self.histObj.append(c)  

    def setHist(self, size):
        for e in self.histObj:
            self.scene().removeItem(e)
        self.histObj = []
        c = QGraphicsEllipseItem(-size/2, -size/2, size, size, self, self.scene())
        c.setBrush(QBrush(DefColor))
        c.setPen(QPen(Qt.white))
        c.show()
        self.histObj.append(c)

    def setLabel(self, label):
        self.labelText = label

    def setColor(self, color):
        self.color = color
        if color.value() < 100:
            self.textColor = Qt.white
        self.setBrush(QBrush(color))

    def setNode(self, node):
        self.node = node
        self.hasNode = True
        self.buildToolTip()

    def setSelected(self, bool=True):
        self.isSelected = bool
        self.setPen(QPen(Qt.red, 2) if bool else self.defaultPen)
        self.setZValue(1 if bool else 0)

    def advancement(self):
        pass

    def buildToolTip(self):
        if self.node and self.scene().showToolTip:
            node = self.node
            text = "Items: %i" % len(self.node.examples)
            if self.scene().includeCodebook:
                text += "<hr><b>Codebook vector:</b><br>" + "<br>".join(\
                    [a.variable.name + ": " + str(a) for a in node.referenceExample \
                     if a.variable != node.referenceExample.domain.classVar])
            
            if node.examples.domain.classVar and len(node.examples):
                dist = orange.Distribution(node.examples.domain.classVar, node.examples)
                if node.examples.domain.classVar.varType == orange.VarTypes.Continuous:
                    text += "<hr>Avg " + node.examples.domain.classVar.name + ":" + ("%.3f" % dist.average())
                else:
                    colors = OWColorPalette.ColorPaletteHSV(len(node.examples.domain.classVar.values))
                    text += "<hr>" + "<br>".join(["<span style=\"color:%s\">%s</span>" %(colors[i].name(), str(value) + ": " + str(dist[i])) \
                                                 for i, value in enumerate(node.examples.domain.classVar.values)])
            self.setToolTip(text)
        else:
            self.setToolTip("")
       
##    def drawShape(self, painter):
##        QCanvasPolygon.drawShape(self, painter)
##        if self.canvas().showGrid or self.isSelected:
##            color=self.isSelected and Qt.blue or Qt.black
##            p=QPen(color)
##            if self.isSelected:
##                p.setWidth(3)
##            painter.setPen(p)
##            painter.drawPolyline(self.outlinePoints)
##            painter.drawLine(self.outlinePoints.point(0)[0],self.outlinePoints.point(0)[1],
##                             self.outlinePoints.point(self.segments-1)[0],self.outlinePoints.point(self.segments-1)[1])
        
class GraphicsSOMHexagon(GraphicsSOMItem):
    startAngle = 0
    segments = 6
    def advancement(self):
        width = self.outlinePoints[0].x() - self.outlinePoints[3].x()
        line = self.outlinePoints[1].x() - self.outlinePoints[2].x()
        x = width - (width-line)/2
        y = self.outlinePoints[2].y() - self.outlinePoints[5].y()
        return (x, y)

class GraphicsSOMRectangle(GraphicsSOMItem):
    startAngle = math.pi/4
    segments = 4
    def advancement(self):
        x = self.outlinePoints[0].x() - self.outlinePoints[1].x()
        y = self.outlinePoints[0].y() - self.outlinePoints[3].y()
        return (x,y)
    
baseColor = QColor(20,20,20)  

class SOMScene(QGraphicsScene):
    def __init__(self, master, *args):
        QGraphicsScene.__init__(self, *args)
        self.master = master
        self.drawMode = 1
        self.showGrid = True
        self.component = 0
        self.objSize = 25
        self.canvasObj = []
        self.selectionList = []
        self.selectionRect = None
        self.showToolTip = True
        self.includeCodebook = True
        self.somMap = None

    def drawHistogram(self):
        if self.parent().inputSet:
            maxVal = max([len(n.mappedExamples) for n in self.somMap.map] + [1])
        else:
            maxVal = max([len(n.examples) for n in self.somMap.map] + [1])
        if self.drawMode == 1:
            maxSize = 4*int(self.somMap.map_shape[0]*self.objSize/(2*self.somMap.map_shape[0] - 1)*0.7)
        else:
            maxSize = 4*int(self.objSize*0.7)
        for n in self.canvasObj:
            if n.hasNode:
                if self.parent().inputSet:
                    size = float(len(n.node.mappedExamples))/maxVal*maxSize
                else:
                    size = float(len(n.node.examples))/maxVal*maxSize
                if self.parent().drawPies():
                    n.setPie(self.parent().attribute, size)
                else:
                    n.setHist(size)
                    
        self.updateHistogramColors()            

    def updateHistogramColors(self):
        if self.parent().drawPies():
            return
        attr = self.somMap.examples.domain.variables[self.parent().attribute]
        for n in self.canvasObj:
            if n.hasNode:
                if attr.varType == orange.VarTypes.Discrete:
                    if self.parent().inputSet:
                        dist = orange.Distribution(attr, n.node.mappedExamples)
                    else:
                        dist = orange.Distribution(attr, n.node.examples)
                    colors = OWColorPalette.ColorPaletteHSV(len(dist))
                    maxProb = max(dist)
                    majValInd = filter(lambda i:dist[i] == maxProb, range(len(dist)))[0]
                    if self.parent().discHistMode == 1:
                        n.histObj[0].setBrush(QBrush(colors[majValInd]))
                    elif self.parent().discHistMode == 2:
                        light = 180 - 80*float(dist[majValInd])/max(sum(dist), 1)
                        n.histObj[0].setBrush(QBrush(colors[majValInd].light(light)))
                else:
                    if self.parent().inputSet:
                        dist=orange.Distribution(attr, n.node.mappedExamples)
                        fullDist=orange.Distribution(attr, self.parent().examples)
                    else:
                        dist=orange.Distribution(attr, n.node.examples)
                        fullDist=orange.Distribution(attr, self.somMap.examples)
                    if len(dist)==0:
                        continue
                    
                    if self.parent().contHistMode==0:
                        n.histObj[0].setBrush(QBrush(DefColor))
                    if self.parent().contHistMode==1:
                        std=(dist.average()-fullDist.average())/max(fullDist.dev(),1)
                        std=min(max(std,-1),1)
                        #print std
                        n.histObj[0].setBrush(QBrush(QColor(70*(std+1)+50, 70*(std+1)+50, 0)))                           
                    if self.parent().contHistMode==2:
                        light = 300-200*dist.var()/fullDist.var()
                        n.histObj[0].setBrush(QBrush(QColor(0,0,20).light(light)))

    def updateToolTips(self):
        for item in self.canvasObj:
            item.buildToolTip()
    
    def setSom(self, somMap=None):
        self.oldSom=self.somMap
        self.somMap=somMap
        self.clear()
        if not self.somMap:
            return
        
        self.somNodeMap={}
        for n in somMap.map:
##            self.somNodeMap[(n.x,n.y)]=n
            self.somNodeMap[tuple(n.pos)]=n

        if self.drawMode==1:
##            self.uMat=orngSOM.getUMat(somMap)
            self.uMat = somMap.map.getUMat()
            if somMap.topology==orngSOM.HexagonalTopology:
                self.drawUMatHex()
            else:
                self.drawUMatRect()
        else:
            if somMap.topology == orngSOM.HexagonalTopology:
                self.drawHex()
            else:
                self.drawRect()
            if self.drawMode!=0:
                minVal=min([n.vector[self.component] for n in self.somMap.map])
                maxVal=max([n.vector[self.component] for n in self.somMap.map])
                for o in self.canvasObj:
                    val=255-max(min(255*(o.node.vector[self.component]-minVal)/(maxVal-minVal),245),10)
                    o.setColor(QColor(val,val,val))
        if (self.parent().inputSet==0 or self.parent().examples) and self.parent().histogram:
            self.drawHistogram()
        self.updateLabels()
        
    def redrawSom(self):    #for redrawing without clearing the selection 
        if not self.somMap:
            return
        oldSelection = self.parent().scene.selectionList
        self.setSom(self.somMap)
        nodeList = [n.node for n in oldSelection]
        newSelection = []
        for o in self.canvasObj:
            if o.node in nodeList:
                o.setSelected(True)
                newSelection.append(o)
        self.parent().sceneView.selectionList = newSelection
        self.update()
    
    def drawHex(self):
        #size=self.objSize*2-1
        #size=int((self.objSize-9)/10.0*20)*2-1
        size = 2*self.objSize - 1
        x, y = size, size*2
        for n in self.somMap.map:
            offset = 1 - abs(int(n.pos[0])%2 - 2)
            h=GraphicsSOMHexagon(None, self)
            h.setSize(size)
            h.setNode(n)
            (xa, ya) = h.advancement()
##            h.move(x+n.x*xa, y+n.y*ya+offset*ya/2)
            h.setPos(x + n.pos[0]*xa, y + n.pos[1]*ya + offset*ya/2)
            h.show()
            self.canvasObj.append(h)
        self.setSceneRect(self.itemsBoundingRect())
        self.update()
    
    def drawRect(self):
        size=self.objSize*2-1
        x,y=size, size
##        self.resize(1,1)    # crashes at update without this line !!!
        for n in self.somMap.map:
            r=GraphicsSOMRectangle(None, self)
            r.setSize(size)
            r.setNode(n)
            (xa,ya) = r.advancement()
##            r.move(x+n.x*xa, y+n.y*ya)
            r.setPos(x + n.pos[0]*xa, y + n.pos[1]*ya)
            r.show()
            self.canvasObj.append(r)
        self.setSceneRect(self.itemsBoundingRect())
        self.update()
    
    def drawUMatHex(self):
        #size=2*(int(self.objSize*1.15)/2)-1
        size=2*int(self.somMap.map_shape[0]*self.objSize/(2*self.somMap.map_shape[0]-1))-1
        x,y=size, size
        maxDist=max(reduce(numpy.maximum, [a for a in self.uMat]))
        minDist=max(reduce(numpy.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            offset=2-abs(i%4-2)
            for j in range(len(self.uMat[i])):
                h=GraphicsSOMHexagon(None, self)
                h.setSize(size)
                (xa,ya)=h.advancement()
                h.setPos(x+i*xa, y+j*ya+offset*ya/2)
                if i%2==0 and j%2==0:
                    h.setNode(self.somNodeMap[(i/2,j/2)])
                h.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                h.setColor(QColor(val, val, val))
                self.canvasObj.append(h)
        self.setSceneRect(self.itemsBoundingRect())
        self.update()
        
    def drawUMatRect(self):
        #size=self.objSize-1
        size=2*int(self.somMap.map_shape[0]*self.objSize/(2*self.somMap.map_shape[0]-1))-1
        x,y=size, size
        
        maxDist=max(reduce(numpy.maximum, [a for a in self.uMat]))
        minDist=max(reduce(numpy.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            for j in range(len(self.uMat[i])):
                r=GraphicsSOMRectangle(None, self)
                r.setSize(size)
                if i%2==0 and j%2==0:
                    r.setNode(self.somNodeMap[(i/2,j/2)])
                (xa,ya)=r.advancement()
                r.setPos(x+i*xa, y+j*ya)
                r.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                r.setColor(QColor(val, val, val))
                self.canvasObj.append(r)
        self.setSceneRect(self.itemsBoundingRect())
        self.update()
        
    def updateLabels(self):
        for o in self.canvasObj:
            if o.hasNode and len(o.node.examples):
                if self.parent().labelNodes and o.node.classifier:
                    o.setLabel(str(o.node.classifier.defaultValue))
                else:
                    o.setLabel("")
            else:
                o.setLabel("")
        self.updateAll()

    def updateAll(self):
##        self.setAllChanged()
        self.update()

    def updateGrid(self):
        pen = QPen(Qt.black) if self.showGrid else QPen(Qt.NoPen)
        for item in self.canvasObj:
            item.defaultPen = pen
            item.setSelected(item.isSelected)
        
    def clear(self):
        for o in self.canvasObj:
            self.removeItem(o) #o.setCanvas(None)
        self.canvasObj=[]
            
    def mouseMoveEvent(self, event):
        pos = event.scenePos()
        if self.selectionRect:
            rect = self.selectionRect.rect()
            self.selectionRect.setRect(QRectF(rect.x(), rect.y(), pos.x() - rect.x(), pos.y() - rect.y()))
        
    def mousePressEvent(self, event):
        pos = event.scenePos()
        self.selectionRect = r = QGraphicsRectItem(pos.x(), pos.y(), 1, 1, None, self)
        r.show()
        r.setZValue(19)
        self.oldSelection=self.selectionList
        if not self.master.ctrlPressed:
            self.clearSelection()
        
    def mouseReleaseEvent(self, event):
        self.updateSelection()
        self.removeItem(self.selectionRect)
        self.selectionRect=None
        self.master.updateSelection([a.node for a in self.selectionList])
        self.update()
    
    def updateSelection(self):
        obj = self.items(self.selectionRect.rect())
        obj = [a for a in obj if isinstance(a, GraphicsSOMItem) and a.hasNode]
        if self.master.ctrlPressed:
            set1 = set(obj)
            set2 = set(self.oldSelection)
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            for e in intersection:
                self.removeSelection(e)
            for e in set1.difference(set2):
                self.addSelection(e)
        else:
            self.clearSelection()
            for e in obj:
                self.addSelection(e)          

    def addSelection(self, obj):
        obj.setSelected(True)
        self.selectionList.append(obj)
           
    def removeSelection(self, obj):
        obj.setSelected(False)
        self.selectionList = [a for a in self.selectionList if a.isSelected]
    
    def clearSelection(self):
        for o in self.selectionList:
            o.setSelected(False)
        self.selectionList=[]
        
    def invertSelection(self):
        for n in self.canvasObj:
            if n.hasNode:
                if not n.isSelected:
                    self.addSelection(n)
                else:
                    self.removeSelection(n)
        self.master.updateSelection([a.node for a in self.selectionList])
        self.update()


class OWSOMVisualizer(OWWidget):
    settingsList = ["scene.drawMode","scene.objSize","commitOnChange", "backgroundMode", "backgroundCheck", "scene.includeCodebook", "scene.showToolTip"]
    contextHandlers = {"":DomainContextHandler("", [ContextField("attribute", DomainContextHandler.Optional),
                                                  ContextField("discHistMode", DomainContextHandler.Optional),
                                                  ContextField("contHistMode", DomainContextHandler.Optional),
                                                  ContextField("targetValue", DomainContextHandler.Optional),
                                                  ContextField("histogram", DomainContextHandler.Optional),
                                                  ContextField("inputSet", DomainContextHandler.Optional),
                                                  ContextField("scene.component", DomainContextHandler.Optional),
                                                  ContextField("scene.includeCodebook", DomainContextHandler.Optional)])}
    
    drawModes = ["None", "U-Matrix", "Component planes"]
    
    def __init__(self, parent=None, signalManager=None, name="SOM visualizer"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs = [("SOMMap", orngSOM.SOMMap, self.setSomMap), ("Examples", ExampleTable, self.data)]
        self.outputs = [("Examples", ExampleTable)]
        
        self.labelNodes = 0
        self.commitOnChange = 0
        self.backgroundCheck = 1
        self.backgroundMode = 0
        self.histogram = 1
        self.attribute = 0
        self.discHistMode = 0
        self.targetValue = 0
        self.contHistMode = 0
        self.inputSet = 0

        self.somMap = None
        self.examples = None
        
        
##        layout = QVBoxLayout(self.mainArea) #,QVBoxLayout.TopToBottom,0)
        self.scene = SOMScene(self, self)
        self.sceneView = QGraphicsView(self.scene, self.mainArea)
        self.sceneView.viewport().setMouseTracking(True)

        self.mainArea.layout().addWidget(self.sceneView)
        
        self.loadSettings()
        call = lambda:self.scene.redrawSom()

        histTab = mainTab = self.controlArea

        self.mainTab = mainTab
        self.histTab = histTab

        self.backgroundBox = OWGUI.widgetBox(mainTab, "Background")
        #OWGUI.checkBox(self.backgroundBox, self, "backgroundCheck","Show background", callback=self.setBackground)
        b = OWGUI.radioButtonsInBox(self.backgroundBox, self, "scene.drawMode", self.drawModes, callback=self.setBackground)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.componentCombo=OWGUI.comboBox(OWGUI.indentedBox(b), self,"scene.component", callback=self.setBackground)
        self.componentCombo.setEnabled(self.scene.drawMode==2)
        OWGUI.checkBox(self.backgroundBox, self, "scene.showGrid", "Show grid", callback=self.scene.updateGrid)
        #b=OWGUI.widgetBox(mainTab, "Histogram")
        OWGUI.separator(mainTab)
        
        b = OWGUI.widgetBox(mainTab, "Histogram") ##QVButtonGroup("Histogram", mainTab)
#        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(b, self, "histogram", "Show histogram", callback=self.setHistogram)
        OWGUI.radioButtonsInBox(OWGUI.indentedBox(b), self, "inputSet", ["Use training set", "Use input subset"], callback=self.setHistogram)
        OWGUI.separator(mainTab)
        
        b1= OWGUI.widgetBox(mainTab) ##QVBox(mainTab)
#        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        b=OWGUI.hSlider(b1, self, "scene.objSize","Plot size", 10,100,step=10,ticks=10, callback=call)
#        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        #OWGUI.checkBox(b1, self, "labelNodes", "Node Labeling", callback=self.canvas.updateLabels)
        OWGUI.separator(b1)

        b1 = OWGUI.widgetBox(b1, "Tooltip Info")
        OWGUI.checkBox(b1, self, "scene.showToolTip","Show tooltip", callback=self.scene.updateToolTips)
        OWGUI.checkBox(b1, self, "scene.includeCodebook", "Include codebook vector", callback=self.scene.updateToolTips)
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.separator(mainTab)
        
        self.histogramBox = OWGUI.widgetBox(histTab, "Coloring")
        self.attributeCombo = OWGUI.comboBox(self.histogramBox, self, "attribute", callback=self.setHistogram)

        self.discTab = OWGUI.radioButtonsInBox(OWGUI.indentedBox(self.histogramBox), self, "discHistMode", ["Pie chart", "Majority value", "Majority value prob."], box=1, callback=self.setHistogram)
        self.discTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        #self.targetValueCombo=OWGUI.comboBox(b, self, "targetValue", callback=self.setHistogram)
##        QVBox(discTab)
##        b=QVButtonGroup(contTab)
##        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.contTab = OWGUI.radioButtonsInBox(OWGUI.indentedBox(self.histogramBox), self, "contHistMode", ["Default", "Average value"], box=1, callback=self.setHistogram)
        self.contTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
##        QVBox(contTab)

##        OWGUI.rubber(mainTab)
 
        b = OWGUI.widgetBox(self.controlArea, "Selection")
        OWGUI.button(b, self, "&Invert selection", callback=self.scene.invertSelection)
        OWGUI.button(b, self, "&Commit", callback=self.commit)
        OWGUI.checkBox(b, self, "commitOnChange", "Commit on change")

        OWGUI.separator(self.controlArea)
        OWGUI.button(self.controlArea, self, "&Save Graph", callback=self.saveGraph, debuggingEnabled = 0)
        
        self.selectionList = []
        self.ctrlPressed = False
##        self.setFocusPolicy(QWidget.StrongFocus)
        self.resize(600,500)

    def sendReport(self):
        self.reportSettings("Visual settings",
                           [("Background", self.drawModes[self.scene.drawMode]+(" - "+self.componentCombo.currentText() if self.scene.drawMode==2 else "")),
                            ("Histogram", ["data from training set", "data from input subset"][self.inputSet] if self.histogram else "none"),
                            ("Coloring",  "%s for %s" % (["pie chart", "majority value", "majority value probability"][self.discHistMode], self.attributeCombo.currentText()))
                           ])
        self.reportSection("Plot")
        self.reportImage(OWChooseImageSizeDlg(self.scene).saveImage)
        
    def setMode(self):
        self.componentCombo.setEnabled(self.scene.drawMode == 2)
        self.scene.redrawSom()

    def setBackground(self):
        self.setMode()

    def setDiscCont(self):
        if self.somMap.examples.domain.variables[self.attribute].varType == orange.VarTypes.Discrete:
            self.discTab.show()
            self.contTab.hide()
        else:
            self.discTab.hide()
            self.contTab.show()
        
    def setHistogram(self):
        if self.somMap:
            self.setDiscCont()
            self.scene.redrawSom()

    def drawPies(self):
        return self.discHistMode == 0 and self.somMap.examples.domain.variables[self.attribute].varType == orange.VarTypes.Discrete
        
    def setSomMap(self, somMap=None):
        self.somType = "Map"
        self.setSom(somMap)
        
    def setSomClassifier(self, somMap=None):
        self.somType = "Classifier"
        self.setSom(somMap)
        
    def setSom(self, somMap=None):
        self.closeContext()
        self.somMap = somMap
        if not somMap:
            self.clear()
            return
        self.componentCombo.clear()
        self.attributeCombo.clear()
        
        self.targetValue = 0
        self.scene.component = 0
        self.attribute = 0
        for v in somMap.examples.domain.attributes:
            self.componentCombo.addItem(v.name)
        for v in somMap.examples.domain.variables:
            self.attributeCombo.addItem(v.name)

        self.openContext("", somMap.examples)
        self.setDiscCont()
        self.scene.setSom(somMap)
       
    def data(self, data=None):
        self.examples = data
        if data and self.somMap:
            for n in self.somMap.map:
                setattr(n,"mappedExamples", orange.ExampleTable(data.domain))
            for e in data:
                bmu = self.somMap.getBestMatchingNode(e)
                bmu.mappedExamples.append(e)
            if self.inputSet == 1:
                self.setHistogram()
    
    def clear(self):
        self.scene.clearSelection()
        self.componentCombo.clear()
        self.scene.component = 0
        self.scene.setSom(None)
        self.send("Examples", None)
        
    def updateSelection(self, nodeList):
        self.selectionList = nodeList
        if self.commitOnChange:
            self.commit()
            
    def commit(self):
        if not self.somMap:
            return
        ex = orange.ExampleTable(self.somMap.examples.domain)
        for n in self.selectionList:
            if self.inputSet == 0 and n.examples:
                ex.extend(n.examples)
            elif self.inputSet == 1 and n.mappedExamples:
                ex.extend(n.mappedExamples)
        if len(ex):
            self.send("Examples",ex)
        else:
            self.send("Examples",None)

    def saveGraph(self):
        sizeDlg = OWChooseImageSizeDlg(self.scene)
        sizeDlg.exec_()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrlPressed = True
        else:
            OWWidget.keyPressEvent(self, event)
      
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrlPressed = False
        else:
            OWWidget.keyReleaseEvent(self, event)
        

        
if __name__=="__main__":
    ap = QApplication(sys.argv)
    data = orange.ExampleTable("../../doc/datasets/brown-selected.tab")
##    l=orngSOM.SOMLearner(batch_train=False)
    l = orngSOM.SOMLearner(batch_train=True, initialize=orngSOM.InitializeLinear)
##    l = orngSOM.SOMLearner(batch_train=True, initialize=orngSOM.InitializeRandom)
    l = l(data)
    l.data = data
    w = OWSOMVisualizer()
##    ap.setMainWidget(w)
    w.setSomClassifier(l)
    w.data(orange.ExampleTable(data[:50]))
    w.show()
    ap.exec_()
    w.saveSettings()

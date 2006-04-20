"""
<name>SOMVisualizer</name>
<description>SOM visualizer</description>
<icon>SOMVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact> 
<priority>5020</priority>
"""
import orange, orngSOM, orangeom
import math, Numeric, sets
import OWGUI, OWGraphTools
from OWWidget import *
from qt import *
from qtcanvas import *
from OWTreeViewer2D import CanvasBubbleInfo
from OWDlgs import OWChooseImageSizeDlg

DefColor=QColor(200, 200, 0)
BaseColor=QColor(0, 0, 200)
class CanvasSOMItem(QCanvasPolygon):
    startAngle=0
    segment=6
    def __init__(self, *args):
        apply(QCanvasPolygon.__init__,(self,)+args)
        self.node=None
        self.hasNode=False
        self.isSelected=False
        self.labelText=""
        self.textColor=Qt.black
        self.outlinePoints=None
        self.histObj=[]
        self.setSize(10)
        self.setZ(0)

    def boundingRect(self):
        return self.outlinePoints.boundingRect()

    def move(self, x, y):
        QCanvasPolygon.move(self, x, y)
        for e in self.histObj:
            e.move(x, y)

    def areaPoints(self):
        return self.outlinePoints
    
    def setSize(self, size):
        self.outlinePoints=QPointArray(self.segments)
        for i in range(self.segments):
            x=int(math.cos(2*math.pi/self.segments*i+self.startAngle)*size)
            y=int(math.sin(2*math.pi/self.segments*i+self.startAngle)*size)
            self.outlinePoints.setPoint(i,x,y)
        self.setPoints(self.outlinePoints)

    def setPie(self, attrIndex, size):
        for e in self.histObj:
            e.setCanvas(None)
        self.histObj=[]
        if len(self.node.examples)<0:
            return
        dist=orange.Distribution(attrIndex, self.node.examples)
        colors=OWGraphTools.ColorPaletteHSV(len(dist))
        distSum=max(sum(dist),1)
        startAngle=0
        for i in range(len(dist)):
            angle=360/distSum*dist[i]*16
            c=QCanvasEllipse(size, size, startAngle, angle, self.canvas())
            c.setBrush(QBrush(colors[i]))
            c.setZ(10)
            c.move(self.x(), self.y())
            c.show()
            startAngle+=angle
            self.histObj.append(c)

    def setHist(self, size):
        for e in self.histObj:
            e.setCanvas(None)
        self.histObj=[]
        c=QCanvasEllipse(size, size, self.canvas())
        c.setZ(10)
        c.move(self.x(), self.y())
        c.setBrush(QBrush(DefColor))
        c.setPen(QPen(Qt.white))
        c.show()
        self.histObj.append(c)

    def setLabel(self, label):
        self.labelText=label

    def setColor(self, color):
        self.color=color
        if color.hsv()[2]<100:
            self.textColor=Qt.white
        self.setBrush(QBrush(color))

    def setNode(self, node):
        self.node=node
        self.hasNode=True

    def setSelected(self, bool=True):
        if self.isSelected!=bool:
            self.isSelected=bool
            self.setChanged()
            self.setZ(bool and (self.z()+1) or (self.z()-1))

    def advancement(self):
        pass

    def setChanged(self):
        self.canvas().setChanged(self.boundingRect())
       
    def drawShape(self, painter):
        QCanvasPolygon.drawShape(self, painter)
        if self.canvas().showGrid or self.isSelected:
            color=self.isSelected and Qt.blue or Qt.black
            p=QPen(color)
            if self.isSelected:
                p.setWidth(3)
            painter.setPen(p)
            painter.drawPolyline(self.outlinePoints)
            painter.drawLine(self.outlinePoints.point(0)[0],self.outlinePoints.point(0)[1],
                             self.outlinePoints.point(self.segments-1)[0],self.outlinePoints.point(self.segments-1)[1])
        """if self.node:
            painter.setPen(QPen(self.isSelected and Qt.red or self.textColor))
            if self.labelText:
                painter.drawText(self.boundingRect(), Qt.AlignVCenter | Qt.AlignLeft, " "+self.labelText)
            else:
                #return 
                painter.setBrush(QBrush(self.isSelected and Qt.red or self.textColor))
                painter.drawPie(self.x()-2,self.y()-2,4,4,0,5760)
        #self.setChanged()"""

    def setCanvas(self, canvas):
        QCanvasPolygon.setCanvas(self, canvas)
        for e in self.histObj:
            e.setCanvas(canvas)
        
class CanvasHexagon(CanvasSOMItem):
    startAngle=0
    segments=6
    def __init__(self, *args):
        apply(CanvasSOMItem.__init__,(self,)+args)
    def advancement(self):
        width=self.outlinePoints.point(0)[0]-self.outlinePoints.point(3)[0]
        line=self.outlinePoints.point(1)[0]-self.outlinePoints.point(2)[0]
        x=width-(width-line)/2
        y=self.outlinePoints.point(2)[1]-self.outlinePoints.point(5)[1]
        return (x,y)

class CanvasRectangle(CanvasSOMItem):
    startAngle=math.pi/4
    segments=4
    def advancement(self):
        x=self.outlinePoints.point(0)[0]-self.outlinePoints.point(1)[0]
        y=self.outlinePoints.point(0)[1]-self.outlinePoints.point(3)[1]
        return (x,y)

class SOMCanvasView(QCanvasView):
    def __init__(self, master, canvas, *args):
        apply(QCanvasView.__init__, (self,canvas)+args)
        self.master=master
        self.selectionList=[]
        self.bubble=None
        self.bubbleNode=None
        self.showBubbleInfo=True
        self.includeCodebook=True
        self.selectionRect=None
        self.viewport().setMouseTracking(True)
        
    def buildBubble(self, node):
        b=CanvasBubbleInfo(node,None,self.canvas())
        b.setZ(20)
        s="Items: "+str(len(node.examples))
        b.addTextLine(s)
        
        if self.includeCodebook:
            b.addTextLine()
            b.addTextLine("Codebook vector:")
            for a in node.referenceExample:
                if a.variable!=node.referenceExample.domain.classVar:
                    b.addTextLine(a.variable.name+": "+str(a))
            #b.addTextLine()    
            
        if node.examples.domain.classVar and len(node.examples):
            b.addTextLine()
            dist=orange.Distribution(node.examples.domain.classVar, node.examples)
            if node.examples.domain.classVar.__class__==orange.FloatVariable:
                s="Avg "+node.examples.domain.classVar.name+":"+("%.3f" % dist.average())
                b.addTextLine(s)
            else:
                colors=OWGraphTools.ColorPaletteHSV(len(node.examples.domain.classVar.values))
                for i in range(len(node.examples.domain.classVar.values)):
                    s=str(node.examples.domain.classVar.values[i])+": "+str(dist[i])
                    b.addTextLine(s, colors[i])
        b.fitSquare()
        b.show()
        return b
    
    def fitBubble(self):
        bRect=self.bubble.boundingRect()
        #cRect=self.canvas().rect()
        flipX=flipY=False
        if self.canvas().width()<bRect.right():
            flipX=True
        if self.canvas().height()<bRect.bottom():
            flipY=True
        self.bubble.move(self.bubble.x()-(flipX and self.bubble.width()+20 or 0),
                         self.bubble.y()-(flipY and self.bubble.height()+20 or 0))
        
    def contentsMouseMoveEvent(self, event):
        pos=event.pos()
        if self.selectionRect:
            rect=self.selectionRect.rect()
            self.selectionRect.setSize(pos.x()-rect.x(),pos.y()-rect.y())
            #self.updateSelection()
        obj=self.canvas().collisions(pos)
        if obj and (obj[-1].__class__==CanvasHexagon or obj[-1].__class__==CanvasRectangle) and obj[-1].hasNode:
            if not self.showBubbleInfo:
                if self.bubble:
                    self.bubble.hide()
                self.canvas().update()
                return
            node=obj[-1].node
            if self.bubbleNode:
                if node==self.bubbleNode:
                    self.bubble.move(pos.x()+10,pos.y()+10)
                else:
                    self.bubble.setCanvas(None)
                    self.bubble=self.buildBubble(node)
                    self.bubble.move(pos.x()+10,pos.y()+10)
                    self.bubbleNode=node
            else:
                self.bubble=self.buildBubble(node)
                self.bubble.move(pos.x()+10,pos.y()+10)
                self.bubbleNode=node
            self.fitBubble()
        elif self.bubble:
            self.bubble.setCanvas(None)
            self.bubble=None
            self.bubbleNode=None
        self.canvas().update()
        
    def contentsMousePressEvent(self, event):
        pos=event.pos()
        self.selectionRect=r=QCanvasRectangle(pos.x(), pos.y(), 1, 1, self.canvas())
        r.show()
        r.setZ(19)
        self.oldSelection=self.selectionList
        if not self.master.ctrlPressed:
            self.clearSelection()            
        """
        obj=self.canvas().collisions(event.pos())
        if obj and obj[-1].hasNode:
            if obj[-1] in self.selectionList:
                if self.master.ctrlPressed:
                    self.removeSelection(obj[-1])
                else:
                    self.clearSelection()
                    self.addSelection(obj[-1])
            else:
                if self.master.ctrlPressed:
                    self.addSelection(obj[-1])
                else:
                    self.clearSelection()
                    self.addSelection(obj[-1])
            obj[-1].setChanged()
        else:
            self.clearSelection()
        self.master.updateSelection([a.node for a in self.selectionList])
        self.canvas().update()
        """

    def contentsMouseReleaseEvent(self, event):
        #obj=self.canvas().collisions(event.pos())
        #if obj and obj[-1]
        self.updateSelection()
        self.selectionRect.setCanvas(None)
        self.selectionRect=None
        self.master.updateSelection([a.node for a in self.selectionList])
        self.canvas().update()
        pass
    
    def updateSelection(self):
        obj=self.canvas().collisions(self.selectionRect.rect())
        obj=filter(lambda a:isinstance(a, CanvasSOMItem) and a.hasNode, obj)
        if self.master.ctrlPressed:
            set1=sets.Set(obj)
            set2=sets.Set(self.oldSelection)
            intersection=set1.intersection(set2)
            union=set1.union(set2)
            for e in list(intersection):
                self.removeSelection(e)
            for e in list(set1.difference(set2)):
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
        self.selectionList=filter(lambda a:a.isSelected ,self.selectionList)
    
    def clearSelection(self):
        for o in self.selectionList:
            o.setSelected(False)
        self.selectionList=[]
        
    def invertSelection(self):
        for n in self.canvas().canvasObj:
            if n.hasNode:
                if not n.isSelected:
                    self.addSelection(n)
                else:
                    self.removeSelection(n)
        self.master.updateSelection([a.node for a in self.selectionList])
        self.canvas().update()

baseColor=QColor(20,20,20)  

class SOMCanvas(QCanvas):
    def __init__(self, *args):
        apply(QCanvas.__init__, (self,)+args)
        self.drawMode=1
        self.showGrid=True
        self.component=0
        self.objSize=25
        self.canvasObj=[]
        self.somMap=None

    def drawHistogram(self):
        if self.parent().inputSet:
            maxVal=max([len(n.mappedExamples) for n in self.somMap.nodes]+[1])
        else:
            maxVal=max([len(n.examples) for n in self.somMap.nodes]+[1])
        if self.drawMode==1:
            maxSize=4*int(self.somMap.xDim*self.objSize/(2*self.somMap.xDim-1)*0.7)
        else:
            maxSize=4*int(self.objSize*0.7)
        for n in self.canvasObj:
            if n.hasNode:
                if self.parent().inputSet:
                    size=float(len(n.node.mappedExamples))/maxVal*maxSize
                else:
                    size=float(len(n.node.examples))/maxVal*maxSize
                if self.parent().drawPies():
                    n.setPie(self.parent().attribute, size)
                else:
                    n.setHist(size)
                    
        self.updateHistogramColors()            

    def updateHistogramColors(self):
        if self.parent().drawPies():
            return
        attr=self.somMap.examples.domain.variables[self.parent().attribute]
        for n in self.canvasObj:
            if n.hasNode:
                if attr.varType==orange.VarTypes.Discrete:
                    if self.parent().inputSet:
                        dist=orange.Distribution(attr, n.node.mappedExamples)
                    else:
                        dist=orange.Distribution(attr, n.node.examples)
                    colors=OWGraphTools.ColorPaletteHSV(len(dist))
                    maxProb=max(dist)
                    majValInd=filter(lambda i:dist[i]==maxProb, range(len(dist)))[0]
                    if self.parent().discHistMode==1:
                        n.histObj[0].setBrush(QBrush(colors[majValInd]))
                    elif self.parent().discHistMode==2:
                        light=180-80*float(dist[majValInd])/max(sum(dist),1)
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
    
    def setSom(self, somMap=None):
        self.oldSom=self.somMap
        self.somMap=somMap
        self.clear()
        if not self.somMap:
            return
        
        self.somNodeMap={}
        for n in somMap.nodes:
            self.somNodeMap[(n.x,n.y)]=n

        if self.drawMode==1:
            self.uMat=orngSOM.getUMat(somMap)
            if somMap.topology==orangeom.SOMLearner.HexagonalTopology:
                self.drawUMatHex()
            else:
                self.drawUMatRect()
        else:
            if somMap.topology==orangeom.SOMLearner.HexagonalTopology:
                self.drawHex()
            else:
                self.drawRect()
            if self.drawMode!=0:
                minVal=min([n.vector[self.component] for n in self.somMap.nodes])
                maxVal=max([n.vector[self.component] for n in self.somMap.nodes])
                for o in self.canvasObj:
                    val=255-max(min(255*(o.node.vector[self.component]-minVal)/(maxVal-minVal),245),10)
                    o.setColor(QColor(val,val,val))
        if (self.parent().inputSet==0 or self.parent().examples) and self.parent().histogram:
            self.drawHistogram()
        self.updateLabels()
        
    def redrawSom(self):    #for redrawing without clearing the selection 
        if not self.somMap:
            return
        oldSelection=self.parent().canvasView.selectionList
        self.setSom(self.somMap)
        nodeList=[n.node for n in oldSelection]
        newSelection=[]
        for o in self.canvasObj:
            if o.node in nodeList:
                o.setSelected(True)
                newSelection.append(o)
        self.parent().canvasView.selectionList=newSelection
        self.update()
    
    def drawHex(self):
        #size=self.objSize*2-1
        #size=int((self.objSize-9)/10.0*20)*2-1
        size=2*self.objSize-1
        x,y=size, size*2
        for n in self.somMap.nodes:
            offset=1-abs(n.x%2-2)
            h=CanvasHexagon(self)
            h.setSize(size)
            h.setNode(n)
            (xa,ya)=h.advancement()
            h.move(x+n.x*xa, y+n.y*ya+offset*ya/2)
            h.show()
            self.canvasObj.append(h)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
    
    def drawRect(self):
        size=self.objSize*2-1
        x,y=size, size
        self.resize(1,1)    # crashes at update without this line !!!
        for n in self.somMap.nodes:
            r=CanvasRectangle(self)
            r.setSize(size)
            r.setNode(n)
            (xa,ya)=r.advancement()
            r.move(x+n.x*xa, y+n.y*ya)
            r.show()
            self.canvasObj.append(r)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
    
    def drawUMatHex(self):
        #size=2*(int(self.objSize*1.15)/2)-1
        size=2*int(self.somMap.xDim*self.objSize/(2*self.somMap.xDim-1))-1
        x,y=size, size
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            offset=2-abs(i%4-2)
            for j in range(len(self.uMat[i])):
                h=CanvasHexagon(self)
                h.setSize(size)
                (xa,ya)=h.advancement()
                h.move(x+i*xa, y+j*ya+offset*ya/2)
                if i%2==0 and j%2==0:
                    h.setNode(self.somNodeMap[(i/2,j/2)])
                h.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                h.setColor(QColor(val, val, val))
                self.canvasObj.append(h)
        self.resize(x+self.somMap.xDim*xa*2, y+self.somMap.yDim*ya*2)
        self.update()
        
    def drawUMatRect(self):
        #size=self.objSize-1
        size=2*int(self.somMap.xDim*self.objSize/(2*self.somMap.xDim-1))-1
        x,y=size, size
        
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            for j in range(len(self.uMat[i])):
                r=CanvasRectangle(self)
                r.setSize(size)
                if i%2==0 and j%2==0:
                    r.setNode(self.somNodeMap[(i/2,j/2)])
                (xa,ya)=r.advancement()
                r.move(x+i*xa, y+j*ya)
                r.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                r.setColor(QColor(val, val, val))
                self.canvasObj.append(r)
        self.resize(x+self.somMap.xDim*xa*2, y+self.somMap.yDim*ya*2)
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
        self.setAllChanged()
        self.update()
        
    def clear(self):
        for o in self.canvasObj:
            o.setCanvas(None)
        self.canvasObj=[]


class OWSOMVisualizer(OWWidget):
    settingsList=["canvas.drawMode","canvas.objSize","commitOnChange", "backgroundMode", "backgroundCheck"]
    def __init__(self, parent=None, signalManager=None, name="SOMVisualizer"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs=[("SOMMap", orangeom.SOMMap, self.setSomMap), ("SOMClassifier", orangeom.SOMClassifier, self.setSomClassifier), ("Examples", ExampleTable, self.data)]
        self.outputs=[("Examples", ExampleTable)]
        
        self.labelNodes=0
        self.commitOnChange=0
        self.backgroundCheck=1
        self.backgroundMode=0
        self.histogram=1
        self.attribute=0
        self.discHistMode=0
        self.targetValue=0
        self.contHistMode=0
        self.inputSet=0

        self.somMap=None
        self.examples=None
        
        
        layout=QVBoxLayout(self.mainArea,QVBoxLayout.TopToBottom,0)
        self.canvas=SOMCanvas(self)
        self.canvasView=SOMCanvasView(self, self.canvas, self.mainArea)
        self.canvasView.setCanvas(self.canvas)
        layout.addWidget(self.canvasView)
        
        self.loadSettings()
        call=lambda:self.canvas.redrawSom()
        tabW=QTabWidget(self.controlArea)
        mainTab=OWGUI.widgetBox(self.controlArea)
        histTab=OWGUI.widgetBox(self.controlArea)
        mainTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        histTab.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.mainTab=mainTab
        self.histTab=histTab
        tabW.addTab(mainTab, "Options")
        tabW.addTab(histTab, "Histogram Coloring")
        self.backgroundBox=QVButtonGroup("Background", mainTab)
        #OWGUI.checkBox(self.backgroundBox, self, "backgroundCheck","Show background", callback=self.setBackground)
        b=OWGUI.radioButtonsInBox(self.backgroundBox, self, "canvas.drawMode", ["None", "U-Matrix", "Component Planes"], callback=self.setBackground)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.componentCombo=OWGUI.comboBox(b,self,"canvas.component", callback=self.setBackground)
        self.componentCombo.setEnabled(self.canvas.drawMode==2)
        OWGUI.checkBox(self.backgroundBox, self, "canvas.showGrid", "Show Grid", callback=self.canvas.updateAll)
        #b=OWGUI.widgetBox(mainTab, "Histogram")
        b=QVButtonGroup("Histogram", mainTab)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(b, self, "histogram", "Show histogram", callback=self.setHistogram)
        OWGUI.radioButtonsInBox(b, self, "inputSet", ["Use training set", "Use input subset"], callback=self.setHistogram)
        
        b1=QVBox(mainTab)
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        b=OWGUI.hSlider(b1, self, "canvas.objSize","Plot size", 10,100,step=10,ticks=10, callback=call)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        #OWGUI.checkBox(b1, self, "labelNodes", "Node Labeling", callback=self.canvas.updateLabels)
        b1=OWGUI.widgetBox(b1, "Bubble Info")
        OWGUI.checkBox(b1, self, "canvasView.showBubbleInfo","Show")
        OWGUI.checkBox(b1, self, "canvasView.includeCodebook", "Include codebook vector")
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        
        #OWGUI.checkBox(mainTab, self, "commitOnChange", "Commit on change")
        QVBox(mainTab)
        self.histogramBox=OWGUI.widgetBox(histTab, "Coloring")
        self.attributeCombo=OWGUI.comboBox(self.histogramBox, self, "attribute", "Attribute", callback=self.setHistogram)
        self.tabWidget=QTabWidget(self.histogramBox)
        self.discTab=discTab=OWGUI.widgetBox(self.histogramBox)
        self.contTab=contTab=OWGUI.widgetBox(self.histogramBox)
        self.tabWidget.addTab(discTab, "Discrete")
        self.tabWidget.addTab(contTab, "Continous")
        b=QVButtonGroup(discTab)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.radioButtonsInBox(b, self, "discHistMode", ["Pie chart", "Majority value", "Majority value prob."], callback=self.setHistogram)
        #self.targetValueCombo=OWGUI.comboBox(b, self, "targetValue", callback=self.setHistogram)
        QVBox(discTab)
        b=QVButtonGroup(contTab)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.radioButtonsInBox(b, self, "contHistMode", ["Default", "Avg. value"],callback=self.setHistogram)
        QVBox(contTab)
 
        b=OWGUI.widgetBox(self.controlArea, "Selection")
        OWGUI.button(b, self, "&Invert selection", callback=self.canvasView.invertSelection)
        OWGUI.button(b, self, "&Commit", callback=self.commit)
        OWGUI.checkBox(b, self, "commitOnChange", "Commit on change")
        OWGUI.button(self.controlArea, self, "&Save Graph", callback=self.saveGraph)
        
        self.selectionList=[]
        self.ctrlPressed=False
        self.setFocusPolicy(QWidget.StrongFocus)
        self.resize(600,600)

    def setMode(self):
        self.componentCombo.setEnabled(self.canvas.drawMode==2)
        self.canvas.redrawSom()

    def setBackground(self):
        self.setMode()

    def setHistogram(self):
        if self.somMap.examples.domain.variables[self.attribute].varType==orange.VarTypes.Discrete:
            self.tabWidget.setTabEnabled(self.discTab,True)
            self.tabWidget.setTabEnabled(self.contTab,False)
            self.tabWidget.showPage(self.discTab)
        else:
            self.tabWidget.setTabEnabled(self.discTab,False)
            self.tabWidget.setTabEnabled(self.contTab,True)
            self.tabWidget.showPage(self.contTab)
        self.canvas.redrawSom()

    def drawPies(self):
        return self.discHistMode==0 and self.somMap.examples.domain.variables[self.attribute].varType==orange.VarTypes.Discrete
        
    def setSomMap(self, somMap=None):
        self.somType="Map"
        self.setSom(somMap)
        
    def setSomClassifier(self, somMap=None):
        self.somType="Classifier"
        self.setSom(somMap)
        
    def setSom(self, somMap=None):
        self.somMap=somMap
        if not somMap:
            self.clear()
            return
        self.componentCombo.clear()
        self.attributeCombo.clear()
        #self.targetValueCombo.clear()
        self.targetValue=0
        self.canvas.component=0
        self.attribute=0
        for v in somMap.examples.domain.attributes:
            self.componentCombo.insertItem(v.name)
        for v in somMap.examples.domain.variables:
            self.attributeCombo.insertItem(v.name)
        #for v in somMap.examples.domain.attributes[self.attribute].values:
        #    self.targetValueCombo.insertItem(str(v))

        if self.somMap.examples.domain.variables[self.attribute].varType==orange.VarTypes.Discrete:
            self.tabWidget.setTabEnabled(self.discTab,True)
            self.tabWidget.setTabEnabled(self.contTab,False)
            self.tabWidget.showPage(self.discTab)
        else:
            self.tabWidget.setTabEnabled(self.discTab,False)
            self.tabWidget.setTabEnabled(self.contTab,True)
            self.tabWidget.showPage(self.contTab)        
            
        self.canvas.setSom(somMap)
       
    def data(self, data=None):
        self.examples=data
        if data and self.somMap:
            for n in self.somMap.nodes:
                setattr(n,"mappedExamples", orange.ExampleTable(data.domain))
            for e in data:
                bmu=self.somMap.getWinner(e)
                bmu.mappedExamples.append(e)
            if self.inputSet==1:
                self.setHistogram()
                

    def clear(self):
        self.componentCombo.clear()
        self.canvas.component=0
        self.canvas.setSom(None)
        self.send("Examples", None)
        
    def updateSelection(self, nodeList):
        self.selectionList=nodeList
        if self.commitOnChange:
            self.commit()
            
    def commit(self):
        ex=orange.ExampleTable(self.somMap.examples.domain)
        for n in self.selectionList:
            if self.inputSet==0 and n.examples:
                ex.extend(n.examples)
            elif self.inputSet==1 and n.mappedExamples:
                ex.extend(n.mappedExamples)
        if len(ex):
            self.send("Examples",ex)
        else:
            self.send("Examples",None)

    def saveGraph(self):
        sizeDlg = OWChooseImageSizeDlg(self.canvas)
        sizeDlg.exec_loop()
        return
        
    def keyPressEvent(self, event):
        if event.key()==Qt.Key_Control:
            self.ctrlPressed=True
        else:
            event.ignore()
      
    def keyReleaseEvent(self, event):
        if event.key()==Qt.Key_Control:
            self.ctrlPressed=False
        else:
            event.ignore()
        
        
if __name__=="__main__":
    ap=QApplication(sys.argv)
    data=orange.ExampleTable("../../doc/datasets/iris.tab")
    l=orngSOM.SOMLearner( topology=orangeom.SOMLearner.RectangularTopology)
    l=l(data)
    l.data=data
    w=OWSOMVisualizer()
    ap.setMainWidget(w)
    w.setSomClassifier(l)
    w.data(orange.ExampleTable(data[:50]))
    w.show()
    ap.exec_loop()

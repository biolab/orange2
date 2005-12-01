"""
<name>SOMVisualizer</name>
<description>SOM visualizer</description>
<icon>SOMVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjevec(@at@)fri.uni-lj.si)</contact> 
<priority>5020</priority>
"""
import orngSOM, orange, orangeom
import math, Numeric
import OWGUI, OWGraphTools
from OWWidget import *
from qt import *
from qtcanvas import *
from OWTreeViewer2D import CanvasBubbleInfo
from OWDlgs import OWChooseImageSizeDlg

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
        self.innerPoints=None
        self.setSize(10)

    def boundingRect(self):
        return self.outlinePoints.boundingRect()

    def move(self, x, y):
        ox, oy=self.x(), self.y()
        dx, dy=ox-x, oy-y
        self.outlinePoints.translate(-dx,-dy)
        QCanvasPolygon.move(self, x, y)

    def areaPoints(self):
        return self.outlinePoints
    
    def setSize(self, size):
        self.outlinePoints=QPointArray(self.segments)
        for i in range(self.segments):
            x=int(math.cos(2*math.pi/self.segments*i+self.startAngle)*size)
            y=int(math.sin(2*math.pi/self.segments*i+self.startAngle)*size)
            self.outlinePoints.setPoint(i,x,y)
        if not self.innerPoints:
            self.setInnerSize(size)
            
    def setInnerSize(self, size):
        self.innerPoints=QPointArray(self.segments)
        for i in range(self.segments):
            x=int(math.cos(2*math.pi/self.segments*i+self.startAngle)*size)
            y=int(math.sin(2*math.pi/self.segments*i+self.startAngle)*size)
            self.innerPoints.setPoint(i,x,y)
        self.setPoints(self.innerPoints)
        if not self.outlinePoints:
            self.setSize(size)
        
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
        if self.canvas().showGrid:
            color=self.isSelected and Qt.red or Qt.black
            p=QPen(color)
            if self.isSelected:
                p.setWidth(2)
            painter.setPen(p)
            painter.drawPolyline(self.outlinePoints)
            painter.drawLine(self.outlinePoints.point(0)[0],self.outlinePoints.point(0)[1],
                             self.outlinePoints.point(self.segments-1)[0],self.outlinePoints.point(self.segments-1)[1])
        if self.node:
            painter.setPen(QPen(self.isSelected and Qt.red or self.textColor))
            if self.labelText:
                painter.drawText(self.boundingRect(), Qt.AlignVCenter | Qt.AlignLeft, " "+self.labelText)
            else:
                #return 
                painter.setBrush(QBrush(self.isSelected and Qt.red or self.textColor))
                painter.drawPie(self.x()-2,self.y()-2,4,4,0,5760)
        #self.setChanged()
        
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
        
    def setSom(self, somMap=None):
        self.oldSom=self.somMap
        self.somMap=somMap
        self.clear()
        if not self.somMap:
            return
        
        self.somNodeMap={}
        for n in somMap.nodes:
            self.somNodeMap[(n.x,n.y)]=n
            
        if self.drawMode==0:
            self.uMat=orngSOM.getUMat(somMap)
            if somMap.topology==orangeom.SOMLearner.HexagonalTopology:
                self.drawUMatHex()
            else:
                self.drawUMatRect()
        elif self.drawMode==1:
            if somMap.topology==orangeom.SOMLearner.HexagonalTopology:
                self.drawHistogramHex()
            else:
                self.drawHistogramRect()
        else:
            if somMap.topology==orangeom.SOMLearner.HexagonalTopology:
                self.drawHex()
            else:
                self.drawRect()
            minVal=min([n.vector[self.component] for n in self.somMap.nodes])
            maxVal=max([n.vector[self.component] for n in self.somMap.nodes])
            for o in self.canvasObj:
                val=255-max(min(255*(o.node.vector[self.component]-minVal)/(maxVal-minVal),245),10)
                o.setColor(QColor(val,val,val))
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
        size=self.objSize*2-1
        x,y=size*2, size*2
        for n in self.somMap.nodes:
            offset=1-abs(n.x%2-2)
            h=CanvasHexagon(self)
            h.setSize(size)
            h.setInnerSize(size)
            h.setNode(n)
            (xa,ya)=h.advancement()
            h.move(x+n.x*xa, y+n.y*ya+offset*ya/2)
            h.show()
            self.canvasObj.append(h)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
    
    def drawRect(self):
        size=self.objSize*2-1
        x,y=size*2, size*2
        self.resize(1,1)    # crashes at update without this line !!!
        for n in self.somMap.nodes:
            r=CanvasRectangle(self)
            r.setSize(size)
            r.setInnerSize(size)
            r.setNode(n)
            (xa,ya)=r.advancement()
            r.move(x+n.x*xa, y+n.y*ya)
            r.show()
            self.canvasObj.append(r)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
    
    def drawHistogramHex(self):
        size=self.objSize*2-1
        x,y=size*2, size*2
        maxVal=max([len(n.examples) for n in self.somMap.nodes])
        colors=OWGraphTools.ColorPaletteHSV(len(n.examples.domain.classVar.values))
        self.resize(1,1)    # crashes at update without this line !!!
        for n in self.somMap.nodes:
            offset=offset=1-abs(n.x%2-2)
            h=CanvasHexagon(self)
            h.setNode(n)
            h.setSize(size)
            h.setInnerSize(size*float(len(n.examples))/maxVal)
            (xa,ya)=h.advancement()
            h.move(x+n.x*xa, y+n.y*ya+offset*ya/2)
            h.show()
            h.setColor(Qt.lightGray) #colors[int(n.classifier.defaultVal)])
            self.canvasObj.append(h)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
    
    def drawHistogramRect(self):
        size=self.objSize*2-1
        x,y=size*2, size*2
        maxVal=max([len(n.examples) for n in self.somMap.nodes]+[1])
        colors=OWGraphTools.ColorPaletteHSV(len(n.examples.domain.classVar.values))
        for n in self.somMap.nodes:
            r=CanvasRectangle(self)
            r.setSize(size)
            r.setInnerSize(size*len(n.examples)/maxVal)
            r.setNode(n)
            (xa, ya)=r.advancement()
            r.move(y+n.x*xa, y+n.y*ya)
            r.show()
            r.setColor(colors[int(n.classifier.defaultVal)])
            self.canvasObj.append(r)
        self.resize(x+self.somMap.xDim*xa, y+self.somMap.yDim*ya)
        self.update()
        
    
    def drawUMatHex(self):
        size=self.objSize*2-1
        x,y=size*2, size*2
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            offset=2-abs(i%4-2)
            for j in range(len(self.uMat[i])):
                h=CanvasHexagon(self)
                h.setSize(size)
                h.setInnerSize(size)
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
        size=self.objSize*2-1
        x,y=size*2, size*2
        
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            for j in range(len(self.uMat[i])):
                r=CanvasRectangle(self)
                r.setSize(size)
                r.setInnerSize(size)
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
    settingsList=["canvas.drawMode","canvas.objSize","commitOnChange"]
    def __init__(self, parent=None, signalManager=None, name="SOMVisualizer"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs=[("SOMMap", orangeom.SOMMap, self.setSomMap), ("SOMClassifier", orangeom.SOMClassifier, self.setSomClassifier), ("Examples", ExampleTable, self.data)]
        self.outputs=[("Examples", ExampleTable)]
        
        self.labelNodes=0
        self.commitOnChange=0
        
        
        layout=QVBoxLayout(self.mainArea,QVBoxLayout.TopToBottom,0)
        self.canvas=SOMCanvas(self)
        self.canvasView=SOMCanvasView(self, self.canvas, self.mainArea)
        self.canvasView.setCanvas(self.canvas)
        layout.addWidget(self.canvasView)
        
        self.loadSettings()
        call=lambda:self.canvas.redrawSom()
        box=QVBox(self.controlArea)
        b=OWGUI.radioButtonsInBox(box, self, "canvas.drawMode", ["U-Matrix", "Histogram", "Component Planes"], box="Visualization Method", callback=self.setMode)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.componentCombo=OWGUI.comboBox(b,self,"canvas.component", callback=call)
        self.componentCombo.setEnabled(self.canvas.drawMode==2)
        QRadioButton
        b1=QVBox(box)
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        b=OWGUI.hSlider(b1, self, "canvas.objSize","Size", 10,20,step=2,ticks=10, callback=call)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(box, self, "labelNodes", "Node Labeling", callback=self.canvas.updateLabels)
        OWGUI.checkBox(box, self, "canvas.showGrid", "Show Grid", callback=self.canvas.updateAll)
        b1=OWGUI.widgetBox(box, "Bubble Info")
        OWGUI.checkBox(b1, self, "canvasView.showBubbleInfo","Show")
        OWGUI.checkBox(b1, self, "canvasView.includeCodebook", "Include codebook vector")
        b1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(box, self, "commitOnChange", "Commit on change")
        OWGUI.button(box, self, "&Invert selection", callback=self.canvasView.invertSelection)
        OWGUI.button(box, self, "&Commit", callback=self.commit)
        OWGUI.button(self.controlArea, self, "&Save Graph", callback=self.saveGraph)
        
        self.selectionList=[]
        self.ctrlPressed=False
        self.setFocusPolicy(QWidget.StrongFocus)
        self.resize(600,600)

    def setMode(self):
        self.componentCombo.setEnabled(self.canvas.drawMode==2)
        self.canvas.redrawSom()
        
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
        self.canvas.component=0
        for v in somMap.examples.domain.attributes:
            self.componentCombo.insertItem(v.name)
        self.canvas.setSom(somMap)
       
    def data(self, data=None):
        self.examples=data

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
            if n.examples:
                ex.extend(n.examples)
        if len(ex):
            self.send("Examples",ex)
        else:
            self.send("Examples",None)

    def saveGraph(self):
        sizeDlg = OWChooseImageSizeDlg(self.canvas)
        sizeDlg.exec_loop()
        return
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        dSize=self.canvas.size()
        buffer = QPixmap(dSize.width(),dSize.height()) # any size can do, now using the window size
        painter=QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        self.canvasView.drawContents(painter,0,0,dSize.width(), dSize.height())
        painter.end()
        buffer.save(fileName, ext)
        
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
    l=orngSOM.SOMLearner(data)#, topology=orangeom.SOMLearner.RectangularTopology)
    l.data=data
    w=OWSOMVisualizer()
    ap.setMainWidget(w)
    w.setSomClassifier(l)
    w.show()
    ap.exec_loop()

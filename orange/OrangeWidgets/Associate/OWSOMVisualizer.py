"""
<name>SOMVisualizer</name>
<description>SOM visualizer</description>
<icon>SOMVisualizer.png</icon>
<contact>Ales Erjavec (ales.erjevec324(@at@)email.si)</contact> 
<priority>5020</priority>
"""
import orngSOM, orange, orangeom
import math, Numeric
import OWGUI
from OWWidget import *
from qt import *
from qtcanvas import * 


class Hexagon(QCanvasPolygon):
    def __init__(self, *args):
        apply(QCanvasPolygon.__init__,(self,)+args)
        self.hasNode=False
        self.node=None
        self.isSelected=False
        self.label=QCanvasText("",self.canvas())
        self.label.setTextFlags(Qt.AlignCenter)
        self.dot=QCanvasEllipse(5,5,self.canvas())
        self.dot.setBrush(QBrush(Qt.black))
        self.obj=[self.label, self.dot]
        for o in self.obj:
            o.setZ(self.z()+10)
            
    def setSize(self, size):
        r=size #*2/(math.cos(math.pi/3)+1)
        points=QPointArray(6)
        for i in range(6):
            x=math.cos(i*math.pi/3.0)*r
            y=math.sin(i*math.pi/3.0)*r
            points.setPoint(i, x,y)
        self.setPoints(points)
        
    def move(self, x, y):
        apply(QCanvasPolygon.move, (self, x,y))
        for o in self.obj:
            o.move(x,y)
            
    def setNode(self, node=None):
        self.node=node
        self.hasNode=True
        if node:
            for o in self.obj:
                o.show()
                
    def setSelected(self, bool=False):
        self.isSelected=bool
        if bool:
            self.dot.setBrush(QBrush(Qt.red))
        else:
            self.dot.setBrush(QBrush(Qt.black))

    def setLabel(self, label):
        self.label.setText(label)
    
    def setCanvas(self, canvas):
        apply(QCanvasPolygon.setCanvas,(self, canvas))
        for o in self.obj:
            o.setCanvas(canvas)
            
class Rectangle(QCanvasRectangle):
    def __init__(self, *args):
        apply(QCanvasRectangle.__init__,(self,)+args)
        self.hasNode=False
        self.node=None
        self.isSelected=False
        self.label=QCanvasText("",self.canvas())
        self.label.setTextFlags(Qt.AlignCenter)
        self.dot=QCanvasEllipse(5,5,self.canvas())
        self.dot.setBrush(QBrush(Qt.black))
        self.obj=[self.label, self.dot]
        for o in self.obj:
            o.setZ(self.z()+10)
            
    def setLabel(self, label):
        self.label.setText(label)
        
    def move(self, x, y):
        apply(QCanvasRectangle.move, (self, x-self.width()/2, y-self.height()/2))
        for o in self.obj:
            o.move(x, y)
      
    def setSize(self, w,h):
        x=self.x()+self.width()/2
        y=self.y()+self.height()/2
        apply(QCanvasRectangle.setSize, (self,w,h))
        self.move(x,y)
        
    def setNode(self, node=None):
        self.node=node
        self.hasNode=True
        if node:
            for o in self.obj:
                o.show()
                
    def setSelected(self, bool=False):
        self.isSelected=bool
        if bool:
            self.dot.setBrush(QBrush(Qt.red))
        else:
            self.dot.setBrush(QBrush(Qt.black))
                      
    def setCanvas(self, canvas):
        apply(QCanvasRectangle.setCanvas,(self, canvas))
        for o in self.obj:
            o.setCanvas(canvas)
    
class SOMCanvasView(QCanvasView):
    def __init__(self, master, canvas, *args):
        apply(QCanvasView.__init__, (self,canvas)+args)
        self.master=master
        self.selectionList=[]
        
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
            if hasattr(n, "hasNode") and n.hasNode:
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
            minVal=min([float(n.referenceExample[self.component]) for n in self.somMap.nodes])
            maxVal=max([float(n.referenceExample[self.component]) for n in self.somMap.nodes])
            for o in self.canvasObj:
                val=255-max(min(255*(float(o.node.referenceExample[self.component])-minVal)/(maxVal-minVal),245),10)
                o.setBrush(QBrush(QColor(val,val,val)))
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
        size=self.objSize
        x,y=size, size*2
        for n in self.somMap.nodes:
            offset=offset=1-abs(n.x%2-2)
            h=Hexagon(self)
            h.move(x+n.x*size*2*0.733, y+(n.y*size*2+offset*size)*0.834)
            h.setNode(n)
            h.setSize(size)
            h.show()
            #h.setBrush(QBrush(Qt.darkGray.light(160-60*len(n.examples)/maxVal)))
            self.canvasObj.append(h)
        self.resize((self.somMap.xDim+1)*size*2*0.733, (self.somMap.yDim+1)*size*2*0.834)
        self.update()
            
    
    def drawRect(self):
        size=self.objSize
        x,y=size, size
        for n in self.somMap.nodes:
            r=Rectangle(self)
            r.move(x+n.x*size*2, y+n.y*size*2)
            r.setSize(size*2, size*2)
            r.setNode(n)
            r.show()
            r.setPen(QPen(Qt.NoPen))
            self.canvasObj.append(r)
        self.resize((self.somMap.xDim+1)*size*2, (self.somMap.yDim+1)*size*2)
        self.update()
          
    def drawHistogramHex(self):
        size=self.objSize
        x,y=size, size*2
        maxVal=max([len(n.examples) for n in self.somMap.nodes])
        for n in self.somMap.nodes:
            offset=offset=1-abs(n.x%2-2)
            h=Hexagon(self)
            h.move(x+n.x*size*2*0.733, y+(n.y*size*2+offset*size)*0.834)
            h.setNode(n)
            h.setSize(size*max(float(len(n.examples))/maxVal,1.0/5.0))
            h.show()
            h.setBrush(QBrush(Qt.darkGray.light(160-60*len(n.examples)/maxVal)))
            self.canvasObj.append(h)
        self.resize((self.somMap.xDim+1)*size*2*0.733, (self.somMap.yDim+1)*size*2*0.834)
        self.update()
            
    
    def drawHistogramRect(self):
        size=self.objSize
        x,y=size, size
        maxVal=max([len(n.examples) for n in self.somMap.nodes]+[1])
        for n in self.somMap.nodes:
            r=Rectangle(self)
            r.move(x+n.x*size*2, y+n.y*size*2)
            s=max(float(len(n.examples))/maxVal,1.0/5.0)
            r.setSize(size*2*s, size*2*s)
            r.setNode(n)
            r.show()
            r.setBrush(QBrush(Qt.darkGray.light(160-60*len(n.examples)/maxVal)))
            r.setPen(QPen(Qt.NoPen))
            self.canvasObj.append(r)
        self.resize((self.somMap.xDim+1)*size*2, (self.somMap.yDim+1)*size*2)
        self.update()
    
    def drawUMatHex(self):
        size=self.objSize
        x,y=size, size
        #rr=math.sin(2*size/(math.cos(math.pi/3)+1))
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            offset=2-abs(i%4-2)
            for j in range(len(self.uMat[i])):
                h=Hexagon(self)
                h.move(x+i*size*2*0.733,y+(j*size*2+offset*size)*0.834)
                h.setSize(size)
                if i%2==0 and j%2==0:
                    h.setNode(self.somNodeMap[(i/2,j/2)])
                h.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                h.setBrush(QBrush(QColor(val, val, val)))
                self.canvasObj.append(h)
        self.resize(2*size*(2*self.somMap.xDim)*0.733, 2*size*(2*self.somMap.yDim)*0.834)
        self.update()
        
    def drawUMatRect(self):
        size=self.objSize
        x,y=size, size
        
        maxDist=max(reduce(Numeric.maximum, [a for a in self.uMat]))
        minDist=max(reduce(Numeric.minimum, [a for a in self.uMat]))
        for i in range(len(self.uMat)):
            for j in range(len(self.uMat[i])):
                r=Rectangle(self)
                r.move(x+i*size*2,y+j*size*2)
                r.setSize(size*2, size*2)
                if i%2==0 and j%2==0:
                    r.setNode(self.somNodeMap[(i/2,j/2)])
                r.show()
                val=255-min(max(255*(self.uMat[i][j]-minDist)/(maxDist-minDist),10),245)
                r.setBrush(QBrush(QColor(val, val, val)))
                r.setPen(QPen(Qt.NoPen))
                self.canvasObj.append(r)
        self.resize(2*size*(2*self.somMap.xDim), 2*size*(2*self.somMap.yDim))
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
        b=OWGUI.hSlider(b1, self, "canvas.objSize","Size", 25,40, callback=call)
        b.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed))
        OWGUI.checkBox(box, self, "labelNodes", "Node Labeling", callback=self.canvas.updateLabels)
        OWGUI.checkBox(box, self, "commitOnChange", "Commit on change")
        OWGUI.button(box, self, "&Invert selection", callback=self.canvasView.invertSelection)
        OWGUI.button(box, self, "&Commit", callback=self.commit)
        
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

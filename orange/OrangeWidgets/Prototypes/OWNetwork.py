"""
<name>Network</name>
<description>Network Widget visualizes graphs.</description>
<icon>icons/Outlier.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2040</priority>
"""
import OWGUI

from OWWidget import *
from qwt import *
from qt import *
from OWGraphDrawerCanvas import *
from orngNetwork import * 

class OWNetwork(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'GraphDrawer')

        self.inputs = [("Graph with ExampleTable", orange.Graph, self.setGraph)]
        self.outputs=[("Selected Examples", ExampleTable), ("Selected Graph", orange.Graph)]
        
        self.graphShowGrid = 1  # show gridlines in the graph

        # GUI       
        self.optimizeBox = QVGroupBox("Optimize", self.controlArea)
        OWGUI.button(self.optimizeBox, self, "Random", callback=self.random)
        OWGUI.button(self.optimizeBox, self, "F-R", callback=self.ff)
        OWGUI.button(self.optimizeBox, self, "Circular", callback=self.circular)
        
        self.styleBox = QVGroupBox("Style", self.controlArea)
        QLabel("Color Attribute", self.styleBox)
        self.colorCombo = OWGUI.comboBox(self.styleBox, self, "color", callback=self.setVertexColor)
        self.colorCombo.insertItem("(none)")
        
        QLabel("Select Marker Attributes", self.styleBox)
        self.attListBox = QListBox(self.styleBox, "markerAttributes")
        self.attListBox.setMultiSelection(True)
        self.styleBox.connect(self.attListBox, SIGNAL("clicked( QListBoxItem * )"), self.clickedAttLstBox)
        
        QLabel("Select ToolTip Attributes", self.styleBox)
        self.tooltipListBox = QListBox(self.styleBox, "tooltipAttributes")
        self.tooltipListBox.setMultiSelection(True)
        self.styleBox.connect(self.tooltipListBox, SIGNAL("clicked( QListBoxItem * )"), self.clickedTooltipLstBox)
        
        self.selectBox = QVGroupBox("Select", self.controlArea)
        QLabel("Select Hubs", self.selectBox)
        self.spinExplicit = 0
        self.spinPercentage = 0
        OWGUI.spin(self.selectBox, self, "spinPercentage", 0, 100, 1, label="Percentage:", callback=self.selectHubsPercentage)
        OWGUI.spin(self.selectBox, self, "spinExplicit", 0, 1000000, 1, label="Explicit value:", callback=self.selectHubsExplicit)

        QLabel("Select Connected Nodes", self.selectBox)
        self.connectDistance = 0
        OWGUI.spin(self.selectBox, self, "connectDistance", 0, 100, 1, label="Distance:", callback=self.selectConnectedNodes)
 
        OWGUI.button(self.selectBox, self, "Select all Connected Nodes", callback=self.selectAllConnectedNodes)
        
        pics=pixmaps()
        
        self.cgb = QHGroupBox(self.controlArea)
        self.btnZomm = OWGUI.button(self.cgb, self, "", callback=self.btnZommClicked)
        self.btnZomm.setToggleButton(1)
        self.btnZomm.setPixmap(QPixmap(pics.LOOKG))
        
        self.btnMpos = OWGUI.button(self.cgb, self, "", callback=self.btnMposClicked)
        self.btnMpos.setToggleButton(1)
        self.btnMpos.setPixmap(QPixmap(pics.HAND))
        
        self.btnRctSel = OWGUI.button(self.cgb, self, "", callback=self.btnRctSelClicked)
        self.btnRctSel.setToggleButton(1)
        self.btnRctSel.setPixmap(QPixmap(pics.RECTG))
        
        self.btnPolySel = OWGUI.button(self.cgb, self, "", callback=self.btnPolySelClicked)
        self.btnPolySel.setToggleButton(1)
        self.btnPolySel.setPixmap(QPixmap(pics.POLYG))
        
        OWGUI.button(self.controlArea, self, "Send", callback=self.sendData)
        
        self.graph = OWGraphDrawerCanvas(self, self.mainArea, "ScatterPlot")
        #self.optimize = OWGraphDrawingOptimize(parent=self);
        #start of content (right) area
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
        
    def selectConnectedNodes(self):
        self.graph.selectConnectedNodes(self.connectDistance)
        
    def selectAllConnectedNodes(self):
        self.graph.selectConnectedNodes(1000000)
            
    def selectHubsExplicit(self):
        if self.spinExplicit > self.visualize.nVertices():
            self.spinExplicit = self.visualize.nVertices()
            
        self.spinPercentage = 100 * self.spinExplicit / self.visualize.nVertices()
        self.graph.selectHubs(self.spinExplicit)
    
    def selectHubsPercentage(self):
        self.spinExplicit = self.spinPercentage * self.visualize.nVertices() / 100
        self.graph.selectHubs(self.spinExplicit)
    
    def sendData(self):
        graph = self.graph.getSelectedGraph()
        
        if graph != None:
            if graph.items != None:
                self.send("Selected Examples", graph.items)
            else:
                self.send("Selected Examples", self.graph.getSelectedExamples())
                
            self.send("Selected Graph", graph)
        else:
            items = self.graph.getSelectedExamples()
            if items != None:
                self.send("Selected Examples", items)
   
    def btnZommClicked(self):
        self.btnZomm.setOn(1)
        self.btnMpos.setOn(0)
        self.btnRctSel.setOn(0)
        self.btnPolySel.setOn(0)
        self.graph.state = ZOOMING
        self.graph.canvas().setCursor(Qt.crossCursor)
    
    def btnMposClicked(self):
        self.btnZomm.setOn(0)
        self.btnMpos.setOn(1)
        self.btnRctSel.setOn(0)
        self.btnPolySel.setOn(0)
        self.graph.state = MOVE_SELECTION
        self.graph.canvas().setCursor(Qt.pointingHandCursor)
        
    def btnRctSelClicked(self):
        self.btnZomm.setOn(0)
        self.btnMpos.setOn(0)
        self.btnRctSel.setOn(1)
        self.btnPolySel.setOn(0)
        self.graph.state = SELECT_RECTANGLE
        self.graph.canvas().setCursor(Qt.arrowCursor)
    
    def btnPolySelClicked(self):
        self.btnZomm.setOn(0)
        self.btnMpos.setOn(0)
        self.btnRctSel.setOn(0)
        self.btnPolySel.setOn(1)
        self.graph.state = SELECT_POLYGON
        self.graph.canvas().setCursor(Qt.arrowCursor)
    
    def clickedAttLstBox(self, item):
        attributes = []
        i = self.attListBox.firstItem()
        while i:
            if self.attListBox.isSelected(i):
                attributes.append(str(i.text()))
            i = i.next()
       
        self.graph.setLabelText(attributes)
        self.updateCanvas()
        
    def clickedTooltipLstBox(self, item):
        attributes = []
        i = self.tooltipListBox.firstItem()
        while i:
            if self.tooltipListBox.isSelected(i):
                attributes.append(str(i.text()))
            i = i.next()
       
        self.graph.setTooltipText(attributes)
        self.updateCanvas()
    
    def setGraph(self, graph):
        self.visualize = GraphVisualizer(graph, self)
        
        self.colorCombo.clear()
        self.attListBox.clear()
        self.tooltipListBox.clear()
        self.colorCombo.insertItem("(one color)")
        
        for var in self.visualize.getVars():
            self.colorCombo.insertItem(unicode(var.name))
            self.attListBox.insertItem(unicode(var.name))
            self.tooltipListBox.insertItem(unicode(var.name))

        self.graph.addVisualizer(self.visualize)
        self.displayRandom(firstTime = True)
        
    def random(self):
        self.displayRandom(firstTime=False); 
        
    def ff(self):
        if self.visualize == None:   #grafa se ni
            return

        #najprej nakljucne koordinate za vsa vozlisca
        self.visualize.fruchtermanReingold()
        
    def circular(self):
        pass

    def setVertexColor(self):
        self.graph.setVertexColor(self.colorCombo.currentText())
        self.updateCanvas()
        
    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)
            
    def displayRandom(self, firstTime=False):
        if self.visualize == None:   #grafa se ni
            return

        #najprej nakljucne koordinate za vsa vozlisca
        self.visualize.random()

        #ko graf preberemo iz datoteke, upostevamo podane koordinate
#        if firstTime == True: 
#            for i in range(0, self.visualize.nVertices()):
#                if self.verticesDescriptions[i].inFileDefinedCoors[0] != None:
#                    self.visualize.xCoors[i] = self.verticesDescriptions[i].inFileDefinedCoors[0]
#                    
#                if self.verticesDescriptions[i].inFileDefinedCoors[1] != None:
#                    self.visualize.yCoors[i] = self.verticesDescriptions[i].inFileDefinedCoors[1]        
        self.updateCanvas(); 
                    
    def updateCanvas(self):
        #ce imamo graf
        if self.visualize != None:
            self.graph.updateCanvas()#self.visualize.xCoors, self.visualize.yCoors)
        
if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWNetwork()
    appl.setMainWidget(ow)
    ow.show()
    appl.exec_loop()

#ta razred je potreben, da lahko narisemo pixmap z painterjem
class pixmaps(QWidget):
    def __init__(self):
        apply(QWidget.__init__, (self,))

        #risanje lupe
        self.LOOKG=QPixmap(20,20)
        self.LOOKG.fill()
        painter=QPainter(self.LOOKG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(Qt.red))
        pa=QPointArray([2,5,2,10,5,13,10,13,13,10,13,5,10,2,5,2,2,5])
        painter.drawPolygon(pa)
        painter.drawLine(12,11,19,18)
        painter.drawLine(11,12,18,19)
        painter.drawLine(12,12,18,18)
        painter.end()

        #risanje roke
        self.HAND=QPixmap(20,20)
        self.HAND.fill()
        painter=QPainter(self.HAND)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(Qt.red))
        pa=QPointArray([1,4,1,6,4,6,4,8,6,8,6,10,8,10,8,12,11,12,15,8,15,7,12,4,1,4])
        painter.drawPolygon(pa)
        painter.drawLine(5,6,7,6)
        painter.drawLine(7,8,9,8)
        painter.drawPoint(9,10)
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([11,12,14,15,18,11,15,8,11,12])
        painter.drawPolygon(pa)
        painter.end()

        #risanje pravokotnega izbiralnika
        self.RECTG=QPixmap(20,20)
        self.RECTG.fill()
        painter=QPainter(self.RECTG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([2,2,2,17,17,17,17,2,2,2])
        painter.drawPolygon(pa)
        painter.setPen(QPen(Qt.white,1))
        pa=QPointArray([2,5,2,7,2,9,2,11,2,13,2,15,4,17,6,17,8,17,10,17,12,17,14,17,17,14,17,12,17,10,17,8,17,6,17,4,15,2,13,2,11,2,9,2,7,2,5,2])
        painter.drawPoints(pa,0,-1)
        painter.setPen(QPen(Qt.black,1))
        pa=QPointArray([6,6,6,13,13,13,13,6,6,6])
        painter.drawPolygon(pa)
        painter.drawLine(6,6,13,13)
        painter.drawLine(6,13,13,6)
        painter.setPen(QPen(QColor(192,192,255),1))
        pa=QPointArray([6,9,6,10,9,13,10,13,13,10,13,9,10,6,9,6])
        painter.drawPoints(pa,0,-1)
        painter.end()

        #risanje poligonskega izbiralnika
        self.POLYG=QPixmap(20,20)
        self.POLYG.fill()
        painter=QPainter(self.POLYG)
        painter.setPen(QPen(Qt.black,1))
        painter.setBrush(QBrush(QColor(192,192,255)))
        pa=QPointArray([1,5,3,15,9,13,14,18,18,12,18,11,13,2,13,1,1,5])
        painter.drawPolygon(pa)
        painter.end()

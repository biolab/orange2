"""
<name>Graph Canvas</name>
<description>The Graph Canvas Widget enables users to visualize graph schemas.</description>
<icon>icons/Outlier.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2030</priority>
"""
import OWGUI

from OWWidget import *
from qwt import *
from qt import *
from OWGraphDrawerCanvas import *
from OWGraphVisualizer import * 

class OWGraphDrawer(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'GraphDrawer')

        self.inputs = [("Graph with ExampleTable", orange.Graph, self.setGraph)]
        self.outputs = []
        
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
        
        QLabel("Select attributes", self.styleBox)
        self.attListBox = QListBox(self.styleBox, "selectAttributes")
        self.attListBox.setMultiSelection(True)
        self.styleBox.connect(self.attListBox, SIGNAL("clicked( QListBoxItem * )"), self.clickedAttLstBox)
        
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
        
        self.graph = OWGraphDrawerCanvas(self, self.mainArea, "ScatterPlot")
        #self.optimize = OWGraphDrawingOptimize(parent=self);
        #start of content (right) area
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
    
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
                attributes[:0] = [i.text()]
            i = i.next()
            
        self.graph.setLabelText(attributes)
        self.updateCanvas()
    
    def setGraph(self, graph):
        self.visualize = GraphVisualizer(graph, self)
        
        self.colorCombo.clear()
        self.attListBox.clear()
        self.colorCombo.insertItem("(one color)")
        
        if isinstance(graph.items, orange.ExampleTable):
            for var in graph.items.domain.variables:
                self.colorCombo.insertItem(unicode(var.name))
                self.attListBox.insertItem(unicode(var.name))
            
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
    ow = OWGraphDrawer()
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

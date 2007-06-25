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
from time import *

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


class OWNetwork(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Network')

        self.inputs = [("Graph with ExampleTable", orange.Graph, self.setGraph), ("Example Subset", orange.ExampleTable, self.setExampleSubset)]
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
        
        self.hideBox = QVGroupBox("Hide", self.controlArea)
        OWGUI.button(self.hideBox, self, "Hide selected", callback=self.hideSelected)
        OWGUI.button(self.hideBox, self, "Hide all but selected", callback=self.hideAllButSelected)
        OWGUI.button(self.hideBox, self, "Show all", callback=self.showAllNodes)
        
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
        
        OWGUI.button(self.controlArea, self, "Save network", callback=self.saveNetwork)
        OWGUI.button(self.controlArea, self, "Send", callback=self.sendData)
        OWGUI.button(self.controlArea, self, "test replot", callback=self.testRefresh)
        
        self.graph = OWGraphDrawerCanvas(self, self.mainArea, "ScatterPlot")
        #self.optimize = OWGraphDrawingOptimize(parent=self);
        #start of content (right) area
        self.box = QVBoxLayout(self.mainArea)
        self.box.addWidget(self.graph)
    
    def testRefresh(self):
        start = time()
        self.graph.replot()
        stop = time()    
        print "replot in " + str(stop - start)
        
    def saveNetwork(self):
        filename = QFileDialog.getSaveFileName(QString.null,'PAJEK networks (*.net)')
        if filename:
            fn = ""
            head, tail = os.path.splitext(str(filename))
            if not tail:
                fn = head + ".net"
            else:
                fn = str(filename)
            
            self.graph.visualizer.saveNetwork(fn)
    
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
            self.send("Selected Graph", None)
   
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
        if graph == None:
            return
        print "OWNetwork/setGraph: new visualizer..."
        self.visualize = NetworkVisualizer(graph, self)
        print "done."
        self.colorCombo.clear()
        self.attListBox.clear()
        self.tooltipListBox.clear()
        self.colorCombo.insertItem("(one color)")
        
        for var in self.visualize.getVars():
            self.colorCombo.insertItem(unicode(var.name))
            self.attListBox.insertItem(unicode(var.name))
            self.tooltipListBox.insertItem(unicode(var.name))

        print "OWNetwork/setGraph: add visualizer..."
        self.graph.addVisualizer(self.visualize)
        print "done."
        print "OWNetwork/setGraph: display random..."
        self.random()
        print "done."
    
    def setExampleSubset(self, subset):
        if self.graph == None:
            return
        
        hiddenNodes = []
        
        if subset != None:
            try:
                expected = 1
                for row in subset:
                    index = int(row['index'].value)
                    if expected != index:
                        hiddenNodes += range(expected-1, index-1)
                        expected = index + 1
                    else:
                        expected += 1
                        
                hiddenNodes += range(expected-1, self.graph.nVertices)
                
                self.graph.setHiddenNodes(hiddenNodes)
            except:
                print "Error. Index column does not exists."
        
        #print "hiddenNodes:"
        #print hiddenNodes
        
    def hideSelected(self):
        #print self.graph.selection
        toHide = self.graph.selection + self.graph.hiddenNodes
        self.graph.setHiddenNodes(toHide)
        self.graph.removeSelection()
        
    def hideAllButSelected(self):
        allNodes = set(range(self.graph.nVertices))
        allButSelected = list(allNodes - set(self.graph.selection))
        toHide = allButSelected + self.graph.hiddenNodes
        self.graph.setHiddenNodes(toHide)
    
    def showAllNodes(self):
        self.graph.setHiddenNodes([])
        
    def random(self):
        print "OWNetwork/random.."
        if self.visualize == None:   #grafa se ni
            return    
            
        self.visualize.random()
        
        print "OWNetwork/random: updating canvas..."
        self.updateCanvas();
        print "done."
        
        
    def ff(self):
        print "OWNetwork/ff..."
        if self.visualize == None:   #grafa se ni
            return
        
        k = 1.13850193174e-008
        #k = 1.61735442033e-008
        nodes = self.visualize.nVertices()
        t = k * nodes * nodes
        refreshRate = int(5.0 / t)
        if refreshRate <   1: refreshRate = 1;
        if refreshRate > 1500: refreshRate = 1500;
        print "refreshRate: " + str(refreshRate)
        #najprej nakljucne koordinate za vsa vozlisca
        #- self.visualize.nVertices() / 50 + 100
        #if refreshRate < 5:
        #    refreshRate = 5;
        
        tolerance = 5
        initTemp = 1000
        #refreshRate = 1
        initTemp = self.visualize.fruchtermanReingold(refreshRate, initTemp, self.graph.hiddenNodes)
        self.updateCanvas()
        
#        self.visualize.fruchtermanReingold(refreshRate, initTemp)
        
#        while True:
#            print initTemp
#            initTemp = self.visualize.fruchtermanReingold(refreshRate, initTemp)
#            
#            if (initTemp <= tolerance):
#                #self.visualize.postProcess()
#                print "OWNetwork/ff: updating canvas..."
#                self.updateCanvas()
#                return
#            print "OWNetwork/ff: updating canvas..."
#            self.updateCanvas()
        print "done."
        
    def circular(self):
        pass

    def setVertexColor(self):
        self.graph.setVertexColor(self.colorCombo.currentText())
        self.updateCanvas()
        
    def setGraphGrid(self):
        self.graph.enableGridY(self.graphShowGrid)
        self.graph.enableGridX(self.graphShowGrid)
                    
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

"""
<name>Interaction Graph</name>
<description>Show interaction graph</description>
<category>Classification</category>
<icon>icons/InteractionGraph.png</icon>
<priority>4000</priority>
"""
# InteractionGraph.py
#
# 

from OWWidget import *
from OWInteractionGraphOptions import *
from OWScatterPlotGraph import OWScatterPlotGraph
from OData import *
from qt import *
from qtcanvas import *
import orngInteract
import statc
import os
from re import *


class IntGraphView(QCanvasView):
    def __init__(self, parent, *args):
        apply(QCanvasView.__init__,(self,) + args)
        self.parent = parent

    # mouse button was pressed
    def contentsMousePressEvent(self, ev):
        self.parent.mousePressed(ev)


###########################################################################################
##### WIDGET : Interaction graph
###########################################################################################
class OWInteractionGraph(OWWidget):
    settingsList = []
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Interaction graph", 'show interaction graph', TRUE, TRUE)

        #set default settings
        self.data = None
        self.interactionMatrix = None
        self.rectIndices = {}   # QRect rectangles
        self.rectNames   = {}   # info about rectangle names (attributes)
        self.lines = []     # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])

        self.addInput("cdata")
        self.addOutput("cdata")
        self.addOutput("view")      # when user clicks on a link label we can send information about this two attributes to a scatterplot
        self.addOutput("selection") # when user clicks on "show selection" button we can send information about selected attributes

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        #self.options = OWInteractionGraphOptions()

        self.canvas = QCanvas(2000,2000)
        self.canvasView = IntGraphView(self, self.canvas, self.mainArea)
        
        self.canvasView.show()

        #GUI
        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.selectionButton = QPushButton("Show selection", self.space)

        #connect controls to appropriate functions
        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttributeClick)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttributeClick)
        self.connect(self.selectionButton, SIGNAL("clicked()"), self.selectionClick)


        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        #self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        
        #self.loadGraphButton = QPushButton("Load interaction graph", self.controlArea)
        #self.convertGraphButton = QPushButton("convert graph", self.controlArea)
        #self.applyButton = QPushButton("Apply changes", self.controlArea)
        
        #self.connect(self.loadGraphButton, SIGNAL("clicked()"), self.loadGraphMethod)
        #self.connect(self.convertGraphButton, SIGNAL("clicked()"), self.convertGraphMethod)
        #self.connect(self.applyButton, SIGNAL("clicked()"), self.applyMethod)

    def selectionClick(self):
        if self.data == None: return
        list = []
        for i in range(self.shownAttribsLB.count()):
            list.append(str(self.shownAttribsLB.text(i)))
        self.send("selection", list)

    # we catch mouse release event so that we can send the "view" signal
    def onMouseReleased(self, e):
        for i in range(len(self.graphs)):
            if self.graphs[i].blankClick == 1:
                (attr1, attr2, className, string) = self.graphParameters[i]
                self.send("view", (attr1, attr2))
                self.graphs[i].blankClick = 0

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.interactionMatrix = orngInteract.InteractionMatrix(self.data)

        f = open('interaction.dot','w')
        self.interactionMatrix.exportGraph(f, significant_digits=3,positive_int=8,negative_int=8,absolute_int=0,url=1)
        f.flush()
        f.close()

        # execute dot and save otuput to pipes
        (pipePngOut, pipePngIn) = os.popen2("dot interaction.dot -Tpng", "b")
        (pipePlainOut, pipePlainIn) = os.popen2("dot interaction.dot -Tismap", "t")
        textPng = pipePngIn.read()
        textPlainList = pipePlainIn.readlines()
        pipePngIn.close()
        pipePlainIn.close()
        pipePngOut.close()
        pipePlainOut.close()
        
        pixmap = QPixmap()
        pixmap.loadFromData(textPng)
        canvasPixmap = QCanvasPixmap(pixmap, QPoint(0,0))
        width = canvasPixmap.width()
        height = canvasPixmap.height()

        # hide all rects
        for rectInd in self.rectIndices.keys():
            self.rectIndices[rectInd].hide()

        self.send("cdata", data)
        
        self.canvas.setTiles(pixmap, 1, 1, width, height)
        self.canvas.resize(width, height)
        
        self.rectIndices = {}     # QRect rectangles
        self.rectNames   = {} # info about rectangle names (attributes)
        self.lines = []     # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])

        
        self.parseGraphData(textPlainList, width, height)
        self.initLists(self.data)
        self.canvas.update()

    # parse info from plain file. picWidth and picHeight are sizes in pixels
    def parseGraphData(self, textPlainList, picWidth, picHeight):
        scale = 0
        w = 1; h = 1
        for line in textPlainList:
            if line[:9] == "rectangle":
                list = line.split()
                topLeftRectStr = list[1]
                bottomRightRectStr = list[2]
                attrIndex = list[3]
                
                isAttribute = 0     # does rectangle represent attribute
                if attrIndex.find("-") < 0:
                    isAttribute = 1
                
                topLeftRectStr = topLeftRectStr.replace("(","")
                bottomRightRectStr = bottomRightRectStr.replace("(","")
                topLeftRectStr = topLeftRectStr.replace(")","")
                bottomRightRectStr = bottomRightRectStr.replace(")","")
                
                topLeftRectList = topLeftRectStr.split(",")
                bottomRightRectList = bottomRightRectStr.split(",")
                xLeft = int(topLeftRectList[0])
                yTop = int(topLeftRectList[1])
                width = int(bottomRightRectList[0]) - xLeft
                height = int(bottomRightRectList[1]) - yTop

                rect = QCanvasRectangle(xLeft, yTop, width, height, self.canvas)
                pen = QPen(Qt.green)
                pen.setWidth(4)
                rect.setPen(pen)
                rect.hide()
                
                if isAttribute == 1:
                    name = self.data.domain[int(attrIndex)].name
                    self.rectIndices[int(attrIndex)] = rect
                    self.rectNames[name] = rect
                else:
                    attrs = attrIndex.split("-")
                    attr1 = self.data.domain[int(attrs[0])].name
                    attr2 = self.data.domain[int(attrs[1])].name
                    pen.setStyle(Qt.NoPen)
                    rect.setPen(pen)
                    self.lines.append((attr1, attr2, rect))
    
    def resizeEvent(self, e):
        if self.canvasView != None:
            self.canvasView.resize(self.mainArea.size())

    def initLists(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        for key in self.rectNames.keys():
            self.setAttrVisible(key, 1)


    #################################################
    ### showing and hiding attributes
    #################################################
    def _showAttribute(self, name):
        self.shownAttribsLB.insertItem(name)    # add to shown

        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from hidden
            if str(self.hiddenAttribsLB.text(i)) == name:
                self.hiddenAttribsLB.removeItem(i)

    def _hideAttribute(self, name):
        self.hiddenAttribsLB.insertItem(name)    # add to hidden

        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from shown
            if str(self.shownAttribsLB.text(i)) == name:
                self.shownAttribsLB.removeItem(i)

    def setAttrVisible(self, name, visible = 1):
        if visible == 1:
            self.rectNames[name].show();
            self._showAttribute(name)
        else:
            self.rectNames[name].hide();
            self._hideAttribute(name)
        self.canvas.update()

    def getAttrVisible(self, name):
        return self.rectNames[name].visible()

    #################################################
    # controls processing
    #################################################
    def addAttributeClick(self):
        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                name = str(self.hiddenAttribsLB.text(i))
                self.setAttrVisible(name, 1)

    def removeAttributeClick(self):
        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                name = str(self.shownAttribsLB.text(i))
                self.setAttrVisible(name, 0)
    

    def clickInside(self, rect, point):
        x = point.x()
        y = point.y()
        
        if rect.left() > x: return 0
        if rect.right() < x: return 0
        if rect.top() > y: return 0
        if rect.bottom() < y: return 0

        return 1
        
    
    def mousePressed(self, ev):
        if ev.button() == QMouseEvent.LeftButton:
            for name in self.rectNames:
                clicked = self.clickInside(self.rectNames[name].rect(), ev.pos())
                if clicked == 1:
                    self.setAttrVisible(name, not self.getAttrVisible(name))
                    return
            for (attr1, attr2, rect) in self.lines:
                clicked = self.clickInside(rect.rect(), ev.pos())
                if clicked == 1:
                    self.send("view", (attr1, attr2))
                    return

                


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWInteractionGraph()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

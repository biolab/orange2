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
        self.rects = {}     # QRect rectangles
        self.rectNames = {} # info about rectangle names (attributes)
        self.rectIds   = {} 
        self.lines = {}     # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])
        self.lineNames = {} # info about which rectangles are connected in form (rectName1, rectName2)
        self.lineCaptionPos = {} # QPoint with info on position of line captions

        self.addInput("cdata")
        self.addOutput("cdata")
        self.addOutput("view")      # when user right clicks on one graph we can send information about this graph to a scatterplot

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        #self.options = OWInteractionGraphOptions()

        self.canvas = QCanvas(2000,2000)
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        
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

        #connect controls to appropriate functions
        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)


        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        #self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        
        #self.loadGraphButton = QPushButton("Load interaction graph", self.controlArea)
        #self.convertGraphButton = QPushButton("convert graph", self.controlArea)
        #self.applyButton = QPushButton("Apply changes", self.controlArea)
        
        #self.connect(self.loadGraphButton, SIGNAL("clicked()"), self.loadGraphMethod)
        #self.connect(self.convertGraphButton, SIGNAL("clicked()"), self.convertGraphMethod)
        #self.connect(self.applyButton, SIGNAL("clicked()"), self.applyMethod)

    def applyMethod(self):
        pass

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
        self.interactionMatrix.exportGraph(f, significant_digits=3)
        f.flush()
        f.close()

        # execute dot and save otuput to pipes
        (pipePngOut, pipePngIn) = os.popen2("dot interaction.dot -Tpng", "b")
        (pipePlainOut, pipePlainIn) = os.popen2("dot interaction.dot -Tplain", "t")
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

        self.hideAllRects()        

        self.send("cdata", data)
        
        self.canvas.setTiles(pixmap, 1, 1, width, height)
        self.canvas.resize(width, height)
        
        self.rects = {}     # QRect rectangles
        self.rectNames = {} # info about rectangle names (attributes)
        self.rectIds   = {} 
        self.lines = {}     # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])
        self.lineNames = {} # info about which rectangles are connected in form (rectName1, rectName2)
        self.lineCaptionPos = {} # QPoint with info on position of line captions
        
        self.parseGraphData(textPlainList, width, height)
        self.initLists(self.data)
        self.canvas.update()

    # parse info from plain file. picWidth and picHeight are sizes in pixels
    def parseGraphData(self, textPlainList, picWidth, picHeight):
        scale = 0
        w = 1; h = 1
        for line in textPlainList:
            # GRAPH
            if line[:5] == "graph":
                list = line.split()
                if len(list) != 4:
                    print "error in input. read line = ", line
                    continue
                scale = float(list[1])
                w = float(list[2])
                h = float(list[3])
            # NODE
            elif line[:4] == "node":
                list = line.split()
                id = int(list[1])
                x = float(list[2])
                y = float(list[3])
                xsize = float(list[4])
                ysize = float(list[5])
                m = match('"{([^\s]+)\|.+"', list[6])    # regular expression that will extract only node name
                name = m.group(1)
                name = name.replace("\\n", " ")
                xLeft = round((x-(xsize/2.0)) * float(picWidth) / w)
                yTop  = picHeight - round((y+(ysize/2.0)) * float(picHeight)/ h)
                width = round(xsize * float(picWidth) / w)
                height = round(ysize * float(picHeight) / h)
                rect = QCanvasRectangle(xLeft, yTop, width, height, self.canvas)
                rect.setPen(QPen(Qt.green))
                rect.hide()
                self.rects[id] = rect
                self.rectNames[name] = id
                self.rectIds[id] = name
                #print "name = %s. xLeft = %d, yTop = %d, width = %d, height = %d (%d)" % (name, xLeft, yTop, width, height, x*picHeight/h)

            # EDGE
            elif line[:4] == "edge":
                list = line.split()
                rect1 = int(list[1])
                rect2 = int(list[2])
                rectName1 = self.rectIds[rect1]
                rectName2 = self.rectIds[rect2]
                count = int(list[3])
                points = []
                for i in range(count):
                    x = round(float(list[4 + 2*i]) * float(picWidth)/w)
                    y = picHeight - round(float(list[4 + 2*i +1]) * float(picHeight)/h)
                    points.append(QPoint(x,y))
                labelX = round(float(list[5 + 2*count]) * float(picWidth)/w)
                labelY = round(float(list[5 + 2*count+1]) * float(picHeight)/h)
                self.lines[(rectName1, rectName2)] = (QPoint(labelX, labelY), points)

            # STOP
            elif line[:4] == "stop":
                return
            else:
                print "error in input. read line = ", line
            
       
    def resizeEvent(self, e):
        if self.canvasView != None:
            self.canvasView.resize(self.mainArea.size())

    def hideAllRects(self):
        for rectName in self.rects.keys():
            self.rects[rectName].hide()
            self.rects[rectName] = None

    def setRectVisible(self, name, visible = 1):
        if visible == 1: self.rects[self.rectNames[name]].show();
        else:            self.rects[self.rectNames[name]].hide();
        self.canvas.update()

    def initLists(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        for key in self.rectNames.keys():
            self.shownAttribsLB.insertItem(key)
        for key in self.rects.keys():
            self.rects[key].show()
        

    #################################################
    # controls processing
    #################################################
    def addAttribute(self):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                text = self.hiddenAttribsLB.text(i)
                self.hiddenAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, pos)
                self.setRectVisible(str(text), 1)


    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)
                self.setRectVisible(str(text), 0)



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWInteractionGraph()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

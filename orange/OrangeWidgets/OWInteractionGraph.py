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

        self.addInput("cdata")
        self.addOutput("cdata")
        self.addOutput("view")      # when user right clicks on one graph we can send information about this graph to a scatterplot

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        #self.options = OWInteractionGraphOptions()

        self.canvas = QCanvas(2000,2000)
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        self.array = None
        
        self.canvasView.show()
        self.canvasView.setResizePolicy(QCanvasView.AutoOneFit)

        #GUI
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

        # execute dot and wait until it completes
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
        
        if self.array == None:
            self.array = QCanvasPixmapArray([canvasPixmap], [QPoint(0,0)])
        else:
            self.array.setImage(0, canvasPixmap)
                                
        self.canvasSprite = QCanvasSprite(self.array, self.canvas)
        self.canvasSprite.show()
        self.canvas.update()

        self.send("cdata", data)

        self.parseGraphData()

    def parseGraphData(self):
        #f = open(
        pass
        
    def resizeEvent(self, e):
        if self.canvasView != None:
            self.canvasView.resize(self.mainArea.size())

    #################################################

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWInteractionGraph()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

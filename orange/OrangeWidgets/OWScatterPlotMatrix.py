"""
<name>Scatterplot matrix</name>
<description>Show all possible projections of the data</description>
<category>Visualization</category>
<icon>icons/ScatterPlotMatrix.png</icon>
<priority>120</priority>
"""
# ScatterPlotMatrix.py
#
# Show data using parallel coordinates visualization method
# 

from OWWidget import *
from OWScatterPlotMatrixOptions import *
from OWScatterPlotGraph import OWScatterPlotGraph
from OData import *
import qt
import orngInteract
import statc
from math import sqrt
from OWGraph import *

class QMyLabel(QLabel):
    def __init__(self, size, *args):
        apply(QLabel.__init__,(self,) + args)
        self.size = size

    def setSize(self, size):
        self.size = size
 
    def sizeHint(self):
        return self.size

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWScatterPlotMatrix(OWWidget):
    settingsList = ["pointWidth", "jitteringType", "showXAxisTitle", "showYAxisTitle", "showTitle", "showAttributeValues",
                    "showLegend", "graphGridColor", "graphCanvasColor", "jitterSize", "jitterContinuous", "showFilledSymbols"]
    spreadType=["none","uniform","triangle","beta"]
    jitterSizeList = ['0.1','0.5','1','2','5','10', '15', '20']
    jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Scatterplot matrix", 'Show all possible projections of the data', TRUE, TRUE)

        #set default settings
        self.data = None
        self.pointWidth = 5
        self.jitteringType = "uniform"
        self.showTitle = 0
        self.showAttributeValues = 0
        self.showXAxisTitle = 0
        self.showYAxisTitle = 0
        self.showVerticalGridlines = 0
        self.showHorizontalGridlines = 0
        self.showLegend = 0
        self.jitterContinuous = 0
        self.jitterSize = 2
        self.showFilledSymbols = 1
        self.shownAttrCount = 0
        self.graphGridColor = str(Qt.black.name())
        self.graphCanvasColor = str(Qt.white.name())

        self.addInput("cdata")
        self.addInput("selection")
        #self.addOutput("cdata")
        self.addOutput("view")      # when user right clicks on one graph we can send information about this graph to a scatterplot

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.options = OWScatterPlotMatrixOptions()
        self.activateLoadedSettings()

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

        self.createMatrixButton = QPushButton("Create matrix", self.space)

        #connect controls to appropriate functions
        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)
        self.connect(self.createMatrixButton, SIGNAL("clicked()"), self.createGraphs)
        self.connect(self.options.apply, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.connect(self.options, PYSIGNAL("gridColorChange(QColor &)"), self.setGridColor)
        self.connect(self.options, PYSIGNAL("canvasColorChange(QColor &)"), self.setCanvasColor)
        self.connect(self.options.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.options.jitterContinuous, SIGNAL("clicked()"), self.setJitterContinuous)
        self.connect(self.options.gShowDistributions, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.options.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)


        self.grid = QGridLayout(self.mainArea)
        self.graphs = []
        self.labels = []
        self.graphParameters = []

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.options.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.options.gShowTitle.setChecked(self.showTitle)
        self.options.gShowAttributeValues.setChecked(self.showAttributeValues)
        self.options.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.options.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.options.gSetGridColor.setNamedColor(str(self.graphGridColor))
        self.options.gSetCanvasColor.setNamedColor(str(self.graphCanvasColor))
        self.options.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)

        self.options.jitterContinuous.setChecked(self.jitterContinuous)
        for i in range(len(self.jitterSizeList)):
            self.options.jitterSize.insertItem(self.jitterSizeList[i])
        self.options.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.options.widthSlider.setValue(self.pointWidth)
        self.options.widthLCD.display(self.pointWidth)

    def setGraphOptions(self, graph, title):
        graph.updateSettings(showDistributions = self.options.gShowDistributions.isChecked(), showAttributeValues = self.showAttributeValues)
        graph.setJitteringOption(self.jitteringType)
        graph.setShowXaxisTitle(self.showXAxisTitle)
        graph.setShowYLaxisTitle(self.showYAxisTitle)
        graph.setGridColor(self.options.gSetGridColor)
        graph.setCanvasColor(self.options.gSetCanvasColor)
        graph.setPointWidth(self.pointWidth)
        graph.setShowFilledSymbols(self.showFilledSymbols)
        graph.setShowMainTitle(self.showTitle)
        graph.setMainTitle(title)
        graph.setPointWidth(self.pointWidth)

    def setPointWidth(self, n):
        self.pointWidth = n
        for graph in self.graphs:
            graph.setPointWidth(n)
        self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.updateJitteringSettings()

    def setJitterContinuous(self):
        self.jitterContinuous = self.options.jitterContinuous.isChecked()
        self.updateJitteringSettings()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.setJitterContinuous()

    def updateJitteringSettings(self):
        if self.graphs == []: return
        self.graphs[0].setJitteringOption(self.jitteringType)
        self.graphs[0].setJitterContinuous(self.jitterContinuous)
        self.graphs[0].setJitterSize(self.jitterSize)
        for graph in self.graphs[1:]:
            graph.setJitterSize(self.jitterSize)
            graph.setJitterContinuous(self.jitterContinuous)
            graph.setJitteringOption(self.jitteringType)
            graph.scaledData = self.graphs[0].scaledData
            graph.coloringScaledData = self.graphs[0].coloringScaledData
        self.updateGraph()
    
    def setCanvasColor(self, c):
        self.graphCanvasColor = c
        for graph in self.graphs:
            graph.setCanvasColor(c)

    def setGridColor(self, c):
        self.graphGridColor = c
        for graph in self.graphs:
            graph.setGridColor(c)

    # #########################
    # GRAPH MANIPULATION
    # #########################
    def updateSettings(self):
        self.showTitle = self.options.gShowTitle.isChecked()
        self.showAttributeValues = self.options.gShowAttributeValues.isChecked()
        self.showXAxisTitle = self.options.gSetXaxisCB.isChecked()
        self.showYAxisTitle = self.options.gSetYaxisCB.isChecked()
        self.showFilledSymbols = self.options.gShowFilledSymbolsCB.isChecked()
        
        self.pointWidth = self.options.widthSlider.value()

        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.setGraphOptions(self.graphs[i], title)
            
        self.updateGraph()
    
    def updateGraph(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.graphs[i].updateData(attr1, attr2, className)
            
    def removeAllGraphs(self):
        for graph in self.graphs:
            graph.hide()
        self.graphs = []
        self.graphParameters = []

        for label in self.labels:
            label.hide()
        self.labels = []
    

    def createGraphs(self):
        self.removeAllGraphs()

        list = []
        for i in range(self.shownAttribsLB.count()): list.append(str(self.shownAttribsLB.text(i)))
        list.reverse()
        count = len(list)
        
        self.shownAttrCount = count-1
        for i in range(count-1, -1, -1):
            for j in range(i):
                graph = OWScatterPlotGraph(self.mainArea)
                graph.setMinimumSize(QSize(10,10))
                if self.graphs == []:
                    graph.setData(self.data)
                else:
                    graph.rawdata = self.graphs[0].rawdata
                    graph.domainDataStat = self.graphs[0].domainDataStat
                    graph.scaledData = self.graphs[0].scaledData
                    graph.noJitteringScaledData = self.graphs[0].noJitteringScaledData
                    graph.coloringScaledData = self.graphs[0].coloringScaledData
                    graph.attrValues = self.graphs[0].attrValues
                    graph.attributeNames = self.graphs[0].attributeNames

                self.setGraphOptions(graph, "")
                self.grid.addWidget(graph, count-i-1, j)
                self.graphs.append(graph)
                self.connect(graph, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)
                params = (list[j], list[i], self.data.domain.classVar.name, "")
                self.graphParameters.append(params)
                graph.show()

        self.updateGraph()

        w = self.mainArea.width()
        h = self.mainArea.height()
        for i in range(len(list)):
            label = QMyLabel(QSize(w/(count+1), h/(count+1)), list[i], self.mainArea)
            self.grid.addWidget(label, len(list)-i-1, i, Qt.AlignCenter)
            label.setAlignment(Qt.AlignCenter)
            label.show()
            self.labels.append(label)


    def saveToFile(self):
        sizeDlg = OWChooseImageSizeDlg(self, "", TRUE)
        sizeDlg.exec_loop()
        if sizeDlg.result() != QDialog.Accepted: return

        if sizeDlg.sizeOriginal.isChecked(): size = self.size()
        elif sizeDlg.size400.isChecked(): size = QSize(400,400)
        elif sizeDlg.size600.isChecked(): size = QSize(600,600)
        elif sizeDlg.size800.isChecked(): size = QSize(800,800)
        elif sizeDlg.custom.isChecked():  size = QSize(int(str(sizeDlg.xSize.text())), int(str(sizeDlg.ySize.text())))
        else:
            print "error"; return

        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        if not sizeDlg.allSizes.isChecked():
            self.saveToFileDirect(fileName, ext, size)
        else:
            dirName, shortFileName = os.path.split(fileName)
            if not os.path.isdir(dirName + "\\400\\"): os.mkdir(dirName + "\\400\\")
            if not os.path.isdir(dirName + "\\600\\"): os.mkdir(dirName + "\\600\\")
            if not os.path.isdir(dirName + "\\800\\"): os.mkdir(dirName + "\\800\\")
            if not os.path.isdir(dirName + "\\Original\\"): os.mkdir(dirName + "\\Original\\")
            self.saveToFileDirect(dirName + "\\400\\" + shortFileName, ext, QSize(400,400))
            self.saveToFileDirect(dirName + "\\600\\" + shortFileName, ext, QSize(600,600))
            self.saveToFileDirect(dirName + "\\800\\" + shortFileName, ext, QSize(800,800))
            self.saveToFileDirect(dirName + "\\Original\\" + shortFileName, ext, QSize())


    # saving scatterplot matrix is a bit harder than the rest of visualization widgets. we have to save each scatterplot separately
    def saveToFileDirect(self, fileName, ext, size = QSize()):
        if self.graphs == []: return
        count = self.shownAttrCount
        attrNameSpace = 30
        dist = 4
        topOffset = 5
        if size.isEmpty():
            size = self.graphs[0].size()
            size = QSize(size.width()*count + attrNameSpace + count*dist + topOffset, size.height()*count + attrNameSpace + count*dist)

        fullBuffer = QPixmap(size)
        fullPainter = QPainter(fullBuffer)
        fullPainter.fillRect(fullBuffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background

        smallSize = QSize((size.width()-attrNameSpace - count*dist - topOffset)/count, (size.height()-attrNameSpace - count*dist)/count)

        # draw scatterplots
        for i in range(len(self.graphs)):
            buffer = QPixmap(smallSize)
            painter = QPainter(buffer)
            painter.fillRect(buffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background
            self.graphs[i].printPlot(painter, buffer.rect())
            painter.end()

            # here we have i-th scatterplot printed in a QPixmap. we have to print this pixmap into the right position in the big pixmap
            (found, y, x) = self.grid.findWidget(self.graphs[i])
            fullPainter.drawPixmap(attrNameSpace + x*(smallSize.width() + dist), y*(smallSize.height() + dist) + topOffset, buffer)
        

        list = []
        for i in range(self.shownAttribsLB.count()): list.append(str(self.shownAttribsLB.text(i)))

        # draw vertical text
        fullPainter.rotate(-90)
        for i in range(len(self.labels)-1):
            y1 = topOffset + i*(smallSize.height() + dist)
            newRect = fullPainter.xFormDev(QRect(0,y1,30, smallSize.height()));
            fullPainter.drawText(newRect, Qt.AlignCenter, str(self.labels[len(self.labels)-i-1].text()))
            
        # draw hortizonatal text
        fullPainter.rotate(90)
        for i in range(len(self.labels)-1):
            x1 = attrNameSpace + i*(smallSize.width() + dist)
            y1 = (len(self.labels)-1-i)*(smallSize.height() + dist)
            rect = QRect(x1, y1, smallSize.width(), 30)
            fullPainter.drawText(rect, Qt.AlignCenter, str(self.labels[i].text()))
            

        fullPainter.end()
        fullBuffer.save(fileName, ext)


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

        if data == None: return

        #self.send("cdata", data)

        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        for attr in self.data.domain.attributes:
            self.shownAttribsLB.insertItem(attr.name)

        self.createGraphs()

    #################################################

    def selection(self, list):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if self.data == None: return

        for attr in list:
            self.shownAttribsLB.insertItem(attr)

        for attr in self.data.domain.attributes:
            if attr.name not in list:
                self.hiddenAttribsLB.insertItem(attr.name)

        self.createGraphs()



    ################################################
    # adding and removing interesting attributes
    def addAttribute(self):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                text = self.hiddenAttribsLB.text(i)
                self.hiddenAttribsLB.removeItem(i)
                self.shownAttribsLB.insertItem(text, pos)


    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                text = self.shownAttribsLB.text(i)
                self.shownAttribsLB.removeItem(i)
                self.hiddenAttribsLB.insertItem(text, pos)


    def resizeEvent(self, e):
        w = self.mainArea.width()
        h = self.mainArea.height()
        size = QSize(w/(len(self.labels)+1), h/(len(self.labels)+1))
        for label in self.labels:
            label.setSize(size)



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlotMatrix()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

"""
<name>Scatterplot matrix</name>
<description>Show all possible projections of the data</description>
<category>Visualization</category>
<icon>icons/ScatterPlotMatrix.png</icon>
<priority>120</priority>
"""
# ScatterPlotMatrix.py
#
# Show data using scatterplot matrix visualization method
# 

from OWWidget import *
from OWScatterPlotGraph import OWScatterPlotGraph
import qt
import orngInteract
import statc
import OWDlgs
from math import sqrt

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
        OWWidget.__init__(self, parent, "Scatterplot matrix", 'Show all possible projections of the data', FALSE, TRUE, icon = "ScatterPlotMatrix.png")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1), ("Selection", list, self.selection, 1)]
        self.outputs = [("Attribute selection", list)] 


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

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = QTabWidget(self.space, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = OWScatterPlotMatrixOptions(self, "Settings")
        self.tabs.insertTab(self.GeneralTab, "General")
        self.tabs.insertTab(self.SettingsTab, "Settings")
        
        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.GeneralTab)
        self.addRemoveGroup = QHButtonGroup(self.GeneralTab)
        self.hiddenAttribsGroup = QVGroupBox(self.GeneralTab)
        self.shownAttribsGroup.setTitle("Shown attributes")
        self.hiddenAttribsGroup.setTitle("Hidden attributes")

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)
        
        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        self.createMatrixButton = QPushButton("Create matrix", self.GeneralTab)

        #connect controls to appropriate functions
        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttribute)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttribute)
        self.connect(self.createMatrixButton, SIGNAL("clicked()"), self.createGraphs)
        self.connect(self.SettingsTab.apply, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        self.connect(self.SettingsTab.jitterSize, SIGNAL("activated(int)"), self.setJitteringSize)
        self.connect(self.SettingsTab.jitterContinuous, SIGNAL("clicked()"), self.setJitterContinuous)
        self.connect(self.SettingsTab.gShowDistributions, SIGNAL("clicked()"), self.updateSettings)
        self.connect(self.SettingsTab.jitteringButtons, SIGNAL("clicked(int)"), self.setSpreadType)


        self.grid = QGridLayout(self.mainArea)
        self.graphs = []
        self.labels = []
        self.graphParameters = []

        # add a settings dialog and initialize its values
        self.activateLoadedSettings()
        self.resize(900, 700)
        

    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        self.SettingsTab.jitteringButtons.setButton(self.spreadType.index(self.jitteringType))
        self.SettingsTab.gShowTitle.setChecked(self.showTitle)
        self.SettingsTab.gShowAttributeValues.setChecked(self.showAttributeValues)
        self.SettingsTab.gSetXaxisCB.setChecked(self.showXAxisTitle)
        self.SettingsTab.gSetYaxisCB.setChecked(self.showYAxisTitle)
        self.SettingsTab.gShowFilledSymbolsCB.setChecked(self.showFilledSymbols)

        self.SettingsTab.jitterContinuous.setChecked(self.jitterContinuous)
        for i in range(len(self.jitterSizeList)):
            self.SettingsTab.jitterSize.insertItem(self.jitterSizeList[i])
        self.SettingsTab.jitterSize.setCurrentItem(self.jitterSizeNums.index(self.jitterSize))

        self.SettingsTab.widthSlider.setValue(self.pointWidth)
        self.SettingsTab.widthLCD.display(self.pointWidth)

    def setGraphOptions(self, graph, title):
        graph.updateSettings(showDistributions = self.SettingsTab.gShowDistributions.isChecked(), showAttributeValues = self.showAttributeValues)
        graph.setJitteringOption(self.jitteringType)
        graph.setShowXaxisTitle(self.showXAxisTitle)
        graph.setShowYLaxisTitle(self.showYAxisTitle)
        graph.updateSettings(showFilledSymbols = self.showFilledSymbols)
        graph.setShowMainTitle(self.showTitle)
        graph.setMainTitle(title)
        graph.pointWidth = self.pointWidth
        graph.setCanvasBackground(QColor(self.graphCanvasColor))
        graph.setGridPen(QPen(QColor(self.graphGridColor)))

    def setPointWidth(self, n):
        self.pointWidth = n
        for graph in self.graphs:
            graph.pointWidth = n
        self.updateGraph()
        
    # jittering options
    def setSpreadType(self, n):
        self.jitteringType = self.spreadType[n]
        self.updateJitteringSettings()

    def setJitterContinuous(self):
        self.jitterContinuous = self.SettingsTab.jitterContinuous.isChecked()
        self.updateJitteringSettings()

    # jittering options
    def setJitteringSize(self, n):
        self.jitterSize = self.jitterSizeNums[n]
        self.setJitterContinuous()

    def updateJitteringSettings(self):
        if self.graphs == []: return
        self.graphs[0].setJitteringOption(self.jitteringType)
        self.graphs[0].setJitterContinuous(self.jitterContinuous)
        self.graphs[0].jitterSize = self.jitterSize
        for graph in self.graphs[1:]:
            graph.jitterSize = self.jitterSize
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
        self.showTitle = self.SettingsTab.gShowTitle.isChecked()
        self.showAttributeValues = self.SettingsTab.gShowAttributeValues.isChecked()
        self.showXAxisTitle = self.SettingsTab.gSetXaxisCB.isChecked()
        self.showYAxisTitle = self.SettingsTab.gSetYaxisCB.isChecked()
        self.showFilledSymbols = self.SettingsTab.gShowFilledSymbolsCB.isChecked()
        
        self.pointWidth = self.SettingsTab.widthSlider.value()

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
                graph = OWScatterPlotGraph(self, self.mainArea)
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
        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self)
        self.sizeDlg.disconnect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.sizeDlg.accept)
        self.sizeDlg.connect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_loop()

    def saveToFileAccept(self):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        self.saveToFileDirect(fileName, ext, self.sizeDlg.getSize())
        QDialog.accept(self.sizeDlg)

    # saving scatterplot matrix is a bit harder than the rest of visualization widgets. we have to save each scatterplot separately
    def saveToFileDirect(self, fileName, ext, size):
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


    # we catch mouse release event so that we can send the "Attribute selection" signal
    def onMouseReleased(self, e):
        for i in range(len(self.graphs)):
            if self.graphs[i].blankClick == 1:
                (attr1, attr2, className, string) = self.graphParameters[i]
                self.send("Attribute selection", [attr1, attr2])
                self.graphs[i].blankClick = 0

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        exData = self.data
        
        if data == None:
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            self.removeAllGraphs()
            return
        
        self.data = orange.Preprocessor_dropMissingClasses(data)

        if self.data and exData and str(exData.domain.attributes) == str(self.data.domain.attributes): # preserve attribute choice if the domain is the same
            if self.graphs != []: self.createGraphs()   # if we had already created graphs, redraw them with new data
            return  

        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()
        
        for attr in self.data.domain.attributes:
            self.shownAttribsLB.insertItem(attr.name)

        #self.createGraphs()

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

class OWScatterPlotMatrixOptions(QVGroupBox):
    def __init__(self,parent=None,name=None):
        QVGroupBox.__init__(self, parent, name)
        self.parent = parent

        # point width
        widthBox = QHGroupBox("Point Width", self)
        QToolTip.add(widthBox, "The width of points")
        self.widthSlider = QSlider(2, 20, 1, 3, QSlider.Horizontal, widthBox)
        self.widthSlider.setTickmarks(QSlider.Below)
        self.widthLCD = QLCDNumber(2, widthBox)

        #####
        # jittering
        self.jitteringButtons = QVButtonGroup("Jittering type", self)
        QToolTip.add(self.jitteringButtons, "Selected the type of jittering for discrete variables")
        self.jitteringButtons.setExclusive(TRUE)
        self.spreadNone = QRadioButton('none', self.jitteringButtons)
        self.spreadUniform = QRadioButton('uniform', self.jitteringButtons)
        self.spreadTriangle = QRadioButton('triangle', self.jitteringButtons)
        self.spreadBeta = QRadioButton('beta', self.jitteringButtons)

        ######
        # jittering options
        self.jitteringOptionsBG = QVButtonGroup("Jittering options", self)
        QToolTip.add(self.jitteringOptionsBG, "Percents of a discrete value to be jittered")
        self.hbox = QHBox(self.jitteringOptionsBG, "jittering size")
        self.jitterLabel = QLabel('Jittering size (% of size)', self.hbox)
        self.jitterSize = QComboBox(self.hbox)

        self.jitterContinuous = QCheckBox('jitter continuous attributes', self.jitteringOptionsBG)        

        #####
        self.graphSettings = QVButtonGroup("General graph settings", self)
        QToolTip.add(self.graphSettings, "Enable/disable main title, axis title and grid")
        self.gShowTitle = QCheckBox('Show title', self.graphSettings)
        self.gShowAttributeValues = QCheckBox('Show attribute values', self.graphSettings)
        self.gSetXaxisCB = QCheckBox('X axis title ', self.graphSettings)
        self.gSetYaxisCB = QCheckBox('Y axis title ', self.graphSettings)
        self.gShowFilledSymbolsCB = QCheckBox('show filled symbols', self.graphSettings)
        self.gShowDistributions = QCheckBox('Show distributions', self.graphSettings)

        self.apply = QPushButton("Apply changes", self)
        self.gSetGridColorB = QPushButton("Grid Color", self)
        self.gSetCanvasColorB = QPushButton("Canvas Color", self)
        self.connect(self.widthSlider, SIGNAL("valueChanged(int)"), self.widthLCD, SLOT("display(int)"))
        self.connect(self.gSetGridColorB, SIGNAL("clicked()"), self.setGraphGridColor)
        self.connect(self.gSetCanvasColorB, SIGNAL("clicked()"), self.setGraphCanvasColor)

    def setGraphGridColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphGridColor))
        if newColor.isValid():
            self.parent.graphGridColor = str(newColor.name())
            self.parent.setGridColor(newColor)

    def setGraphCanvasColor(self):
        newColor = QColorDialog.getColor(QColor(self.parent.graphCanvasColor))
        if newColor.isValid():
            self.parent.graphCanvasColor = str(newColor.name())
            self.parent.setCanvasColor(QColor(newColor))



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlotMatrix()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

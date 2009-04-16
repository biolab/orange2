"""
<name>Scatterplot matrix</name>
<description>Scatterplot matrix visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/ScatterPlotMatrix.png</icon>
<priority>1100</priority>
"""
# ScatterPlotMatrix.py
#
# Show data using scatterplot matrix visualization method
#
from OWVisWidget import *
import OWColorPalette
from OWScatterPlotGraph import OWScatterPlotGraph
import OWDlgs, OWGUI, OWColorPalette

#class QMyLabel(QLabel):
#    def __init__(self, size, *args):
#        apply(QLabel.__init__,(self,) + args)
#        self.size = size
#
#    def setSize(self, size):
#        self.size = size
#
#    def sizeHint(self):
#        return self.size

###########################################################################################
##### WIDGET : Parallel coordinates visualization
###########################################################################################
class OWScatterPlotMatrix(OWVisWidget):
    settingsList = ["pointWidth", "showAxisScale", "showXaxisTitle", "showYLaxisTitle",
                    "showLegend", "jitterSize", "jitterContinuous", "showFilledSymbols", "colorSettings", "showAllAttributes"]
    jitterSizeNums = [0.0, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20]
    contextHandlers = {"": DomainContextHandler("", [ContextField("shownAttributes", DomainContextHandler.RequiredList, selected="selectedShown", reservoir="hiddenAttributes")])}

    def __init__(self,parent=None, signalManager = None):
        OWVisWidget.__init__(self, parent, signalManager, "Scatterplot Matrix", TRUE)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Example Subset", ExampleTable, self.setSubsetData), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = [("Attribute Selection List", AttributeList)]

        #set default settings
        self.data = None
        self.subsetData = None
        self.visualizedAttributes = []

        self.showAllAttributes = 0
        self.pointWidth = 5
        self.showAxisScale = 1
        self.showXaxisTitle = 0
        self.showYLaxisTitle = 0
        self.showLegend = 0
        self.jitterContinuous = 0
        self.jitterSize = 2
        self.showFilledSymbols = 1
        self.shownAttrCount = 0
        self.graphCanvasColor = str(QColor(QColor(Qt.white)).name())
        self.attributeSelectionList = None
        self.colorSettings = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "General")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        #add controls to self.controlArea widget
        self.createShowHiddenLists(self.GeneralTab)
        self.createMatrixButton = OWGUI.button(self.GeneralTab, self, "Create matrix", callback = self.createGraphs, tooltip="Create scatterplot matrix using shown attributes")

        # ####################################
        # settings tab
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box=' Point size ', minValue=1, maxValue=20, step=1, callback = self.setPointWidth)

        box2 = OWGUI.widgetBox(self.SettingsTab, "Jittering options")
        self.jitterSizeCombo = OWGUI.comboBox(box2, self, "jitterSize", label = "Jittering size (% of size): ", orientation = "horizontal", callback = self.updateJitteringSettings, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box2, self, 'jitterContinuous', 'Jitter continuous attributes', callback = self.updateJitteringSettings, tooltip = "Does jittering apply also on continuous attributes?")

        box4 = OWGUI.widgetBox(self.SettingsTab, "General graph settings")
        OWGUI.checkBox(box4, self, 'showAxisScale', 'Show axis scale', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showXaxisTitle', 'X axis title', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showYLaxisTitle', 'Y axis title', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showLegend', 'Show legend', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showFilledSymbols', 'Show filled symbols', callback = self.updateSettings)

        hbox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(hbox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color and color palette for coloring continuous variables", debuggingEnabled = 0)
        OWGUI.rubber(self.SettingsTab)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFile)

        import sip
        sip.delete(self.mainArea.layout())
        self.grid = QGridLayout()
        self.mainArea.setLayout(self.grid)
        self.graphs = []
        self.labels = []
        self.graphParameters = []

        # add a settings dialog and initialize its values
        self.icons = self.createAttributeIconDict()

        dlg = self.createColorDialog()
        self.contPalette = dlg.getContinuousPalette("contPalette")
        self.discPalette = dlg.getDiscretePalette("discPalette")
        self.graphCanvasColor = dlg.getColor("Canvas")
        self.resize(900, 700)

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color palette")
        c.createDiscretePalette("discPalette", "Discrete Palette")
        c.createContinuousPalette("contPalette", "Continuous Palette")
        box = c.createBox("otherColors", "Other Colors")
        c.createColorButton(box, "Canvas", "Canvas color", QColor(Qt.white))
        box.layout().addSpacing(5)
        c.setColorSchemas(self.colorSettings)
        return c

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_():
            self.colorSettings = dlg.getColorSchemas()
            self.contPalette = dlg.getContinuousPalette("contPalette")
            self.discPalette = dlg.getDiscretePalette("discPalette")
            self.graphCanvasColor = dlg.getColor("Canvas")
            self.updateSettings()

    def setPointWidth(self):
        for graph in self.graphs:
            graph.pointWidth = self.pointWidth
        self.updateGraph()

    def updateShowLegend(self):
        for graph in self.graphs:
            graph.showLegend = self.showLegend

    def updateJitteringSettings(self):
        if self.graphs == []: return
        self.graphs[0].jitterSize = self.jitterSize
        self.graphs[0].jitterContinuous = self.jitterContinuous
        self.graphs[0].rescaleData()

        for graph in self.graphs[1:]:
            for attr in ["rawData", "domainDataStat", "scaledData", "scaledSubsetData", "noJitteringScaledData", "noJitteringScaledSubsetData",
                                     "validDataArray", "validSubsetDataArray", "attrValues", "originalData", "originalSubsetData",
                                     "attributeNames", "domainDataStat", "attributeNameIndex", "dataDomain", "dataHasClass", "dataHasContinuousClass",
                                     "dataHasDiscreteClass", "dataClassName", "dataClassIndex", "haveData", "haveSubsetData",
                                     "jitterSize", "jitterContinuous" ]:
                setattr(graph, attr, getattr(self.graphs[0], attr))
        self.updateGraph()

    # #########################
    # GRAPH MANIPULATION
    # #########################
    def updateSettings(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.setGraphOptions(self.graphs[i], title)
        self.updateGraph()

    def setGraphOptions(self, graph, title):
        graph.showAxisScale = self.showAxisScale
        graph.showXaxisTitle = self.showXaxisTitle
        graph.showYLaxisTitle = self.showYLaxisTitle
        graph.showLegend = self.showLegend
        graph.showFilledSymbols = self.showFilledSymbols
        graph.setCanvasBackground(self.graphCanvasColor)
        graph.contPalette = self.contPalette
        graph.discPalette = self.discPalette


    def updateGraph(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.graphs[i].updateData(attr1, attr2, className)
            self.graphs[i].updateLayout()

    def removeAllGraphs(self):
        for graph in self.graphs:
            graph.hide()
            graph.destroy()
        self.graphs = []
        self.graphParameters = []

        for label in self.labels:
            label.hide()
        self.labels = []


    def createGraphs(self):
        self.removeAllGraphs()

        attrs = [str(self.shownAttribsLB.item(i).text()) for i in range(self.shownAttribsLB.count())]
        if len(attrs) < 2:
            return
        self.visualizedAttributes = attrs
        attrs.reverse()
        count = len(attrs)
        w = self.mainArea.width()/(len(self.visualizedAttributes)-1)
        h = self.mainArea.height()/(len(self.visualizedAttributes)-1)

        self.shownAttrCount = count-1
        for i in range(count-1, -1, -1):
            for j in range(i):
                graph = OWScatterPlotGraph(self, self.mainArea)
                graph.setMinimumSize(QSize(10,10))
                graph.jitterSize = self.jitterSize
                graph.jitterContinuous = self.jitterContinuous

                if self.graphs == []:
                    graph.setData(self.data)
                else:
                    for attr in ["rawData", "domainDataStat", "scaledData", "scaledSubsetData", "noJitteringScaledData", "noJitteringScaledSubsetData",
                                 "validDataArray", "validSubsetDataArray", "attrValues", "originalData", "originalSubsetData",
                                 "attributeNames", "domainDataStat", "attributeNameIndex", "dataDomain", "dataHasClass", "dataHasContinuousClass",
                                 "dataHasDiscreteClass", "dataClassName", "dataClassIndex", "haveData", "haveSubsetData"]:
                        setattr(graph, attr, getattr(self.graphs[0], attr))

                self.setGraphOptions(graph, "")
                self.grid.addWidget(graph, count-i-1, j)
                self.graphs.append(graph)
                self.connect(graph, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)
                params = (attrs[j], attrs[i], self.data.domain.classVar.name, "")
                self.graphParameters.append(params)
                graph.setFixedSize(w,h)
                self.grid.addWidget(graph, count-i, j+1)
        self.updateGraph()


##        w = self.mainArea.width()
##        h = self.mainArea.height()
##        for i in range(len(attrs)):
##            label = QMyLabel(QSize(w/(count+1), h/(count+1)), attrs[i], self.mainArea)
##            self.grid.addWidget(label, len(attrs)-i-1, i, Qt.AlignCenter)
##            label.setAlignment(Qt.AlignCenter)
##            label.show()
##            self.labels.append(label)


    def saveToFile(self):
        self.sizeDlg = OWDlgs.OWChooseImageSizeDlg(self)
        self.sizeDlg.disconnect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.sizeDlg.accept)
        self.sizeDlg.connect(self.sizeDlg.okButton, SIGNAL("clicked()"), self.saveToFileAccept)
        self.sizeDlg.exec_()

    def saveToFileAccept(self):
        qfileName = QFileDialog.getSaveFileName(None, "Save to..", "graph.png", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)")
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
        fullPainter.fillRect(fullBuffer.rect(), QBrush(QColor(Qt.white))) # make background same color as the widget's background

        smallSize = QSize((size.width()-attrNameSpace - count*dist - topOffset)/count, (size.height()-attrNameSpace - count*dist)/count)

        # draw scatterplots
        for i in range(len(self.graphs)):
            buffer = QPixmap(smallSize)
            painter = QPainter(buffer)
            painter.fillRect(buffer.rect(), QBrush(QColor(Qt.white))) # make background same color as the widget's background
            self.graphs[i].printPlot(painter, buffer.rect())
            painter.end()

            # here we have i-th scatterplot printed in a QPixmap. we have to print this pixmap into the right position in the big pixmap
            (found, y, x) = self.grid.findWidget(self.graphs[i])
            fullPainter.drawPixmap(attrNameSpace + x*(smallSize.width() + dist), y*(smallSize.height() + dist) + topOffset, buffer)


        list = []
        for i in range(self.shownAttribsLB.count()): list.append(str(self.shownAttribsLB.item(i).text()))

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


    # we catch mouse release event so that we can send the "Attribute Selection List" signal
    def onMouseReleased(self, e):
        for i in range(len(self.graphs)):
            if self.graphs[i].staticClick == 1:
                (attr1, attr2, className, string) = self.graphParameters[i]
                self.send("Attribute Selection List", [attr1, attr2])
                self.graphs[i].staticClick = 0


    # receive new data and update all fields
    def setData(self, data):
        if data and (len(data) == 0 or len(data.domain) == 0):
            data = None
        if self.data and data and self.data.checksum() == data.checksum():
            return    # check if the new data set is the same as the old one

        self.closeContext()
        sameDomain = self.data and data and data.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        self.data = data
        if not sameDomain:
            self.setShownAttributeList(self.attributeSelectionList)
            self.removeAllGraphs()
        self.openContext("", self.data)
        self.resetAttrManipulation()

    def setSubsetData(self, subsetData):
        self.subsetData = subsetData

    def setShownAttributes(self, attrList):
        self.attributeSelectionList = attributeSelectionList

    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.graph.setData(self.data, self.subsetData)
        if self.attributeSelectionList and 0 not in [self.graph.attributeNameIndex.has_key(attr) for attr in self.attributeSelectionList]:
            self.setShownAttributeList(self.attributeSelectionList)
        else:
            self.setShownAttributeList()
        self.attributeSelectionList = None
        self.updateGraph()


#    def resizeEvent(self, e):
#        OWWidget.resizeEvent(self,e)
#        if len(self.visualizedAttributes) >= 2:
#            w = (self.mainArea.width()-40)/(len(self.visualizedAttributes)-1)
#            h = (self.mainArea.height()-40)/(len(self.visualizedAttributes)-1)
#            for graph in self.graphs:
#                graph.setFixedSize(w,h)



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWScatterPlotMatrix()
    ow.show()
    data = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\wine.tab")
    ow.setData(data)
    a.exec_()

    #save settings
    ow.saveSettings()

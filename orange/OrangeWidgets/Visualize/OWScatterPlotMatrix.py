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
import orngOrangeFoldersQt4
from OWWidget import *
import OWColorPalette
from OWScatterPlotGraph import OWScatterPlotGraph
import orngInteract
import statc
import OWDlgs, OWGUI
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
    settingsList = ["pointWidth", "showAxisScale", "showXaxisTitle", "showYLaxisTitle",
                    "showLegend", "jitterSize", "jitterContinuous", "showFilledSymbols", "colorSettings"]
    jitterSizeNums = [0.1,   0.5,  1,  2,  5,  10, 15, 20]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Scatterplot Matrix", TRUE)

        self.inputs = [("Examples", ExampleTable, self.setData), ("Attribute Selection List", AttributeList, self.setShownAttributes)]
        self.outputs = [("Attribute Selection List", AttributeList)]

        #set default settings
        self.data = None
        self.visualizedAttributes = []

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
        self.attributeSelection = None
        self.colorSettings = None

        #load settings
        self.loadSettings()

        #GUI
        self.tabs = OWGUI.tabWidget(self.controlArea)
        self.GeneralTab = OWGUI.createTabPage(self.tabs, "General")
        self.SettingsTab = OWGUI.createTabPage(self.tabs, "Settings")

        #add controls to self.controlArea widget
        self.shownAttribsGroup = OWGUI.widgetBox(self.GeneralTab, "Shown attributes")
        self.shownAttribsGroup.setMinimumWidth(200)
        hbox = OWGUI.widgetBox(self.shownAttribsGroup, orientation = 'horizontal')
        self.addRemoveGroup = OWGUI.widgetBox(self.GeneralTab, 1, orientation = "horizontal" )
        self.hiddenAttribsGroup = OWGUI.widgetBox(self.GeneralTab, "Hidden attributes")

        self.shownAttribsLB = OWGUI.listBox(hbox, self, selectionMode = QListWidget.ExtendedSelection)
        self.hiddenAttribsLB = OWGUI.listBox(self.hiddenAttribsGroup, self, selectionMode = QListWidget.ExtendedSelection)

        vbox = OWGUI.widgetBox(hbox, orientation = 'vertical')
        self.buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attributes up")
        self.buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attributes down")
        self.buttonUPAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonUPAttr.setMaximumWidth(20)
        self.buttonDOWNAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonDOWNAttr.setMaximumWidth(20)
        self.buttonUPAttr.setMaximumWidth(20)

        self.attrAddButton =    OWGUI.button(self.addRemoveGroup, self, "", callback = self.addAttribute, tooltip="Add (show) selected attributes")
        self.attrAddButton.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_up2.png")))
        self.attrRemoveButton = OWGUI.button(self.addRemoveGroup, self, "", callback = self.removeAttribute, tooltip="Remove (hide) selected attributes")
        self.attrRemoveButton.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_down2.png")))

        self.createMatrixButton = OWGUI.button(self.GeneralTab, self, "Create matrix", callback = self.createGraphs, tooltip="Create scatterplot matrix using shown attributes")

        # ####################################
        # settings tab
        OWGUI.hSlider(self.SettingsTab, self, 'pointWidth', box=' Point size ', minValue=1, maxValue=20, step=1, callback = self.setPointWidth)

        box2 = OWGUI.widgetBox(self.SettingsTab, "Jittering options")
        box3 = OWGUI.widgetBox(box2, orientation = "horizontal")
        self.jitterLabel = QLabel('Jittering size (% of size)'+'  ', box3)
        self.jitterSizeCombo = OWGUI.comboBox(box3, self, "jitterSize", callback = self.updateJitteringSettings, items = self.jitterSizeNums, sendSelectedValue = 1, valueType = float)
        OWGUI.checkBox(box2, self, 'jitterContinuous', 'Jitter continuous attributes', callback = self.updateJitteringSettings, tooltip = "Does jittering apply also on continuous attributes?")

        box4 = OWGUI.widgetBox(self.SettingsTab, "General graph settings")
        OWGUI.checkBox(box4, self, 'showAxisScale', 'Show axis scale', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showXaxisTitle', 'X axis title', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showYLaxisTitle', 'Y axis title', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showLegend', 'Show legend', callback = self.updateSettings)
        OWGUI.checkBox(box4, self, 'showFilledSymbols', 'Show filled symbols', callback = self.updateSettings)

        hbox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(hbox, self, "Set Colors", self.setColors, tooltip = "Set the canvas background color and color palette for coloring continuous variables", debuggingEnabled = 0)

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
        self.activateLoadedSettings()
        self.resize(900, 700)


    # #########################
    # OPTIONS
    # #########################
    def activateLoadedSettings(self):
        dlg = self.createColorDialog()
        self.contPalette = dlg.getContinuousPalette("contPalette")
        self.discPalette = dlg.getDiscretePalette("discPalette")
        self.graphCanvasColor = dlg.getColor("Canvas")

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

    def setGraphOptions(self, graph, title):
        graph.showAxisScale = self.showAxisScale
        graph.showXaxisTitle = self.showXaxisTitle
        graph.showYLaxisTitle = self.showYLaxisTitle
        graph.showLegend = self.showLegend
        graph.showFilledSymbols = self.showFilledSymbols
        graph.setCanvasBackground(self.graphCanvasColor)
        graph.contPalette = self.contPalette
        graph.discPalette = self.discPalette


    def setPointWidth(self):
        for graph in self.graphs:
            graph.pointWidth = n
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
            graph.jitterSize = self.jitterSize
            graph.jitterContinuous = self.jitterContinuous
            graph.scaledData = self.graphs[0].scaledData
            graph.validDataArray = self.graphs[0].validDataArray
            graph.attributeNameIndex = self.graphs[0].attributeNameIndex
            graph.domainDataStat = self.graphs[0].domainDataStat
            graph.attributeNames = self.graphs[0].attributeNames
        self.updateGraph()

    # #########################
    # GRAPH MANIPULATION
    # #########################
    def updateSettings(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.setGraphOptions(self.graphs[i], title)
        self.updateGraph()

    def updateGraph(self):
        for i in range(len(self.graphs)):
            (attr1, attr2, className, title) = self.graphParameters[i]
            self.graphs[i].updateData(attr1, attr2, className)
            self.graphs[i].repaint()

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
                #graph.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
                graph.setMinimumSize(QSize(10,10))
                graph.jitterSize = self.jitterSize
                graph.jitterContinuous = self.jitterContinuous

                if self.graphs == []:
                    graph.setData(self.data)
                else:
                    for attr in ["rawData", "domainDataStat", "scaledData", "noJitteringScaledData", "validDataArray", "attrValues", "attributeNames", "domainDataStat", "attributeNameIndex"]:
                        setattr(graph, attr, getattr(self.graphs[0], attr))

                self.setGraphOptions(graph, "")
                self.grid.addWidget(graph, count-i-1, j)
                self.graphs.append(graph)
                self.connect(graph, SIGNAL('plotMouseReleased(const QMouseEvent&)'),self.onMouseReleased)
                params = (attrs[j], attrs[i], self.data.domain.classVar.name, "")
                self.graphParameters.append(params)
                graph.setFixedSize(w,h)
                self.grid.addWidget(graph, count-i, j+1)
                #graph.show()
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
        qfileName = QFileDialog.getSaveFileName(None, "Save to...", "graph.png", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)")
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
        exData = self.data

        if data:
            name = getattr(data, "name", "")
            data = data.filterref(orange.Filter_hasClassValue())
            data.name = name
            if len(data) == 0 or len(data.domain) == 0:        # if we don't have any examples or attributes then this is not a valid data set
                data = None

        if data == None:
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()
            self.removeAllGraphs()
            return

        self.data = data

        sameDomain = self.data and exData and exData.domain.checksum() == self.data.domain.checksum() # preserve attribute choice if the domain is the same
        if sameDomain:
            if self.graphs != []:
                self.createGraphs()   # if we had already created graphs, redraw them with new data
            return

        if not self.setShownAttributes(self.attributeSelection):
            self.shownAttribsLB.clear()
            self.hiddenAttribsLB.clear()

            for attr in self.data.domain.attributes:
                self.shownAttribsLB.addItem(QListWidgetItem(self.icons[attr.varType], attr.name))

            #self.createGraphs()

    #################################################

    def setShownAttributes(self, attrList):
        self.attributeSelection = attrList

        if not self.data or not attrList: return 0

        domain = [attr.name for attr in self.data.domain]
        for attr in attrList:
            if attr not in domain: return 0  # this attribute list belongs to a new dataset that has not come yet

        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        for attr in attrList:
            self.shownAttribsLB.addItem(QListWidgetItem(self.icons[self.data.domain[attr].varType], attr))

        for attr in self.data.domain.attributes:
            if attr.name not in attrList:
                self.hiddenAttribsLB.addItem(QListWidgetItem(self.icons[attr.varType], attr.name))

        self.createGraphs()
        return 1


    ################################################
    # adding and removing interesting attributes
    def addAttribute(self):
        count = self.hiddenAttribsLB.count()
        pos   = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.item(i).isSelected():
                self.shownAttribsLB.insertItem(pos, QListWidgetItem(self.hiddenAttribsLB.item(i).icon(), self.hiddenAttribsLB.item(i).text()))
                self.hiddenAttribsLB.takeItem(i)

    def removeAttribute(self):
        count = self.shownAttribsLB.count()
        pos   = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.item(i).isSelected():
                self.hiddenAttribsLB.insertItem(pos, QListWidgetItem(self.shownAttribsLB.item(i).icon(), self.shownAttribsLB.item(i).text()))
                self.shownAttribsLB.takeItem(i)

    def moveAttrUP(self):
        for i in range(1, self.shownAttribsLB.count()):
            if self.shownAttribsLB.item(i).isSelected():
                self.shownAttribsLB.insertItem(i-1, QListWidgetItem(self.shownAttribsLB.item(i).icon(), self.shownAttribsLB.item(i).text()))
                self.shownAttribsLB.takeItem(i+1)
                self.shownAttribsLB.item(i-1).setSelected(TRUE)

    def moveAttrDOWN(self):
        count = self.shownAttribsLB.count()
        for i in range(count-2,-1,-1):
            if self.shownAttribsLB.item(i).isSelected():
                self.shownAttribsLB.insertItem(i+2, QListWidgetItem(self.shownAttribsLB.item(i).icon(), self.shownAttribsLB.item(i).text()))
                self.shownAttribsLB.takeItem(i)
                self.shownAttribsLB.item(i+1).setSelected(TRUE)

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

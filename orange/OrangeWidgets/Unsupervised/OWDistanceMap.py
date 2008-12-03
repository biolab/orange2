"""
<name>Distance Map</name>
<description>Displays distance matrix as a heat map.</description>
<icon>icons/DistanceMap.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1200</priority>
"""

import orange, math, sys
import OWGUI, OWToolbars
from OWWidget import *

from ColorPalette import *
import OWColorPalette
import OWToolbars

#####################################################################
# parameters that determine the canvas layout

c_offsetX = 10; c_offsetY = 10  # top and left border
c_spaceX = 10; c_spaceY = 10    # space btw graphical elements
c_legendHeight = 15             # height of the legend
c_averageStripeWidth = 12       # width of the stripe with averages
c_smallcell = 8                 # below this threshold cells are
                                # considered small and grid dissapears

#####################################################################
# canvas with events

class EventfulGraphicsView(QGraphicsView):
    def __init__(self, scene, parent, master):
        QGraphicsView.__init__(self, scene, parent)
        self.master = master
        self.viewport().setMouseTracking(True)

##    def mousePressEvent (self, event):
##        self.master.mousePress(event.pos().x(), event.pos().y())
##
##    def mouseReleaseEvent (self, event):
##        self.master.mouseRelease(event.pos().x(), event.pos().y())
##
##    def mouseMoveEvent (self, event):
##        self.master.mouseMove(event.pos().x(), event.pos().y())

class EventfulGraphicsScene(QGraphicsScene):
    def __init__(self, master):
        QGraphicsScene.__init__(self)
        self.master = master
##        self.setMouseTracking(True)

    def mousePressEvent (self, event):
        self.master.mousePress(event.scenePos().x(), event.scenePos().y())

    def mouseReleaseEvent (self, event):
        self.master.mouseRelease(event.scenePos().x(), event.scenePos().y())

    def mouseMoveEvent (self, event):
        self.master.mouseMove(event)

#####################################################################
# main class
v_sel_width = 2
v_legend_width = 104
v_legend_height = 18
v_legend_offsetX = 5
v_legend_offsetY = 15

class OWDistanceMap(OWWidget):
    settingsList = ["CellWidth", "CellHeight", "Merge", "Gamma", "CutLow",
                    "CutHigh", "CutEnabled", "Sort", "SquareCells",
                    "ShowLegend", "ShowLabels", "ShowBalloon",
                    "Grid", "savedGrid",
                    "ShowItemsInBalloon", "SendOnRelease", "colorSettings", "selectedSchemaIndex", "palette"]

    def __init__(self, parent=None, signalManager = None):
        self.callbackDeposit = [] # deposit for OWGUI callback function
        OWWidget.__init__(self, parent, signalManager, 'Distance Map')

        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix)]
        self.outputs = [("Examples", ExampleTable), ("Attribute List", orange.VarList)]

        self.clicked = False
        self.offsetX = 5
        self.offsetY = 5
        self.imageWidth = 0
        self.imageHeight = 0
        self.distanceImage = None
        self.legendImage = None
        self.colorSettings = None
        self.selectedSchemaIndex = 0

        self.palette = None        

        self.shiftPressed = False

        #set default settings
        self.CellWidth = 15; self.CellHeight = 15
        self.Merge = 1;
        self.savedMerge = self.Merge
        self.Gamma = 1
        self.Grid = 1
        self.savedGrid = 1
        self.CutLow = 0; self.CutHigh = 0; self.CutEnabled = 0
        self.Sort = 0
        self.SquareCells = 0
        self.ShowLegend = 1;
        self.ShowLabels = 1;
        self.ShowBalloon = 1;
        self.ShowItemsInBalloon = 1
        self.SendOnRelease = 1

        self.loadSettings()

        self.maxHSize = 30; self.maxVSize = 30
        self.sorting = [("No sorting", self.sortNone),
                        ("Adjacent distance", self.sortAdjDist),
                        ("Random order", self.sortRandom),
                        ("Clustering", self.sortClustering),
                        ("Clustering with ordered leafs", self.sortClusteringOrdered)]

        self.matrix = self.order = None
        self.rootCluster = None

        # GUI definition
        self.tabs = OWGUI.tabWidget(self.controlArea)

        # SETTINGS TAB
        tab = OWGUI.createTabPage(self.tabs, "Settings")
        box = OWGUI.widgetBox(tab, "Cell Size (Pixels)", addSpace=True)
        OWGUI.qwtHSlider(box, self, "CellWidth", label='Width: ',
                         labelWidth=38, minValue=1, maxValue=self.maxHSize,
                         step=1, precision=0,
                         callback=[lambda f="CellWidth", t="CellHeight": self.adjustCellSize(f,t), self.drawDistanceMap, self.manageGrid])
        OWGUI.qwtHSlider(box, self, "CellHeight", label='Height: ',
                         labelWidth=38, minValue=1, maxValue=self.maxVSize,
                         step=1, precision=0,
                         callback=[lambda f="CellHeight", t="CellWidth": self.adjustCellSize(f,t), self.drawDistanceMap,self.manageGrid])
        OWGUI.checkBox(box, self, "SquareCells", "Cells as squares",
                         callback = [self.setSquares, self.drawDistanceMap])
        self.gridChkBox = OWGUI.checkBox(box, self, "Grid", "Show grid", callback = self.createDistanceMap, disabled=lambda: min(self.CellWidth, self.CellHeight) <= c_smallcell)

        OWGUI.qwtHSlider(tab, self, "Merge", box="Merge" ,label='Elements:', labelWidth=50,
                         minValue=1, maxValue=100, step=1,
                         callback=self.createDistanceMap, ticks=0, addSpace=True)
        
        self.labelCombo = OWGUI.comboBox(tab, self, "Sort", box="Sort",
                         items=[x[0] for x in self.sorting],
                         tooltip="Sorting method for items in distance matrix.",
                         callback=self.sortItems)
        OWGUI.rubber(tab)

##        self.tabs.insertTab(tab, "Settings")

        # FILTER TAB
        tab = OWGUI.createTabPage(self.tabs, "Colors")
        box = OWGUI.widgetBox(tab, "Color settings", addSpace=True)
        OWGUI.widgetLabel(box, "Gamma")
        OWGUI.qwtHSlider(box, self, "Gamma", minValue=0.1, maxValue=1,
                         step=0.1, maxWidth=100, callback=self.drawDistanceMap)

        OWGUI.separator(box)

        OWGUI.checkBox(box, self, 'CutEnabled', "Enable thresholds", callback=self.setCutEnabled)
        self.sliderCutLow = OWGUI.qwtHSlider(box, self, 'CutLow', label='Low:',
                              labelWidth=33, minValue=-100, maxValue=0, step=0.1,
                              precision=1, ticks=0, maxWidth=80,
                              callback=self.drawDistanceMap)
        self.sliderCutHigh = OWGUI.qwtHSlider(box, self, 'CutHigh', label='High:',
                              labelWidth=33, minValue=0, maxValue=100, step=0.1,
                              precision=1, ticks=0, maxWidth=80,
                              callback=self.drawDistanceMap)
        if not self.CutEnabled:
            self.sliderCutLow.box.setDisabled(1)
            self.sliderCutHigh.box.setDisabled(1)


##        self.colorPalette = ColorPalette(box, self, "",
##                         additionalColors =["Cell outline", "Selected cells"],
##                         callback = self.setColor)
##        box.layout().addWidget(self.colorPalette)
        box = OWGUI.widgetBox(box, "Colors", orientation="horizontal")
        self.colorCombo = OWColorPalette.PaletteSelectorComboBox(self)
        try:
            self.colorCombo.setPalettes("palette", self.createColorDialog())
        except Exception, ex:
            print >> sys.stderr, ex, "Error loading saved color palettes!\nCreating new default palette!"
            self.colorSettings = None
            self.colorCombo.setPalettes("palette", self.createColorDialog())
        self.colorCombo.setCurrentIndex(self.selectedSchemaIndex)
        self.connect(self.colorCombo, SIGNAL("activated(int)"), self.setColor)
        box.layout().addWidget(self.colorCombo, 2)
        OWGUI.button(box, self, "Edit colors", callback=self.openColorDialog)
        OWGUI.rubber(tab)

        self.setColor(self.selectedSchemaIndex)        

##        self.tabs.insertTab(tab, "Colors")

        # INFO TAB
        tab = OWGUI.createTabPage(self.tabs, "Info")
        box = OWGUI.widgetBox(tab, "Annotation && Legends")
        OWGUI.checkBox(box, self, 'ShowLegend', 'Show legend',
                       callback=self.drawDistanceMap)
        OWGUI.checkBox(box, self, 'ShowLabels', 'Show labels',
                       callback=self.drawDistanceMap)

        box = OWGUI.widgetBox(tab, "Balloon")
        OWGUI.checkBox(box, self, 'ShowBalloon', "Show balloon")
        OWGUI.checkBox(box, self, 'ShowItemsInBalloon', "Display item names")

        box = OWGUI.widgetBox(tab, "Select")
        box2 = OWGUI.widgetBox(box, orientation = "horizontal")
        self.box2 = box2
        self.buttonUndo = OWToolbars.createButton(box2, 'Undo', self.actionUndo,
                              QIcon(OWToolbars.dlg_undo), toggle = 0)
        self.buttonRemoveAllSelections = OWToolbars.createButton(box2,
                              'Remove all selections', self.actionRemoveAllSelections,
                              QIcon(OWToolbars.dlg_clear), toggle = 0)

        self.buttonSendSelections = OWToolbars.createButton(box2, 'Send selections',
                              self.sendOutput, QIcon(OWToolbars.dlg_send), toggle = 0)
        OWGUI.checkBox(box, self, 'SendOnRelease', "Send after mouse release")
        OWGUI.rubber(tab)

##        self.tabs.insertTab(tab, "Info")

        self.resize(700,400)

        self.scene = EventfulGraphicsScene(self)
        self.sceneView = EventfulGraphicsView(self.scene, self.mainArea, self)
        self.mainArea.layout().addWidget(self.sceneView)

        #construct selector
        self.selector = QGraphicsRectItem(0, 0, self.CellWidth, self.CellHeight, None, self.scene)
##        color = self.colorPalette.getCurrentColorSchema().getAdditionalColors()["Cell outline"]
        color = self.cellOutlineColor
        self.selector.setPen(QPen(self.qrgbToQColor(color),v_sel_width))
        self.selector.setZValue(20)

##        self.bubble = BubbleInfo(self.scene)
        self.selection = SelectionManager()

        self.selectionLines = []
        self.annotationText = []
        self.clusterItems = []

        self.legendText1 = QGraphicsSimpleTextItem(None, self.scene)
        self.legendText2 = QGraphicsSimpleTextItem(None, self.scene)

        self.errorText = QGraphicsSimpleTextItem("Bitmap is too large.", None, self.scene)
        self.errorText.setPos(10,10)

        #restore color schemas from settings
##        if self.ColorSchemas:
##            self.colorPalette.setColorSchemas(self.ColorSchemas)

    def createColorStripe(self, palette, offsetX):
        dx = v_legend_width
        dy = v_legend_height
        bmp = chr(252)*dx*2 + reduce(lambda x,y:x+y, [chr(i*250/dx) for i in range(dx)] * (dy-4)) + chr(252)*dx*2

        image = ImageItem(bmp, self.scene, dx, dy, palette, x=offsetX, y=v_legend_offsetY, z=0)
        return image

    def colFromMousePos(self, x, y):
        if (x <= self.offsetX or x >= self.offsetX + self.imageWidth):
            return -1
        else:
            return int((x - self.offsetX)/self.CellWidth)

    def rowFromMousePos(self, x,y):
        if (y <= self.offsetY or y >= self.offsetY + self.imageHeight):
            return -1
        else:
            return int((y - self.offsetY)/self.CellHeight)


    def qrgbToQColor(self, color):
        # we could also use QColor(positiveColor(rgb), 0xFFFFFFFF)
        return QColor(qRed(positiveColor(color)), qGreen(positiveColor(color)), qBlue(positiveColor(color))) # if color cannot be negative number we convert it manually

    def getItemFromPos(self, i):
        if (len(self.distanceMap.elementIndices)==0):
            j = i
        else:
            j = self.distanceMap.elementIndices[i]

        if self.distanceMapConstructor.order:
           j = self.distanceMapConstructor.order[j]

        return j

    def sendOutput(self):
        if len(self.matrix.items)<1:
            return

        selectedIndices = []
        tmp = []

        if len(self.selection.getSelection())==0:
            self.send("Attribute List", None)
            self.send("Examples", None)
        else:
            selection = self.selection.getSelection()
            for sel in selection:
                if (len(self.distanceMap.elementIndices)==0):
                    tmp += range(sel[0].x(), sel[1].x()+1)
                    tmp +=range(sel[0].y(), sel[1].y()+1)
                else:
                    tmp += range(self.distanceMap.elementIndices[sel[0].x()], self.distanceMap.elementIndices[sel[1].x()+1])
                    tmp +=range(self.distanceMap.elementIndices[sel[0].y()], self.distanceMap.elementIndices[sel[1].y()+1])

            for i in tmp:
                if self.distanceMapConstructor.order:
                    if not (self.distanceMapConstructor.order[i] in selectedIndices):
                        selectedIndices += [self.distanceMapConstructor.order[i]]

                if not (i in selectedIndices):
                    selectedIndices += [i]

            items = self.matrix.items
            if issubclass(orange.EnumVariable, type(items[0])):
                selected = orange.VarList()
                for i in selectedIndices:
                    selected.append(items[i])
                self.send("Attribute List", selected)


            if isinstance(items[0], orange.Example):
                ex = [items[x] for x in selectedIndices]
                selected = orange.ExampleTable(items[0].domain, ex)
                self.send("Examples", selected)


    def setColor(self, index=0):
        self.selectedSchemaIndex = index
        dialog = self.createColorDialog()
        self.colorCombo.setPalettes("palette", dialog)
        self.colorCombo.setCurrentIndex(self.selectedSchemaIndex)
        
        palette = dialog.getExtendedContinuousPalette("palette")
        unknown = dialog.getColor("unknown").rgb()
        underflow = dialog.getColor("underflow").rgb()
        overflow = dialog.getColor("overflow").rgb()

        background = dialog.getColor("background").rgb()
        self.cellOutlineColor = dialog.getColor("cellOutline").rgb()
        self.selectionColor = dialog.getColor("selection").rgb()
##        color = self.colorPalette.getCurrentColorSchema().getAdditionalColors()["Cell outline"]
##        self.selector.setPen(QPen(self.qrgbToQColor(color),v_sel_width))

##        self.ColorSchemas = self.colorPalette.getColorSchemas()
        self.palette = [palette[float(i)/252].rgb() for i in range(250)] + [background]*3 + [underflow, overflow, unknown]
        self.drawDistanceMap()

    def openColorDialog(self):
        dialog = self.createColorDialog()
        if dialog.exec_():
            self.colorSettings = dialog.getColorSchemas()
            self.selectedSchemaIndex = dialog.selectedSchemaIndex
            self.colorCombo.setCurrentIndex(self.selectedSchemaIndex)
            self.setColor(self.selectedSchemaIndex)

    def createColorDialog(self):
        c = OWColorPalette.ColorPaletteDlg(self, "Color Palette")
        c.createExtendedContinuousPalette("palette", "Continuous Palette", initialColor1=QColor(Qt.blue), initialColor2=QColor(255, 255, 0).rgb())
        box = c.createBox("otherColors", "Other Colors")
        
        c.createColorButton(box, "unknown", "Unknown", Qt.gray)
        box.layout().addSpacing(5)
        c.createColorButton(box, "overflow", "Overflow", Qt.black)
        box.layout().addSpacing(5)
        c.createColorButton(box, "underflow", "Underflow", Qt.white)

        box = c.createBox("cellColors", "Cell colors")
        c.createColorButton(box, "background", "Background", Qt.white)
        box.layout().addSpacing(5)
        c.createColorButton(box, "cellOutline", "Cell outline", Qt.gray)
        box.layout().addSpacing(5)
        c.createColorButton(box, "selection", "Selection", Qt.black)
        
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c
        
    def setCutEnabled(self):
        self.sliderCutLow.box.setDisabled(not self.CutEnabled)
        self.sliderCutHigh.box.setDisabled(not self.CutEnabled)
        self.drawDistanceMap()

    def constructDistanceMap(self):
        if self.matrix:
            self.distanceMapConstructor = orange.DistanceMapConstructor(distanceMatrix = self.matrix)
            self.createDistanceMap()

    def createDistanceMap(self):
        """creates distance map objects"""
        merge = min(self.Merge, float(self.matrix.dim))
        squeeze = 1. / merge

        self.distanceMapConstructor.order = self.order
        self.distanceMap, self.lowerBound, self.upperBound = self.distanceMapConstructor(squeeze)

        self.sliderCutLow.setRange(self.lowerBound, self.upperBound, 0.1)
        self.sliderCutHigh.setRange(self.lowerBound, self.upperBound, 0.1)
        self.CutLow = max(self.CutLow, self.lowerBound)
        self.CutHigh = min(self.CutHigh, self.upperBound)
        self.sliderCutLow.setValue(self.CutLow)
        self.sliderCutHigh.setValue(self.CutHigh)

        self.selection.clear()
        self.drawDistanceMap()

    def drawDistanceMap(self):
        """renders distance map object on canvas"""
        if not self.matrix:
            return

        if self.matrix.dim * max(int(self.CellWidth), int(self.CellHeight)) > 32767:
            self.errorText.show()
            return

        self.errorText.hide()

        lo = self.CutEnabled and self.CutLow   or self.lowerBound
        hi = round(self.CutEnabled and self.CutHigh  or self.upperBound, 1)

        self.offsetX = 5

        if self.distanceImage:
            self.scene.removeItem(self.distanceImage)

        if self.legendImage:
            self.scene.removeItem(self.legendImage)

        if self.ShowLegend==1:
            self.offsetY = v_legend_height + 30
        else:
            self.offsetY = 5

##        palette = self.colorPalette.getCurrentColorSchema().getPalette()
        palette = self.palette
        bitmap, width, height = self.distanceMap.getBitmap(int(self.CellWidth),
                            int(self.CellHeight), lo, hi, self.Gamma, self.Grid)

##        self.scene.setSceneRect(0, 0, 2000, 2000) # this needs adjustment

        for tmpText in self.annotationText:
##            tmpText.setScene(None)
            self.scene.removeItem(tmpText)

        for cluster in self.clusterItems:
            self.scene.removeItem(cluster)

        if self.rootCluster and self.order:
            from OWClustering import HierarchicalClusterItem
            clusterTop = HierarchicalClusterItem(self.rootCluster, None, self.scene)
            clusterLeft = HierarchicalClusterItem(self.rootCluster, None, self.scene)
            clusterHeight = 100.0
            clusterTop.setTransform(QTransform().scale(width/float(len(self.rootCluster)), -clusterHeight/clusterTop.rect().height()).\
                                    translate(0, -clusterTop.rect().height()))
            clusterTop.setPos(0, self.offsetY)
            clusterLeft.setTransform(QTransform().scale(clusterHeight/clusterLeft.rect().height(), width/float(len(self.rootCluster))).\
                                    rotate(90).translate(0, -clusterTop.rect().height()))
            clusterLeft.setPos(self.offsetX, 0)
            self.offsetX += clusterHeight + 10
            self.offsetY += clusterHeight + 10

            self.clusterItems += [clusterTop, clusterLeft]

            clusterTop.show()            
            clusterLeft.show()
            

        # determine the font size to fit the cell width
        fontrows = self.getfont(self.CellHeight)
        fontcols = self.getfont(self.CellWidth)
    
        # labels rendering
        self.annotationText = []
        if self.ShowLabels==1 and self.Merge<=1:
            # show labels, no merging (one item per line)
            items = self.matrix.items
            if len(self.distanceMap.elementIndices)==0:
                tmp = [i for i in range(0, len(items))]
            else:
                tmp = [self.distanceMap.elementIndices[i] for i in range(0, len(items))]

            if self.distanceMapConstructor.order:
                indices = [self.distanceMapConstructor.order[i] for i in tmp]
            else:
                indices = tmp

            maxHeight = 0
            maxWidth = 0
            for i in range(0, len(indices)):
                text = items[indices[i]]
                if type(text) not in [str, unicode]:
                    text = text.name
                if text<>"":
                    tmpText = QCustomGraphicsText(text, self.scene, -90.0, font=fontcols)
                    tmpText.show()
                    if tmpText.boundingRect().height() > maxHeight:
                        maxHeight = tmpText.boundingRect().height()
                    self.annotationText += [tmpText]

                    tmpText = QGraphicsSimpleTextItem(text, None, self.scene)
                    tmpText.setFont(fontrows)
                    tmpText.show()
                    if tmpText.boundingRect().width() > maxWidth:
                        maxWidth = tmpText.boundingRect().width()
                    self.annotationText += [tmpText]

            for i in range(0, len(self.annotationText)/2):
##                self.annotationText[i*2].setX(self.offsetX + maxWidth + 3 + (i+0.5)*self.CellWidth)
##                self.annotationText[i*2].setY(self.offsetY)
                self.annotationText[i*2].setPos(self.offsetX + maxWidth + 3 + (i+0.5)*self.CellWidth, self.offsetY + maxWidth)
##                self.annotationText[i*2 + 1].setX(self.offsetX)
##                self.annotationText[i*2 + 1].setY(self.offsetY + maxHeight + 3 + (i+0.5)*self.CellHeight)
##                self.annotationText[i*2 + 1].setPos(self.offsetX, self.offsetY + maxHeight + 3 + (i+0.5)*self.CellHeight)
                self.annotationText[i*2 + 1].setPos(self.offsetX, self.offsetY + maxWidth + 3 + (i+0.5)*self.CellHeight)

            self.offsetX += maxWidth + 10
##            self.offsetY += maxHeight + 10
            self.offsetY += maxWidth + 10

        # rendering of legend
        if self.ShowLegend==1:
##            self.legendImage = self.createColorStripe(self.colorPalette.getCurrentColorSchema().getPalette(), offsetX=self.offsetX)
            self.legendImage = self.createColorStripe(palette, offsetX=self.offsetX)
            self.legendText1.setText("%4.2f" % lo)
            self.legendText2.setText("%4.2f" % hi)
            self.legendText1.setPos(self.offsetX, 0)
            self.legendText2.setPos(self.offsetX + v_legend_width - self.legendText2.boundingRect().width(), 0)
            self.legendText1.show()
            self.legendText2.show()
        else:
            self.legendText1.hide()
            self.legendText2.hide()

        # paint distance map

        if self.rootCluster and self.order:
            ## We now know the location of bitmap
            clusterTop.setPos(self.offsetX, clusterTop.y())
            clusterLeft.setPos(clusterLeft.x(), self.offsetY)
            
        self.distanceImage = ImageItem(bitmap, self.scene, width, height,
                                       palette, x=self.offsetX, y=self.offsetY, z=0)
        self.distanceImage.height = height
        self.distanceImage.width = width

        self.imageWidth = width
        self.imageHeight = height

##        color = self.colorPalette.getCurrentColorSchema().getAdditionalColors()["Cell outline"]
        color = self.cellOutlineColor
        self.selector.setPen(QPen(self.qrgbToQColor(color),v_sel_width))
        self.selector.setRect(QRectF(0, 0, self.CellWidth, self.CellHeight))

        self.updateSelectionRect()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.scene.update()

    def addSelectionLine(self, x, y, direction):
        selLine = QGraphicsLineItem(None, self.scene)
        if direction==0:
            #horizontal line
            selLine.setLine(self.offsetX + x*self.CellWidth, self.offsetY + y*self.CellHeight,
                              self.offsetX + (x+1)*self.CellWidth, self.offsetY + y*self.CellHeight)
        else:
            #vertical line
            selLine.setLine(self.offsetX + x*self.CellWidth, self.offsetY + y*self.CellHeight,
                              self.offsetX + x*self.CellWidth, self.offsetY + (y+1)*self.CellHeight)
##        color = self.colorPalette.getCurrentColorSchema().getAdditionalColors()["Selected cells"]
        color = self.selectionColor
        selLine.setPen(QPen(self.qrgbToQColor(color),v_sel_width))
        selLine.setZValue(20)
        selLine.show();
        self.selectionLines += [selLine]

    def getfont(self, height):
        """finds the font that for a given height"""
        dummy = QGraphicsSimpleTextItem("123", None, self.scene)
        for fontsize in range(8, 2, -1):
            font = QFont("", fontsize)
            dummy.setFont(font)
            if dummy.boundingRect().height() <= height:
                break
        return font

    def updateSelectionRect(self):
        entireSelection = []
        newSel = False
        for selLine in self.selectionLines:
            self.scene.removeItem(selLine)
        
        self.selectionLines = []
        if len(self.selection.getSelection())>0:
            for sel in self.selection.getSelection():
                for i in range(sel[0].x(), sel[1].x()):
                    for j in range(sel[0].y(), sel[1].y()):
                        selTuple = (i, j)
                        if not (selTuple in entireSelection):
                            entireSelection += [selTuple]
            for selTuple in entireSelection:
                #check left
                if (not (selTuple[0] - 1, selTuple[1]) in entireSelection):
                    self.addSelectionLine(selTuple[0], selTuple[1], 1)

                #check up
                if (not (selTuple[0], selTuple[1] - 1) in entireSelection):
                    self.addSelectionLine(selTuple[0], selTuple[1], 0)

                #check down
                if (not (selTuple[0], selTuple[1] + 1) in entireSelection):
                    self.addSelectionLine(selTuple[0], selTuple[1] + 1, 0)

                #check right
                if (not (selTuple[0] + 1, selTuple[1]) in entireSelection):
                    self.addSelectionLine(selTuple[0] + 1, selTuple[1], 1)
        self.scene.update()

    def mouseMove(self, event):
        x, y = event.scenePos().x(), event.scenePos().y()
        row = self.rowFromMousePos(x,y)
        col = self.colFromMousePos(x,y)

        if (self.clicked==True):
            self.selection.UpdateSel(col, row)

        if (row==-1 or col==-1):
            self.selector.hide()
##            self.bubble.hide()
        else:
##            self.selector.setX(self.offsetX + col * self.CellWidth)
##            self.selector.setY(self.offsetY + row * self.CellHeight)
            self.selector.setPos(self.offsetX + col * self.CellWidth, self.offsetY + row * self.CellHeight)
            self.selector.show()

            if self.ShowBalloon == 1:
##                self.bubble.move(x + 20, y + 20)

                i = self.getItemFromPos(col)
                j = self.getItemFromPos(row)
##                self.bubble.head.setText(str(self.matrix[i, j]))
                head = str(self.matrix[i, j])

                if (self.ShowItemsInBalloon == 1):
                    namei, namej = self.matrix.items[i], self.matrix.items[j]
                    if type(namei) not in [str, unicode]:
                        namei = namei.name
                    if type(namej) not in [str, unicode]:
                        namej = namej.name
                    if namei or namej:
##                        self.bubble.body.setText(namei + "\n" + namej)
                        body = namei + "\n" + namej
                    else:
##                        self.bubble.body.setText("")
                        body = ""
                else:
##                    self.bubble.body.setText("")
                    body = ""

##                self.bubble.show()
##                QToolTip.showText(QPoint(x,y), "")
                QToolTip.showText(event.screenPos(), "")
                QToolTip.showText(event.screenPos(), "%s\n%s" % (head, body))
##            else:
##                self.bubble.hide()

            self.updateSelectionRect()

        self.scene.update()

    def keyPressEvent(self, e):
        if e.key() == 4128:
            self.shiftPressed = True
        else:
            OWWidget.keyPressEvent(self, e)

    def keyReleaseEvent(self, e):
        if e.key() == 4128:
            self.shiftPressed = False
        else:
            OWWidget.keyReleaseEvent(self, e)

    def mousePress(self, x,y):
        self.clicked = True
        row = self.rowFromMousePos(x,y)
        col = self.colFromMousePos(x,y)
        if not (self.shiftPressed == True):
            self.selection.clear()
        self.selection.SelStart(col, row)

    def mouseRelease(self, x,y):
        if self.clicked==True:
            self.clicked = False
            row = self.rowFromMousePos(x,y)
            col = self.colFromMousePos(x,y)

            if (row<>-1 and col<>-1):
                self.selection.SelEnd()
            else:
                self.selection.CancelSel()

            self.updateSelectionRect()
            if self.SendOnRelease==1:
                self.sendOutput()

    def actionUndo(self):
        self.selection.undo()
        self.updateSelectionRect()

    def actionRemoveAllSelections(self):
        self.selection.clear()
        self.updateSelectionRect()

    # input signal management

    def sortNone(self):
        self.order = None

    def sortAdjDist(self):
        self.order = None

    def sortRandom(self):
        import random
        self.order = range(len(self.matrix.items))
        random.shuffle(self.order)
        self.rootCluster = None

    def sortClustering(self):
        self.rootCluster=orange.HierarchicalClustering(self.matrix,
                linkage=orange.HierarchicalClustering.Average)
        self.order = list(self.rootCluster.mapping)

    def sortClusteringOrdered(self):
        self.rootCluster=orange.HierarchicalClustering(self.matrix,
                linkage=orange.HierarchicalClustering.Average)
        import orngClustering
        orngClustering.orderLeafs(self.rootCluster, self.matrix)
        self.order = list(self.rootCluster.mapping)
        

    def sortItems(self):
        if not self.matrix:
            return
        self.sorting[self.Sort][1]()
        self.createDistanceMap()

    def setMatrix(self, matrix):
        self.send("Examples", None)
        self.send("Attribute List", None)

        if not matrix:
            return

        # check if the same length
        self.matrix = matrix
        self.constructDistanceMap()

    def setSquares(self):
        if self.SquareCells:
            if self.CellWidth < self.CellHeight:
                self.CellHeight = self.CellWidth
            else:
                self.CellWidth = self.CellHeight

    def adjustCellSize(self, frm, to):
        if self.SquareCells:
            setattr(self, to, getattr(self, frm))

    def manageGrid(self):
        if min(self.CellWidth, self.CellHeight) <= c_smallcell:
            if self.gridChkBox.isEnabled():
                self.savedGrid = self.Grid # remember the state
                self.Grid = 0
                self.gridChkBox.setDisabled(True)
        else:
            if not self.gridChkBox.isEnabled():
                self.gridChkBox.setEnabled(True)
                self.Grid = self.savedGrid

#####################################################################
# new canvas items

class ImageItem(QGraphicsRectItem):
    def __init__(self, bitmap, scene, width, height, palette, depth=8, numColors=256, x=0, y=0, z=0):
        QGraphicsRectItem.__init__(self, None, scene)
##        self.image = QImage(bitmap, width, height, depth, signedPalette(palette), numColors, QImage.LittleEndian) # we take care palette has proper values with proper types
        self.image = QImage(bitmap, width, height, QImage.Format_Indexed8)
        self.image.bitmap = bitmap # this is tricky: bitmap should not be freed, else we get mess. hence, we store it in the object
        self.image.setColorTable(signedPalette(palette))
        self.scene = scene
        self.setRect(0, 0 ,width, height)
##        self.setX(x); self.setY(y); self.setZValue(z)
        self.setPos(x, y)
        self.setZValue(z)
        self.show()

    def paint(self, painter, option, widget=None):
##        painter.drawImage(self.x(), self.y(), self.image, 0, 0, -1, -1)
        painter.drawImage(0, 0, self.image, 0, 0, -1, -1)

class QCustomGraphicsText(QGraphicsSimpleTextItem):
    def __init__(self, text, scene = None, rotateAngle = 0.0, font=None):
        QGraphicsSimpleTextItem.__init__(self, None, scene)
        self.scene = scene
        self.font = font
        self.rotateAngle = rotateAngle
        if font:
            self.setFont(font)
        self.rotate(rotateAngle)
        self.setText(text)
        
##class QCustomCanvasText(QCanvasRectangle):
##    def __init__(self, text, canvas = None, rotateAngle = 0.0, font=None):
##        QCanvasRectangle.__init__(self, canvas)
##        self.canvas = canvas
##        self.font = font
##        self.rotateAngle = rotateAngle
##        self.setText(text)
##        
##    def setText(self, text):
##        self.text = text
##        self.hiddenText = QCanvasText(text, self.canvas)
##        self.hiddenText.setFont(self.font)
##        if self.font:
##            self.hiddenText.setFont(self.font)
##        xsize = self.hiddenText.boundingRect().height()
##        ysize = self.hiddenText.boundingRect().width()
##        self.setSize(xsize, ysize)
##
##    def draw(self, painter):
##        pixmap = QPixmap()
##        xsize = self.hiddenText.boundingRect().height()
##        ysize = self.hiddenText.boundingRect().width()
##        pixmap.resize(xsize, ysize)
##
##        helpPainter = QPainter()
##        helpPainter.begin(pixmap)
##        helpPainter.setFont(self.font)
##
##        helpPainter.setPen( Qt.black );
##        helpPainter.setBrush( Qt.white );
##        helpPainter.drawRect( -1, -1, xsize + 2, ysize + 2);
##        helpPainter.rotate(self.rotateAngle)
##        helpPainter.drawText(-ysize, xsize, self.text)
##        helpPainter.end()
##
##        painter.drawPixmap(self.x(), self.y(), pixmap)

#####################################################################
# selection manager class

class SelectionManager:
    def __init__(self):
        self.selection = []
        self.selecting = False
        self.currSelEnd = None
        self.currSel = None

    def SelStart(self, x, y):
        if x < 0: x=0
        if y < 0: y=0
        self.currSel = QPoint(x,y)
        self.currSelEnd = QPoint(x,y)
        self.selecting = True

    def UpdateSel(self, x, y):
        self.currSelEnd = QPoint(x,y)

    def CancelSel(self):
        self.selecting = False

    def SelEnd(self):
        minx = min(self.currSel.x(), self.currSelEnd.x())
        maxx = max(self.currSel.x(), self.currSelEnd.x())

        miny = min(self.currSel.y(), self.currSelEnd.y())
        maxy = max(self.currSel.y(), self.currSelEnd.y())

        if (minx==maxx) and (miny==maxy):
            maxx+=1
            maxy+=1

        self.selection += [(QPoint(minx, miny),QPoint(maxx,maxy))]
        self.selecting = False

    def clear(self):
        self.selection = []

    def undo(self):
        if len(self.selection)>0:
            del self.selection[len(self.selection)-1]

    def getSelection(self):
        res = self.selection + []
        if self.selecting==True:
            minx = min(self.currSel.x(), self.currSelEnd.x())
            maxx = max(self.currSel.x(), self.currSelEnd.x())

            miny = min(self.currSel.y(), self.currSelEnd.y())
            maxy = max(self.currSel.y(), self.currSelEnd.y())

            res += [(QPoint(minx, miny),QPoint(maxx,maxy))]
        return res

#####################################################################
# bubble info class

bubbleBorder = 4

class BubbleInfo(QGraphicsRectItem):
    def __init__(self, *args):
        QGraphicsRectItem.__init__(self, *args)
        self.scene = args[0]
        self.setBrush(QBrush(Qt.white))
        #self.setPen(QPen(Qt.black, v_sel_width))
        self.bubbleShadow = QGraphicsRectItem(self, self.scene)
        self.bubbleShadow.setBrush(QBrush(Qt.black))
        self.bubbleShadow.setPen(QPen(Qt.black))
        self.head = QCanvasText(self.canvas)
        self.line = QCanvasLine(self.canvas)
        self.body = QCanvasText(self.canvas)
        self.items = [self.head, self.line, self.body]
        self.setZ(110)
        self.bubbleShadow.setZ(109)
        for i in self.items:
            i.setZ(111)

    def move(self, x, y):
        QCanvasRectangle.move(self, x, y)
        self.setX(x); self.setY(y)
        self.bubbleShadow.move(x+5, y+5)
        for item in self.items:
            item.setX(x + bubbleBorder)
        w = max(100, self.head.boundingRect().width() + 2 * bubbleBorder, self.body.boundingRect().width() + 2 * bubbleBorder)
        y += 2
        self.head.setY(y)
        y += self.head.boundingRect().height()
        self.line.setPoints(0,0,w,0)
        self.line.setX(x); self.line.setY(y)
        y += 2
        self.body.setY(y)
        h = 2 * (2 + (self.body.text()<>None)) + self.head.boundingRect().height() + (self.body.text()<>None) * self.body.boundingRect().height()
        self.setSize(w,h)
        self.bubbleShadow.setSize(w,h)

    def show(self):
        QCanvasRectangle.show(self)
        self.bubbleShadow.show()
        self.head.show()
        if self.body.text():
            self.line.show()
            self.body.show()

    def hide(self):
        QCanvasRectangle.hide(self)
        self.bubbleShadow.hide()
        for item in self.items:
            item.hide()


#############################################################
# color palette

class ColorPalette(QWidget):
    def __init__(self, parent, master, value, label = "Colors", additionalColors = None, callback = None):
        QWidget.__init__(self, parent)

        self.constructing = TRUE
        self.callback = callback
        self.schema = ""
        self.passThroughBlack = 0

        self.colorSchemas = {}

        self.setMinimumHeight(300)
        self.setMinimumWidth(200)

        self.box = OWGUI.widgetBox(self, label, orientation = "vertical")

        self.schemaCombo = OWGUI.comboBox(self.box, self, "schema", callback = self.onComboBoxChange)

        self.interpolationHBox = OWGUI.widgetBox(self.box, orientation = "horizontal")
        self.colorButton1 = ColorButton(self, self.interpolationHBox)
        self.interpolationView = InterpolationView(self.interpolationHBox)
        self.colorButton2 = ColorButton(self, self.interpolationHBox)

        self.chkPassThroughBlack = OWGUI.checkBox(self.box, self, "passThroughBlack", "Pass through black", callback = self.onCheckBoxChange)
        #OWGUI.separator(self.box, 10, 10)
        self.box.layout().addSpacing(10)

        #special colors buttons

        self.NAColorButton = ColorButton(self, self.box, "N/A")
        self.underflowColorButton = ColorButton(self, self.box, "Underflow")
        self.overflowColorButton = ColorButton(self, self.box, "Overflow")
        self.backgroundColorButton = ColorButton(self, self.box, "Background (Grid)")

        #set up additional colors
        self.additionalColorButtons = {}

        if additionalColors<>None:
            for colorName in additionalColors:
                self.additionalColorButtons[colorName] = ColorButton(self, self.box, colorName)

        #set up new and delete buttons
        self.buttonHBox = OWGUI.widgetBox(self.box, orientation = "horizontal")
        self.newButton = OWGUI.button(self.buttonHBox, self, "New", self.OnNewButtonClicked)
        self.deleteButton = OWGUI.button(self.buttonHBox, self, "Delete", self.OnDeleteButtonClicked)

        self.setInitialColorPalettes()
        self.paletteSelected()
        self.constructing = FALSE

    def onComboBoxChange(self, string):
        self.paletteSelected()

    def onCheckBoxChange(self, state):
        self.colorSchemaChange()

    def OnNewButtonClicked(self):
        message = "Please enter new color schema name"
        ok = FALSE
        while (not ok):
            s = QInputDialog.getText(self, "New Schema", message)
            ok = TRUE
            if (s[1]==TRUE):
                for i in range(self.schemaCombo.count()):
                    if s[0].lower().compare(self.schemaCombo.itemText(i).lower())==0:
                        ok = FALSE
                        message = "Color schema with that name already exists, please enter another name"
                if (ok):
                    self.colorSchemas[str(s[0])] = ColorSchema(self.getCurrentColorSchema().getName(),
                                                               self.getCurrentColorSchema().getPalette(),
                                                               self.getCurrentColorSchema().getAdditionalColors(),
                                                               self.getCurrentColorSchema().getPassThroughBlack())
                    self.schemaCombo.addItem(s[0])
                    self.schemaCombo.setCurrentIndex(self.schemaCombo.count()-1)
            self.deleteButton.setEnabled(self.schemaCombo.count()>1)


    def OnDeleteButtonClicked(self):
        i = self.schemaCombo.currentIndex()
        self.schemaCombo.removeItem(i)
        self.schemaCombo.setCurrentIndex(i)
        self.deleteButton.setEnabled(self.schemaCombo.count()>1)
        self.paletteSelected()

    def getCurrentColorSchema(self):
        return self.colorSchemas[str(self.schemaCombo.currentText())]

    def setCurrentColorSchema(self, schema):
        self.colorSchemas[str(self.schemaCombo.currentText())] = schema


    def getColorSchemas(self):
        return self.colorSchemas

    def setColorSchemas(self, schemas):
        self.colorSchemas = schemas
        self.schemaCombo.clear()
        self.schemaCombo.addItems(schemas)
        self.paletteSelected()

    def createPalette(self,color1,color2, passThroughBlack):
        palette = []
        if passThroughBlack:
            for i in range(paletteInterpolationColors/2):
                palette += [qRgb(color1.red() - color1.red()*i*2./paletteInterpolationColors,
                                 color1.green() - color1.green()*i*2./paletteInterpolationColors,
                                 color1.blue() - color1.blue()*i*2./paletteInterpolationColors)]

            for i in range(paletteInterpolationColors - (paletteInterpolationColors/2)):
                palette += [qRgb(color2.red()*i*2./paletteInterpolationColors,
                                 color2.green()*i*2./paletteInterpolationColors,
                                 color2.blue()*i*2./paletteInterpolationColors)]
        else:
            for i in range(paletteInterpolationColors):
                palette += [qRgb(color1.red() + (color2.red()-color1.red())*i/paletteInterpolationColors,
                                 color1.green() + (color2.green()-color1.green())*i/paletteInterpolationColors,
                                 color1.blue() + (color2.blue()-color1.blue())*i/paletteInterpolationColors)]
        return palette

    def paletteSelected(self):
        schema = self.getCurrentColorSchema()
        self.interpolationView.setPalette1(schema.getPalette())
        self.colorButton1.setColor(self.rgbToQColor(schema.getPalette()[0]))
        self.colorButton2.setColor(self.rgbToQColor(schema.getPalette()[249]))

        self.chkPassThroughBlack.setChecked(schema.getPassThroughBlack())

        self.NAColorButton.setColor(self.rgbToQColor(schema.getPalette()[255]))
        self.overflowColorButton.setColor(self.rgbToQColor(schema.getPalette()[254]))
        self.underflowColorButton.setColor(self.rgbToQColor(schema.getPalette()[253]))
        self.backgroundColorButton.setColor(self.rgbToQColor(schema.getPalette()[252]))

        for buttonName in self.additionalColorButtons:
            self.additionalColorButtons[buttonName].setColor(self.rgbToQColor(schema.getAdditionalColors()[buttonName]))

        if not self.constructing:
            self.callback()

    def rgbToQColor(self, rgb):
        # we could also use QColor(positiveColor(rgb), 0xFFFFFFFF) but there is probably a reason
        # why this was not used before so I am leaving it as it is

        return QColor(qRed(positiveColor(rgb)), qGreen(positiveColor(rgb)), qBlue(positiveColor(rgb))) # on Mac color cannot be negative number in this case so we convert it manually

    def qRgbFromQColor(self, qcolor):
        return qRgb(qcolor.red(), qcolor.green(), qcolor.blue())

    def colorSchemaChange(self):
        white = qRgb(255,255,255)
        gray = qRgb(200,200,200)
        name = self.getCurrentColorSchema().getName()
        passThroughBlack = self.chkPassThroughBlack.isChecked()
        palette = self.createPalette(self.colorButton1.getColor(), self.colorButton2.getColor(), passThroughBlack)
        palette += [white]*2 + [self.qRgbFromQColor(self.backgroundColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.underflowColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.overflowColorButton.getColor())] + \
                               [self.qRgbFromQColor(self.NAColorButton.getColor())]

        self.interpolationView.setPalette1(palette)

        additionalColors = {}
        for buttonName in self.additionalColorButtons:
            additionalColors[buttonName] = self.qRgbFromQColor(self.additionalColorButtons[buttonName].getColor())

        schema = ColorSchema(name, palette, additionalColors, passThroughBlack)
        self.setCurrentColorSchema(schema)

        if not self.constructing and self.callback:
            self.callback()


    def setInitialColorPalettes(self):
        white = qRgb(255,255,255)
        gray = qRgb(200,200,200)

        additionalColors = {}
        for buttonName in self.additionalColorButtons:
            additionalColors[buttonName] = gray


        self.schemaCombo.addItem("Blue - Yellow")
        palette = self.createPalette(QColor(0,0,255), QColor(255,255,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0., 255.), qRgb(255., 255., 0.), gray]
        self.colorSchemas["Blue - Yellow"] = ColorSchema("Blue - Yellow", palette, additionalColors, FALSE)

        self.schemaCombo.addItem("Black - Red")
        palette = self.createPalette(QColor(0,0,0), QColor(255,0,0),FALSE)
        palette += [white]*3 + [qRgb(0., 0, 0), qRgb(255., 0, 0), gray]
        self.colorSchemas["Black - Red"] = ColorSchema("Black - Red", palette, additionalColors, FALSE)

        self.schemaCombo.addItem("Green - Black - Red")
        palette = self.createPalette(QColor(0,255,0), QColor(255,0,0),TRUE)
        palette += [white]*3 + [qRgb(0, 255., 0), qRgb(255., 0, 0), gray]
        self.colorSchemas["Green - Black - Red"] = ColorSchema("Green - Black - Red", palette, additionalColors, TRUE)




#############################################################
# test script

if __name__=="__main__":
    def distanceMatrix(data):
        dist = orange.ExamplesDistanceConstructor_Euclidean(data)
        matrix = orange.SymMatrix(len(data))
        matrix.setattr('items', data)
        for i in range(len(data)):
            for j in range(i+1):
                matrix[i, j] = dist(data[i], data[j])
        return matrix

    import orange
    a = QApplication(sys.argv)
    ow = OWDistanceMap()
    ow.show()

    data = orange.ExampleTable(r'../../doc/datasets/iris.tab')
    data = data.select(orange.MakeRandomIndices2(p0=20)(data), 0)
    for d in data:
        d.name = str(d["sepal length"])
    matrix = distanceMatrix(data)
    ow.setMatrix(matrix)

    a.exec_()

    ow.saveSettings()

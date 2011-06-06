"""
<name>Distance Map</name>
<description>Displays distance matrix as a heat map.</description>
<icon>icons/DistanceMap.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>1200</priority>
"""
from __future__ import with_statement
import orange, math, sys
import OWGUI, OWToolbars
from OWWidget import *

from ColorPalette import *
from OWDlgs import OWChooseImageSizeDlg
import OWColorPalette
import OWToolbars
#from OWHierarchicalClustering import recursion_limit

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
#        self.setRenderHints(QPainter.Antialiasing)
        self.viewport().setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

class EventfulGraphicsScene(QGraphicsScene):
    def __init__(self, master):
        QGraphicsScene.__init__(self)
        self.master = master

    def mousePressEvent (self, event):
        if self.master.matrix:
            self.master.mousePress(event.scenePos().x(), event.scenePos().y())

    def mouseReleaseEvent (self, event):
        if self.master.matrix:
            self.master.mouseRelease(event.scenePos().x(), event.scenePos().y())

    def mouseMoveEvent (self, event):
        if self.master.matrix:
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
        OWWidget.__init__(self, parent, signalManager, 'Distance Map', wantGraph=True)

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
        self.CellWidth = 15
        self.CellHeight = 15
        self.Merge = 1
        self.savedMerge = self.Merge
        self.Gamma = 1
        self.Grid = 1
        self.savedGrid = 1
        self.CutLow = 0
        self.CutHigh = 0
        self.CutEnabled = 0
        self.Sort = 0
        self.SquareCells = 0
        self.ShowLegend = 1
        self.ShowLabels = 1
        self.ShowBalloon = 1
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
        box = OWGUI.widgetBox(tab, "Cell Size (Pixels)")
        OWGUI.qwtHSlider(box, self, "CellWidth", label='Width: ',
                         labelWidth=38, minValue=1, maxValue=self.maxHSize,
                         step=1, precision=0,
                         callback=[lambda f="CellWidth", t="CellHeight": self.adjustCellSize(f,t),
                                   self.drawDistanceMap,
                                   self.manageGrid])
        OWGUI.qwtHSlider(box, self, "CellHeight", label='Height: ',
                         labelWidth=38, minValue=1, maxValue=self.maxVSize,
                         step=1, precision=0,
                         callback=[lambda f="CellHeight", t="CellWidth": self.adjustCellSize(f,t),
                                   self.drawDistanceMap,
                                   self.manageGrid])
        OWGUI.checkBox(box, self, "SquareCells", "Cells as squares",
                         callback = [self.setSquares, self.drawDistanceMap])
        self.gridChkBox = OWGUI.checkBox(box, self, "Grid", "Show grid",
                                         callback = self.createDistanceMap,
                                         disabled=lambda: min(self.CellWidth, self.CellHeight) <= c_smallcell)

        OWGUI.separator(tab)
        OWGUI.qwtHSlider(tab, self, "Merge", box="Merge" ,label='Elements:', labelWidth=50,
                         minValue=1, maxValue=100, step=1,
                         callback=self.createDistanceMap, ticks=0)
        
        OWGUI.separator(tab)
        self.labelCombo = OWGUI.comboBox(tab, self, "Sort", box="Sort",
                         items=[x[0] for x in self.sorting],
                         tooltip="Sorting method for items in distance matrix.",
                         callback=self.sortItems)
        OWGUI.rubber(tab)

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
        OWGUI.button(box, self, "Edit colors", callback=self.openColorDialog, debuggingEnabled = 0)
        OWGUI.rubber(tab)

        self.setColor(self.selectedSchemaIndex)

        # INFO TAB
        tab = OWGUI.createTabPage(self.tabs, "Info")
        box = OWGUI.widgetBox(tab, "Annotation && Legends")
        OWGUI.checkBox(box, self, 'ShowLegend', 'Show legend',
                       callback=self.drawDistanceMap)
        OWGUI.checkBox(box, self, 'ShowLabels', 'Show labels',
                       callback=self.drawDistanceMap)

        box = OWGUI.widgetBox(tab, "Tool Tips")
        OWGUI.checkBox(box, self, 'ShowBalloon', "Show tool tips")
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

        self.resize(700,400)

        self.scene = EventfulGraphicsScene(self)
        self.sceneView = EventfulGraphicsView(self.scene, self.mainArea, self)
        self.mainArea.layout().addWidget(self.sceneView)

        #construct selector
        self.selector = QGraphicsRectItem(0, 0, self.CellWidth, self.CellHeight, None, self.scene)
        color = self.cellOutlineColor
        self.selector.setPen(QPen(self.qrgbToQColor(color),v_sel_width))
        self.selector.setZValue(20)

        self.selection = SelectionManager()

        self.selectionLines = []
        self.annotationText = []
        self.clusterItems = []
        self.selectionRects = []

        self.legendText1 = QGraphicsSimpleTextItem(None, self.scene)
        self.legendText2 = QGraphicsSimpleTextItem(None, self.scene)

        self.errorText = QGraphicsSimpleTextItem("Bitmap is too large.", None, self.scene)
        self.errorText.setPos(10,10)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), lambda:OWChooseImageSizeDlg(self.scene, parent=self).exec_())

        self._clustering_cache = {}

    def sendReport(self):
        self.reportSettings("Data",
                            [("Matrix dimension", self.matrix.dim)])
        self.reportSettings("Settings",
                            [("Merge", "%i elements in a cell" % self.Merge if self.Merge > 1 else "none"),
                             ("Sorting", self.sorting[self.Sort][0].lower()),
                             ("Thresholds", "low %.1f, high %.1f" % (self.CutLow, self.CutHigh) if self.CutEnabled else "none"),
                             ("Gamma", "%.2f" % self.Gamma)])
        self.reportRaw("<br/>")
        buffer = QPixmap(self.scene.width(), self.scene.height())
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        self.scene.render(painter)
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))

                             
        
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
                x1, y1 = sel.x(), sel.y()
                x2, y2 = x1 + sel.width(), y1 + sel.height()
                 
                if (len(self.distanceMap.elementIndices)==0):
                    tmp += range(x1, x2)
                    tmp += range(y1, y2)
                else:   
                    tmp += range(self.distanceMap.elementIndices[x1], self.distanceMap.elementIndices[x2])
                    tmp += range(self.distanceMap.elementIndices[y1], self.distanceMap.elementIndices[y2])

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

    def getGammaCorrectedPalette(self):
        return [QColor(*self.contPalette.getRGB(float(i)/250, gamma=self.Gamma)).rgb() for i in range(250)] + self.palette[-6:]

    def setColor(self, index=0):
        self.selectedSchemaIndex = index
        dialog = self.createColorDialog()
        self.colorCombo.setPalettes("palette", dialog)
        self.colorCombo.setCurrentIndex(self.selectedSchemaIndex)
        
        self.contPalette = palette = dialog.getExtendedContinuousPalette("palette")
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
            self.sortItems()
            self.createDistanceMap()

    def createDistanceMap(self):
        """creates distance map objects"""
        if not self.matrix:
            return
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
        
        self.clearScene()

        if self.matrix.dim * max(int(self.CellWidth), int(self.CellHeight)) > 32767:
            self.errorText.show()
            return

        self.errorText.hide()

        lo = self.CutEnabled and self.CutLow or round(self.lowerBound, 1) + \
            (-0.1 if round(self.lowerBound, 1) > self.lowerBound else 0)
        hi = self.CutEnabled and self.CutHigh or round(self.upperBound, 1) + \
            (0.1 if round(self.upperBound, 1) < self.upperBound else 0)

        self.offsetX = 5

        if self.distanceImage:
            self.scene.removeItem(self.distanceImage)

        if self.legendImage:
            self.scene.removeItem(self.legendImage)

        if self.ShowLegend==1:
            self.offsetY = v_legend_height + 30
        else:
            self.offsetY = 5

        palette = self.getGammaCorrectedPalette() #self.palette
        bitmap, width, height = self.distanceMap.getBitmap(int(self.CellWidth),
                            int(self.CellHeight), lo, hi, 1.0, self.Grid)
##                            int(self.CellHeight), lo, hi, self.Gamma, self.Grid)

        for tmpText in self.annotationText:
            self.scene.removeItem(tmpText)

        for cluster in self.clusterItems:
            self.scene.removeItem(cluster)

        if self.rootCluster and self.order:
#            from OWClustering import HierarchicalClusterItem
#            with recursion_limit(len(self.rootCluster) * 3 + 20 + sys.getrecursionlimit()): #extend the recursion limit
#                clusterTop = HierarchicalClusterItem(self.rootCluster, None, None)
#                clusterLeft = HierarchicalClusterItem(self.rootCluster, None, None)
#                self.scene.addItem(clusterTop)
#                self.scene.addItem(clusterLeft)
            from OWClustering import DendrogramWidget
            clusterTop = DendrogramWidget(self.rootCluster, orientation=Qt.Horizontal, scene=self.scene)
            clusterLeft = DendrogramWidget(self.rootCluster, orientation=Qt.Horizontal, scene=self.scene)
            
            margin = self.CellWidth / 2.0 / self.Merge
            
            clusterLeft.layout().setContentsMargins(margin, 0, margin, 10)
            clusterTop.layout().setContentsMargins(margin, 0, margin, 10)
            
            clusterHeight = 100.0
#            clusterTop.resize(width, clusterHeight)
            clusterTop.setMinimumSize(QSizeF(width, clusterHeight))
            clusterTop.setMaximumSize(QSizeF(width, clusterHeight))
            clusterTop.scale(1.0, -1.0)
#            clusterTop.setSize(width, -clusterHeight)
#            clusterLeft.setSize(height, clusterHeight)
#            clusterLeft.resize(height, clusterHeight)
            clusterLeft.setMinimumSize(QSizeF(height, clusterHeight))
            clusterLeft.setMaximumSize(QSizeF(height, clusterHeight))
            
            clusterTop.setPos(0 + self.CellWidth / 2.0, self.offsetY + clusterHeight)


            clusterLeft.rotate(90)
            clusterLeft.setPos(self.offsetX + clusterHeight, 0 + self.CellHeight / 2.0)
            self.offsetX += clusterHeight + 10
            self.offsetY += clusterHeight + 10

            self.clusterItems += [clusterTop, clusterLeft]

            clusterTop.show()
            clusterLeft.show()
            
#            anchors = list(clusterLeft.leaf_anchors())
#            minx = min([a.x() for a in anchors])
#            maxx = max([a.x() for a in anchors])
#            print maxx - minx, width
            

        # determine the font size to fit the cell width
        fontrows = self.getfont(self.CellHeight)
        fontcols = self.getfont(self.CellWidth)
        
        fontmetrics_row = QFontMetrics(fontrows)
        fontmetrics_col = QFontMetrics(fontcols)
    
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
            exampleTableHasNames = type(items) == orange.ExampleTable and any(ex.name for ex in items)
            for i in range(0, len(indices)):
                text = items[indices[i]]
                
                if type(text) not in [str, unicode]:
                    if exampleTableHasNames or isinstance(text, orange.Variable): 
                        text = text.name
                    else:
                        text = repr(text)
                        
                if text != "":
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
            
            fix = 0.0 
                        
            for i in range(0, len(self.annotationText)/2):
##                self.annotationText[i*2].setX(self.offsetX + maxWidth + 3 + (i+0.5)*self.CellWidth)
##                self.annotationText[i*2].setY(self.offsetY)

                self.annotationText[i*2].setPos(self.offsetX + maxWidth + 3 + (i+0.5)*self.CellWidth + \
                                                fontmetrics_row.height()/2.0 * fix,
                                                self.offsetY + maxWidth)
##                self.annotationText[i*2 + 1].setX(self.offsetX)
##                self.annotationText[i*2 + 1].setY(self.offsetY + maxHeight + 3 + (i+0.5)*self.CellHeight)
##                self.annotationText[i*2 + 1].setPos(self.offsetX, self.offsetY + maxHeight + 3 + (i+0.5)*self.CellHeight)
                self.annotationText[i*2 + 1].setPos(self.offsetX, self.offsetY + maxWidth + 3 + (i+0.5)*self.CellHeight +\
                                                    fontmetrics_col.height()/2.0 * fix)

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
#            clusterTop.setPos(self.offsetX + self.CellWidth/2.0/self.Merge, clusterTop.y())
#            clusterLeft.setPos(clusterLeft.x(), self.offsetY + self.CellHeight/2.0/self.Merge)
            clusterTop.setPos(self.offsetX , clusterTop.y())
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
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-5, -5, 5, 5))
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
        return QFont("", max(min(height, 16), 2))
        
    def updateSelectionRect(self):
        selections = self.selection.getSelection()
        for rect in self.selectionRects:
            self.scene.removeItem(rect)
        self.selectionRects = []
            
        color = self.selectionColor
        pen = QPen(self.qrgbToQColor(color),v_sel_width)
        pen.setCosmetic(True)
        for selection in selections:
            rect = QGraphicsRectItem(QRectF(selection), self.distanceImage, self.scene)
            rect.setPen(pen)
            rect.setBrush(QBrush(Qt.NoBrush))
            rect.setZValue(20)
            rect.scale(self.CellWidth, self.CellHeight)
            self.selectionRects.append(rect)

    def clearScene(self):
        if self.distanceImage:
            self.scene.removeItem(self.distanceImage)
            self.distanceImage = None
            self.selectionRects = [] # selectionRects are children of distanceImage
            
        if self.legendImage:
            self.scene.removeItem(self.legendImage)
            self.legendImage = None
            
        for tmpText in getattr(self, "annotationText", []):
            self.scene.removeItem(tmpText)
            
        self.annotationText = []

        for cluster in getattr(self, "clusterItems", []):
            self.scene.removeItem(cluster)
            
        self.clusterItems = []
            
        for line in self.selectionLines:
            self.scene.removeItem(line)
            
        self.selectionLines = []
        
        self.selector.hide()
        self.errorText.hide()
        self.legendText1.hide()
        self.legendText2.hide()
        
        
        self.scene.setSceneRect(QRectF(0, 0, 10, 10))
        
    def mouseMove(self, event):
        x, y = event.scenePos().x(), event.scenePos().y()
        row = self.rowFromMousePos(x,y)
        col = self.colFromMousePos(x,y)
        if (self.clicked==True):
            self.selection.UpdateSel(col, row)

        if (row==-1 or col==-1):
            self.selector.hide()
        else:
            bb = self.selector.sceneBoundingRect()
            self.selector.setPos(self.offsetX + col * self.CellWidth, self.offsetY + row * self.CellHeight)
            self.selector.show()
            self.scene.update(bb) #Update the old boundingRect (left unchanged when scrolling the view. Why?)

            if self.ShowBalloon == 1:

                i = self.getItemFromPos(col)
                j = self.getItemFromPos(row)
                
                head = str(self.matrix[i, j])

                if (self.ShowItemsInBalloon == 1):
                    namei, namej = self.matrix.items[i], self.matrix.items[j]
                    if type(namei) not in [str, unicode]:
                        namei = namei.name
                    if type(namej) not in [str, unicode]:
                        namej = namej.name
                    if namei or namej:
                        body = namei + "\n" + namej
                    else:
                        body = ""
                else:
                    body = ""

                QToolTip.showText(event.screenPos(), "")
                QToolTip.showText(event.screenPos(), "%s\n%s" % (head, body))

            self.updateSelectionRect()

#        self.scene.update()

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
        cluster = self._clustering_cache.get("sort clustering", None)
        if cluster is None:
            cluster = orange.HierarchicalClustering(self.matrix,
                linkage=orange.HierarchicalClustering.Average)
            # Cache the cluster
            self._clustering_cache["sort clustering"] = cluster
        self.rootCluster = cluster
        self.order = list(self.rootCluster.mapping)
        
    def sortClusteringOrdered(self):
        cluster = self._clustering_cache.get("sort ordered clustering", None)
        if cluster is None:
            cluster = orange.HierarchicalClustering(self.matrix,
                linkage=orange.HierarchicalClustering.Average)
            import orngClustering
            self.progressBarInit()
            orngClustering.orderLeaves(cluster, self.matrix, self.progressBarSet)
            self.progressBarFinished()
            # Cache the cluster
            self._clustering_cache["sort ordered clustering"] = cluster
        self.rootCluster = cluster
        self.order = list(self.rootCluster.mapping)

    def sortItems(self):
        if not self.matrix:
            return
        self.sorting[self.Sort][1]()
        self.createDistanceMap()

    def setMatrix(self, matrix):
        self.send("Examples", None)
        self.send("Attribute List", None)
        self._clustering_cache.clear()

        if not matrix:
            self.matrix = None
            self.clearScene()
        self.sceneView.viewport().setMouseTracking(bool(matrix))

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
        
class ImageItem(QGraphicsPixmapItem):
    def __init__(self, bitmap, scene, width, height, palette, depth=8, numColors=256, x=0, y=0, z=0):
        image = QImage(bitmap, width, height, QImage.Format_Indexed8)
        image.bitmap = bitmap # this is tricky: bitmap should not be freed, else we get mess. hence, we store it in the object
        if qVersion() <= "4.5":
            image.setColorTable(signedPalette(palette))
        else:
            image.setColorTable(palette)
        pixmap = QPixmap.fromImage(image)
        QGraphicsPixmapItem.__init__(self, pixmap, None, scene)
        self.setPos(x, y)
        self.setZValue(z)
#        self.show()

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
        self.currSel = QRect(x, y, 1, 1)
        self.currSelEnd = QRect(x, y, 1, 1)
        self.selecting = True

    def UpdateSel(self, x, y):
        self.currSelEnd = QRect(x, y, 1, 1)

    def CancelSel(self):
        self.selecting = False
        
    def SelEnd(self):
        self.selection += [self.currSel.united(self.currSelEnd).normalized()]
        self.selecting = False

    def clear(self):
        self.selection = []

    def undo(self):
        if len(self.selection) > 0:
            del self.selection[len(self.selection)-1]
    
    def getSelection(self):
        res = list(self.selection)
        if self.selecting == True:
            res += [self.currSel.united(self.currSelEnd).normalized()]
        return res

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
    ow.setMatrix(None)
    ow.setMatrix(matrix)
    a.exec_()

    ow.saveSettings()

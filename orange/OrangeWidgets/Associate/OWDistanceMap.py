"""
<name>Distance Map</name>
<description>Displays distance matrix as a heat map.</description>
<category>Test</category>
<icon>icons/DistanceMap.png</icon>
<priority>1500</priority>
"""

import orange, math
import OWGUI, OWToolbars
from qt import *
from qtcanvas import *
from OWWidget import *

##############################################################################
# parameters that determine the canvas layout

c_offsetX = 10; c_offsetY = 10  # top and left border
c_spaceX = 10; c_spaceY = 10    # space btw graphical elements
c_legendHeight = 15             # height of the legend
c_averageStripeWidth = 12       # width of the stripe with averages

##############################################################################
# main class

class OWDistanceMap(OWWidget):	
    settingsList = ["CellWidth", "CellHeight", "Merge", "Gamma", "CutLow", "CutHigh", "CutEnabled", "Sort",
                    "ShowLegend", "ShowAnnotations", "ShowBalloon", "ShowItemsInBalloon", "SendOnRelease"]

    def __init__(self, parent=None, name='DistanceMap'):
        self.callbackDeposit = [] # deposit for OWGUI callback function
        OWWidget.__init__(self, parent, name, 'Distance Map', FALSE, FALSE) 

        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setMatrix, 1)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass),
                        ("Attribute List", orange.VarList)]
        
        #set default settings
        self.CellWidth = 3; self.CellHeight = 3
        self.Merge = 1; self.savedMerge = self.Merge
        self.Gamma = 1
        self.Grid = 1
        self.CutLow = 0; self.CutHigh = 0; self.CutEnabled = 0
        self.Sort = 0
        self.setColorPalette()
        self.SquareCells = 0
        self.ShowLegend = 1; self.ShowAnnotations = 1; self.ShowBalloon = 1; self.ShowItemsInBalloon = 1
        self.SendOnRelease = 1
        
        self.loadSettings()
        
        self.maxHSize = 30; self.maxVSize = 30
        self.sorting = [("None", self.sortNone), ("Adjacent distance", self.sortAdjDist), ("Random", self.sortRandom)]

        self.matrix = self.order = None
        
        # GUI definition
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')

        # SETTINGS TAB
        tab = QVGroupBox(self)
        box = QVButtonGroup("Cell Size (Pixels)", tab)
        OWGUI.qwtHSlider(box, self, "CellWidth", label='Width: ', labelWidth=38, minValue=1, maxValue=self.maxHSize, step=1, precision=0, callback=self.drawDistanceMap)
        self.sliderVSize = OWGUI.qwtHSlider(box, self, "CellHeight", label='Height: ', labelWidth=38, minValue=1, maxValue=self.maxVSize, step=1, precision=0, callback=self.createDistanceMap)
        OWGUI.checkBox(box, self, "SquareCells", "Cells as squares", callback = None)
        OWGUI.checkBox(box, self, "Grid", "Show grid", callback = self.createDistanceMap)
        
        OWGUI.qwtHSlider(tab, self, "Gamma", box="Gamma", minValue=0.1, maxValue=1, step=0.1, callback=self.drawDistanceMap)

        # define the color stripe to show the current palette
        colorItems = [self.createColorStripe(i) for i in range(len(self.ColorPalettes))]
        palc = OWGUI.comboBox(tab, self, "CurrentPalette", box="Colors", items=None, tooltip=None, callback=self.setColor)
        for cit in colorItems:
            palc.insertItem(cit) ## because of a string cast in the comboBox constructor

        self.tabs.insertTab(tab, "Settings")

        # FILTER TAB
        tab = QVGroupBox(self)
        box = QVButtonGroup("Threshold Values", tab)
        OWGUI.checkBox(box, self, 'CutEnabled', "Enabled", callback=self.setCutEnabled)
        self.sliderCutLow = OWGUI.qwtHSlider(box, self, 'CutLow', label='Low:', labelWidth=33, minValue=-100, maxValue=0, step=0.1, precision=1, ticks=0, maxWidth=80, callback=self.drawDistanceMap)
        self.sliderCutHigh = OWGUI.qwtHSlider(box, self, 'CutHigh', label='High:', labelWidth=33, minValue=0, maxValue=100, step=0.1, precision=1, ticks=0, maxWidth=80, callback=self.drawDistanceMap)
        if not self.CutEnabled:
            self.sliderCutLow.box.setDisabled(1)
            self.sliderCutHigh.box.setDisabled(1)

        box = QVButtonGroup("Merge", tab)
        OWGUI.qwtHSlider(box, self, "Merge", label='Elements:', labelWidth=50, minValue=1, maxValue=100, step=1, callback=self.createDistanceMap, ticks=0)
        self.labelCombo = OWGUI.comboBox(tab, self, "Sort", box="Sort", items=[x[0] for x in self.sorting],
                                         tooltip="Choose method to sort items in distance matrix.", callback=self.sortItems)
        self.tabs.insertTab(tab, "Filter")

        # INFO TAB
        tab = QVGroupBox(self)
        box = QVButtonGroup("Annotation && Legends", tab)
        OWGUI.checkBox(box, self, 'ShowLegend', 'Show legend', callback=None)
        OWGUI.checkBox(box, self, 'ShowAnnotations', 'Show annotations', callback=None)
        
        box = QVButtonGroup("Balloon", tab)
        OWGUI.checkBox(box, self, 'ShowBalloon', "Show balloon", callback=None)
        OWGUI.checkBox(box, self, 'ShowItemsInBalloon', "Display item names", callback=None)

        box = QVButtonGroup("Select", tab)
        QLabel("BUTTONS HERE", box)
        QLabel("BUTTONS HERE", box)
        OWGUI.checkBox(box, self, 'SendOnRelease', "Send after mouse release", callback=None)

        self.tabs.insertTab(tab, "Info")
        
        self.resize(700,400)

        self.layout = QVBoxLayout(self.mainArea)
        self.canvas = QCanvas()
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        self.layout.add(self.canvasView)

    def createColorStripe(self, palette):
        dx = 104; dy = 18
        bmp = chr(252)*dx*2 + reduce(lambda x,y:x+y, [chr(i*250/dx) for i in range(dx)] * (dy-4)) + chr(252)*dx*2 
        image = QImage(bmp, dx, dy, 8, self.ColorPalettes[palette], 256, QImage.LittleEndian)
        pm = QPixmap()
        pm.convertFromImage(image, QPixmap.Color);
        return pm

    # set the default palettes used in the program
    # palette defines 256 colors, 250 are used for distance map, remaining 6 are extra
    # color indices for unknown is 255, underflow 253, overflow 254, white 252
    def setColorPalette(self):
        white = qRgb(255,255,255)
        gray = qRgb(200,200,200)
        red = qRgb(255, 0, 0)
        green = qRgb(0, 255, 0)
        self.ColorPalettes = \
          ([qRgb(255.*i/250., 255.*i/250., 255-(255.*i/250.)) for i in range(250)] + [white]*3 + [red, green, gray],
           [qRgb(0, 255.*i*2/250., 0) for i in range(125, 0, -1)] + [qRgb(255.*i*2/250., 0, 0) for i in range(125)] + [white]*3 + [qRgb(0, 255., 0), qRgb(255., 0, 0), gray],
           [qRgb(255.*i/250., 0, 0) for i in range(250)] + [white]*3 + [qRgb(0., 0, 0), qRgb(255., 0, 0), gray])
#          ([qRgb(255.*i/250., 255.*i/250., 255-(255.*i/250.)) for i in range(250)] + [white]*3 + [qRgb(0., 0., 255.), qRgb(255., 255., 0.), gray],
        self.SelectionColors = [QColor(0,0,0), QColor(255,255,128), QColor(0,255,255)]
        self.CurrentPalette = 0

    ##########################################################################
    # callbacks (rutines called after some GUI event, like click on a button)

    def setColor(self):
        if self.CurrentPalette == len(self.ColorPalettes):
            self.CurrentPalette = 0
            # put a code here that allows to define ones own colors
        else:
            pm = self.createColorStripe(self.CurrentPalette)
            self.drawDistanceMap()

    def setCutEnabled(self):
        self.sliderCutLow.box.setDisabled(not self.CutEnabled)
        self.sliderCutHigh.box.setDisabled(not self.CutEnabled)
        self.drawDistanceMap()

    def constructDistanceMap(self):
        if self.matrix:
            self.distanceMapConstructor = orange.DistanceMapConstructor(distanceMatrix = self.matrix)
            self.createDistanceMap()

    def createDistanceMap(self):
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
        self.drawDistanceMap()

    def drawDistanceMap(self):
        lo = self.CutEnabled and self.CutLow   or self.lowerBound
        hi = self.CutEnabled and self.CutHigh  or self.upperBound

        for i in self.canvas.allItems():
            i.setCanvas(None)

        palette = self.ColorPalettes[self.CurrentPalette]
        bitmap, width, height = self.distanceMap.getBitmap(int(self.CellWidth), int(self.CellHeight), lo, hi, self.Gamma, self.Grid)

        self.canvas.resize(2000, 2000) # this needs adjustment
        x = c_offsetX; y0 = c_offsetY

        image = ImageItem(bitmap, self.canvas, width, height, palette, x=0, y=0, z=0)
        image.height = height
        image.width = width

        self.canvas.update()

    ##########################################################################
    # input signal management

    def sortNone(self):
        self.order = None

    def sortAdjDist(self):
        self.order = None

    def sortRandom(self):
        import random
        self.order = range(len(self.matrix.items))
        random.shuffle(self.order)

    def sortItems(self):
        if not self.matrix:
            return
        self.sorting[self.Sort][1]()
        self.createDistanceMap()

    ##########################################################################
    # input signal management

    def setMatrix(self, matrix):
        if not matrix:
            # should remove the data where necessary
            return
        # check if the same length
        self.matrix = matrix
        self.constructDistanceMap()


        # following is just to demonstrate some points
        # REMOVE FROM THE FINAL VERSION
        
        # just a try: print out the first 10 labels of the items
        print 'first 5 of %d matrix labels:' % len(self.matrix.items),
        print reduce(lambda x,y: x+','+y, [self.matrix.items[x].name for x in range(5)])

        if len(self.matrix.items)<2:
            return
        items = self.matrix.items
        print items[0]
        print type(items[0])
        if issubclass(orange.EnumVariable, type(items[0])):
            print 'Attributes, first one:', items[0].name
            selected = orange.VarList()
            for i in range(2):
                selected.append(items[i])
            self.send("Attribute List", selected)
            print 'send', selected
            
        elif isinstance(items[0], orange.Example):
            print 'Examples'
            ex = [items[x] for x in range(2)]
            selected = orange.ExampleTable(items[0].domain, ex)
            self.send("Examples", selected)
            print "send", selected
            if selected.domain.classVar:
                self.send("Classified Examples", selected)
        

##################################################################################################
# new canvas items

class ImageItem(QCanvasRectangle):
    def __init__(self, bitmap, canvas, width, height, palette, depth=8, numColors=256, x=0, y=0, z=0):
        QCanvasRectangle.__init__(self, canvas)
        self.image = QImage(bitmap, width, height, depth, palette, numColors, QImage.LittleEndian)
        self.image.bitmap = bitmap # this is tricky: bitmap should not be freed, else we get mess. hence, we store it in the object
        self.canvas = canvas
        self.setSize(width, height)
        self.setX(x); self.setY(y); self.setZ(z)
        self.show()

    def drawShape(self, painter):
        painter.drawImage(self.x(), self.y(), self.image, 0, 0, -1, -1)
        
##################################################################################################
# test script

if __name__=="__main__":

    def computeMatrix(data):
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
    a.setMainWidget(ow)

    ow.show()

    data = orange.ExampleTable(r'../../doc/datasets/glass')
    
    matrix = computeMatrix(data)
    ow.setMatrix(matrix)

    a.exec_loop()

    ow.saveSettings()

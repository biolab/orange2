"""
<name>Mosaic Display</name>
<description>Show mosaic display</description>
<category>Visualization</category>
<icon>icons/MosaicDisplay.png</icon>
<priority>4200</priority>
"""
# OWMosaicDisplay.py
#
# 

from OWWidget import *
from qt import *
from qtcanvas import *
import orngInteract
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct
from copy import copy


###########################################################################################
##### WIDGET : 
###########################################################################################
class OWMosaicDisplay(OWWidget):
    settingsList = ["showLines"]
    #settingsList = ["showLines", "kvoc"]
    #kvocList = ['1.5','2','3','5','10','20']
    #kvocNums = [1.5,   2,  3,  5,  10,  20]

    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Mosaic display", 'show Mosaic display', FALSE, TRUE, icon = "MosaicDisplay.png")

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []
        self.lines = []
        self.tooltips = []

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, 1)]
        self.outputs = []
    
        #load settings
        self.showLines = 1
        self.cellspace = 6
        self.loadSettings()

        # add a settings dialog and initialize its values
        #self.options = OWInteractionGraphOptions()

        self.box = QVBoxLayout(self.mainArea)
        self.canvas = QCanvas(2000, 2000)
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        self.box.addWidget(self.canvasView)
        self.canvasView.show()
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        
        #GUI
        #add controls to self.controlArea widget
        """
        self.shownCriteriaGroup = QVGroupBox(self.controlArea)
        self.shownCriteriaGroup.setTitle("Shown criteria")
        self.criteriaCombo = QComboBox(self.shownCriteriaGroup)
        self.connect(self.criteriaCombo, SIGNAL('activated ( const QString & )'), self.updateData)
        """
        
        self.attrSelGroup = QVGroupBox(self.controlArea)
        self.attrSelGroup.setTitle("Shown attributes")

        self.attr1Group = QVButtonGroup("1st attribute", self.attrSelGroup)
        self.attr1 = QComboBox(self.attr1Group)
        self.connect(self.attr1, SIGNAL('activated ( const QString & )'), self.updateData)

        self.attr2Group = QVButtonGroup("2nd attribute", self.attrSelGroup)
        self.attr2 = QComboBox(self.attr2Group)
        self.connect(self.attr2, SIGNAL('activated ( const QString & )'), self.updateData)

        self.attr3Group = QVButtonGroup("3rd attribute", self.attrSelGroup)
        self.attr3 = QComboBox(self.attr3Group)
        self.connect(self.attr3, SIGNAL('activated ( const QString & )'), self.updateData)

        self.attr4Group = QVButtonGroup("4th attribute", self.attrSelGroup)
        self.attr4 = QComboBox(self.attr4Group)
        self.connect(self.attr4, SIGNAL('activated ( const QString & )'), self.updateData)        

        self.showLinesCB = QCheckBox('Show lines', self.controlArea)
        self.connect(self.showLinesCB, SIGNAL("toggled(bool)"), self.updateData)

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)

        #self.resize(800, 480)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

    def activateLoadedSettings(self):
        self.showLinesCB.setChecked(self.showLines)

        
        
    ##################################################
    # initialize combo boxes with discrete attributes
    def initCombos(self, data):
        self.attr1.clear(); self.attr2.clear(); self.attr3.clear(); self.attr4.clear()

        if data == None: return

        self.attr3.insertItem("(none)")
        self.attr4.insertItem("(none)")

        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                self.attr1.insertItem(attr.name)
                self.attr2.insertItem(attr.name)
                self.attr3.insertItem(attr.name)
                self.attr4.insertItem(attr.name)

        if self.attr1.count() > 0:
            self.attr1.setCurrentItem(0)
        if self.attr2.count() > 1:
            self.attr2.setCurrentItem(1)
        self.attr3.setCurrentItem(0)
        self.attr4.setCurrentItem(0)

    ######################################################################
    ##  when we resize the widget, we have to redraw the data
    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateData()

    ######################################################################
    ## VIEW signal
    def view(self, (attr1, attr2)):
        if self.data == None:
            return

        ind1 = 0; ind2 = 0; classInd = 0
        for i in range(self.attr1.count()):
            if str(self.attr1.text(i)) == attr1: ind1 = i
            if str(self.attr1.text(i)) == attr2: ind2 = i

        if ind1 == ind2 == 0:
            print "no valid attributes found"
            return    # something isn't right

        self.attr1.setCurrentItem(ind1)
        self.attr2.setCurrentItem(ind2)
        self.updateData()

    ######################################################################
    ## DATA signal
    # receive new data and update all fields
    def cdata(self, data):
        self.data = None
        if data:
            self.data = orange.Preprocessor_dropMissing(data)
        self.initCombos(self.data)
        self.updateData()


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        # hide all rectangles
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for line in self.lines: line.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.lines = []; self.tooltips = []
        
        if self.data == None : return

        self.showLines = self.showLinesCB.isOn()
        #self.kvoc = float(str(self.kvocCombo.currentText()))

        attrList = [str(self.attr1.currentText()), str(self.attr2.currentText())]
        if str(self.attr3.currentText()) != "(none)": attrList.append(str(self.attr3.currentText()))
        if str(self.attr4.currentText()) != "(none)": attrList.append(str(self.attr4.currentText()))

        # get the maximum width of rectangle
        text = QCanvasText(self.data.domain[attrList[1]].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        width = text.boundingRect().right() - text.boundingRect().left() + 30 + 20
        xOff = width
        if len(attrList) == 4:
            text = QCanvasText(self.data.domain[attrList[3]].name, self.canvas);
            font = text.font(); font.setBold(1); text.setFont(font)
            width += text.boundingRect().right() - text.boundingRect().left() + 30 + 20
        
        # get the maximum height of rectangle        
        height = 90
        yOff = 40
        squareSize = min(self.canvasView.size().width() - width, self.canvasView.size().height() - height)
        if squareSize < 0: return    # canvas is too small to draw rectangles

        self.legend = {}        # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList: self.legend[attr] = 0
        self.DrawData(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 1)
        self.DrawText(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize))
       
        self.canvas.update()

    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, data, attrList, (x0, x1), (y0, y1)):
        # save values for all attributes
        values = []
        for attr in attrList:
            vals = data.domain[attr].values
            values.append(vals)

        width = x1-x0 - self.cellspace*len(attrList)*(len(values[0])-1)
        height = y1-y0 - self.cellspace*(len(attrList)-1)*(len(values[1])-1)
        #print x1-x0, width, y1-y0, height
        
        #calculate position of first attribute
        self.addText(attrList[0], x0+(x1-x0)/2, y1+30, Qt.AlignCenter, 1)
        currPos = 0        
        for val in values[0]:
            tempData = data.select({attrList[0]:val})
            perc = float(len(tempData))/float(len(data))
            self.addText(str(val), x0+currPos+(x1-x0)*0.5*perc, y1 + 15, Qt.AlignCenter, 1)
            currPos += perc*width + self.cellspace*len(attrList)

        #calculate position of second attribute
        self.addText(attrList[1], x0-30, y0+(y1-y0)/2, Qt.AlignRight + Qt.AlignVCenter, 1)
        currPos = 0
        tempData = data.select({attrList[0]:values[0][0]})
        for val in list(values[1])[::-1]:
            tempData2 = tempData.select({attrList[1]:val})
            perc = float(len(tempData2))/float(len(tempData))
            self.addText(str(val), x0-10, y0+currPos+(y1-y0)*0.5*perc, Qt.AlignRight + Qt.AlignVCenter, 1)
            currPos += perc*height + self.cellspace*(len(attrList)-1)

        if len(attrList) < 3: return

        #calculate position of third attribute
        self.addText(attrList[2], x0 + width*float(len(tempData))/float(2*len(data)), y0 - 25, Qt.AlignCenter, 1)
        currPos = 0
        tempData2 = data.select({attrList[0]:values[0][0], attrList[1]:values[1][0]})
        width2 = width*float(len(tempData))/float(len(data)) - (len(values[2])-1)*self.cellspace*(len(attrList)-2)
        if len(tempData2) == 0: print "Error: Division by zero"; return
        for val in values[2]:
            tempData3 = tempData2.select({attrList[2]:val})
            perc = (float(len(tempData3))/float(len(tempData2)))#*(float(len(tempData))/float(len(data)))
            #print perc
            self.addText(str(val), x0+currPos+width2*perc*0.5, y0 - 10, Qt.AlignCenter, 1)
            currPos += perc*width2 + self.cellspace*(len(attrList)-2)

        if len(attrList) < 4: return

        #calculate position of fourth attribute
        tempData0 = data.select({attrList[0]:values[0][len(values[0])-1]})
        tempData1 = tempData0.select({attrList[1]:values[1][0]})
        tempData2 = tempData1.select({attrList[2]:values[2][len(values[2])-1]})
        if len(tempData0) == 0: print "Error: Division by zero"; return
        self.addText(attrList[3], x1 + 30, y0+ float(height*len(tempData1))/float(2*len(tempData0)), Qt.AlignLeft + Qt.AlignVCenter, 1)
        currPos = 0
        height2 = height * float(len(tempData1))/float(len(tempData0)) - self.cellspace*(len(values[3])-1)
        if len(tempData2) == 0: print "Error: Division by zero"; return
        for val in list(values[3])[::-1]:
            tempData3 = tempData2.select({attrList[3]:val})
            perc = float(len(tempData3))/float(len(tempData2)) #* (float(len(tempData1))/float(len(tempData0)))
            self.addText(str(val), x1+10, y0 + currPos + height2*0.5*perc, Qt.AlignLeft + Qt.AlignVCenter, 1)
            currPos += perc*height2 + self.cellspace
            
    ######################################################################
    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, data, attrList, (x0, x1), (y0, y1), bHorizontal):
        if len(data) == 0:
            self.addRect(x0, x1, y0, y1)
            return
        attr = attrList[0]
        edge = len(attrList) * self.cellspace  # how much smaller rectangles do we draw
        if bHorizontal: vals = self.data.domain[attr].values
        else:           vals = list(self.data.domain[attr].values)[::-1]
        currPos = 0
        if bHorizontal: whole = (x1-x0)-edge*(len(vals)-1)  # we remove the space needed for separating different attr. values
        else:           whole = (y1-y0)-edge*(len(vals)-1)
        
        for val in vals:
            tempData = data.select({attr:val})
            perc = float(len(tempData))/float(len(data))
            if bHorizontal:
                size = ceil(whole*perc);
                if len(attrList) == 1:  self.addRect(x0+currPos, x0+currPos+size, y0, y1)
                else:                   self.DrawData(tempData, attrList[1:], (x0+currPos, x0+currPos+size), (y0, y1), not bHorizontal)
            else:
                size = ceil(whole*perc)
                if len(attrList) == 1:  self.addRect(x0, x1, y0+currPos, y0+currPos+size)
                else:                   self.DrawData(tempData, attrList[1:], (x0, x1), (y0+currPos, y0+currPos+size), not bHorizontal)
            currPos += size + edge

    # draws text with caption name at position x,y with alignment and style
    def addText(self, name, x, y, alignment, bold):
        text = QCanvasText(name, self.canvas);
        text.setTextFlags(alignment);
        font = text.font(); font.setBold(bold); text.setFont(font)
        text.move(x, y)
        text.show()
        self.texts.append(text)

    # draw a rectangle, set it to back and add it to rect list                
    def addRect(self, x0, x1, y0, y1):
        if x0==x1: x1+=1
        if y0==y1: y1+=1
        rect = QCanvasRectangle(x0, y0, x1-x0, y1-y0, self.canvas)
        rect.setZ(-10)
        rect.show()
        #pen = rect.pen(); pen.setWidth(2); rect.setPen(pen)
        self.rects.append(rect)
        return rect

   
     ##################################################
    ## SAVING GRAPHS
    ##################################################
    def saveToFileCanvas(self):
        size = self.canvas.size()
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG);;Windows Bitmap (.BMP);;Graphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        
        buffer = QPixmap(size) # any size can do, now using the window size
        #buffer = QPixmap(QSize(200,200)) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        self.canvasView.drawContents(painter, 0,0, buffer.rect().width(), buffer.rect().height())
        painter.end()
        buffer.save(fileName, ext)

        

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWMosaicDisplay()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
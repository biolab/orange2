"""
<name>Mosaic Display</name>
<description>Show mosaic display</description>
<icon>icons/MosaicDisplay.png</icon>
<priority>4200</priority>
"""
# OWMosaicDisplay.py
#
# 

from OWWidget import *
#from qt import *
from qtcanvas import *
import orngInteract
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct
from copy import copy
import OWGraphTools
from OWGUI import checkBox


###########################################################################################
##### WIDGET : 
###########################################################################################
class OWMosaicDisplay(OWWidget):
    settingsList = ["horizontalDistribution", "showDistribution", "showAprioriDistribution"]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Mosaic display", TRUE)

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []
        self.tooltips = []
        self.names = []     # class values
        self.symbols = []   # squares for class values
        

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata)]
        self.outputs = []
    
        #load settings
        self.showDistribution = 1
        self.showAprioriDistribution = 1
        self.horizontalDistribution = 1
        self.cellspace = 6
        self.attributeNameOffset = 30
        self.attributeValueOffset = 15
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
        self.group = QVGroupBox(self.controlArea)
        self.group.setMinimumWidth(200)
        self.attrSelGroup = QVGroupBox(self.group)
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

        checkBox(self.group, self, 'showDistribution', 'Show distribution', callback = self.updateData, tooltip = "Do you wish to see class distribution or only class purity?")
        checkBox(self.group, self, 'showAprioriDistribution', 'Show apriori distribution', callback = self.updateData, tooltip = "Do you wish to see lines showing apriori class distribution?")
        checkBox(self.group, self, 'horizontalDistribution', 'Show distribution horizontally', callback = self.updateData, tooltip = "Do you wish to see distribution drawn horizontally or vertically?")

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)

        self.resize(680, 480)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

        
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
            if data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                self.colorPalette = OWGraphTools.ColorPaletteBrewer(len(data.domain.classVar.values))
            
        self.initCombos(self.data)
        
        self.updateData()


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        # hide all rectangles
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.tooltips = []
        
        if self.data == None : return

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
        squareSize = min(self.canvasView.size().width() - width - 20, self.canvasView.size().height() - height - 20)
        if squareSize < 0: return    # canvas is too small to draw rectangles

        self.legend = {}        # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList: self.legend[attr] = 0

        # draw rectangles
        self.DrawData(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 1)

        # draw labels
        self.DrawText(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize))

        # draw class legend
        self.DrawClasses(self.data, (xOff, xOff+squareSize), (yOff, yOff+squareSize))
       
        self.canvas.update()

   
    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, data, attrList, (x0, x1), (y0, y1)):
        # save values for all attributes
        values = [data.domain[attr].values for attr in attrList]

        width = x1-x0 - self.cellspace*len(attrList)*(len(values[0])-1)
        height = y1-y0 - self.cellspace*(len(attrList)-1)*(len(values[1])-1)
        
        #calculate position of first attribute
        self.addText(attrList[0], x0+(x1-x0)/2, y1 + self.attributeNameOffset, Qt.AlignCenter, 1)
        currPos = 0        
        for val in values[0]:
            tempData = data.select({attrList[0]:val})
            perc = float(len(tempData))/float(len(data))
            self.addText(str(val), x0+currPos+(x1-x0)*0.5*perc, y1 + self.attributeValueOffset, Qt.AlignCenter, 0)
            currPos += perc*width + self.cellspace*len(attrList)

        #calculate position of second attribute 
        self.addText(attrList[1], x0 - self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignRight + Qt.AlignVCenter, 1)
        currPos = 0
        tempData = data.select({attrList[0]:values[0][0]})
        for val in list(values[1])[::-1]:
            tempData2 = tempData.select({attrList[1]:val})
            perc = float(len(tempData2))/float(len(tempData))
            self.addText(str(val), x0 - self.attributeValueOffset, y0+currPos+(y1-y0)*0.5*perc, Qt.AlignRight + Qt.AlignVCenter, 0)
            currPos += perc*height + self.cellspace*(len(attrList)-1)

        if len(attrList) < 3: return

        #calculate position of third attribute
        currPos = 0
        i = len(values[1])-1
        tempData2 = []
        while i >= 0 and len(tempData2) == 0:
            tempData2 = data.select({attrList[0]:values[0][0], attrList[1]:values[1][i]})
        width2 = width*float(len(tempData))/float(len(data)) - (len(values[2])-1)*self.cellspace*(len(attrList)-2)
        if len(tempData2) == 0: self.warning("Unable to draw attribute labels due to empty example subsetError: Division by zero."); return
        self.addText(attrList[2], x0 + width*float(len(tempData))/float(2*len(data)), y0 - self.attributeNameOffset, Qt.AlignCenter, 1)
        for val in values[2]:
            tempData3 = tempData2.select({attrList[2]:val})
            perc = (float(len(tempData3))/float(len(tempData2)))#*(float(len(tempData))/float(len(data)))
            self.addText(str(val), x0+currPos+width2*perc*0.5, y0 - self.attributeValueOffset, Qt.AlignCenter, 0)
            currPos += perc*width2 + self.cellspace*(len(attrList)-2)

        if len(attrList) < 4: return

        #calculate position of fourth attribute
        currPos = 0
        tempData0 = data.select({attrList[0]:values[0][-1]})
        tempData1 = tempData0.select({attrList[1]:values[1][-1]})
        tempData2 = tempData1.select({attrList[2]:values[2][-1]})
        if len(tempData0) == 0: self.warning("Unable to draw attribute labels due to empty example subsetError: Division by zero."); return
        height2 = height * float(len(tempData1))/float(len(tempData0)) - self.cellspace*(len(values[3])-1)

        if len(tempData2) == 0: self.warning("Unable to draw attribute labels due to empty example subsetError: Division by zero."); return
        self.addText(attrList[3], x1 + self.attributeNameOffset, y0+ height2/2, Qt.AlignLeft + Qt.AlignVCenter, 1)
        for val in list(values[3])[::-1]:
            tempData3 = tempData2.select({attrList[3]:val})
            perc = float(len(tempData3))/float(len(tempData2)) #* (float(len(tempData1))/float(len(tempData0)))
            self.addText(str(val), x1 + self.attributeValueOffset, y0 + currPos + height2*0.5*perc, Qt.AlignLeft + Qt.AlignVCenter, 0)
            currPos += perc*height2 + self.cellspace
            
    ######################################################################
    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, data, attrList, (x0, x1), (y0, y1), bHorizontal, condition = ""):
        if len(data) == 0:
            self.addRect(x0, x1, y0, y1, None)
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
                if len(attrList) == 1:  self.addRect(x0+currPos, x0+currPos+size, y0, y1, tempData, condition + "<b>" + attr + ":</b> " + val + "<br>")
                else:                   self.DrawData(tempData, attrList[1:], (x0+currPos, x0+currPos+size), (y0, y1), not bHorizontal, condition + "<b>" + attr + ":</b> " + val + "<br>")
            else:
                size = ceil(whole*perc)
                if len(attrList) == 1:  self.addRect(x0, x1, y0+currPos, y0+currPos+size, tempData, condition + "<b>" + attr + ":</b> " + val + "<br>")
                else:                   self.DrawData(tempData, attrList[1:], (x0, x1), (y0+currPos, y0+currPos+size), not bHorizontal, condition + "<b>" + attr + ":</b> " + val + "<br>")
            currPos += size + edge


     # draw the class legend below the square
    def DrawClasses(self, data, (x0, x1), (y0, y1)):
        # compute the x position of the center of the legend
        if not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous or not self.showDistribution: return

        for name in self.names: name.hide()
        self.names = []
        for symbol in self.symbols: symbol.hide()
        self.symbols = []
        
        x = (x0+x1)/2
        y = y1 + self.attributeNameOffset + 20

        self.names = []
        totalWidth = 0
        for name in data.domain.classVar.values:
            item = QCanvasText(name, self.canvas)
            self.names.append(item)
            totalWidth += item.boundingRect().width()

        distance = 30
        startX = x - (totalWidth + (len(data.domain.classVar.values)-1)*distance)/2
        xOffset = 0
        for i in range(len(data.domain.classVar.values)):
            symbol = QCanvasRectangle (startX + xOffset, y, 8, 8, self.canvas)
            symbol.setBrush(QBrush(self.colorPalette[i])); symbol.setPen(QPen(self.colorPalette[i]))
            symbol.show()
            self.symbols.append(symbol)
            self.names[i].move(startX + xOffset + 13, y-4)
            self.names[i].show()
            xOffset += distance + self.names[i].boundingRect().width()
            


    # draws text with caption name at position x,y with alignment and style
    def addText(self, name, x, y, alignment, bold):
        text = QCanvasText(name, self.canvas);
        text.setTextFlags(alignment);
        font = text.font(); font.setBold(bold); text.setFont(font)
        text.move(x, y)
        text.show()
        self.texts.append(text)

    # draw a rectangle, set it to back and add it to rect list                
    def addRect(self, x0, x1, y0, y1, data = None, condition = ""):
        if x0==x1: x1+=1
        if y0==y1: y1+=1
        rect = QCanvasRectangle(x0, y0, x1-x0, y1-y0, self.canvas)
        rect.setZ(-10)
        rect.show()
        #pen = rect.pen(); pen.setWidth(2); rect.setPen(pen)
        self.rects.append(rect)
        
        if not data: return rect

        originalDist = orange.Distribution(data.domain.classVar.name, self.data)
        dist = orange.Distribution(data.domain.classVar.name, data)

        if self.showDistribution:
            total = 0
            for i in range(len(dist)):
                val = dist[i]
                if self.horizontalDistribution:
                    v = (x1-x0)/len(data) * val
                    r = QCanvasRectangle(x0+total, y0+1, v, y1-y0-1, self.canvas)
                else:
                    v = (y1-y0)/len(data) * val
                    r = QCanvasRectangle(x0, y0+total, x1-x0, v, self.canvas)
                r.setPen(QPen(self.colorPalette[i])); r.setBrush(QBrush(self.colorPalette[i]))
                r.setZ(-20); r.show()
                self.rects.append(r)
                total += v

            if self.showAprioriDistribution:
                total = 0
                for i in range(len(originalDist)-1):
                    r = QCanvasLine(self.canvas)
                    if self.horizontalDistribution:
                        total += (x1-x0)/len(self.data) * originalDist[i]
                        r.setPoints(x0+total, y0+1, x0+total, y1-1)
                    else:
                        total += (y1-y0)/len(self.data) * originalDist[i]
                        r.setPoints(x0+1, y0+total, x1-1, y0+total)
                    r.setZ(10); r.show()
                    self.rects.append(r)
        else:
            pass
            
        self.addTooltip(x0, y0, x1-x0, y1-y0, condition, originalDist, dist)

    #################################################
    # add tooltips
    def addTooltip(self, x, y, w, h, condition, apriori, actual):
        examples = sum(list(actual))
        apriori = [val*100.0/float(sum(apriori)) for val in apriori]
        actual = [val*100.0/float(sum(actual)) for val in actual]
        aprioriText = ""; actualText = ""
        for i in range(len(apriori)):
            aprioriText += "%.2f, " %(apriori[i])
            actualText += "%.2f, " %(actual[i])
        aprioriText = "[ " + aprioriText[:-2] + " ]"
        actualText = "[ " +  actualText[:-2] + " ]"
       
        tooltipText = condition + "<hr>Number of examples: " + str(int(examples)) + "<br>Apriori distribution: " + aprioriText + "<br>Actual distribution: " + actualText
        tipRect = QRect(x, y, w, h)
        QToolTip.add(self.canvasView, tipRect, tooltipText)
        self.tooltips.append(tipRect)

   
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
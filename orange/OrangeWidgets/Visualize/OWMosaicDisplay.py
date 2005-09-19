"""
<name>Mosaic Display</name>
<description>Shows a mosaic display.</description>
<author>Gregor Leban (gregor.leban@fri.uni-lj.si)</author>
<icon>icons/MosaicDisplay.png</icon>
<priority>4200</priority>
"""
# OWMosaicDisplay.py
#
# 

from OWWidget import *
#from qt import *
from qtcanvas import *
from OWMosaicOptimization import *
import orngInteract
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct
from copy import copy
import OWGraphTools, OWGUI 


BOTTOM = 0
LEFT = 1
TOP = 2
RIGHT = 3

###########################################################################################
##### WIDGET : 
###########################################################################################
class OWMosaicDisplay(OWWidget):
    settingsList = ["horizontalDistribution", "showDistribution", "showAprioriDistribution"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Mosaic display", TRUE)

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
        self.attr1 = ""
        self.attr2 = ""
        self.attr3 = ""
        self.attr4 = ""
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
        self.controlArea.setMinimumWidth(220)
        box1 = OWGUI.widgetBox(self.controlArea, " 1st Attribute ")
        box2 = OWGUI.widgetBox(self.controlArea, " 2nd Attribute ")
        box3 = OWGUI.widgetBox(self.controlArea, " 3rd Attribute ")
        box4 = OWGUI.widgetBox(self.controlArea, " 4th Attribute ")
        self.attr1Combo = OWGUI.comboBox(box1, self, "attr1", None, callback = self.updateData, sendSelectedValue = 1, valueType = str)
        self.attr2Combo = OWGUI.comboBox(box2, self, "attr2", None, callback = self.updateData, sendSelectedValue = 1, valueType = str)
        self.attr3Combo = OWGUI.comboBox(box3, self, "attr3", None, callback = self.updateData, sendSelectedValue = 1, valueType = str)
        self.attr4Combo = OWGUI.comboBox(box4, self, "attr4", None, callback = self.updateData, sendSelectedValue = 1, valueType = str)

        box1.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        box2.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        box3.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        box4.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        self.optimizationDlg = MosaicOptimization(self, self.signalManager)
        optimizationButtons = OWGUI.widgetBox(self.controlArea, " Optimization Dialog ", orientation = "horizontal")
        optimizationButtons.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        OWGUI.button(optimizationButtons, self, "VizRank", callback = self.optimizationDlg.reshow)
        
        
        box5 = OWGUI.widgetBox(self.controlArea, "Visual Settings")
        box5.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        OWGUI.checkBox(box5, self, 'showDistribution', 'Show Distribution', callback = self.updateData, tooltip = "Show the class distribution for each bar")
        OWGUI.checkBox(box5, self, 'showAprioriDistribution', 'Show Apriori Distribution', callback = self.updateData, tooltip = "Show the lines that represent the apriori class distribution")
        OWGUI.checkBox(box5, self, 'horizontalDistribution', 'Show Distribution Horizontally', callback = self.updateData, tooltip = "Do you wish to see class distribution drawn horizontally or vertically?")

        b = QVBox(self.controlArea)

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = self.createAttributeIconDict()
        self.resize(680, 480)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

        
    ##################################################
    # initialize combo boxes with discrete attributes
    def initCombos(self, data):
        self.attr1Combo.clear(); self.attr2Combo.clear(); self.attr3Combo.clear(); self.attr4Combo.clear()

        if data == None: return

        self.attr2Combo.insertItem("(None)")
        self.attr3Combo.insertItem("(None)")
        self.attr4Combo.insertItem("(None)")

        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                self.attr1Combo.insertItem(self.icons[orange.VarTypes.Discrete], attr.name)
                self.attr2Combo.insertItem(self.icons[orange.VarTypes.Discrete], attr.name)
                self.attr3Combo.insertItem(self.icons[orange.VarTypes.Discrete], attr.name)
                self.attr4Combo.insertItem(self.icons[orange.VarTypes.Discrete], attr.name)

        if self.attr1Combo.count() > 0:
            self.attr1 = str(self.attr1Combo.text(0))
            self.attr2 = str(self.attr2Combo.text(1 + (self.attr2Combo.count() > 1)))
        self.attr3 = str(self.attr3Combo.text(0))
        self.attr4 = str(self.attr4Combo.text(0))
        

    ######################################################################
    ##  when we resize the widget, we have to redraw the data
    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateData()

    ######################################################################
    ## DATA signal
    # receive new data and update all fields
    def cdata(self, data):
        self.data = None
        self.optimizationDlg.setData(data)
        
        if data:
            #self.data = orange.Preprocessor_dropMissing(data)
            self.data = data
            if data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                self.colorPalette = OWGraphTools.ColorPaletteBrewer(len(data.domain.classVar.values))
            
        self.initCombos(self.data)
        
        self.updateData()


    def setShownAttributes(self, attrList):
        if not attrList: return
        self.attr1 = attrList[0]
        
        if len(attrList) > 1: self.attr2 = attrList[1]
        else: self.attr2 = "(None)"

        if len(attrList) > 2: self.attr3 = attrList[2]
        else: self.attr3 = "(None)"

        if len(attrList) > 3: self.attr4 = attrList[3]
        else: self.attr4 = "(None)"

        self.updateData()
        

    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        # hide all rectangles
        self.warning()
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.tooltips = []
        
        if self.data == None : return

        attrList = [self.attr1, self.attr2, self.attr3, self.attr4]
        while "(None)" in attrList:
            attrList.remove("(None)")

        selectList = attrList
        data = self.optimizationDlg.getData()   # get the selected class values
        if self.data.domain.classVar: data = data.select(attrList + [data.domain.classVar.name])
        else: data = data.select(attrList)
        data = orange.Preprocessor_dropMissing(data)
        
        # get the maximum width of rectangle
        xOff = 50
        width = 50
        if len(attrList) > 1:
            text = QCanvasText(attrList[1], self.canvas);
            font = text.font(); font.setBold(1); text.setFont(font)
            width = text.boundingRect().right() - text.boundingRect().left() + 30 + 20
            xOff = width
            if len(attrList) == 4:
                text = QCanvasText(attrList[3], self.canvas);
                font = text.font(); font.setBold(1); text.setFont(font)
                width += text.boundingRect().right() - text.boundingRect().left() + 30 + 20
        
        # get the maximum height of rectangle        
        height = 90
        yOff = 40
        squareSize = min(self.canvasView.size().width() - width - 20, self.canvasView.size().height() - height - 20)
        if squareSize < 0: return    # canvas is too small to draw rectangles

        self.legend = {}        # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList: self.legend[attr] = 0

        self.drawnSides = dict([(0,0),(1,0),(2,0),(3,0)])

        # draw rectangles
        self.DrawData(data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 0, "", len(attrList))

        # draw labels
        #self.DrawText(data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize))

        # draw class legend
        self.DrawClasses(data, (xOff, xOff+squareSize), (yOff, yOff+squareSize))
       
        self.canvas.update()

   
    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, data, side, attr, (x0, x1), (y0, y1), totalAttrs, lastValueForFirstAttribute):
        if self.drawnSides[side] or not data or len(data) == 0: return
        if side == RIGHT and lastValueForFirstAttribute != 2: return
        
        self.drawnSides[side] = 1

        if side % 2 == 0: values = data.domain[attr].values
        else            : values = list(data.domain[attr].values)[::-1]

        width  = x1-x0 - (side % 2 == 0) * self.cellspace*(totalAttrs-side)*(len(values)-1)
        height = y1-y0 - (side % 2 == 1) * self.cellspace*(totalAttrs-side)*(len(values)-1)
        
        #calculate position of first attribute
        if side == 0:    self.addText(attr, x0+(x1-x0)/2, y1 + self.attributeNameOffset, Qt.AlignCenter, 1)
        elif side == 1:  self.addText(attr, x0 - self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignRight + Qt.AlignVCenter, 1)
        elif side == 2:  self.addText(attr, x0+(x1-x0)/2, y0 - self.attributeNameOffset, Qt.AlignCenter, 1)
        else:            self.addText(attr, x1 + self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignLeft + Qt.AlignVCenter, 1)
                
        currPos = 0        
        for val in values:
            tempData = data.select({attr:val})
            perc = float(len(tempData))/float(len(data))
            if side == 0:    self.addText(str(val), x0+currPos+(x1-x0)*0.5*perc, y1 + self.attributeValueOffset, Qt.AlignCenter, 0)
            elif side == 1:  self.addText(str(val), x0-self.attributeValueOffset, y0+currPos+(y1-y0)*0.5*perc, Qt.AlignRight + Qt.AlignVCenter, 0)
            elif side == 2:  self.addText(str(val), x0+currPos+(x1-x0)*perc*0.5, y0 - self.attributeValueOffset, Qt.AlignCenter, 0)
            else:            self.addText(str(val), x1+self.attributeValueOffset, y0 + currPos + (y1-y0)*0.5*perc, Qt.AlignLeft + Qt.AlignVCenter, 0)

            if side % 2 == 0: currPos += perc*width + self.cellspace*(totalAttrs-side)
            else :            currPos += perc*height+ self.cellspace*(totalAttrs-side)
            
        """
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
        """
            
    ######################################################################
    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, data, attrList, (x0, x1), (y0, y1), side, condition, totalAttrs, lastValueForFirstAttribute = 0):
        if len(data) == 0:
            self.addRect(x0, x1, y0, y1, None)
            return
        attr = attrList[0]
        edge = len(attrList) * self.cellspace  # how much smaller rectangles do we draw
        if side%2 == 0: vals = self.data.domain[attr].values
        else:           vals = list(self.data.domain[attr].values)[::-1]
        currPos = 0
        if side%2 == 0: whole = max(0, (x1-x0)-edge*(len(vals)-1))  # we remove the space needed for separating different attr. values
        else:           whole = max(0, (y1-y0)-edge*(len(vals)-1))

        for val in vals:
            tempData = data.select({attr:val})
            perc = float(len(tempData))/float(len(data))
            if side % 2 == 0:   # if drawing horizontal
                size = ceil(whole*perc);
                if len(attrList) == 1:  self.addRect(x0+currPos, x0+currPos+size, y0, y1, tempData, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + val + "<br>")
                else:                   self.DrawData(tempData, attrList[1:], (x0+currPos, x0+currPos+size), (y0, y1), side +1, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + val + "<br>", totalAttrs, lastValueForFirstAttribute + (side%2==0 and val == vals[-1]))
            else:
                size = ceil(whole*perc)
                if len(attrList) == 1:  self.addRect(x0, x1, y0+currPos, y0+currPos+size, tempData, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + val + "<br>")
                else:                   self.DrawData(tempData, attrList[1:], (x0, x1), (y0+currPos, y0+currPos+size), side +1, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + val + "<br>", totalAttrs, lastValueForFirstAttribute + (side%2==0 and val == vals[-1]))
            currPos += size + edge

        self.DrawText(data, side, attrList[0], (x0, x1), (y0, y1), totalAttrs, lastValueForFirstAttribute)


     # draw the class legend below the square
    def DrawClasses(self, data, (x0, x1), (y0, y1)):
        for name in self.names: name.hide()
        self.names = []
        for symbol in self.symbols: symbol.hide()
        self.symbols = []

        # compute the x position of the center of the legend
        if not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous or not self.showDistribution: return
        
        x = (x0+x1)/2
        y = y1 + self.attributeNameOffset + 20

        self.names = []
        totalWidth = 0

        for name in data.domain.classVar.values:
            item = QCanvasText(name, self.canvas)
            self.names.append(item)
            totalWidth += item.boundingRect().width()

        item = QCanvasText(data.domain.classVar.name, self.canvas)
        self.names.append(item)
        totalWidth += item.boundingRect().width()

        distance = 30
        startX = x - (totalWidth + (len(data.domain.classVar.values))*distance)/2
        xOffset = 0

        self.names[-1].move(startX, y-4)
        self.names[-1].show()
        xOffset += self.names[-1].boundingRect().width() + distance + 4
        
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
        text = QCanvasText(name, self.canvas)
        text.setTextFlags(alignment)
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
        
        if not data or not data.domain.classVar or not data.domain.classVar.varType == orange.VarTypes.Discrete:
            return rect

        originalDist = orange.Distribution(data.domain.classVar.name, self.data)
        dist = orange.Distribution(data.domain.classVar.name, data)
        self.addTooltip(x0, y0, x1-x0, y1-y0, condition, originalDist, dist)

        if self.showDistribution:
            total = 0
            for i in range(len(dist)):
                val = dist[i]
                if self.horizontalDistribution:
                    v = ((x1-x0)* val)/len(data)
                    r = QCanvasRectangle(x0+total, y0+1, v, y1-y0-1, self.canvas)
                else:
                    v = ((y1-y0)* val)/len(data) 
                    r = QCanvasRectangle(x0, y0+total, x1-x0, v, self.canvas)
                r.setPen(QPen(self.colorPalette[i])); r.setBrush(QBrush(self.colorPalette[i]))
                r.setZ(-20); r.show()
                self.rects.append(r)
                total += v

            if self.showAprioriDistribution and abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
                total = 0
                for i in range(len(originalDist)-1):
                    r = QCanvasLine(self.canvas)
                    if self.horizontalDistribution:
                        total += ((x1-x0)* originalDist[i])/len(self.data) 
                        r.setPoints(x0+total, y0+1, x0+total, y1-1)
                    else:
                        total += ((y1-y0)* originalDist[i])/len(self.data)
                        r.setPoints(x0+1, y0+total, x1-1, y0+total)
                    r.setZ(10); r.show()
                    self.rects.append(r)

    #################################################
    # add tooltips
    def addTooltip(self, x, y, w, h, condition, apriori, actual):
        #print "actual = ", actual, type(actual)
        examples = sum(list(actual))
        apriori = [val*100.0/float(sum(apriori)) for val in apriori]
        actual = [val*100.0/float(sum(actual)) for val in actual]
        aprioriText = ""; actualText = ""
        for i in range(len(apriori)):
            aprioriText += "%.1f, " %(apriori[i])
            actualText += "%.1f, " %(actual[i])
        aprioriText = "[ " + aprioriText[:-2] + " ]"
        actualText = "[ " +  actualText[:-2] + " ]"
       
        tooltipText = "<b>Examples in this area have:</b><br>" + condition + "<hr>Number of examples: " + str(int(examples)) + "<br>Apriori distribution: " + aprioriText + "<br>Actual distribution: " + actualText
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

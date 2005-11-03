"""
<name>Mosaic Display</name>
<description>Shows a mosaic display.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
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
from orngScaleData import getVariableValuesSorted, getVariableValueIndices
import random
import OWGraphTools, OWGUI
from OWQCanvasFuncts import *
from OWTools import getHtmlCompatibleString

PEARSON = 0
CLASS_DISTRIBUTION = 1

BOTTOM = 0
LEFT = 1
TOP = 2
RIGHT = 3


###########################################################################################
##### WIDGET : 
###########################################################################################
class OWMosaicDisplay(OWWidget):
    settingsList = ["horizontalDistribution", "showAprioriDistribution", "interiorColoring" ]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Mosaic display", TRUE, TRUE)

        #set default settings
        self.data = None
        self.subsetData = None
        self.tooltips = []
        self.names = []     # class values
        
        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default), ("Example Subset", ExampleTable, self.subsetdata)]
        self.outputs = [("Learner", orange.Learner)]
    
        #load settings
        self.interiorColoring = 0
        self.showAprioriDistribution = 1
        self.horizontalDistribution = 1
        self.attr1 = ""
        self.attr2 = ""
        self.attr3 = ""
        self.attr4 = ""
        self.cellspace = 6
        self.attributeNameOffset = 30
        self.attributeValueOffset = 15
        self.residuals = [] # residual values if the residuals are visualized
        self.aprioriDistributions = []
        self.colorPalette = None

        #self.blueColors = [QColor(255, 255, 255), QColor(117, 149, 255), QColor(38, 43, 232), QColor(1,5,173)]
        self.blueColors = [QColor(255, 255, 255), QColor(210, 210, 255), QColor(110, 110, 255), QColor(0,0,255)]
        self.redColors = [QColor(255, 255, 255), QColor(255, 200, 200), QColor(255, 100, 100), QColor(255, 0, 0)]
        
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

        box6 = OWGUI.widgetBox(self.controlArea, "Visualize...")
        OWGUI.comboBox(box6, self, "interiorColoring", None, items = ["Standardized (Pearson) residuals", "Class distribution"], callback = self.changedInteriorColoring)
        box6.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        
        self.box5 = OWGUI.widgetBox(self.controlArea, "Class Distribution Settings")
        OWGUI.checkBox(self.box5, self, 'showAprioriDistribution', 'Show Apriori Distribution', callback = self.updateData, tooltip = "Show the lines that represent the apriori class distribution")
        OWGUI.checkBox(self.box5, self, 'horizontalDistribution', 'Show Distribution Horizontally', callback = self.updateData, tooltip = "Do you wish to see class distribution drawn horizontally or vertically?")
        self.box5.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        b = QVBox(self.controlArea)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = self.createAttributeIconDict()
        self.resize(680, 480)

        #connect controls to appropriate functions
        self.activateLoadedSettings()
        self.changedInteriorColoring()

        self.VizRankLearner = MosaicVizRankLearner(self.optimizationDlg)
        self.send("Learner", self.VizRankLearner)

        
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
            self.attr2 = str(self.attr2Combo.text(0 + 2*(self.attr2Combo.count() > 2)))
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
        
        #self.updateData()

    def subsetdata(self, data):
        self.subsetData = data
                
        if data and len(data) > 1: self.setStatusBarText("The data set received on the 'Example subset' input contains more than one example. Only the first example will be considered.")
        else:                      self.setStatusBarText("")

        #self.updateData()
        


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
        

    def changedInteriorColoring(self):
        self.box5.setEnabled(self.interiorColoring)
        self.updateData()

    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        # hide all rectangles
        self.warning()
        for item in self.canvas.allItems(): item.setCanvas(None)    # remove all canvas items
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.names = []; self.tooltips = []
        
        if self.data == None : return

        attrList = [self.attr1, self.attr2, self.attr3, self.attr4]
        while "(None)" in attrList:
            attrList.remove("(None)")

        selectList = attrList
        data = self.optimizationDlg.getData()   # get the selected class values
        if data.domain.classVar: data = data.select(attrList + [data.domain.classVar.name])
        else: data = data.select(attrList)
        data = orange.Preprocessor_dropMissing(data)

        self.aprioriDistributions = []
        if self.interiorColoring == PEARSON:
            for attr in attrList:
                self.aprioriDistributions = [orange.Distribution(attr, data) for attr in attrList]

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
        self.drawPositions = {}

        # draw rectangles
        self.DrawData(data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 0, "", len(attrList))

        # draw class legend
        self.DrawLegend(data, (xOff, xOff+squareSize), (yOff, yOff+squareSize))

        del data
        self.canvas.update()


    ######################################################################
    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, data, attrList, (x0, x1), (y0, y1), side, condition, totalAttrs, lastValueForFirstAttribute = 0, usedAttrs = []):
        if len(data) == 0:
            self.addRect(x0, x1, y0, y1, None)
            self.DrawText(data, side, attrList[0], (x0, x1), (y0, y1), totalAttrs, lastValueForFirstAttribute)  # store coordinates for later drawing of labels
            return
        
        attr = attrList[0]
        edge = len(attrList) * self.cellspace  # how much smaller rectangles do we draw
        vals = getVariableValuesSorted(self.data, attr)
        if side%2: vals = vals[::-1]        # reverse names if necessary

        if side%2 == 0:
            whole = max(0, (x1-x0)-edge*(len(vals)-1))  # we remove the space needed for separating different attr. values
            if whole == 0: edge = (x1-x0)/float(len(vals)-1)
        else:
            whole = max(0, (y1-y0)-edge*(len(vals)-1))
            if whole == 0: edge = (y1-y0)/float(len(vals)-1)

        currPos = 0.0
        for val in vals:
            tempData = data.select({attr:val})
            perc = float(len(tempData))/float(len(data))
            size = whole*perc
            htmlVal = getHtmlCompatibleString(val)

            if side % 2 == 0:   # if drawing horizontal
                if len(attrList) == 1:  self.addRect(x0+currPos, x0+currPos+size, y0, y1, tempData, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + htmlVal + "<br>", usedAttrs + [attr, val])
                else:                   self.DrawData(tempData, attrList[1:], (x0+currPos, x0+currPos+size), (y0, y1), side +1, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + htmlVal + "<br>", totalAttrs, lastValueForFirstAttribute + (val == vals[-1]), usedAttrs + [attr, val])
            else:
                if len(attrList) == 1:  self.addRect(x0, x1, y0+currPos, y0+currPos+size, tempData, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + htmlVal + "<br>", usedAttrs + [attr, val])
                else:                   self.DrawData(tempData, attrList[1:], (x0, x1), (y0+currPos, y0+currPos+size), side +1, condition + 4*" &nbsp " + "<b>" + attr + ":</b> " + htmlVal + "<br>", totalAttrs, lastValueForFirstAttribute, usedAttrs + [attr, val])
            currPos += size + edge
            del tempData

        self.DrawText(data, side, attrList[0], (x0, x1), (y0, y1), totalAttrs, lastValueForFirstAttribute)

   
    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, data, side, attr, (x0, x1), (y0, y1), totalAttrs, lastValueForFirstAttribute):
        if self.drawnSides[side]: return
        if side == RIGHT and lastValueForFirstAttribute != 2: return
        
        if not data or len(data) == 0:
            if not self.drawPositions.has_key(side): self.drawPositions[side] = (x0, x1, y0, y1)
            return
        else:
            if self.drawPositions.has_key(side): (x0, x1, y0, y1) = self.drawPositions[side]        # restore the positions where we have to draw the attribute values and attribute name
            
        self.drawnSides[side] = 1

        values = getVariableValuesSorted(data, attr)
        if side % 2:  values = values[::-1]

        width  = x1-x0 - (side % 2 == 0) * self.cellspace*(totalAttrs-side)*(len(values)-1)
        height = y1-y0 - (side % 2 == 1) * self.cellspace*(totalAttrs-side)*(len(values)-1)
        
        #calculate position of first attribute
        if side == 0:    OWCanvasText(self.canvas, attr, x0+(x1-x0)/2, y1 + self.attributeNameOffset, Qt.AlignCenter, bold = 1)
        elif side == 1:  OWCanvasText(self.canvas, attr, x0 - self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignRight + Qt.AlignVCenter, bold = 1)
        elif side == 2:  OWCanvasText(self.canvas, attr, x0+(x1-x0)/2, y0 - self.attributeNameOffset, Qt.AlignCenter, bold = 1)
        else:            OWCanvasText(self.canvas, attr, x1 + self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignLeft + Qt.AlignVCenter, bold = 1)
                
        currPos = 0
        dist = orange.Distribution(attr, data)
        for val in values:
            perc = dist[val]/float(len(data))
            if side == 0:    OWCanvasText(self.canvas, str(val), x0+currPos+width*0.5*perc, y1 + self.attributeValueOffset, Qt.AlignCenter, bold = 0)
            elif side == 1:  OWCanvasText(self.canvas, str(val), x0-self.attributeValueOffset, y0+currPos+height*0.5*perc, Qt.AlignRight + Qt.AlignVCenter, bold = 0)
            elif side == 2:  OWCanvasText(self.canvas, str(val), x0+currPos+width*perc*0.5, y0 - self.attributeValueOffset, Qt.AlignCenter, bold = 0)
            else:            OWCanvasText(self.canvas, str(val), x1+self.attributeValueOffset, y0 + currPos + height*0.5*perc, Qt.AlignLeft + Qt.AlignVCenter, bold = 0)

            if side % 2 == 0: currPos += perc*width + self.cellspace*(totalAttrs-side)
            else :            currPos += perc*height+ self.cellspace*(totalAttrs-side)

            
     # draw the class legend below the square
    def DrawLegend(self, data, (x0, x1), (y0, y1)):
        if self.interiorColoring == CLASS_DISTRIBUTION and (not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous): return

        if self.interiorColoring == PEARSON:
            names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8", "Residuals:"]
            colors = self.redColors[::-1] + self.blueColors[1:]
        else:
            names = getVariableValuesSorted(data, data.domain.classVar.name) + [data.domain.classVar.name+":"]
            colors = [self.colorPalette[i] for i in range(len(data.domain.classVar.values))]
        
        for name in names:
            self.names.append(OWCanvasText(self.canvas, name))
            
        totalWidth = sum([self.names[i].boundingRect().width() for i in range(len(self.names))])

        # compute the x position of the center of the legend
        y = y1 + self.attributeNameOffset + 15
        distance = 30
        startX = (x0+x1)/2 - (totalWidth + (len(names))*distance)/2

        self.names[-1].move(startX+15, y+1); self.names[-1].show()
        xOffset = self.names[-1].boundingRect().width() + distance

        size = 16 # 8 + 8*(self.interiorColoring == PEARSON)
        
        for i in range(len(names)-1):
            if self.interiorColoring == PEARSON: edgeColor = Qt.black
            else: edgeColor = colors[i]

            OWCanvasRectangle(self.canvas, startX + xOffset, y, size, size, edgeColor, colors[i])

            self.names[i].move(startX + xOffset + 18, y+1)

            xOffset += distance + self.names[i].boundingRect().width()
            

    # draw a rectangle, set it to back and add it to rect list                
    def addRect(self, x0, x1, y0, y1, data = None, condition = "", usedAttrs = []):
        x0 = int(x0); x1 = int(x1); y0 = int(y0); y1 = int(y1)
        if x0 == x1: x1+=1
        if y0 == y1: y1+=1

        if x1-x0 + y1-y0 == 2: y1+=1        # if we want to show a rectangle of width and height 1 it doesn't show anything. in such cases we therefore have to increase size of one edge

        rect = OWCanvasRectangle(self.canvas, x0, y0, x1-x0, y1-y0, z = -10)            
        

        if not data: return rect
        if self.interiorColoring == CLASS_DISTRIBUTION and (not data.domain.classVar or not data.domain.classVar.varType == orange.VarTypes.Discrete):
            return rect

        originalDist = None; dist = None; pearson = None; expected = None
        
        if self.interiorColoring == PEARSON or not data.domain.classVar or data.domain.classVar.varType != orange.VarTypes.Discrete:
            vals = usedAttrs[1::2]
            s = sum(self.aprioriDistributions[0])
            expected = s * reduce(lambda x, y: x*y, [self.aprioriDistributions[i][vals[i]]/float(s) for i in range(len(vals))])
            actual = len(data)
            pearson = float(actual - expected) / sqrt(expected)
            if abs(pearson) < 2:   ind = 0
            elif abs(pearson) < 4: ind = 1
            elif abs(pearson) < 8: ind = 2
            else:                  ind = 3

            if pearson > 0: color = self.blueColors[ind]
            else: color = self.redColors[ind]
            rect = OWCanvasRectangle(self.canvas, x0, y0+1, x1-x0, y1-y0, color, color, z = -20)
        else:
            originalDist = orange.Distribution(self.data.domain.classVar.name, self.data)
            dist = orange.Distribution(data.domain.classVar.name, data)
            values = getVariableValuesSorted(data, data.domain.classVar.name)
            
            total = 0
            for i in range(len(dist)):
                val = dist[values[i]]
                if self.horizontalDistribution:
                    v = ((x1-x0)* val)/len(data)
                    r = OWCanvasRectangle(self.canvas, x0+total, y0+1, v, y1-y0-2, self.colorPalette[i], self.colorPalette[i], z = -20)

                else:
                    v = ((y1-y0)* val)/len(data)
                    r = OWCanvasRectangle(self.canvas, x0, y0+total, x1-x0, v, self.colorPalette[i], self.colorPalette[i], z = -20)

                total += v

            if self.showAprioriDistribution and abs(x1 - x0) > 1 and abs(y1 - y0) > 1:
                total = 0
                for i in range(len(originalDist)-1):
                    if self.horizontalDistribution:
                        total += ((x1-x0)* originalDist[values[i]])/len(self.data) 
                        OWCanvasLine(self.canvas, x0+total, y0+1, x0+total, y1-1, z = 10)
                    else:
                        total += ((y1-y0)* originalDist[values[i]])/len(self.data)
                        OWCanvasLine(self.canvas, x0+1, y0+total, x1-1, y0+total, z = 10)

        if self.subsetData and self.subsetData.domain == self.data.domain:
            correctBox = 1
            for i in range(len(usedAttrs)/2):
                if self.subsetData[0][usedAttrs[2*i]] != usedAttrs[2*i+1]:
                    correctBox = 0
                    break
            if correctBox:
                rect = OWCanvasRectangle(self.canvas, x0-3, y0-3, x1-x0+7, y1-y0+7, QColor(0,255,0), Qt.white, penWidth = 3, z=-50)

        self.addTooltip(x0, y0, x1-x0, y1-y0, condition, originalDist, dist, pearson, expected)

    #################################################
    # add tooltips
    def addTooltip(self, x, y, w, h, condition, apriori = None, actual = None, pearson = None, expected = None):
        tooltipText = "<b>Examples in this area have:</b><br>" + condition
        if apriori and actual:
            examples = sum(list(actual))
            apriori = [val*100.0/float(sum(apriori)) for val in apriori]
            actual = [val*100.0/float(sum(actual)) for val in actual]
            aprioriText = ""; actualText = ""
            for i in range(len(apriori)):
                aprioriText += "%.1f%%, " %(apriori[i])
                actualText += "%.1f%%, " %(actual[i])
            aprioriText = "[ " + aprioriText[:-2] + " ]"
            actualText = "[ " +  actualText[:-2] + " ]"
        
            tooltipText += "<hr>Number of examples: " + str(int(examples)) + "<br>Apriori distribution: " + aprioriText + "<br>Actual distribution: " + actualText
        if pearson and expected:
            tooltipText += "<hr>Expected number of examples: %.1f<br>Standardized (Pearson) residual: %.1f" % (expected, pearson)
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

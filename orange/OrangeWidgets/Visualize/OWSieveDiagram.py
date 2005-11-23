"""
<name>Sieve Diagram</name>
<description>Sieve diagram.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/SieveDiagram.png</icon>
<priority>4100</priority>
"""
# OWSieveDiagram.py
#

from OWWidget import *
from qt import *
from qtcanvas import *
import orngInteract, OWGUI
from OWQCanvasFuncts import *
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct
import random
from OWTools import getHtmlCompatibleString
from OWDlgs import OWChooseImageSizeDlg

###########################################################################################
##### WIDGET : 
###########################################################################################
class OWSieveDiagram(OWWidget):
    settingsList = ["showLines", "showCases", "showInColor"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Sieve diagram", TRUE)

        self.controlArea.setMinimumWidth(250)

        self.inputs = [("Examples", ExampleTable, self.cdata, Default), ("Attribute selection", list, self.attributeSelection)]
        self.outputs = []

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []
        self.lines = []
        self.tooltips = []

        self.attrX = ""
        self.attrY = ""
        self.attrCondition = None
        self.attrConditionValue = None
        self.showLines = 1
        self.showCases = 0
        self.showInColor = 1
        self.attributeSelectionList = None
        self.stopCalculating = 0

        #load settings
        self.loadSettings()

        self.box = QVBoxLayout(self.mainArea)
        self.canvas = QCanvas(2000, 2000)
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        self.box.addWidget(self.canvasView)
        self.canvasView.show()
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        
        #GUI
        self.attrSelGroup = OWGUI.widgetBox(self.controlArea, box = " Shown Attributes ")

        self.attrXCombo = OWGUI.comboBoxWithCaption(self.attrSelGroup, self, "attrX", "X Attribute:", tooltip = "Select an attribute to be shown on the X axis", callback = self.updateData, sendSelectedValue = 1, valueType = str, labelWidth = 70)
        self.attrYCombo = OWGUI.comboBoxWithCaption(self.attrSelGroup, self, "attrY", "Y Attribute:", tooltip = "Select an attribute to be shown on the Y axis", callback = self.updateData, sendSelectedValue = 1, valueType = str, labelWidth = 70)

        self.conditionGroup = OWGUI.widgetBox(self.controlArea, box = " Condition ")
        self.attrConditionCombo      = OWGUI.comboBoxWithCaption(self.conditionGroup, self, "attrCondition", "Attribute:", callback = self.updateConditionAttr, sendSelectedValue = 1, valueType = str, labelWidth = 70)
        self.attrConditionValueCombo = OWGUI.comboBoxWithCaption(self.conditionGroup, self, "attrConditionValue", "Value:", callback = self.updateData, sendSelectedValue = 1, valueType = str, labelWidth = 70)


        box2 = OWGUI.widgetBox(self.controlArea, box = " Visual Settings ")
        OWGUI.checkBox(box2, self, "showLines", "Show Lines", callback = self.updateData)
        hbox = OWGUI.widgetBox(box2, orientation = "horizontal")
        OWGUI.checkBox(hbox, self, "showCases", "Show Data Examples...", callback = self.updateData)
        OWGUI.checkBox(hbox, self, "showInColor", "...In Color", callback = self.updateData)
        
        self.interestingGroupBox = OWGUI.widgetBox(self.controlArea, box = " Interesting Attribute Pairs ")
        
        self.calculateButton = OWGUI.button(self.interestingGroupBox, self, "Calculate Chi Squares", callback = self.calculatePairs)
        self.stopCalculateButton = OWGUI.button(self.interestingGroupBox, self, "Stop Evaluation", callback = self.stopCalculateClick)
        self.stopCalculateButton.hide()
        
        self.interestingList = QListBox(self.interestingGroupBox)
        self.connect(self.interestingList, SIGNAL("selectionChanged()"),self.showSelectedPair)

        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = self.createAttributeIconDict()
        self.resize(800, 550)
        random.seed()

    ###############################################################
    # when clicked on a list box item, show selected attribute pair
    def showSelectedPair(self):
        if self.interestingList.count() == 0: return

        index = self.interestingList.currentItem()
        (chisquare, strName, self.attrX, self.attrY) = self.chisquares[index]        
        self.updateData()

    def calculatePairs(self):
        self.chisquares = []
        self.interestingList.clear()
        self.stopCalculating = 0
        if not self.data: return

        self.calculateButton.hide()
        self.stopCalculateButton.show()

        discAttrs = []
        for attr in self.data.domain:
            if self.data.domain[attr].varType == orange.VarTypes.Discrete: discAttrs.append(attr)

        discData = self.data.select(discAttrs)

        self.progressBarInit()

        total = len(discData.domain)* len(discData.domain) / 2.0
        current = 0

        #for attrX in range(len(data.domain)):
        for attr1 in range(len(discData.domain)):
            attrX = discData.domain[attr1].name

            #for attrY in range(attrX+1, len(data.domain)):
            for attr2 in range(attr1+1, len(discData.domain)):
                attrY = discData.domain[attr2].name
                current += 1
                if self.stopCalculating:
                    self.progressBarFinished()
                    self.calculateButton.show()
                    self.stopCalculateButton.hide()
                    return

                data = self.getConditionalData(attrX, attrY)
                if len(data) == 0: continue

                dcX = orange.ContingencyAttrAttr(attrX, attrX, data)    # distribution of X attribute
                valsX = [sum(dcX[key]) for key in dcX.keys()]          # compute contingency of x attribute

                dcY = orange.ContingencyAttrAttr(attrY, attrY, data)    # distribution of X attribute
                valsY = [sum(dcY[key]) for key in dcY.keys()]          # compute contingency of x attribute
                
                # create cartesian product of selected attributes and compute contingency 
                (cart, profit) = FeatureByCartesianProduct(data, [data.domain[attrX], data.domain[attrY]])
                tempData = data.select(list(data.domain) + [cart])
                contXY = orange.ContingencyAttrAttr(cart, cart, tempData)   # distribution of the merged attribute

                # compute chi-square
                chisquare = 0.0
                for i in range(len(valsX)):
                    valx = valsX[i]
                    for j in range(len(valsY)):
                        valy = valsY[j]

                        actual = 0
                        try:
                            for val in contXY['%s-%s' %(dcX.keys()[i], dcY.keys()[j])]: actual += val
                        except:
                            actual = 0
                        
                        expected = float(valx * valy) / float(len(data))
                        if expected == 0: continue
                        pearson2 = (actual - expected)*(actual - expected) / expected
                        chisquare += pearson2

                i = 0
                while i < len(self.chisquares) and self.chisquares[i][0] > chisquare: i += 1
                self.chisquares.insert(i, (chisquare, "%s - %s" % (attrX, attrY), attrX, attrY))
                self.interestingList.insertItem("%s - %s (%.3f)" % (attrX, attrY, chisquare), i)

                self.progressBarSet(100.0*current/float(total))
                qApp.processEvents()
                

        self.progressBarFinished()
        self.calculateButton.show()
        self.stopCalculateButton.hide()


    def stopCalculateClick(self):
        self.stopCalculating = 1

    ######################################
    # create data subset depending on conditional attribute and value
    def getConditionalData(self, xAttr = None, yAttr = None, dropMissingData = 1):
        if not self.data: return None

        if not xAttr: xAttr = self.attrX
        if not yAttr: yAttr = self.attrY
        
        if self.attrCondition == "[None]":
            data = self.data.select([xAttr, yAttr])
        else:
            data = orange.Preprocessor_dropMissing(self.data.select([xAttr, yAttr, self.attrCondition]))
            data = data.select({self.attrCondition:self.attrConditionValue})
            
        if dropMissingData: return orange.Preprocessor_dropMissing(data)
        else: return data

    ######################################
    # new conditional attribute was set - update graph
    def updateConditionAttr(self):
        self.attrConditionValueCombo.clear()
        
        if self.attrCondition == "[None]":
            self.updateData()
            return

        for val in self.data.domain[self.attrCondition].values:
            self.attrConditionValueCombo.insertItem(val)
        self.attrConditionValue = str(self.attrConditionValueCombo.text(0))
        self.updateData()

    ##################################################
    # initialize lists for shown and hidden attributes
    def initCombos(self):
        self.attrXCombo.clear()
        self.attrYCombo.clear()
        self.attrConditionCombo.clear()        
        self.attrConditionCombo.insertItem("[None]")
        self.attrConditionValueCombo.clear()

        if not self.data: return
        for i in range(len(self.data.domain)):
            if self.data.domain[i].varType == orange.VarTypes.Continuous: continue
            self.attrXCombo.insertItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
            self.attrYCombo.insertItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
            self.attrConditionCombo.insertItem(self.icons[self.data.domain[i].varType], self.data.domain[i].name)
        self.attrCondition = self.attrConditionCombo.text(0)

        if self.attrXCombo.count() > 0:
            self.attrX = str(self.attrXCombo.text(0))
            self.attrY = str(self.attrYCombo.text(self.attrYCombo.count() > 1))

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateData()

    ######################################################################
    ## DATA signal
    # receive new data and update all fields
    def cdata(self, data):
        self.interestingList.clear()
        exData = self.data
        #self.data = None
        #if data: self.data = orange.Preprocessor_dropMissing(data)
        if data: self.data = data

        if not (self.data and exData and str(exData.domain.attributes) == str(self.data.domain.attributes)):  # preserve attribute choice if the domain is the same
            self.initCombos()
        self.attributeSelection(self.attributeSelectionList)


    ######################################################################
    ## Attribute selection signal
    def attributeSelection(self, attrList):
        self.attributeSelectionList = attrList
        if self.data and self.attributeSelectionList and len(attrList) >= 2: 
            try:        # maybe not all attributes in attrList are in current data domain
                self.attrX = attrList[0]
                self.attrY = attrList[1]
            except:
                pass
        
        self.updateData()


    def clearGraph(self):
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for line in self.lines: line.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.lines = []; self.tooltips = []
    


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        self.clearGraph()
        if not self.data: return

        if not self.attrX or not self.attrY: return
        data = self.getConditionalData()

        valsX = []
        valsY = []
        contX = orange.ContingencyAttrAttr(self.attrX, self.attrX, data)   # distribution of X attribute
        contY = orange.ContingencyAttrAttr(self.attrY, self.attrY, data)   # distribution of Y attribute

        # compute contingency of x and y attributes
        for key in contX.keys():
            sum = 0
            try:
                for val in contX[key]: sum += val
            except: pass
            valsX.append(sum)

        for key in contY.keys():
            sum = 0
            try:
                for val in contY[key]: sum += val
            except: pass
            valsY.append(sum)

        # create cartesian product of selected attributes and compute contingency 
        (cart, profit) = FeatureByCartesianProduct(data, [data.domain[self.attrX], data.domain[self.attrY]])
        tempData = data.select(list(data.domain) + [cart])
        contXY = orange.ContingencyAttrAttr(cart, cart, tempData)   # distribution of X attribute

        # compute probabilities
        probs = {}
        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]

                actualProb = 0
                try:
                    for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actualProb += val
                except:
                    actualProb = 0
                probs['%s-%s' %(contX.keys()[i], contY.keys()[j])] = ((contX.keys()[i], valx), (contY.keys()[j], valy), actualProb, len(data))

        # get text width of Y attribute name        
        text = QCanvasText(data.domain[self.attrY].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        xOff = text.boundingRect().right() - text.boundingRect().left() + 40
        yOff = 50
        sqareSize = min(self.canvasView.size().width() - xOff - 35, self.canvasView.size().height() - yOff - 30)
        if sqareSize < 0: return    # canvas is too small to draw rectangles

        # print graph name
        if self.attrCondition == "[None]":
            name  = "P(%s, %s) =\\= P(%s)*P(%s)" %(self.attrX, self.attrY, self.attrX, self.attrY)
        else:
            name = "P(%s, %s | %s = %s) =\\= P(%s | %s = %s)*P(%s | %s = %s)" %(self.attrX, self.attrY, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue), self.attrX, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue), self.attrY, self.attrCondition, getHtmlCompatibleString(self.attrConditionValue))
        self.texts.append(OWCanvasText(self.canvas, name , xOff+ sqareSize/2, 10, Qt.AlignHCenter, bold = 1))
        self.texts.append(OWCanvasText(self.canvas, "N = " + str(len(data)), xOff+ sqareSize/2, 30, Qt.AlignHCenter, bold = 0))

        ######################
        # compute chi-square
        chisquare = 0.0
        for i in range(len(valsX)):
            for j in range(len(valsY)):
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                expected = float(xVal*yVal)/float(sum)
                if expected == 0: continue
                pearson2 = (actual - expected)*(actual - expected) / expected
                chisquare += pearson2

        ######################
        # draw rectangles
        currX = xOff
        for i in range(len(valsX)):
            if valsX[i] == 0: continue
            currY = yOff
            width = int(float(sqareSize * valsX[i])/float(len(data)))

            #for j in range(len(valsY)):
            for j in range(len(valsY)-1, -1, -1):   # this way we sort y values correctly
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                if valsY[j] == 0: continue
                height = int(float(sqareSize * valsY[j])/float(len(data)))

                # create rectangle
                rect = OWCanvasRectangle(self.canvas, currX+2, currY+2, width-4, height-4, z = -10)
                self.rects.append(rect)

                self.addRectIndependencePearson(rect, currX + 1, currY + 1, width-2, height-2, (xAttr, xVal), (yAttr, yVal), actual, sum)
                self.addTooltip(currX+1, currY+1, width-2, height-2, (xAttr, xVal),(yAttr, yVal), actual, sum, chisquare)

                currY += height
                if currX == xOff:
                    self.texts.append(OWCanvasText(self.canvas, data.domain[self.attrY].values[j], xOff - 10, currY - height/2, Qt.AlignRight+Qt.AlignVCenter, bold = 0))

            self.texts.append(OWCanvasText(self.canvas, data.domain[self.attrX].values[i], currX + width/2, yOff + sqareSize + 5, Qt.AlignCenter, bold = 0))
            currX += width

        # show attribute names
        self.texts.append(OWCanvasText(self.canvas, self.attrY, 5, yOff + sqareSize/2, Qt.AlignLeft, bold = 1))
        self.texts.append(OWCanvasText(self.canvas, self.attrX, xOff + sqareSize/2, yOff + sqareSize + 15, Qt.AlignCenter, bold = 1))

        self.canvas.update()

    ######################################################################
    ## show deviations from attribute independence with standardized pearson residuals
    def addRectIndependencePearson(self, rect, x, y, w, h, (xAttr, xVal), (yAttr, yVal), actual, sum):
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)
        
        if pearson > 0:     # if there are more examples that we would expect under the null hypothesis
            intPearson = floor(pearson)
            pen = QPen(QColor(0,0,255)); rect.setPen(pen)
            b = 255
            r = g = 255 - intPearson*20
            r = g = max(r, 55)  #
        elif pearson < 0:
            intPearson = ceil(pearson)
            pen = QPen(QColor(255,0,0))
            rect.setPen(pen)
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            pen = QPen(QColor(255,255,255))
            r = g = b = 255         # white            
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)

        if self.showCases and w > 6 and h > 6:
            if self.showInColor:
                if pearson > 0: c = QColor(0,0,255)
                else: c = QColor(255, 0,0)
            else: c = Qt.black
            for i in range(int(actual)):
                x1 = random.randint(x+1, x + w-4)
                y1 = random.randint(y+1, y + h-4)
                self.rects.append(OWCanvasRectangle(self.canvas, x1, y1, 3, 3, z = 100, penColor = c, brushColor = c))
        
        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson
        if self.showLines == 1: self.addLines(x,y,w,h, kvoc, pen)

    
    #################################################
    # add tooltips
    def addTooltip(self, x,y,w,h, (xAttr, xVal), (yAttr, yVal), actual, sum, chisquare):
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)
        tooltipText = """<b>X Attribute: %s</b><br>Value: <b>%s</b><br>Number of examples (p(x)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Y Attribute: %s</b><br>Value: <b>%s</b><br>Number of examples (p(y)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Number Of Examples (Probabilities):</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b><br>
                        <hr><b>Statistics:</b><br>Chi-square: <b>%.2f</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(self.attrX, getHtmlCompatibleString(xAttr), xVal, 100.0*float(xVal)/float(sum), self.attrY, getHtmlCompatibleString(yAttr), yVal, 100.0*float(yVal)/float(sum), expected, 100.0*float(xVal*yVal)/float(sum*sum), actual, 100.0*float(actual)/float(sum), chisquare, pearson )
        tipRect = QRect(x, y, w, h)
        QToolTip.add(self.canvasView, tipRect, tooltipText)
        self.tooltips.append(tipRect)

    ##################################################
    # add lines
    def addLines(self, x,y,w,h, diff, pen):
        if self.showLines == 0: return
        if diff == 0: return
        #if (xVal == 0) or (yVal == 0) or (actualProb == 0): return

        # create lines
        dist = 20   # original distance between two lines in pixels
        dist = dist * diff
        temp = dist
        while (temp < w):
            line = QCanvasLine(self.canvas)
            line.setPoints(temp+x, y+1, temp+x, y+h-2)
            line.setPen(pen)
            line.show()
            self.lines.append(line)
            temp += dist

        temp = dist
        while (temp < h):
            line = QCanvasLine(self.canvas)
            line.setPoints(x+1, y+temp, x+w-2, y+temp)
            line.setPen(pen)
            line.show()
            self.lines.append(line)
            temp += dist


    ##################################################
    ## SAVING GRAPHS
    ##################################################
    def saveToFileCanvas(self):
        sizeDlg = OWChooseImageSizeDlg(self.canvas)
        sizeDlg.exec_loop()

        
#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    a.setMainWidget(ow)
    ow.show()
    ow.cdata(orange.ExampleTable(r"c:\Development\Python23\Lib\site-packages\Orange\datasets\crush injury - cont.tab"))
    a.exec_loop()
    ow.saveSettings()

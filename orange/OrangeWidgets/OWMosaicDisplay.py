"""
<name>Mosaic Display</name>
<description>Show mosaic display</description>
<category>Classification</category>
<icon>icons/MosaicDisplay.png</icon>
<priority>4110</priority>
"""
# OWMosaicDisplay.py
#
# 

from OWWidget import *
from OData import *
from qt import *
from qtcanvas import *
import orngInteract
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct


###########################################################################################
##### WIDGET : 
###########################################################################################
class OWMosaicDisplay(OWWidget):
    settingsList = ["showLines"]
    #settingsList = ["showLines", "kvoc"]
    #kvocList = ['1.5','2','3','5','10','20']
    #kvocNums = [1.5,   2,  3,  5,  10,  20]

    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Mosaic display", 'show Mosaic display', FALSE, TRUE)

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []
        self.lines = []
        self.tooltips = []

        self.addInput("cdata")
        self.addInput("view")

        #load settings
        self.showLines = 1
        self.kvoc = 2
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

        self.interestingGroupBox = QVGroupBox ("Interesting attribute pairs", self.space)
        self.calculateButton =QPushButton("Calculate chi squares", self.interestingGroupBox)
        self.connect(self.calculateButton, SIGNAL("clicked()"),self.calculatePairs)
        self.interestingList = QListBox(self.interestingGroupBox)
        self.connect(self.interestingList, SIGNAL("selectionChanged()"),self.showSelectedPair)

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)

        self.resize(800, 480)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

    ######################################################################
    ######################################################################
    ##   CALCULATING INTERESTING PAIRS

    ###############################################################
    # when clicked on a list box item, show selected attribute pair
    def showSelectedPair(self):
        if self.interestingList.count() == 0: return

        index = self.interestingList.currentItem()
        (chisquare, strName, xAttr, yAttr) = self.chisquares[index]
        attrXName = self.data.domain[xAttr].name
        attrYName = self.data.domain[yAttr].name
        for i in range(self.attrX.count()):
            if attrXName == str(self.attrX.text(i)): self.attrX.setCurrentItem(i)
            if attrYName == str(self.attrY.text(i)): self.attrY.setCurrentItem(i)
        self.updateData()

    def calculatePairs(self):
        self.chisquares = []
        if self.data == None: return

        self.statusBar.message("Please wait. Computing...")
        total = len(self.data)
        conts = {}
        dc = orange.DomainContingency(self.data)
        for i in range(len(self.data.domain.attributes)):
            if self.data.domain[i].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            cont = dc[i]   # distribution of X attribute
            vals = []
            # compute contingency of x attribute
            for key in cont.keys():
                sum = 0
                for val in cont[key]: sum += val
                vals.append(sum)
            conts[self.data.domain[i].name] = (cont, vals)

        for attrX in range(len(self.data.domain.attributes)):
            if self.data.domain[attrX].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            for attrY in range(attrX+1, len(self.data.domain.attributes)):
                if self.data.domain[attrY].varType == orange.VarTypes.Continuous: continue  # we can only check discrete attributes

                (contX, valsX) = conts[self.data.domain[attrX].name]
                (contY, valsY) = conts[self.data.domain[attrY].name]

                # create cartesian product of selected attributes and compute contingency 
                (cart, profit) = FeatureByCartesianProduct(self.data, [self.data.domain[attrX], self.data.domain[attrY]])
                tempData = self.data.select(list(self.data.domain) + [cart])
                contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute

                # compute chi-square
                chisquare = 0.0
                for i in range(len(valsX)):
                    valx = valsX[i]
                    for j in range(len(valsY)):
                        valy = valsY[j]

                        actual = 0
                        for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actual += val
                        expected = float(valx * valy) / float(total)
                        pearson2 = (actual - expected)*(actual - expected) / expected
                        chisquare += pearson2
                self.chisquares.append((chisquare, "%s - %s" % (self.data.domain[attrX].name, self.data.domain[attrY].name), attrX, attrY))

        ########################
        # populate list box with highest chisquares
        self.chisquares.sort()
        self.chisquares.reverse()
        for (chisquare, attrs, x, y) in self.chisquares:
            str = "%s (%.3f)" % (attrs, chisquare)
            self.interestingList.insertItem(str)        

        self.statusBar.message("")

    ######################################################################
    ######################################################################
        
    def activateLoadedSettings(self):
        self.showLinesCB.setChecked(self.showLines)

        """
        # quotien combo box values        
        for item in self.kvocList: self.kvocCombo.insertItem(item)
        index = self.kvocNums.index(self.kvoc)
        self.kvocCombo.setCurrentItem(index)
        """

        # criteria combo values
        #self.criteriaCombo.insertItem("Attribute independence")
        #self.criteriaCombo.insertItem("Attribute independence (Pearson residuals)")
        #self.criteriaCombo.insertItem("Attribute interactions")
        #self.criteriaCombo.setCurrentItem(0)
        
    ##################################################
    # initialize combo boxes with discrete attributes
    def initCombos(self, data):
        self.attr1.clear(); self.attr2.clear(); self.attr3.clear(); self.attr4.clear()

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
    ## CDATA signal
    # receive new data and update all fields
    def cdata(self, data):
        self.interestingList.clear()
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.initCombos(self.data)
        self.updateData()


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self):
        if self.data == None : return

        self.showLines = self.showLinesCB.isOn()
        #self.kvoc = float(str(self.kvocCombo.currentText()))

        attrList = [str(self.attr1.currentText()), str(self.attr2.currentText())]
        if str(self.attr3.currentText()) != "(none)": attrList.append(str(self.attr3.currentText()))
        if str(self.attr4.currentText()) != "(none)": attrList.append(str(self.attr4.currentText()))

        # hide all rectangles
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for line in self.lines: line.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.lines = []; self.tooltips = []

        # get text width of Y attribute name        
        text = QCanvasText(self.data.domain[str(self.attr2.currentText())].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        xOff = text.boundingRect().right() - text.boundingRect().left() + 25
        yOff = 40
        squareSize = min(self.canvasView.size().width() - xOff - 30 - 30*(len(attrList)>3), self.canvasView.size().height() - yOff - 30 - 30*(len(attrList)>2))
        if squareSize < 0: return    # canvas is too small to draw rectangles

        self.legend = {}        # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList: self.legend[attr] = 0
        self.DrawData(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 1)
        self.DrawText(self.data, attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize))
       
        self.canvas.update()

    def DrawText(self, data, attrList, (x0, x1), (y0, y1)):
        # save values for all attributes
        values = []
        for attr in attrList:
            vals = data.domain[attr].values
            values.append(vals)
        
        #calculate position of first attribute
        self.addText(attrList[0], x0+(x1-x0)/2, y1+30, Qt.AlignCenter, 1)
        currPos = 0        
        for val in values[0]:
            tempData = data.select({attrList[0]:val})
            perc = float(len(tempData))/float(len(data))
            self.addText(str(val), x0+currPos+(x1-x0)*0.5*perc, y1 + 15, Qt.AlignCenter, 1)
            currPos += perc*(x1-x0)

        #calculate position of second attribute
        self.addText(attrList[1], x0-30, y0+(y1-y0)/2, Qt.AlignRight, 1)
        currPos = 0
        tempData = data.select({attrList[0]:values[0][0]})
        for val in values[1]:
            tempData2 = tempData.select({attrList[1]:val})
            perc = float(len(tempData2))/float(len(tempData))
            self.addText(str(val), x0-10, y0+currPos+(y1-y0)*0.5*perc, Qt.AlignRight, 1)
            currPos += perc*(y1-y0)

        if len(attrList) < 3: return

        #calculate position of third attribute
        self.addText(attrList[2], x0 + (x1-x0)*float(len(tempData))/float(2*len(data)), y0 - 25, Qt.AlignCenter, 1)
        currPos = 0
        tempData = data.select({attrList[0]:values[0][0], attrList[1]:values[1][0]})
        for val in values[2]:
            tempData2 = tempData.select({attrList[2]:val})
            perc = float(len(tempData2))/float(len(data))
            self.addText(str(val), x0+currPos+(x1-x0)*0.5*perc, y0 - 10, Qt.AlignCenter, 1)
            currPos += perc*(x1-x0)
        
        if len(attrList) < 4: return

        #calculate position of fourth attribute
        tempData = data.select({attrList[0]:values[0][len(values[0])-1], attrList[1]:values[1][0]})
        self.addText(attrList[3], x1 + 30, (y1-y0)*float(len(tempData))/float(2*len(data)), Qt.AlignLeft, 1)
        currPos = 0
        for val in values[3]:
            tempData2 = tempData.select({attrList[3]:val})
            perc = float(len(tempData2))/float(len(data))
            self.addText(str(val), x1+10, y0 + currPos+ (y1-y0)*0.5*perc, Qt.AlignLeft, 1)
            currPos += perc*(y1-y0)
            
    
    def DrawData(self, data, attrList, (x0, x1), (y0, y1), bHorizontal):
        if len(data) == 0: return
        attr = attrList[0]
        vals = self.data.domain[attr].values
        currPos = 0
        edge = len(attrList)*2  # how much smaller rectangles do we draw

        for val in vals:
            tempData = data.select({attr:val})
            perc = float(len(tempData))/float(len(data))
            if bHorizontal:
                size = (x1-x0)*perc;
                if len(attrList) == 1:  self.addRect(x0+currPos+edge, x0+currPos+size-edge, y0, y1)
                else:                   self.DrawData(tempData, attrList[1:], (x0+currPos+edge, x0+currPos+size-edge), (y0, y1), not bHorizontal)
            else:
                size = (y1-y0)*perc
                if len(attrList) == 1:  self.addRect(x0, x1, y0+currPos+edge, y0+currPos+size-edge)
                else:                   self.DrawData(tempData, attrList[1:], (x0, x1), (y0+currPos+edge, y0+currPos+size-edge), not bHorizontal)
            currPos += size

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
        if x0 > x1: return
        if y0 > y1: return
        rect = QCanvasRectangle(x0, y0, x1-x0, y1-y0, self.canvas)
        rect.setZ(-10)
        rect.show()
        pen = rect.pen(); pen.setWidth(2); rect.setPen(pen)
        self.rects.append(rect)
        return rect

   
    #################################################
    # add tooltips
    def addTooltip(self, x,y,w,h, (xAttr, xVal), (yAttr, yVal), actual, sum, chisquare):
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)
        tooltipText = """<b>X attribute</b><br>Value: <b>%s</b><br>Number of examples (p(x)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Y attribute</b><br>Value: <b>%s</b><br>Number of examples (p(y)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Number of examples (probabilities)</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b><br>
                        <hr><b>Statistics:</b><br>Chi-square: <b>%.2f</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(xAttr, xVal, 100.0*float(xVal)/float(sum), yAttr, yVal, 100.0*float(yVal)/float(sum), expected, 100.0*float(xVal*yVal)/float(sum*sum), actual, 100.0*float(actual)/float(sum), chisquare, pearson )
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
        size = self.canvas.size()
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        
        buffer = QPixmap(size) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        self.canvasView.drawContents(painter, 0,0, size.width(), size.height())
        painter.end()
        buffer.save(fileName, ext)

        

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWMosaicDisplay()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
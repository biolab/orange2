"""
<name>Sieve Diagram</name>
<description>Show sieve diagram</description>
<category>Visualization</category>
<icon>icons/SieveDiagram.png</icon>
<priority>4100</priority>
"""
# OWSieveDiagram.py
#
# 

from OWWidget import *
from qt import *
from qtcanvas import *
import orngInteract
from math import sqrt, floor, ceil, pow
from orngCI import FeatureByCartesianProduct


###########################################################################################
##### WIDGET : 
###########################################################################################
class OWSieveDiagram(OWWidget):
    settingsList = ["showLines"]
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Sieve diagram", 'show sieve diagram', FALSE, TRUE)

        self.inputs = [("Examples", ExampleTable, self.data, 1), ("View", tuple, self.view, 1)]
        self.outputs = []

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []
        self.lines = []
        self.tooltips = []

        #load settings
        self.showLines = 1
        self.loadSettings()

        self.box = QVBoxLayout(self.mainArea)
        self.canvas = QCanvas(2000, 2000)
        self.canvasView = QCanvasView(self.canvas, self.mainArea)
        self.box.addWidget(self.canvasView)
        self.canvasView.show()
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        
        #GUI
        #add controls to self.controlArea widget
        self.controlArea.setMinimumWidth(220)
        self.attrSelGroup = QVGroupBox(self.controlArea)
        self.attrSelGroup.setTitle("Shown attributes")

        self.hbox1 = QHBox(self.attrSelGroup, "x")
        self.attrXCaption = QLabel("X attribute: ", self.hbox1)
        self.attrX = QComboBox(self.hbox1)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateData)

        self.hbox2 = QHBox(self.attrSelGroup, "y")
        self.attrYCaption = QLabel( "Y attribute: ", self.hbox2)
        self.attrY = QComboBox(self.hbox2)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateData)

        self.conditionGroup = QVButtonGroup("Condition", self.controlArea)
        self.box3 = QHBox(self.conditionGroup, "attribute")
        self.box4 = QHBox(self.conditionGroup, "value")
        self.conditionAttrLabel = QLabel("Attribute:", self.box3)
        self.conditionAttr = QComboBox(self.box3)
        self.conditionValLabel = QLabel("Value:", self.box4)
        self.conditionAttrValues  = QComboBox(self.box4)
        self.connect(self.conditionAttr, SIGNAL("activated(int)"), self.updateConditionAttr)
        self.connect(self.conditionAttrValues, SIGNAL("activated(int)"), self.updateConditionAttrValue)

        self.visualSettingsGroup = QVButtonGroup("Visual settings", self.controlArea)        
        self.showLinesCB = QCheckBox('Show lines', self.visualSettingsGroup)
        self.connect(self.showLinesCB, SIGNAL("toggled(bool)"), self.updateData)

        self.interestingGroupBox = QVGroupBox ("Interesting attribute pairs", self.space)
        self.calculateButton =QPushButton("Calculate chi squares", self.interestingGroupBox)
        self.connect(self.calculateButton, SIGNAL("clicked()"),self.calculatePairs)
        self.interestingList = QListBox(self.interestingGroupBox)
        self.connect(self.interestingList, SIGNAL("selectionChanged()"),self.showSelectedPair)

        self.statusBar = QStatusBar(self.mainArea)
        self.box.addWidget(self.statusBar)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.resize(800, 550)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

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
        self.interestingList.clear()
        if self.data == None: return
        data = self.getConditionalData()

        self.statusBar.message("Please wait. Computing...")
        total = len(data)
        conts = {}
        #dc = orange.DomainContingency(data)
        dc = []
        for i in range(len(data.domain)):
            dc.append(orange.ContingencyAttrAttr(data.domain[i], data.domain[i], data))
                      
        for i in range(len(data.domain)):
            if data.domain[i].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            cont = dc[i]   # distribution of X attribute
            vals = []
            # compute contingency of x attribute
            for key in cont.keys():
                sum = 0
                try:
                    for val in cont[key]: sum += val
                except: pass
                vals.append(sum)
            conts[data.domain[i].name] = (cont, vals)

        for attrX in range(len(data.domain)):
            if data.domain[attrX].varType == orange.VarTypes.Continuous: continue      # we can only check discrete attributes

            for attrY in range(attrX+1, len(data.domain)):
                if data.domain[attrY].varType == orange.VarTypes.Continuous: continue  # we can only check discrete attributes

                (contX, valsX) = conts[data.domain[attrX].name]
                (contY, valsY) = conts[data.domain[attrY].name]

                # create cartesian product of selected attributes and compute contingency 
                (cart, profit) = FeatureByCartesianProduct(data, [data.domain[attrX], data.domain[attrY]])
                tempData = data.select(list(data.domain) + [cart])
                contXY = orange.ContingencyAttrAttr(cart, cart, tempData)   # distribution of X attribute

                # compute chi-square
                chisquare = 0.0
                for i in range(len(valsX)):
                    valx = valsX[i]
                    for j in range(len(valsY)):
                        valy = valsY[j]

                        actual = 0
                        try:
                            for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]: actual += val
                        except:
                            actual = 0
                        expected = float(valx * valy) / float(total)
                        if expected == 0: continue
                        pearson2 = (actual - expected)*(actual - expected) / expected
                        chisquare += pearson2
                self.chisquares.append((chisquare, "%s - %s" % (data.domain[attrX].name, data.domain[attrY].name), attrX, attrY))

        ########################
        # populate list box with highest chisquares
        self.chisquares.sort()
        self.chisquares.reverse()
        for (chisquare, attrs, x, y) in self.chisquares:
            str = "%s (%.3f)" % (attrs, chisquare)
            self.interestingList.insertItem(str)        

        self.statusBar.message("")
        
    def activateLoadedSettings(self):
        self.showLinesCB.setChecked(self.showLines)

    ######################################
    # create data subset depending on conditional attribute and value
    def getConditionalData(self):
        attr = str(self.conditionAttr.currentText())
        if attr == "[None]":
            return self.data
        
        attrValue = str(self.conditionAttrValues.currentText())
        return self.data.select({attr:attrValue})

    ######################################
    # new conditional attribute was set - update graph
    def updateConditionAttr(self, index):
        self.conditionAttrValues.clear()
        
        if index == 0:
            self.updateData()
            return
        attrName = str(self.conditionAttr.text(index))
        values = self.data.domain[attrName].values

        for val in values:
            self.conditionAttrValues.insertItem(val)
        self.conditionAttrValues.setCurrentItem(0)
        self.updateConditionAttrValue(0)

    ##########################################
    # new conditional attribute value was set - update graph
    def updateConditionAttrValue(self, index):
        self.updateData()

    ##################################################
    # initialize lists for shown and hidden attributes
    def initCombos(self, data):
        self.attrX.clear()
        self.attrY.clear()
        self.conditionAttr.clear()
        self.conditionAttrValues.clear()
        self.conditionAttr.insertItem("[None]")
    
        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                self.attrX.insertItem(attr.name)
                self.attrY.insertItem(attr.name)
                self.conditionAttr.insertItem(attr.name)
        self.conditionAttr.setCurrentItem(0)
        self.updateConditionAttr(0)

        if self.attrX.count() > 0:  self.attrX.setCurrentItem(0)
        if self.attrY.count() > 1:  self.attrY.setCurrentItem(1)

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateData()

    ######################################################################
    ## VIEW signal
    def view(self, (attr1, attr2)):
        if self.data == None:
            return

        ind1 = -1; ind2 = -1; classInd = 0
        for i in range(self.attrX.count()):
            if str(self.attrX.text(i)) == attr1: ind1 = i
            if str(self.attrX.text(i)) == attr2: ind2 = i

        if ind1 == ind2 == -1:
            print "no valid attributes found"
            return    # something isn't right

        self.attrX.setCurrentItem(ind1)
        self.attrY.setCurrentItem(ind2)
        self.updateData()

    ######################################################################
    ## DATA signal
    # receive new data and update all fields
    def data(self, data):
        self.interestingList.clear()
        self.data = orange.Preprocessor_dropMissing(data)
        self.initCombos(self.data)
        self.updateData()


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self, *args):
        if self.data == None : return

        self.showLines = self.showLinesCB.isOn()
        
        attrX = str(self.attrX.currentText())
        attrY = str(self.attrY.currentText())

        # hide all rectangles
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for line in self.lines: line.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.lines = []; self.tooltips = []
    
        if attrX == "" or attrY == "": return
        data = self.getConditionalData()

        total = len(data)
        valsX = []
        valsY = []
        contX = orange.ContingencyAttrAttr(attrX, attrX, data)   # distribution of X attribute
        contY = orange.ContingencyAttrAttr(attrY, attrY, data)   # distribution of Y attribute

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
        (cart, profit) = FeatureByCartesianProduct(data, [data.domain[attrX], data.domain[attrY]])
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
                probs['%s-%s' %(contX.keys()[i], contY.keys()[j])] = ((contX.keys()[i], valx), (contY.keys()[j], valy), actualProb, total)

        # get text width of Y attribute name        
        text = QCanvasText(data.domain[attrY].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        xOff = text.boundingRect().right() - text.boundingRect().left() + 40
        yOff = 50
        sqareSize = min(self.canvasView.size().width() - xOff - 35, self.canvasView.size().height() - yOff - 30)
        if sqareSize < 0: return    # canvas is too small to draw rectangles

        # print graph name
        condition = str(self.conditionAttr.currentText())
        attr1 = str(self.attrX.currentText())
        attr2 = str(self.attrY.currentText())
        if condition == "[None]":
            name  = "P(%s, %s) =\\= P(%s)*P(%s)" %(attr1, attr2, attr1, attr2)
        else:
            condVal = str(self.conditionAttrValues.currentText())
            name = "P(%s, %s | %s = %s) =\\= P(%s | %s = %s)*P(%s | %s = %s)" %(attr1, attr2, condition, condVal, attr1, condition, condVal, attr2, condition, condVal)
        self.addText(name, xOff+ sqareSize/2, 10, Qt.AlignHCenter, 1)
        self.addText("N = " + str(len(data)), xOff+ sqareSize/2, 30, Qt.AlignHCenter, 0)

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
            width = int(float(sqareSize * valsX[i])/float(total))

            #for j in range(len(valsY)):
            for j in range(len(valsY)-1, -1, -1):   # this way we sort y values correctly
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                if valsY[j] == 0: continue
                height = int(float(sqareSize * valsY[j])/float(total))

                # create rectangle
                rect = QCanvasRectangle(currX+2, currY+2, width-4, height-4, self.canvas)
                rect.setZ(-10)
                rect.show()
                self.rects.append(rect)

                self.addRectIndependencePearson(rect, currX + 1, currY + 1, width-2, height-2, (xAttr, xVal), (yAttr, yVal), actual, sum)
                self.addTooltip(currX+1, currY+1, width-2, height-2, (xAttr, xVal),(yAttr, yVal), actual, sum, chisquare)

                currY += height
                if currX == xOff:
                    self.addText(data.domain[attrY].values[j], xOff - 10, currY - height/2, Qt.AlignRight+Qt.AlignVCenter, 0)

            self.addText(data.domain[attrX].values[i], currX + width/2, yOff + sqareSize + 5, Qt.AlignCenter, 0)
            currX += width

        # show attribute names
        self.addText(data.domain[attrY].name, 5, yOff + sqareSize/2, Qt.AlignLeft, 1)
        self.addText(data.domain[attrX].name, xOff + sqareSize/2, yOff + sqareSize + 15, Qt.AlignCenter, 1)

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
            rect.setPen(QPen(QColor(255,0,0), 1, Qt.DashLine))
            pen = QPen(QColor(255,0,0))
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            pen = QPen(QColor(255,255,255))
            r = g = b = 255         # white            
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)
        
        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson
        if self.showLines == 1: self.addLines(x,y,w,h, kvoc, pen)

    # draws text with caption name at position x,y with alignment and style
    def addText(self, name, x, y, alignment, bold):
        text = QCanvasText(name, self.canvas);
        text.setTextFlags(alignment);
        font = text.font(); font.setBold(bold); text.setFont(font)
        text.move(x, y)
        text.show()
        self.texts.append(text)

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
    ow=OWSieveDiagram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
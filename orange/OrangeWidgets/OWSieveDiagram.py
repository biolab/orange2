"""
<name>Sieve Diagram</name>
<description>Show sieve diagram</description>
<category>Classification</category>
<icon>icons/SieveDiagram.png</icon>
<priority>4100</priority>
"""
# OWSieveDiagram.py
#
# 

from OWWidget import *
from OWSieveDiagramOptions import *
from OData import *
from qt import *
from qtcanvas import *
import orngInteract
from math import sqrt, floor, ceil
from orngCI import FeatureByCartesianProduct

###########################################################################################
##### WIDGET : 
###########################################################################################
class OWSieveDiagram(OWWidget):
    settingsList = ["showLines", "kvoc"]
    kvocList = ['1.5','2','3','5','10','20']
    kvocNums = [1.5,   2,  3,  5,  10,  20]

    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Sieve diagram", 'show sieve diagram', FALSE, FALSE)

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
        self.shownCriteriaGroup = QVGroupBox(self.controlArea)
        self.shownCriteriaGroup.setTitle("Shown criteria")
        self.criteriaCombo = QComboBox(self.shownCriteriaGroup)
        self.connect(self.criteriaCombo, SIGNAL('activated ( const QString & )'), self.updateData)
        
        self.attrSelGroup = QVGroupBox(self.controlArea)
        self.attrSelGroup.setTitle("Shown attributes")

        self.attrXGroup = QVButtonGroup("X axis attribute", self.attrSelGroup)
        self.attrX = QComboBox(self.attrXGroup)
        self.connect(self.attrX, SIGNAL('activated ( const QString & )'), self.updateData)

        self.attrYGroup = QVButtonGroup("Y axis attribute", self.attrSelGroup)
        self.attrY = QComboBox(self.attrYGroup)
        self.connect(self.attrY, SIGNAL('activated ( const QString & )'), self.updateData)

        self.hbox = QHBox(self.controlArea, "kvocient")
        self.kvocLabel = QLabel('Quotient', self.hbox)
        self.kvocCombo = QComboBox(self.hbox)
        self.connect(self.kvocCombo, SIGNAL('activated ( const QString & )'), self.updateData)
        QToolTip.add(self.hbox, "What is maximum expected ratio between p(x,y) and p(x)*p(y). Greater the ratio, brighter the colors.")

        self.showLinesCB = QCheckBox('Show lines', self.controlArea)
        self.connect(self.showLinesCB, SIGNAL("toggled(bool)"), self.updateData)


        self.saveCanvas = QPushButton("Save diagram", self.controlArea)
        self.connect(self.saveCanvas, SIGNAL("clicked()"), self.saveToFileCanvas)

        #connect controls to appropriate functions
        self.activateLoadedSettings()


    def activateLoadedSettings(self):
        self.showLinesCB.setChecked(self.showLines)

        # quotien combo box values        
        for item in self.kvocList: self.kvocCombo.insertItem(item)
        index = self.kvocNums.index(self.kvoc)
        self.kvocCombo.setCurrentItem(index)

        # criteria combo values
        self.criteriaCombo.insertItem("Attribute independence")
        self.criteriaCombo.insertItem("Attribute independence (Pearson residuals)")
        self.criteriaCombo.insertItem("Attribute interactions")
        self.criteriaCombo.setCurrentItem(1)
        
    ##################################################
    # initialize lists for shown and hidden attributes
    def initCombos(self, data):
        self.attrX.clear()
        self.attrY.clear()

        for attr in data.domain.attributes:
            if attr.varType == orange.VarTypes.Discrete:
                self.attrX.insertItem(attr.name)
                self.attrY.insertItem(attr.name)

        if self.attrX.count() > 0:
            self.attrX.setCurrentItem(0)
        if self.attrY.count() > 1:
            self.attrY.setCurrentItem(1)

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
        for i in range(self.attrX.count()):
            if str(self.attrX.text(i)) == attr1: ind1 = i
            if str(self.attrX.text(i)) == attr2: ind2 = i

        if ind1 == ind2 == 0:
            print "no valid attributes found"
            return    # something isn't right

        self.attrX.setCurrentItem(ind1)
        self.attrY.setCurrentItem(ind2)
        self.updateData()

    ######################################################################
    ## CDATA signal
    # receive new data and update all fields
    def cdata(self, data):
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.initCombos(self.data)
        self.updateData()


    ######################################################################
    ## UPDATEDATA - gets called every time the graph has to be updated
    def updateData(self):
        if self.data == None : return

        self.showLines = self.showLinesCB.isOn()
        self.kvoc = float(str(self.kvocCombo.currentText()))
        
        attrX = str(self.attrX.currentText())
        attrY = str(self.attrY.currentText())

        # hide all rectangles
        for rect in self.rects: rect.hide()
        for text in self.texts: text.hide()
        for line in self.lines: line.hide()
        for tip in self.tooltips: QToolTip.remove(self.canvasView, tip)
        self.rects = []; self.texts = [];  self.lines = []; self.tooltips = []
    
        if attrX == "" or attrY == "": return

        total = len(self.data)
        valsX = []
        valsY = []
        contX = orange.ContingencyAttrClass(attrX, self.data)   # distribution of X attribute
        contY = orange.ContingencyAttrClass(attrY, self.data)   # distribution of Y attribute

        # compute contingency of x and y attributes
        for key in contX.keys():
            sum = 0
            for val in contX[key]: sum += val
            valsX.append(sum)

        for key in contY.keys():
            sum = 0
            for val in contY[key]: sum += val
            valsY.append(sum)

        # create cartesian product of selected attributes and compute contingency 
        (cart, profit) = FeatureByCartesianProduct(self.data, [self.data.domain[attrX], self.data.domain[attrY]])
        tempData = self.data.select(list(self.data.domain) + [cart])
        contXY = orange.ContingencyAttrClass(cart, tempData)   # distribution of X attribute

        # compute probabilities
        probs = {}
        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]

                actualProb = 0
                for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]:
                    actualProb += val
                probs['%s-%s' %(contX.keys()[i], contY.keys()[j])] = ((contX.keys()[i], valx), (contY.keys()[j], valy), actualProb, total)
       
        # get text width of Y attribute name        
        text = QCanvasText(self.data.domain[attrY].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        xOff = text.boundingRect().right() - text.boundingRect().left() + 25
        yOff = 20
        sqareSize = min(self.canvasView.size().width() - xOff - 20, self.canvasView.size().height() - yOff - 30)
        if sqareSize < 0: return    # canvas is too small to draw rectangles

        criteriaText = str(self.criteriaCombo.currentText())
        currX = xOff
        for i in range(len(valsX)):
            itemX = valsX[i]
            currY = yOff
            width = int(float(sqareSize * itemX)/float(total))

            for j in range(len(valsY)):
                ((xAttr, xVal), (yAttr, yVal), actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                itemY = valsY[j]
                height = int(float(sqareSize * itemY)/float(total))

                # create rectangle
                rect = QCanvasRectangle(currX+1, currY+1, width-2, height-2, self.canvas)
                rect.setZ(-10)
                rect.show()
                self.rects.append(rect)

                if criteriaText == "Attribute independence":  self.addRectIndependence(rect, currX + 1, currY + 1, width-2, height-2, (xAttr, xVal), (yAttr, yVal), actual, sum)
                elif criteriaText == "Attribute independence (Pearson residuals)": self.addRectIndependencePearson(rect, currX + 1, currY + 1, width-2, height-2, (xAttr, xVal), (yAttr, yVal), actual, sum)

                currY += height
                if currX == xOff:
                    text = QCanvasText(self.data.domain[attrY].values[j], self.canvas);
                    text.setTextFlags(Qt.AlignRight);
                    text.move(xOff - 10, currY - height/2);
                    text.show()
                    self.texts.append(text)
            
            text = QCanvasText(self.data.domain[attrX].values[i], self.canvas);
            text.setTextFlags(Qt.AlignCenter);
            text.move(currX + width/2, yOff + sqareSize + 5);
            text.show()
            self.texts.append(text)
            currX += width

        # show attribute names
        text = QCanvasText(self.data.domain[attrY].name, self.canvas);
        text.setTextFlags(Qt.AlignLeft);
        font = text.font(); font.setBold(1); text.setFont(font)
        text.move(5, yOff + sqareSize/2);
        text.show()
        self.texts.append(text)
        text = QCanvasText(self.data.domain[attrX].name, self.canvas);
        text.setTextFlags(Qt.AlignCenter);
        font = text.font(); font.setBold(1); text.setFont(font)
        text.move(xOff + sqareSize/2, yOff + sqareSize + 15);
        text.show()
        self.texts.append(text)

        self.canvas.update()


    ######################################################################
    ## show deviations from attribute independence
    def addRectIndependence(self, rect, x, y, w, h, (xAttr, xVal), (yAttr, yVal), actual, sum):
        independentProb = float(xVal*yVal)/float(sum*sum)
        actualProb = float(actual) /float(sum)
        if (xVal*yVal < 5) and (actual < 5): return   # in case we have too little examples we don't estimate the deviation from independence

        constA = -205.0 / float(self.kvoc)
        constB = 255 - constA

        # set color
        if actualProb > independentProb:
            pen = QPen(QColor(0,0,255))
            b = 255
            if independentProb == 0: r = g = constA*actualProb*len(self.data)+ 255
            else:                r = g = constA*actualProb/independentProb + 255 - constA   # if actual/independent = 10 --> r=g=255; actual==independent --> r=g=0
            r = g = max(r, 50)   # if actual/independent > 10 --> r=g=50     -- we don't go under 50
        else:
            pen = QPen(QColor(255,0,0))
            r = 255
            if actualProb == 0: g = b = constA*independentProb*len(self.data) + 255  
            else:           g = b = constA*independentProb/actualProb + 255 - constA   # if independent/actual= 10 --> g=b=255; actual==independent --> r=g=0
            g = b = max(g, 50)  # if actual/independent > 10 --> b=g=50     -- we don't go under 50
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)
        self.addTooltip(x,y,w,h, (xAttr, xVal),(yAttr, yVal), actual, sum)
        if self.showLines == 1 and actualProb > 0 and independentProb > 0: self.addLines(x,y,w,h, independentProb/actualProb, pen)


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
            pen = QPen(QColor(255,0,0)); rect.setPen(pen)
            r = 255
            b = g = 255 + intPearson*20
            b = g = max(b, 55)
        else:
            r = g = b = 255         # white            
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)
        self.addTooltip(x,y,w,h, (xAttr, xVal),(yAttr, yVal), actual, sum)
        
        if pearson > 0:
            pearson = min(pearson, 10)
            kvoc = 1 - 0.08 * pearson       #  if pearson in [0..10] --> kvoc in [1..0.2]
        else:
            pearson = max(pearson, -10)
            kvoc = 1 - 0.4*pearson
        if self.showLines == 1: self.addLines(x,y,w,h, kvoc, pen)


    #################################################
    # add tooltips
    def addTooltip(self, x,y,w,h, (xAttr, xVal), (yAttr, yVal), actual, sum):
        expected = float(xVal*yVal)/float(sum)
        pearson = (actual - expected) / sqrt(expected)
        tooltipText = """<b>X attribute</b><br>Value: <b>%s</b><br>Number of examples (p(x)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Y attribute</b><br>Value: <b>%s</b><br>Number of examples (p(y)): <b>%d (%.2f%%)</b><br><hr>
                        <b>Number of examples (probabilities)</b><br>Expected (p(x)p(y)): <b>%.1f (%.2f%%)</b><br>Actual (p(x,y)): <b>%d (%.2f%%)</b><br>
                        <hr><b>Statistics:</b><br>Standardized Pearson residual: <b>%.2f</b>""" %(xAttr, xVal, 100.0*float(xVal)/float(sum), yAttr, yVal, 100.0*float(yVal)/float(sum), expected, 100.0*float(xVal*yVal)/float(sum*sum), actual, 100.0*float(actual)/float(sum), pearson )
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
"""
<name>Sieve Diagram</name>
<description>Show sieve diagram (mosaic plot)</description>
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
import statc
import os
from orngCI import FeatureByCartesianProduct

class QVerticalCanvasText(QCanvasText):
    def __init__(self, *args):
        apply(QCanvasText.__init__,(self,)+ args)

    def draw(self, painter):
        point = QPoint(self.x(),self.y())
        painter.rotate(-90.0)
        point = painter.xFormDev(point)

        oldFont= painter.font()
        painter.setFont(self.font())
        painter.drawText(point,self.text())
        painter.setFont(oldFont)
        painter.rotate(90.0)
    

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

        self.addInput("cdata")

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
        
        self.showLinesCB = QCheckBox('Show lines', self.controlArea)
        self.connect(self.showLinesCB, SIGNAL("toggled(bool)"), self.updateData)


        self.saveCanvas = QPushButton("Save diagram", self.controlArea)
        self.connect(self.saveCanvas, SIGNAL("clicked()"), self.saveToFileCanvas)

        #connect controls to appropriate functions
        self.activateLoadedSettings()


    def activateLoadedSettings(self):
        self.showLinesCB.setChecked(self.showLines)

        for item in self.kvocList: self.kvocCombo.insertItem(item)
        index = self.kvocNums.index(self.kvoc)
        self.kvocCombo.setCurrentItem(index)

    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateData()

    ####### CDATA ################################
    # receive new data and update all fields
    def cdata(self, data):
        self.data = orange.Preprocessor_dropMissing(data.data)
        self.initCombos(self.data)
        self.updateData()
        
    def updateData(self):
        if self.data == None : return

        self.showLines = self.showLinesCB.isOn()
        self.kvoc = float(str(self.kvocCombo.currentText()))
        
        attrX = str(self.attrX.currentText())
        attrY = str(self.attrY.currentText())

        # hide all rectangles
        for rect in self.rects:
            rect.hide()
        self.rects = []

        for text in self.texts:
            text.hide()
        self.texts = []

        for line in self.lines:
            line.hide()
        self.lines = []
    
        if attrX == "" or attrY == "":
            return

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
        sum = 0
        probs = {}
        for val in valsX:
            sum += val

        for i in range(len(valsX)):
            valx = valsX[i]
            for j in range(len(valsY)):
                valy = valsY[j]
                independentProb = float(valx*valy)/float(sum*sum)
                actualProb = 0
                
                for val in contXY['%s-%s' %(contX.keys()[i], contY.keys()[j])]:
                    actualProb += val
                actualProb = float(actualProb) /float(sum)
                probs['%s-%s' %(contX.keys()[i], contY.keys()[j])] = (independentProb, actualProb, sum)
       
        # get text width of Y attribute name        
        text = QCanvasText(self.data.domain[attrY].name, self.canvas);
        font = text.font(); font.setBold(1); text.setFont(font)
        xOff = text.boundingRect().right() - text.boundingRect().left() + 25
        yOff = 20
        sqareSize = min(self.canvasView.size().width() - xOff - 20, self.canvasView.size().height() - yOff - 30)
        if sqareSize < 0: return    # canvas is too small to draw rectangles

        currX = xOff
        for i in range(len(valsX)):
            itemX = valsX[i]
            currY = yOff
            width = int(float(sqareSize * itemX)/float(total))

            for j in range(len(valsY)):
                (independent, actual, sum) = probs['%s-%s' %(contX.keys()[i], contY.keys()[j])]
                itemY = valsY[j]
                height = int(float(sqareSize * itemY)/float(total))
                self.addRectLines(currX + 1, currY + 1, width-2, height-2, independent, actual, sum)
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


    ##################################################
    # SAVING GRAPHS
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

    def addRectLines(self, x, y, w, h, independent, actual, sum):
        # create rectangle
        rect = QCanvasRectangle(x, y, w, h, self.canvas)
        rect.setZ(-10)
        rect.show()
        self.rects.append(rect)

        if (independent*sum < 5) and (actual*sum < 5): return   # in case we have too little examples we don't estimate the deviation from independence

        constA = -205.0 / float(self.kvoc)
        constB = 255 - constA

        # set color
        if actual > independent:
            pen = QPen(QColor(0,0,255))
            b = 255
            if independent == 0: r = g = constA*actual*len(self.data)+ 255
            else:                r = g = constA*actual/independent + 255 - constA   # if actual/independent = 10 --> r=g=255; actual==independent --> r=g=0
            r = g = max(r, 50)   # if actual/independent > 10 --> r=g=50     -- we don't go under 50
        else:
            pen = QPen(QColor(255,0,0))
            r = 255
            if actual == 0: g = b = constA*independent*len(self.data) + 255  
            else:           g = b = constA*independent/actual + 255 - constA   # if independent/actual= 10 --> g=b=255; actual==independent --> r=g=0
            g = b = max(g, 50)  # if actual/independent > 10 --> b=g=50     -- we don't go under 50
        color = QColor(r,g,b)
        brush = QBrush(color); rect.setBrush(brush)

        if self.showLines == 0: return
        if actual == 0: return

        # create lines
        dist = 20   # original distance between two lines in pixels
        dist = dist * (independent/actual)
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
            


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
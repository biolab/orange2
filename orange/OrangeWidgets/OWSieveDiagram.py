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
    settingsList = []
    
    def __init__(self,parent=None):
        OWWidget.__init__(self, parent, "Sieve diagram", 'show sieve diagram', FALSE, FALSE)

        #set default settings
        self.data = None
        self.rects = []
        self.texts = []

        self.addInput("cdata")

        #load settings
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

        self.saveCanvas = QPushButton("Save left canvas", self.controlArea)
        self.connect(self.saveCanvas, SIGNAL("clicked()"), self.saveToFileCanvas)

        #connect controls to appropriate functions
        self.activateLoadedSettings()

    def activateLoadedSettings(self):
        pass

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
        attrX = str(self.attrX.currentText())
        attrY = str(self.attrY.currentText())

        # hide all rectangles
        for rect in self.rects:
            rect.hide()
        self.rects = []

        for text in self.texts:
            text.hide()
        self.texts = []
    
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
            for val in contX[key]:
                sum += val
            valsX.append(sum)

        for key in contY.keys():
            sum = 0
            for val in contY[key]:
                sum += val
            valsY.append(sum)

        xOff = 40
        yOff = 40
        sqareSize = min(self.canvasView.size().width(), self.canvasView.size().height()) - max(xOff, yOff) - 30
        if sqareSize < 0: return    # canvas is too small to draw rectangles

        currX = xOff
        for i in range(len(valsX)):
            itemX = valsX[i]
            currY = yOff
            width = int(float(sqareSize * itemX)/float(total))
            for j in range(len(valsY)):
                itemY = valsY[j]
                height = int(float(sqareSize * itemY)/float(total))
                rect = QCanvasRectangle(currX + 1, currY + 1, width-2, height-2, self.canvas)
                self.rects.append(rect)
                rect.show()
                currY += height
                if currX == xOff:
                    text = QVerticalCanvasText(self.data.domain[attrY].values[j], self.canvas);
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
        text = QVerticalCanvasText(self.data.domain[attrY].name, self.canvas);
        text.setTextFlags(Qt.AlignCenter);
        font = text.font(); font.setBold(1); text.setFont(font)
        text.move(xOff - 25, yOff + sqareSize/2);
        text.show()
        self.texts.append(text)
        text = QCanvasText(self.data.domain[attrX].name, self.canvas);
        text.setTextFlags(Qt.AlignCenter);
        font = text.font(); font.setBold(1); text.setFont(font)
        text.move(xOff + sqareSize/2, yOff + sqareSize + 15);
        text.show()
        self.texts.append(text)

        """
            rect3 = QCanvasRectangle(x3, rectsYOff, x4-x3, rectHeight, self.canvasL)
            if interaction < 0.0:
                color = QColor(200, 0, 0)
                style = Qt.DiagCrossPattern
            else:
                color = QColor(Qt.green)
                style = Qt.Dense5Pattern

            brush3 = QBrush(Qt.blue); brush3.setStyle(Qt.FDiagPattern)
            
            rect3.setBrush(brush3); rect3.setPen(QPen(QColor(Qt.blue)))
      
            tooltipRect = QRect(x1-self.viewXPos, rectsYOff-self.viewYPos, x4-x1, rectHeight)
            tooltipText = "%s : <b>%.1f%%</b><br>%s : <b>%.1f%%</b><br>Interaction : <b>%.1f%%</b><br>Total entropy removed: <b>%.1f%%</b>" %(data.domain[attrIndex1].name, gain1*100, data.domain[attrIndex2].name, gain2*100, interaction*100, total*100)
            QToolTip.add(self.canvasViewL, tooltipRect, tooltipText)

            # compute line width
            rect = text2.boundingRect()
            lineWidth = xOff + xscale*total + 5 + rect.width() + 10
            if  lineWidth > maxWidth:
                maxWidth = lineWidth 

            if rectsYOff + rectHeight + 10 > maxHeight:
                maxHeight = rectsYOff + rectHeight + 10

            self.interactionRects.append((rect1, rect2, rect3, text1, text2, QRect(x1, rectsYOff, x4-x1, rectHeight), tooltipText))
            index += 1

        # resizing of the left canvas to update width
        self.canvasViewL.setMaximumSize(QSize(maxWidth + 30, max(2000, maxHeight)))
        self.canvasViewL.setMinimumWidth(maxWidth + 10)
        self.canvasL.resize(maxWidth + 10, maxHeight)
        self.canvasViewL.setMinimumWidth(0)

        
        """
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


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWSieveDiagram()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()
import os
from qwt import QwtPlot
from qtcanvas import QCanvas
from OWBaseWidget import *
import OWGUI
import OWVisAttrSelection

contMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief()), ("Fisher discriminant", OWVisAttrSelection.MeasureFisherDiscriminant())]
discMeasures = [("None", None), ("ReliefF", orange.MeasureAttribute_relief()), ("Gain ratio", orange.MeasureAttribute_gainRatio()), ("Gini index", orange.MeasureAttribute_gini())]

class OWAttributeOrder(OWBaseWidget):
    def __init__(self, cont, disc):
        OWBaseWidget.__init__(self, None, "Attribute Order", "Choose heuristics to sort attributes by their information", TRUE, FALSE, FALSE, modal = TRUE)

        self.attrCont = cont
        self.attrDisc = disc
                
        self.grid=QGridLayout(self)
        space=QVBox(self)
        self.grid.addWidget(space,0,0)

        self.topText = QLabel("\n  Select the measures of attribute usefulness that  \n  will determine the order of projection evaluation  \n", space)
        
        OWGUI.radioButtonsInBox(space, self, "attrCont", [val for (val, measure) in contMeasures], box = " Ordering of continuous attributes")
        OWGUI.radioButtonsInBox(space, self, "attrDisc", [val for (val, measure) in discMeasures], box = " Ordering of discrete attributes")

        self.okButton = OWGUI.button(space, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(space, self, "Cancel", callback = self.reject)
    
    def evaluateAttributes(self, data):
        return OWVisAttrSelection.evaluateAttributes(data, contMeasures[self.attrCont][1], discMeasures[self.attrDisc][1])

        
        
class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName"]
    def __init__(self, graph):
        OWBaseWidget.__init__(self, None, "Image settings", "Set size of output image", TRUE, FALSE, FALSE, modal = TRUE)

        self.graph = graph
        self.selectedSize = 0
        self.customX = 400
        self.customY = 400
        self.saveAllSizes = 0
        self.lastSaveDirName = os.getcwd() + "/"

        self.loadSettings()
        
        self.space=QVBox(self)
        self.grid=QGridLayout(self)
        self.grid.addWidget(self.space,0,0)
        self.group = QVGroupBox("Image size", self.space)
        self.imageSize = QButtonGroup(5, Qt.Vertical, self.group)
        self.imageSize.setFrameStyle(QFrame.NoFrame)

        self.sizeOriginal = QRadioButton('Original size', self.imageSize)
        self.size400 = QRadioButton('400 x 400', self.imageSize)
        self.size600 = QRadioButton('600 x 600', self.imageSize)
        self.size800 = QRadioButton('800 x 800', self.imageSize)
        self.custom  = QRadioButton('Custom:', self.imageSize)
        self.boxX = QHBox(self.group)
        self.boxY = QHBox(self.group)
        self.customWidth = QLabel('Width:', self.boxX)
        self.xSize = QLineEdit(self.boxX)
        self.customHeight = QLabel('Height:', self.boxY)
        self.ySize = QLineEdit(self.boxY)
        self.sizeOriginal.setChecked(1)
        
        self.printButton = QPushButton("Print", self.space)
        self.okButton = QPushButton("Save image", self.space)
        self.cancelButton = QPushButton("Cancel", self.space)
        self.connect(self.printButton, SIGNAL("clicked()"), self.printPic)
        self.connect(self.okButton, SIGNAL("clicked()"), self.accept)
        self.connect(self.cancelButton, SIGNAL("clicked()"), self.reject)
                

        if self.selectedSize == 0: self.sizeOriginal.setChecked(1)
        elif self.selectedSize == 1: self.size400.setChecked(1)
        elif self.selectedSize == 2: self.size600.setChecked(1)
        elif self.selectedSize == 3: self.size600.setChecked(1)
        self.xSize.setText(str(self.customX))
        self.ySize.setText(str(self.customY))
        self.resize(200,300)

    def accept(self):
        if self.sizeOriginal.isChecked(): self.selectedSize = 0
        elif self.size400.isChecked(): self.selectedSize = 1
        elif self.size600.isChecked(): self.selectedSize = 2
        elif self.size800.isChecked(): self.selectedSize = 3
        self.customX = int(str(self.xSize.text()))
        self.customY = int(str(self.ySize.text()))
        self.saveToFile()
        self.saveSettings()
        QDialog.accept(self)


    def printPic(self):
        printer = QPrinter()

        if self.sizeOriginal.isChecked(): size = self.size()
        elif self.size400.isChecked(): size = QSize(400,400)
        elif self.size600.isChecked(): size = QSize(600,600)
        elif self.size800.isChecked(): size = QSize(800,800)
        elif self.custom.isChecked():  size = QSize(int(str(self.xSize.text())), int(str(self.ySize.text())))
        buffer = QPixmap(size)

        if printer.setup():
            painter = QPainter(printer)
            metrics = QPaintDeviceMetrics(printer)
            height = metrics.height() - 2*printer.margins().height()
            width = metrics.width() - 2*printer.margins().width()
            if height == 0:
                print "Error. Height is zero. Preventing division by zero."
                return
            pageKvoc = width / float(height)
            sizeKvoc = size.width() / float(size.height())
            if pageKvoc < sizeKvoc:     rect = QRect(printer.margins().width(),printer.margins().height(), width, height*pageKvoc/sizeKvoc)
            else:                       rect = QRect(printer.margins().width(),printer.margins().height(), width*sizeKvoc/pageKvoc, height)
            self.fillPainter(painter, rect)
            painter.end()
        QDialog.accept(self)

    def saveToFile(self):
        qfileName = QFileDialog.getSaveFileName(self.lastSaveDirName + "graph.png","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".",""); ext = ext.upper()
        if ext == "" or not (ext == "BMP" or ext == "GIF" or ext == "PNG") :	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"

        dirName, shortFileName = os.path.split(fileName)
        self.lastSaveDirName = dirName + "/"
        self.saveToFileDirect(fileName, ext, self.getSize())

    def getSize(self):
        if self.sizeOriginal.isChecked(): return self.graph.size()
        elif self.size400.isChecked(): return QSize(400,400)
        elif self.size600.isChecked(): return QSize(600,600)
        elif self.size800.isChecked(): return QSize(800,800)
        elif self.custom.isChecked():  return QSize(int(str(self.xSize.text())), int(str(self.ySize.text())))
        else: return QSize(400,400)
        
    def saveToFileDirect(self, fileName, ext, size, overwriteExisting = 0):
        if os.path.exists(fileName) and not overwriteExisting:
            res = QMessageBox.information(self,'Save picture','File already exists. Overwrite?','Yes','No', QString.null,0,1)
            if res == 1: return
        painter = QPainter()
        if size.isEmpty(): buffer = QPixmap(self.graph.size()) # any size can do, now using the window size
        else:              buffer = QPixmap(size)
        painter.begin(buffer)
        painter.fillRect(buffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background
        self.fillPainter(painter, buffer.rect())
        painter.flush()
        painter.end()
        buffer.save(fileName, ext)
        

    def fillPainter(self, painter, rect):
        if isinstance(self.graph, QwtPlot):
            self.graph.printPlot(painter, rect)
        elif isinstance(self.graph, QCanvas):
            # draw background
            self.graph.drawBackground(painter, rect)

            # draw items
            items = self.graph.allItems()
            sortedList = []
            for item in items:
                sortedList.append((item.z(), item))
            sortedList.sort()   # sort items by z value
            for (z, item) in sortedList:
                if item.visible(): item.draw(painter)

            # draw foreground
            self.graph.drawForeground(painter, rect)


if __name__== "__main__":
    a = QApplication(sys.argv)
    c = OWAttributeOrder(0,0)
    
    a.setMainWidget(c)
    c.show()
    a.exec_loop()


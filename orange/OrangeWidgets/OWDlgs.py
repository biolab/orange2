import os
from qwt import QwtPlot
from qtcanvas import QCanvas
from OWBaseWidget import *

class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "saveAllSizes", "lastSaveDirName"]
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
        self.allSizes = QCheckBox("Save all sizes", self.group)
        
        self.printButton = QPushButton("Print", self.space)
        self.okButton = QPushButton("Save image", self.space)
        self.cancelButton = QPushButton("Cancel", self.space)
        self.connect(self.printButton, SIGNAL("clicked()"), self.printPic)
        self.connect(self.okButton, SIGNAL("clicked()"), self.accept)
        self.connect(self.cancelButton, SIGNAL("clicked()"), self.reject)
                
        if self.saveAllSizes == 1: self.allSizes.setChecked(1)

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
        self.saveAllSizes = self.allSizes.isChecked()
        self.saveSettings()
        self.saveToFile()
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
            pageKvoc = width / float(height)
            sizeKvoc = size.width() / float(size.height())
            if pageKvoc < sizeKvoc:     rect = QRect(printer.margins().width(),printer.margins().height(), width, height*pageKvoc/sizeKvoc)
            else:                       rect = QRect(printer.margins().width(),printer.margins().height(), width*sizeKvoc/pageKvoc, height)
            self.fillPainter(painter, rect)
            painter.end()
        QDialog.accept(self)

    def saveToFile(self):
        if self.sizeOriginal.isChecked(): size = self.graph.size()
        elif self.size400.isChecked(): size = QSize(400,400)
        elif self.size600.isChecked(): size = QSize(600,600)
        elif self.size800.isChecked(): size = QSize(800,800)
        elif self.custom.isChecked():  size = QSize(int(str(self.xSize.text())), int(str(self.ySize.text())))
        else:
            print "error"
            return

        qfileName = QFileDialog.getSaveFileName(self.lastSaveDirName + "graph.png","Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", None, "Save to..", "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        if ext == "":	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"
        ext = ext.upper()

        dirName, shortFileName = os.path.split(fileName)
        self.lastSaveDirName = dirName + "/"

        if not self.allSizes.isChecked():
            self.saveToFileDirect(fileName, ext, size)
        else:
            if not os.path.isdir(dirName + "\\400\\"): os.mkdir(dirName + "\\400\\")
            if not os.path.isdir(dirName + "\\600\\"): os.mkdir(dirName + "\\600\\")
            if not os.path.isdir(dirName + "\\800\\"): os.mkdir(dirName + "\\800\\")
            if not os.path.isdir(dirName + "\\Original\\"): os.mkdir(dirName + "\\Original\\")
            self.saveToFileDirect(dirName + "\\400\\" + shortFileName, ext, QSize(400,400))
            self.saveToFileDirect(dirName + "\\600\\" + shortFileName, ext, QSize(600,600))
            self.saveToFileDirect(dirName + "\\800\\" + shortFileName, ext, QSize(800,800))
            self.saveToFileDirect(dirName + "\\Original\\" + shortFileName, ext, self.graph.size())
        
    def saveToFileDirect(self, fileName, ext, size = QSize()):
        if os.path.exists(fileName):
            res = QMessageBox.information(self,'Save picture','File already exists. Overwrite?','Yes','No', QString.null,0,1)
            if res == 1: return

        if size.isEmpty(): buffer = QPixmap(self.graph.size()) # any size can do, now using the window size
        else:              buffer = QPixmap(size)
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background
        self.fillPainter(painter, buffer.rect())
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
    c = OWChooseImageSizeDlg(None)
    
    a.setMainWidget(c)
    c.show()
    a.exec_loop()


import os
from qwt import QwtPlot
from qtcanvas import QCanvas
from OWBaseWidget import *
import OWGUI

        
class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName"]
    def __init__(self, graph):
        OWBaseWidget.__init__(self, None, None, "Image settings", modal = TRUE)

        self.graph = graph
        self.selectedSize = 0
        self.customX = 400
        self.customY = 400
        self.saveAllSizes = 0
        self.lastSaveDirName = os.getcwd() + "/"

        self.loadSettings()
        
        self.space = QVBox(self)
        self.grid=QGridLayout(self)
        self.grid.addWidget(self.space,0,0)
        box = QVButtonGroup("Image Size", self.space)
        size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"])
        
        OWGUI.lineEdit(box, self, "customX", "       Weight:  ", orientation = "horizontal", valueType = int)
        OWGUI.lineEdit(box, self, "customY", "       Height:   ", orientation = "horizontal", valueType = int)

        self.printButton = OWGUI.button(self.space, self, "Print", callback = self.printPic)
        self.okButton = OWGUI.button(self.space, self, "Save image", callback = self.accept)
        self.cancelButton = OWGUI.button(self.space, self, "Cancel", callback = self.reject)
                                       
        self.resize(170,270)

    def accept(self):
        self.saveToFile()
        self.saveSettings()
        QDialog.accept(self)
        self.hide()

    def printPic(self):
        self.saveSettings()
        printer = QPrinter()
        size = self.getSize()
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
        ext = ext[1:].upper()
        if ext == "" or ext not in ("BMP", "GIF", "PNG") :	
        	ext = "PNG"  	# if no format was specified, we choose png
        	fileName = fileName + ".png"

        dirName, shortFileName = os.path.split(fileName)
        self.lastSaveDirName = dirName + "/"
        self.saveToFileDirect(fileName, ext, self.getSize())

    def getSize(self):
        if self.selectedSize == 0: size = self.graph.size()
        elif self.selectedSize == 4: size = QSize(self.customX, self.customY)
        else: size = QSize(200 + self.selectedSize*200, 200 + self.selectedSize*200)
        return size
        
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
    c = OWChooseImageSizeDlg(0)
    
    a.setMainWidget(c)
    c.show()
    a.exec_loop()


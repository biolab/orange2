import os
from qwt import QwtPlot
from qtcanvas import QCanvas
from OWBaseWidget import *
import OWGUI
from ColorPalette import *

        
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
        self.layout = QVBoxLayout(self, 4)
        self.layout.addWidget(self.space)
        
        box = QVButtonGroup("Image Size", self.space)
        size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"], callback = self.updateGUI)
        
        self.customXEdit = OWGUI.lineEdit(box, self, "customX", "       Weight:  ", orientation = "horizontal", valueType = int)
        self.customYEdit = OWGUI.lineEdit(box, self, "customY", "       Height:   ", orientation = "horizontal", valueType = int)

        self.printButton = OWGUI.button(self.space, self, "Print", callback = self.printPic)
        self.okButton = OWGUI.button(self.space, self, "Save image", callback = self.accept)
        self.cancelButton = OWGUI.button(self.space, self, "Cancel", callback = self.reject)
                                       
        self.resize(200,270)
        self.updateGUI()

    def updateGUI(self):
        self.customXEdit.setEnabled(self.selectedSize == 4)
        self.customYEdit.setEnabled(self.selectedSize == 4)

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


class ColorPaletteWidget(QVBox):
    def __init__(self, parent, master, label = "Colors", palettesDict = {}, selectedPaletteIndex = 0, currentState = None,  callback = None, additionalColors = None):
        QVBox.__init__(self, parent)
        self.setSpacing(4)
        
        self.master = master
        self.callback = callback

        self.colorSchemas = {}
        master.selectedPaletteIndex = 0
        master.passThroughBlack = 0
        master.gamma = 1
        master.currentState = currentState
        
        self.schemaCombo = OWGUI.comboBox(self, master, "selectedPaletteIndex", box = "Saved Palettes", callback = self.paletteSelected)

        box = OWGUI.widgetBox(self, "Colors")

        self.interpolationHBox = QHBox(box)
        self.interpolationHBox.setSpacing(5)

        self.colorButton1 = ColorButton(self, self.interpolationHBox)
        self.interpolationView = InterpolationView(self.interpolationHBox)
        self.colorButton2 = ColorButton(self, self.interpolationHBox)

        box.addSpace(6)
        
        self.chkPassThroughBlack = OWGUI.checkBox(box, master, "passThroughBlack", "Pass through black", callback = self.colorSchemaChange)
        box.addSpace(6)

        self.setColorSchemas(palettesDict, selectedPaletteIndex)
        if currentState: self.setCurrentState(currentState)
        else: self.paletteSelected()


    def setColorSchemas(self, schemas, selectedPaletteIndex):
        self.schemaCombo.clear()

        if not schemas or len(schemas.keys()) > 0:
            schemas = {}
            schemas["Blue - Yellow"] = (0,0,255, 255, 255, 0, FALSE)

        for key in schemas.keys():
            self.schemaCombo.insertItem(key)
            (r1, g1, b1, r2, g2, b2, throughblack) = schemas[key]
            self.colorSchemas[key] = ColorSchema(key, self.createPalette(QColor(r1, g1, b1), QColor(r2, g2, b2), throughblack), {}, throughblack)

        self.schemaCombo.insertItem("Save current palette as...")
        self.selectedPaletteIndex = selectedPaletteIndex


    def setCurrentState(self, state):
        col1 = QColor(); col1.setRgb(state[0]); self.colorButton1.setColor(col1)
        col2 = QColor(); col2.setRgb(state[1]); self.colorButton2.setColor(col2)
        self.master.passThroughBlack = state[2]
        self.interpolationView.setPalette(self.createPalette(col1, col2, self.master.passThroughBlack)+ 5*[Qt.white.rgb()])
        self.master.currentState = state

    def getCurrentState(self): return self.master.currentState

    def paletteSelected(self):
        if self.master.selectedPaletteIndex == self.schemaCombo.count()-1:    # if we selected "Save current palette as..." option then add another option to the list
            message = "Please enter new color schema name"
            ok = FALSE
            while (not ok):
                s = QInputDialog.getText("New Schema", message, QLineEdit.Normal)
                ok = TRUE
                if (s[1]==TRUE):
                    for i in range(self.schemaCombo.count()):
                        if s[0].lower().compare(self.schemaCombo.text(i).lower())==0:
                            ok = FALSE
                            message = "Color schema with that name already exists, please enter another name"
                    if (ok):
                        self.colorSchemas[str(s[0])] = ColorSchema(self.getCurrentColorSchema().getName(), self.getCurrentColorSchema().getPalette(), self.getCurrentColorSchema().getAdditionalColors(), self.getCurrentColorSchema().getPassThroughBlack())
                        self.schemaCombo.insertItem(s[0], 0)
                        self.schemaCombo.setCurrentItem(0)
                        self.master.currentState = (self.colorButton1.getColor().rgb(), self.colorButton2.getColor().rgb(), self.master.passThroughBlack)
                else:
                    state = self.getCurrentState()          # if we pressed cancel we have to select a different item than the "Save current palette as..."
                    self.master.selectedPaletteIndex = 0    # this will change the color buttons, so we have to restore the colors
                    self.setCurrentState(state)             
        else:
            schema = self.getCurrentColorSchema()
            self.interpolationView.setPalette(schema.getPalette() + 5*[Qt.white.rgb()])
            self.colorButton1.setColor(self.rgbToQColor(schema.getPalette()[0]))
            self.colorButton2.setColor(self.rgbToQColor(schema.getPalette()[paletteInterpolationColors-1]))
            if self.callback: self.callback()


    # this function is called if one of the color buttons was pressed or there was any other change of the color palette
    def colorSchemaChange(self):
        name = self.getCurrentColorSchema().getName()
        palette = self.createPalette(self.colorButton1.getColor(), self.colorButton2.getColor(), self.master.passThroughBlack)

        self.interpolationView.setPalette(palette + 5*[Qt.white.rgb()])

        self.master.currentState = (self.colorButton1.getColor().rgb(), self.colorButton2.getColor().rgb(), self.master.passThroughBlack)

        if self.callback: self.callback()

    def getCurrentColorSchema(self):
        return self.colorSchemas[str(self.schemaCombo.currentText())]

    def setCurrentColorSchema(self, schema):
        self.colorSchemas[str(self.schemaCombo.currentText())] = schema

    def getColorSchemas(self):
        return self.colorSchemas

    def createPalette(self, color1, color2, passThroughBlack, colorNumber = paletteInterpolationColors):
        if passThroughBlack:
            palette = [qRgb(color1.red() - color1.red()*i*2./colorNumber, color1.green() - color1.green()*i*2./colorNumber, color1.blue() - color1.blue()*i*2./colorNumber) for i in range(colorNumber/2)]
            palette += [qRgb(color2.red()*i*2./colorNumber, color2.green()*i*2./colorNumber, color2.blue()*i*2./colorNumber) for i in range(colorNumber - (colorNumber/2))]
        else:
            palette = [qRgb(color1.red() + (color2.red()-color1.red())*i/colorNumber, color1.green() + (color2.green()-color1.green())*i/colorNumber, color1.blue() + (color2.blue()-color1.blue())*i/colorNumber) for i in range(colorNumber)]

        return palette

    def rgbToQColor(self, rgb):
        return QColor(qRed(rgb), qGreen(rgb), qBlue(rgb))

    def qRgbFromQColor(self, qcolor):
        return qRgb(qcolor.red(), qcolor.green(), qcolor.blue())
    

class ColorPalette(OWBaseWidget):
    def __init__(self,parent, caption = "Color Palette", palettesDict = {}, selectedPaletteIndex = 0, currentState = None, callback = None, additionalColors = None, modal  = TRUE):
        OWBaseWidget.__init__(self, None, None, caption, modal = modal)
        self.layout = QVBoxLayout(self, 4)
        
        self.colorBox = ColorPaletteWidget(self, self, "Colors", palettesDict, selectedPaletteIndex, currentState, callback, additionalColors)
        self.layout.addWidget(self.colorBox)
            
        self.hbox = OWGUI.widgetBox(self, orientation = "horizontal")
        self.layout.addWidget(self.hbox)
        
        self.okButton = OWGUI.button(self.hbox, self, "OK", self.accept)
        self.cancelButton = OWGUI.button(self.hbox, self, "Cancel", self.reject)

    def getColorPalette(self, colorNumber = paletteInterpolationColors):
        return self.colorBox.createPalette(self.colorBox.colorButton1.getColor(), self.colorBox.colorButton2.getColor(), self.passThroughBlack, colorNumber)


if __name__== "__main__":
    a = QApplication(sys.argv)
    c = ColorPalette(None)
    
    a.setMainWidget(c)
    c.show()
    a.exec_loop()


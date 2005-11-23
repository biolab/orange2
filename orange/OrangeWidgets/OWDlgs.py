import os
from qwt import QwtPlot
from qtcanvas import QCanvas
from OWBaseWidget import *
import OWGUI
from ColorPalette import *

        
class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName"]
    def __init__(self, graph, extraButtons = []):
        OWBaseWidget.__init__(self, None, None, "Image settings", modal = TRUE)

        self.graph = graph
        self.selectedSize = 0
        self.customX = 400
        self.customY = 400
        self.saveAllSizes = 0
        self.lastSaveDirName = "./"

        self.loadSettings()
        
        self.space = QVBox(self)
        self.layout = QVBoxLayout(self, 4)
        self.layout.addWidget(self.space)
        
        box = QVButtonGroup("Image Size", self.space)
        if isinstance(graph, QwtPlot):
            size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"], callback = self.updateGUI)
            
            self.customXEdit = OWGUI.lineEdit(box, self, "customX", "       Width:  ", orientation = "horizontal", valueType = int)
            self.customYEdit = OWGUI.lineEdit(box, self, "customY", "       Height:   ", orientation = "horizontal", valueType = int)
        elif isinstance(graph, QCanvas):
            OWGUI.widgetLabel(box, "Image size will be set automatically.")
            
        OWGUI.button(self.space, self, "Print", callback = self.printPic)
        OWGUI.button(self.space, self, "Save Image", callback = self.saveImage)
        OWGUI.button(self.space, self, "Save Graph As matplotlib Script", callback = self.saveToMatplotlib)
        for (text, funct) in extraButtons:
            butt = OWGUI.button(self.space, self, text, callback = funct)
            self.connect(butt, SIGNAL("clicked()"), self.accept)        # also connect the button to accept so that we close the dialog
        OWGUI.button(self.space, self, "Cancel", callback = self.reject)
                                       
        self.resize(200,270)
        self.updateGUI()

    def saveImage(self):
        filename = self.getFileName("graph.png", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", ".png")
        if not filename: return
        
        (fil,ext) = os.path.splitext(filename)
        ext = ext[1:].upper()
        if ext == "" or ext not in ("BMP", "GIF", "PNG") :	
        	ext = "PNG"  	# if no format was specified, we choose png
        	filename = filename + ".png"

        size = self.getSize()
        painter = QPainter()
        if size.isEmpty(): buffer = QPixmap(self.graph.size()) # any size can do, now using the window size
        else:              buffer = QPixmap(size)
        painter.begin(buffer)
        painter.fillRect(buffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background
        self.fillPainter(painter, buffer.rect())
        painter.flush()
        painter.end()
        buffer.save(filename, ext)
        QDialog.accept(self)
        #self.hide()

    def saveToMatplotlib(self):
        filename = self.getFileName("graph.py","Python Script (*.py)", ".py")
        if filename:
            if isinstance(self.graph, QwtPlot):
                self.graph.saveToMatplotlib(filename, self.getSize())
            else:
                minx,maxx,miny,maxy = self.getQCanvasBoundaries()
                f = open(filename, "wt")
                f.write("from pylab import *\nfrom matplotlib.patches import Rectangle\n\nfigure(facecolor = 'w', figsize = (%f,%f), dpi = 80)\na = gca()\nhold(True)\n\n\n" % ((maxx-minx)/80, (maxy-miny)/80))
                print miny
                maxy+= miny
                
                #maxy += 20

                sortedList = [(item.z(), item) for item in self.graph.allItems()]
                sortedList.sort()   # sort items by z value
                
                for (z, item) in sortedList:
                    if not item.visible(): continue
                    if item.__class__ in [QCanvasEllipse, QCanvasLine, QCanvasPolygon and QCanvasRectangle]:
                        penc   = self._getColorFromObject(item.pen())
                        brushc = self._getColorFromObject(item.brush())
                        penWidth = item.pen().width()
                        
                        if   isinstance(item, QCanvasEllipse): continue
                        elif isinstance(item, QCanvasPolygon): continue
                        elif isinstance(item, QCanvasRectangle):
                            x,y,w,h = item.rect().x(), maxy-item.rect().y()-item.rect().height(), item.rect().width(), item.rect().height()
                            f.write("a.add_patch(Rectangle((%d, %d), %d, %d, edgecolor=%s, facecolor = %s, linewidth = %d, fill = %d))\n" % (x,y,w,h, penc, brushc, penWidth, type(brushc) == tuple))
                        elif isinstance(item, QCanvasLine):
                            x1,y1, x2,y2 = item.startPoint().x(), maxy-item.startPoint().y(), item.endPoint().x(), maxy-item.endPoint().y()
                            f.write("plot(%s, %s, marker = 'None', linestyle = '-', color = %s, linewidth = %d)\n" % ([x1,x2], [y1,y2], penc, penWidth))
                    elif item.__class__ == QCanvasText:
                        align = item.textFlags()
                        xalign = (align & Qt.AlignLeft and "left") or (align & Qt.AlignHCenter and "center") or (align & Qt.AlignRight and "right")
                        yalign = (align & Qt.AlignBottom and "bottom") or (align & Qt.AlignTop and "top") or (align & Qt.AlignVCenter and "center")
                        vertAlign = (yalign and ", verticalalignment = '%s'" % yalign) or ""
                        horAlign = (xalign and ", horizontalalignment = '%s'" % xalign) or ""
                        color = tuple([item.color().red()/255., item.color().green()/255., item.color().blue()/255.])
                        weight = item.font().bold() and "bold" or "normal"
                        f.write("text(%f, %f, '%s'%s%s, color = %s, name = '%s', weight = '%s')\n" % (item.x(), maxy-item.y(), str(item.text()), vertAlign, horAlign, color, str(item.font().family()), weight))

                f.write("# disable grid\ngrid(False)\n\n")
                f.write("#hide axis\naxis('off')\naxis([%f, %f, %f, %f])\ngca().set_position([0.01,0.01,0.98,0.98])\n" % (minx, maxx, miny, maxy-miny))
                f.write("show()")
                f.close()

            try:
                import matplotlib
            except:
                QMessageBox.information(self,'Matplotlib missing',"File was saved, but you will not be able to run it because you don't have matplotlib installed.\nYou can download matplotlib for free at matplotlib.sourceforge.net.",'Close')

        QDialog.accept(self)


    def printPic(self):
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
        self.saveSettings()
        QDialog.accept(self)


    def fillPainter(self, painter, rect):
        if isinstance(self.graph, QwtPlot):
            self.graph.printPlot(painter, rect)
        elif isinstance(self.graph, QCanvas):
            # draw background
            self.graph.drawBackground(painter, rect)
            minx,maxx,miny,maxy = self.getQCanvasBoundaries()

            # draw items
            sortedList = [(item.z(), item) for item in self.graph.allItems()]
            sortedList.sort()   # sort items by z value
            for (z, item) in sortedList:
                if item.visible():
                    item.move(item.x()-minx, item.y()-miny)
                    item.draw(painter)
                    item.move(item.x()+minx, item.y()+miny)

            # draw foreground
            self.graph.drawForeground(painter, rect)

    # ############################################################
    # EXTRA FUNCTIONS ############################################
    def getQCanvasBoundaries(self):
        minx,maxx,miny,maxy = 10000000,0,10000000,0
        for item in self.graph.allItems():
            if not item.visible(): continue
            br = item.boundingRect()
            minx = min(br.left(), minx)
            maxx = max(maxx, br.right())
            miny = min(br.top(), miny)
            maxy = max(maxy, br.bottom())
        return minx-10, maxx+10, miny-10, maxy+10

        
    def getFileName(self, defaultName, mask, extension):        
        fileName = str(QFileDialog.getSaveFileName(self.lastSaveDirName + defaultName, mask, None, "Save to..", "Save to.."))
        if not fileName: return None
        if not os.path.splitext(fileName)[1][1:]: fileName = fileName + extension

        if os.path.exists(fileName):
            res = QMessageBox.information(self,'Save picture','File already exists. Overwrite?','Yes','No', QString.null,0,1)
            if res == 1: return None

        self.lastSaveDirName = os.path.split(fileName)[0] + "/"
        self.saveSettings()
        return fileName

    def getSize(self):
        if isinstance(self.graph, QCanvas):
            minx,maxx,miny,maxy = self.getQCanvasBoundaries()
            size = QSize(maxx-minx, maxy-miny)
        elif self.selectedSize == 0: size = self.graph.size()
        elif self.selectedSize == 4: size = QSize(self.customX, self.customY)
        else: size = QSize(200 + self.selectedSize*200, 200 + self.selectedSize*200)
        return size

    def updateGUI(self):
        if isinstance(self.graph, QwtPlot):        
            self.customXEdit.setEnabled(self.selectedSize == 4)
            self.customYEdit.setEnabled(self.selectedSize == 4)

    def _getColorFromObject(self, obj):
        if obj.__class__ == QBrush and obj.style() == Qt.NoBrush: return "'none'"
        if obj.__class__ == QPen   and obj.style() == Qt.NoPen: return "'none'"
        col = [obj.color().red(), obj.color().green(), obj.color().blue()];
        col = tuple([v/float(255) for v in col])
        return col

class ColorPaletteWidget(QVBox):
    def __init__(self, parent, master, label = "Colors", callback = None):
        QVBox.__init__(self, parent)
        self.setSpacing(4)
        
        self.master = master
        self.callback = callback
        self.counter = 1
        self.paletteNames = []
        self.colorButtonNames = []

        self.colorSchemas = {}
        self.master.selectedSchemaIndex = 0
        self.schemaCombo = OWGUI.comboBox(self, self.master, "selectedSchemaIndex", box = "Saved Profiles", callback = self.paletteSelected)
        
    def getColorSchemas(self):
        return self.colorSchemas
    
    def setColorSchemas(self, schemas = None, selectedSchemaIndex = 0):
        self.schemaCombo.clear()

        if not schemas or len(schemas.keys()) == 0:
            schemas = {}
            schemas["Default"] = tuple([self.__dict__[name].getColor().rgb() for name in self.colorButtonNames] + [(self.__dict__[name+"Left"].getColor().rgb(), self.__dict__[name+"Right"].getColor().rgb(), self.master.__dict__[name+"passThroughBlack"]) for name in self.paletteNames])

        self.colorSchemas = schemas
        for key in schemas.keys():
            self.schemaCombo.insertItem(key)

        self.schemaCombo.insertItem("Save current palette as...")
        self.selectedSchemaIndex = selectedSchemaIndex
        self.paletteSelected()

    def getCurrentState(self):
        l1 = [self.qRgbFromQColor(self.__dict__[name].getColor()) for name in self.colorButtonNames]
        l2 = [(self.qRgbFromQColor(self.__dict__[name+"Left"].getColor()), self.qRgbFromQColor(self.__dict__[name+"Right"].getColor()), self.master.__dict__[name+"passThroughBlack"]) for name in self.paletteNames]
        return tuple(l1+l2)

    def setCurrentState(self, state):
        for i in range(len(self.colorButtonNames)):
            self.__dict__[self.colorButtonNames[i]].setColor(self.rgbToQColor(state[i]))
        for i in range(len(self.colorButtonNames),len(state)):
            (l, r, chk) = state[i]
            self.__dict__[self.paletteNames[i-len(self.colorButtonNames)]+"Left"].setColor(self.rgbToQColor(l))
            self.__dict__[self.paletteNames[i-len(self.colorButtonNames)]+"Right"].setColor(self.rgbToQColor(r))
            self.master.__dict__[self.paletteNames[i-len(self.colorButtonNames)]+"passThroughBlack"] = chk
            self.__dict__[self.paletteNames[i-len(self.colorButtonNames)]+"passThroughBlackCheckbox"].setChecked(chk)
            pallete = self.createPalette(self.rgbToQColor(l), self.rgbToQColor(r), chk) + 5*[Qt.white.rgb()]
            self.__dict__[self.paletteNames[i-len(self.colorButtonNames)]+"View"].setPalette1(pallete)
        self.master.currentState = state

    def paletteSelected(self):
        if not self.schemaCombo.count(): return 
        if self.master.selectedSchemaIndex == self.schemaCombo.count()-1:    # if we selected "Save current palette as..." option then add another option to the list
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
                        self.colorSchemas[str(s[0])] = self.getCurrentState()
                        self.schemaCombo.insertItem(s[0], 0)
                        self.schemaCombo.setCurrentItem(0)
                        self.master.currentState = self.colorSchemas[str(s[0])]
                else:
                    state = self.getCurrentState()          # if we pressed cancel we have to select a different item than the "Save current palette as..."
                    self.master.selectedSchemaIndex = 0    # this will change the color buttons, so we have to restore the colors
                    self.setCurrentState(state)             
        else:
            schema = self.getCurrentColorSchema()
            self.setCurrentState(schema)
            if self.callback: self.callback()


    # this function is called if one of the color buttons was pressed or there was any other change of the color palette
    def colorSchemaChange(self):
        state = self.getCurrentState()
        self.setCurrentState(state)
        if self.callback: self.callback()

    def getCurrentColorSchema(self):
        return self.colorSchemas[str(self.schemaCombo.currentText())]

    def setCurrentColorSchema(self, schema):
        self.colorSchemas[str(self.schemaCombo.currentText())] = schema

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

    def createBox(self, boxName, boxCaption = None):
        box = OWGUI.widgetBox(self, boxCaption)
        box.setAlignment(Qt.AlignLeft)
        return box

    def createColorButton(self, box, buttonName, buttonCaption, initialColor = Qt.black):
        self.__dict__["buttonBox"+str(self.counter)] = QHBox(box)
        self.__dict__["buttonBox"+str(self.counter)].setSpacing(5)
        self.__dict__[buttonName] = ColorButton(self, self.__dict__["buttonBox"+str(self.counter)])
        self.__dict__["buttonLabel"+str(self.counter)] = OWGUI.widgetLabel(self.__dict__["buttonBox"+str(self.counter)], buttonCaption)
        self.__dict__[buttonName].setColor(initialColor)
        self.colorButtonNames.append(buttonName)
        self.__dict__["buttonBoxSpacing"+str(self.counter)] = QHBox(self.__dict__["buttonBox"+str(self.counter)])

        self.__dict__[buttonName].setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed ))
        self.__dict__["buttonLabel"+str(self.counter)].setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed ))
        
        #self.__dict__["buttonBoxSpacing"+str(self.counter)].setSizePolicy(QSizePolicy(QSizePolicy.Expanding , QSizePolicy.Fixed ))
        self.counter += 1

    def createColorPalette(self, paletteName, boxCaption, passThroughBlack = 0, initialColor1 = Qt.white, initialColor2 = Qt.black):
        self.__dict__["buttonBox"+str(self.counter)] = OWGUI.widgetBox(self, boxCaption)

        self.__dict__["paletteBox"+str(self.counter)] = OWGUI.widgetBox(self.__dict__["buttonBox"+str(self.counter)], orientation = "horizontal")
        self.__dict__[paletteName+"Left"]  = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+ "View"] = InterpolationView(self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+"Right"] = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)])

        self.__dict__[paletteName+"Left"].setColor(initialColor1)
        self.__dict__[paletteName+"Right"].setColor(initialColor2)
        self.__dict__["buttonBox"+str(self.counter)].addSpace(6)
        
        self.master.__dict__[paletteName+"passThroughBlack"] = passThroughBlack
        self.__dict__[paletteName+"passThroughBlackCheckbox"] = OWGUI.checkBox(self.__dict__["buttonBox"+str(self.counter)], self.master, paletteName+"passThroughBlack", "Pass through black", callback = self.colorSchemaChange)
        self.paletteNames.append(paletteName)
        self.counter += 1


class ColorPalette(OWBaseWidget):
    def __init__(self,parent, caption = "Color Palette", callback = None, modal  = TRUE):
        OWBaseWidget.__init__(self, None, None, caption, modal = modal)
        self.layout = QVBoxLayout(self, 4)
        
        self.mainArea = ColorPaletteWidget(self, self, "Colors", callback)
        self.layout.addWidget(self.mainArea)
            
        self.hbox = OWGUI.widgetBox(self, orientation = "horizontal")
        self.layout.addWidget(self.hbox)
        
        self.okButton = OWGUI.button(self.hbox, self, "OK", self.accept)
        self.cancelButton = OWGUI.button(self.hbox, self, "Cancel", self.reject)
        self.setMinimumWidth(230)

    def getCurrentSchemeIndex(self):
        return self.selectedSchemaIndex

    def getCurrentState(self):
        return self.mainArea.getCurrentState()

    def setCurrentState(self, state):
        self.mainArea.setCurrentState(state)

    def getColorSchemas(self):
        return self.mainArea.colorSchemas
    
    def setColorSchemas(self, schemas = None, selectedSchemaIndex = 0):
        self.mainArea.setColorSchemas(schemas, selectedSchemaIndex)

    def getColor(self, buttonName):
        return self.mainArea.__dict__[buttonName].getColor()

    def getColorPalette(self, paletteName):
        c1 = self.mainArea.__dict__[paletteName+"Left"].getColor()
        c2 = self.mainArea.__dict__[paletteName+"Right"].getColor()
        b = self.mainArea.master.__dict__[paletteName+"passThroughBlack"]
        return self.mainArea.createPalette(c1, c2, b)

    def createBox(self, boxName, boxCaption = None):
        return self.mainArea.createBox(boxName, boxCaption)

    def createColorButton(self, box, buttonName, buttonCaption, initialColor = Qt.black):
        self.mainArea.createColorButton(box, buttonName, buttonCaption, initialColor)

    def createColorPalette(self, paletteName, boxCaption, passThroughBlack = 0, initialColor1 = Qt.white, initialColor2 = Qt.black):
        self.mainArea.createColorPalette(paletteName, boxCaption, passThroughBlack, initialColor1, initialColor2)
    

if __name__== "__main__":
    a = QApplication(sys.argv)
    """
    c = ColorPalette(None, modal = FALSE)
    c.createColorPalette("colorPalette", "Palette")
    box = c.createBox("otherColors", "Colors")
    c.createColorButton(box, "Canvas", "Canvas")
    box.addSpace(5)
    c.createColorButton(box, "Grid", "Grid")
    box.addSpace(5)
    c.createColorButton(box, "test", "ttest")
    box.addSpace(5)
    box.adjustSize()
    c.setColorSchemas()
    a.setMainWidget(c)
    c.show()
    a.exec_loop()
    """
    c = OWChooseImageSizeDlg(None)
    c.show()
    a.exec_loop()
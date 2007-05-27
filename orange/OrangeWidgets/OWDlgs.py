import os
from OWBaseWidget import *
import OWGUI, OWGraphTools, OWTools
try:
   from qwt import QwtPlot
except:
   from Qwt4 import QwtPlot

from qtcanvas import QCanvas
from ColorPalette import *

class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName", "penWidthFactor"]
    def __init__(self, graph, extraButtons = []):
        OWBaseWidget.__init__(self, None, None, "Image settings", modal = TRUE)

        self.graph = graph
        self.selectedSize = 0
        self.customX = 400
        self.customY = 400
        self.saveAllSizes = 0
        self.penWidthFactor = 1
        self.lastSaveDirName = "./"

        self.loadSettings()

        self.space = OWGUI.widgetBox(self)
        self.layout = QVBoxLayout(self, 8)
        self.layout.addWidget(self.space)

        box = QVButtonGroup("Image Size", self.space)
        if isinstance(graph, QwtPlot):
            size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"], callback = self.updateGUI)

            ind1 = OWGUI.indentedBox(box)
            ind2 = OWGUI.indentedBox(box)
            self.customXEdit = OWGUI.lineEdit(ind1, self, "customX", "Width:", orientation = "horizontal", valueType = int)
            self.customYEdit = OWGUI.lineEdit(ind2, self, "customY", "Height:", orientation = "horizontal", valueType = int)

            OWGUI.comboBoxWithCaption(self.space, self, "penWidthFactor", label = 'Factor:', box = "Pen width multiplication factor",  tooltip = "Set the pen width factor for all curves in the plot\n(Useful for example when the lines in the plot look to thin)\nDefault: 1", sendSelectedValue = 1, valueType = int, items = range(1,20))

        elif isinstance(graph, QCanvas):
            OWGUI.widgetLabel(box, "Image size will be set automatically.")

        self.printButton =          OWGUI.button(self.space, self, "Print", callback = self.printPic)
        self.saveImageButton =      OWGUI.button(self.space, self, "Save Image", callback = self.saveImage)
        self.saveMatplotlibButton = OWGUI.button(self.space, self, "Save Graph As matplotlib Script", callback = self.saveToMatplotlib)
        for (text, funct) in extraButtons:
            butt = OWGUI.button(self.space, self, text, callback = funct)
            self.connect(butt, SIGNAL("clicked()"), self.accept)        # also connect the button to accept so that we close the dialog
        OWGUI.button(self.space, self, "Cancel", callback = self.reject)

        self.resize(200,270)
        self.updateGUI()


    def saveImage(self, filename = None, size = None, closeDialog = 1):
        if not filename:
            filename = self.getFileName("graph.png", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", ".png")
            if not filename: return

        (fil,ext) = os.path.splitext(filename)
        ext = ext[1:].upper()
        if ext == "" or ext not in ("BMP", "GIF", "PNG") :
        	ext = "PNG"  	# if no format was specified, we choose png
        	filename = filename + ".png"

        if not size:
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

        if closeDialog:
            QDialog.accept(self)

    def saveToMatplotlib(self):
        filename = self.getFileName("graph.py","Python Script (*.py)", ".py")
        if filename:
            if isinstance(self.graph, QwtPlot):
                self.graph.saveToMatplotlib(filename, self.getSize())
            else:
                minx,maxx,miny,maxy = self.getQCanvasBoundaries()
                f = open(filename, "wt")
                f.write("from pylab import *\nfrom matplotlib.patches import Rectangle\n\n#constants\nx1 = %f; x2 = %f\ny1 = 0.0; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = 0.01\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\na = gca()\nhold(True)\n" % (minx, maxx, maxy, maxx-minx, maxy-miny))

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
                f.write("#hide axis\naxis('off')\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n")
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

        if printer.setup():
            painter = QPainter(printer)
            metrics = QPaintDeviceMetrics(printer)
            height = metrics.height() - 2*printer.margins().height()
            width = metrics.width() - 2*printer.margins().width()

            factor = 1.0
            if isinstance(self.graph, QCanvas):
                minx,maxx,miny,maxy = self.getQCanvasBoundaries()
                factor = min(float(width)/(maxx-minx), float(height)/(maxy-miny))

            if height == 0:
                print "Error. Height is zero. Preventing division by zero."
                return
            pageKvoc = width / float(height)
            sizeKvoc = size.width() / float(size.height())
            if pageKvoc < sizeKvoc:     rect = QRect(printer.margins().width(), printer.margins().height(), width, height)
            else:                       rect = QRect(printer.margins().width(), printer.margins().height(), width, height)

            self.fillPainter(painter, rect, factor)
            painter.end()
        self.saveSettings()
        QDialog.accept(self)


    def fillPainter(self, painter, rect, scale = 1.0):
        if isinstance(self.graph, QwtPlot):
            if self.penWidthFactor != 1:
                for key in self.graph.curveKeys():
                    pen = self.graph.curve(key).pen(); pen.setWidth(self.penWidthFactor*pen.width()); self.graph.curve(key).setPen(pen)

            self.graph.printPlot(painter, rect)

            if self.penWidthFactor != 1:
                for key in self.graph.curveKeys():
                    pen = self.graph.curve(key).pen(); pen.setWidth(pen.width()/self.penWidthFactor); self.graph.curve(key).setPen(pen)

        elif isinstance(self.graph, QCanvas):
            # draw background
            self.graph.drawBackground(painter, rect)
            minx,maxx,miny,maxy = self.getQCanvasBoundaries()

            # draw items
            sortedList = [(item.z(), item) for item in self.graph.allItems()]
            sortedList.sort()   # sort items by z value

            for (z, item) in sortedList:
                if item.visible():
                    item.moveBy(-minx, -miny)
                    if isinstance(item, QCanvasText):
                        rect = item.boundingRect()
                        x,y,w,h = int(rect.x()*scale), int(rect.y()*scale), int(rect.width()*scale), int(rect.height()*scale)
                        painter.setFont(item.font())
                        painter.setPen(item.color())
                        painter.drawText(x,y,w,h,item.textFlags(), item.text())
                        #painter.drawText(int(scale*item.x()), int(scale*item.y()), str(item.text()))
                    else:
                        painter.scale(scale, scale)
                        p = item.pen()
                        oldSize = p.width()
                        p.setWidth(int(oldSize*scale))
                        item.setPen(p)
                        item.draw(painter)
                        p.setWidth(oldSize)
                        item.setPen(p)
                        painter.scale(1.0/scale, 1.0/scale)
                    item.moveBy(minx, miny)

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



class ColorPalette(OWBaseWidget):
    def __init__(self,parent, caption = "Color Palette", callback = None, modal  = TRUE):
        OWBaseWidget.__init__(self, None, None, caption, modal = modal)
        self.layout = QVBoxLayout(self, 4)

        self.callback = callback
        self.counter = 1
        self.paletteNames = []
        self.colorButtonNames = []
        self.colorSchemas = []
        self.discreteColors = []
        self.selectedSchemaIndex = 0

        self.mainArea = OWGUI.widgetBox(self, box = None)
        self.layout.addWidget(self.mainArea)
        self.mainArea.setSpacing(4)
        self.schemaCombo = OWGUI.comboBox(self.mainArea, self, "selectedSchemaIndex", box = "Saved Profiles", callback = self.paletteSelected)

        self.hbox = OWGUI.widgetBox(self, orientation = "horizontal")
        self.layout.addWidget(self.hbox)

        self.okButton = OWGUI.button(self.hbox, self, "OK", self.acceptChanges)
        self.cancelButton = OWGUI.button(self.hbox, self, "Cancel", self.reject)
        self.setMinimumWidth(230)
        self.resize(240, 400)

    def acceptChanges(self):
        state = self.getCurrentState()
        oldState = self.colorSchemas[self.selectedSchemaIndex][1]
        if state == oldState:
            QDialog.accept(self)
        else:
            # if we changed the deafult schema, we must save it under a new name
            if self.colorSchemas[self.selectedSchemaIndex][0] == "Default":
                if QMessageBox.information(self, 'Question', 'The color schema has changed. Do you want to save changes?','Yes','No', '', 0,1):
                    QDialog.reject(self)
                else:
                    self.selectedSchemaIndex = self.schemaCombo.count()-1
                    self.paletteSelected()
                    QDialog.accept(self)
            # simply save the new users schema
            else:
                self.colorSchemas[self.selectedSchemaIndex] = [self.colorSchemas[self.selectedSchemaIndex][0], state]
                QDialog.accept(self)

    def createBox(self, boxName, boxCaption = None):
        box = OWGUI.widgetBox(self.mainArea, boxCaption)
        box.setAlignment(Qt.AlignLeft)
        return box

    def createColorButton(self, box, buttonName, buttonCaption, initialColor = Qt.black):
        newbox = QHBox(box)
        self.__dict__["buttonBox"+str(self.counter)] = newbox
        newbox.setSpacing(5)
        self.__dict__[buttonName] = ColorButton(self, newbox)
        self.__dict__["buttonLabel"+str(self.counter)] = OWGUI.widgetLabel(newbox, buttonCaption)
        self.__dict__[buttonName].setColor(initialColor)
        self.colorButtonNames.append(buttonName)
        self.__dict__["buttonBoxSpacing"+str(self.counter)] = QHBox(newbox)

        self.__dict__[buttonName].setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed ))
        self.__dict__["buttonLabel"+str(self.counter)].setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed ))

        self.counter += 1


    def createContinuousPalette(self, paletteName, boxCaption, passThroughBlack = 0, initialColor1 = Qt.white, initialColor2 = Qt.black):
        self.__dict__["buttonBox"+str(self.counter)] = OWGUI.widgetBox(self.mainArea, boxCaption)

        self.__dict__["paletteBox"+str(self.counter)] = OWGUI.widgetBox(self.__dict__["buttonBox"+str(self.counter)], orientation = "horizontal")
        self.__dict__[paletteName+"Left"]  = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+"Left"].master = self
        self.__dict__[paletteName+ "View"] = InterpolationView(self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+"Right"] = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+"Right"].master = self

        self.__dict__[paletteName+"Left"].setColor(initialColor1)
        self.__dict__[paletteName+"Right"].setColor(initialColor2)
        self.__dict__["buttonBox"+str(self.counter)].addSpace(6)

        self.__dict__[paletteName+"passThroughBlack"] = passThroughBlack
        self.__dict__[paletteName+"passThroughBlackCheckbox"] = OWGUI.checkBox(self.__dict__["buttonBox"+str(self.counter)], self, paletteName+"passThroughBlack", "Pass through black", callback = self.colorSchemaChange)
        self.paletteNames.append(paletteName)
        self.counter += 1

    # #####################################################
    # DISCRETE COLOR PALETTE
    # #####################################################
    def createDiscretePalette(self, boxCaption, colorPalette = OWGraphTools.defaultRGBColors):
        box = OWGUI.widgetBox(self.mainArea, boxCaption)
        hbox = OWGUI.widgetBox(box, orientation = 'horizontal')

        self.discListbox = QListBox(hbox)

        vbox = OWGUI.widgetBox(hbox, orientation = 'vertical')
        self.buttLoad       = OWGUI.button(vbox, self, "D", callback = self.showPopup, tooltip="Load a predefined set of colors")
        self.buttLoad.setMaximumWidth(20)
        self.buttLoad.setMaximumHeight(20)
        buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attributes up")
        buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attributes down")
        buttonUPAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        buttonUPAttr.setMaximumWidth(20)
        buttonDOWNAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        buttonDOWNAttr.setMaximumWidth(20)
        buttonUPAttr.setMaximumWidth(20)
        self.connect(self.discListbox, SIGNAL("doubleClicked ( QListBoxItem * )"), self.changeDiscreteColor)

        self.popupMenu = QPopupMenu(self)
        self.popupMenu.insertItem("Load default RGB palette", self.loadRGBPalette)
        self.popupMenu.insertItem("Load Color Brewer palette", self.loadCBPalette)

        self.discreteColors = [QColor(r,g,b) for (r,g,b) in colorPalette]
        for ind in range(len(self.discreteColors)):
            self.discListbox.insertItem(OWTools.ColorPixmap(self.discreteColors[ind], 15), "Color %d" % (ind))

    def changeDiscreteColor(self, item):
        ind = self.discListbox.index(item)
        color = QColorDialog.getColor(self.discreteColors[ind], self)
        if color.isValid():
            self.discListbox.changeItem(OWTools.ColorPixmap(color, 15), "Color %d" % (ind), ind)
            self.discreteColors[ind] = color

    def loadRGBPalette(self):
        self.discListbox.clear()
        self.discreteColors = [QColor(r,g,b) for (r,g,b) in OWGraphTools.defaultRGBColors]
        for ind in range(len(self.discreteColors)):
            self.discListbox.insertItem(OWTools.ColorPixmap(self.discreteColors[ind], 15), "Color %d" % (ind))

    def loadCBPalette(self):
        self.discListbox.clear()
        self.discreteColors = [QColor(r,g,b) for (r,g,b) in OWGraphTools.ColorBrewerColors]
        for ind in range(len(self.discreteColors)):
            self.discListbox.insertItem(OWTools.ColorPixmap(self.discreteColors[ind], 15), "Color %d" % (ind))

    def showPopup(self):
        point = self.buttLoad.mapToGlobal(QPoint(0, self.buttLoad.height()))
        self.popupMenu.popup(point, 0)

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        for i in range(1, self.discListbox.count()):
            if self.discListbox.isSelected(i):
                pixI, textI = self.discListbox.pixmap(i-1), self.discListbox.text(i)
                pixII, textII = self.discListbox.pixmap(i), self.discListbox.text(i-1)
                self.discListbox.insertItem(pixI, textI, i-1)
                self.discListbox.insertItem(pixII, textII, i-1)
                self.discListbox.removeItem(i+1)
                self.discListbox.removeItem(i+1)
                self.discListbox.setSelected(i-1, TRUE)
                self.discreteColors.insert(i-1, self.discreteColors.pop(i))


    # move selected attribute in "Attribute Order" list one place down
    def moveAttrDOWN(self):
        count = self.discListbox.count()
        for i in range(count-2,-1,-1):
            if self.discListbox.isSelected(i):
                pixI, textI = self.discListbox.pixmap(i+1), self.discListbox.text(i)
                pixII, textII = self.discListbox.pixmap(i), self.discListbox.text(i+1)
                self.discListbox.insertItem(pixI, textI, i)
                self.discListbox.insertItem(pixII, textII, i+1)
                self.discListbox.removeItem(i+2)
                self.discListbox.removeItem(i+2)
                self.discListbox.setSelected(i+1, TRUE)
                self.discreteColors.insert(i+1, self.discreteColors.pop(i))


    # #####################################################

    def getCurrentSchemeIndex(self):
        return self.selectedSchemaIndex

    def getColor(self, buttonName):
        return self.__dict__[buttonName].getColor()

    def getContinuousPalette(self, paletteName):
        c1 = self.__dict__[paletteName+"Left"].getColor()
        c2 = self.__dict__[paletteName+"Right"].getColor()
        b = self.__dict__[paletteName+"passThroughBlack"]
        return ContinuousPaletteGenerator(c1, c2, b)

    def getDiscretePalette(self):
        return OWGraphTools.ColorPaletteGenerator(rgbColors = [(c.red(), c.green(), c.blue()) for c in self.discreteColors])

    def getColorSchemas(self):
        return self.colorSchemas

    def getCurrentState(self):
        l1 = [(name, self.qRgbFromQColor(self.__dict__[name].getColor())) for name in self.colorButtonNames]
        l2 = [(name, (self.qRgbFromQColor(self.__dict__[name+"Left"].getColor()), self.qRgbFromQColor(self.__dict__[name+"Right"].getColor()), self.__dict__[name+"passThroughBlack"])) for name in self.paletteNames]
        l3 = [self.qRgbFromQColor(col) for col in self.discreteColors]
        return [l1, l2, l3]


    def setColorSchemas(self, schemas = None, selectedSchemaIndex = 0):
        self.schemaCombo.clear()

        if not schemas or type(schemas) != list:
            schemas = [("Default", self.getCurrentState()) ]

        self.colorSchemas = schemas
        for (name, sch) in schemas:
            self.schemaCombo.insertItem(name)

        self.schemaCombo.insertItem("Save current palette as...")
        self.selectedSchemaIndex = selectedSchemaIndex
        self.paletteSelected()

    def setCurrentState(self, state):
        [buttons, contPalettes, discPalette] = state
        for (name, but) in buttons:
            self.__dict__[name].setColor(self.rgbToQColor(but))
        for (name, (l,r,chk)) in contPalettes:
            self.__dict__[name+"Left"].setColor(self.rgbToQColor(l))
            self.__dict__[name+"Right"].setColor(self.rgbToQColor(r))
            self.__dict__[name+"passThroughBlack"] = chk
            self.__dict__[name+"passThroughBlackCheckbox"].setChecked(chk)
            palette = self.createPalette(self.rgbToQColor(l), self.rgbToQColor(r), chk) + 5*[Qt.white.rgb()]
            self.__dict__[name+"View"].setPalette1(palette)

        self.discreteColors = [self.rgbToQColor(col) for col in discPalette]
        if self.discreteColors:
            self.discListbox.clear()
            for ind in range(len(self.discreteColors)):
                self.discListbox.insertItem(OWTools.ColorPixmap(self.discreteColors[ind], 15), "Color %d" % (ind))

    def paletteSelected(self):
        if not self.schemaCombo.count(): return

        # if we selected "Save current palette as..." option then add another option to the list
        if self.selectedSchemaIndex == self.schemaCombo.count()-1:
            message = "Please enter a new name for the current color schema:"
            ok = FALSE
            while (not ok):
                s = QInputDialog.getText("New Schema Name", message, QLineEdit.Normal)
                ok = TRUE
                if (s[1]==TRUE):
                    newName = str(s[0])
                    oldNames = [str(self.schemaCombo.text(i)).lower() for i in range(self.schemaCombo.count()-1)]
                    if newName.lower() == "default":
                        ok = FALSE
                        message = "Can not change the 'Default' schema. Please enter a different name:"
                    elif newName.lower() in oldNames:
                        index = oldNames.index(newName.lower())
                        self.colorSchemas.pop(index)

                    if (ok):
                        self.colorSchemas.insert(0, (newName, self.getCurrentState()))
                        self.schemaCombo.insertItem(newName, 0)
                        #self.schemaCombo.setCurrentItem(0)
                        self.selectedSchemaIndex = 0
                else:
                    state = self.getCurrentState()  # if we pressed cancel we have to select a different item than the "Save current palette as..."
                    self.selectedSchemaIndex = 0    # this will change the color buttons, so we have to restore the colors
                    self.setCurrentState(state)
        else:
            schema = self.colorSchemas[self.selectedSchemaIndex][1]
            self.setCurrentState(schema)
            if self.callback: self.callback()


    def rgbToQColor(self, rgb):
        # we could also use QColor(positiveColor(rgb), 0xFFFFFFFF) but there is probably a reason
        # why this was not used before so I am leaving it as it is
        
        return QColor(qRed(positiveColor(rgb)), qGreen(positiveColor(rgb)), qBlue(positiveColor(rgb))) # on Mac color cannot be negative number in this case so we convert it manually

    def qRgbFromQColor(self, qcolor):
        return qRgb(qcolor.red(), qcolor.green(), qcolor.blue())

    def createPalette(self, color1, color2, passThroughBlack, colorNumber = paletteInterpolationColors):
        if passThroughBlack:
            palette = [qRgb(color1.red() - color1.red()*i*2./colorNumber, color1.green() - color1.green()*i*2./colorNumber, color1.blue() - color1.blue()*i*2./colorNumber) for i in range(colorNumber/2)]
            palette += [qRgb(color2.red()*i*2./colorNumber, color2.green()*i*2./colorNumber, color2.blue()*i*2./colorNumber) for i in range(colorNumber - (colorNumber/2))]
        else:
            palette = [qRgb(color1.red() + (color2.red()-color1.red())*i/colorNumber, color1.green() + (color2.green()-color1.green())*i/colorNumber, color1.blue() + (color2.blue()-color1.blue())*i/colorNumber) for i in range(colorNumber)]
        return palette

    # this function is called if one of the color buttons was pressed or there was any other change of the color palette
    def colorSchemaChange(self):
        self.setCurrentState(self.getCurrentState())
        if self.callback: self.callback()


class ContinuousPaletteGenerator:
    def __init__(self, color1, color2, passThroughBlack):
        self.c1Red, self.c1Green, self.c1Blue = color1.red(), color1.green(), color1.blue()
        self.c2Red, self.c2Green, self.c2Blue = color2.red(), color2.green(), color2.blue()
        self.passThroughBlack = passThroughBlack

    def getRGB(self, val):
        if self.passThroughBlack:
            if val < 0.5:
                return (self.c1Red - self.c1Red*val*2, self.c1Green - self.c1Green*val*2, self.c1Blue - self.c1Blue*val*2)
            else:
                return (self.c2Red*(val-0.5)*2., self.c2Green*(val-0.5)*2., self.c2Blue*(val-0.5)*2.)
        else:
            return (self.c1Red + (self.c2Red-self.c1Red)*val, self.c1Green + (self.c2Green-self.c1Green)*val, self.c1Blue + (self.c2Blue-self.c1Blue)*val)

    # val must be between 0 and 1
    def __getitem__(self, val):
        return QColor(*self.getRGB(val))


if __name__== "__main__":
    a = QApplication(sys.argv)

##    c = ColorPalette(None, modal = FALSE)
##    c.createContinuousPalette("continuousPalette", "Continuous Palette")
##    c.createDiscretePalette("Discrete Palette")
##    box = c.createBox("otherColors", "Colors")
##    c.createColorButton(box, "Canvas", "Canvas")
##    box.addSpace(5)
##    c.createColorButton(box, "Grid", "Grid")
##    box.addSpace(5)
##    c.createColorButton(box, "test", "ttest")
##    box.addSpace(5)
##    box.adjustSize()
##    c.setColorSchemas()
##    a.setMainWidget(c)
##    c.show()
##    a.exec_loop()

    c = OWChooseImageSizeDlg(None)
    a.setMainWidget(c)
    c.show()
    a.exec_loop()

import os
from OWBaseWidget import *
import OWGUI, OWGraphTools, OWTools
from PyQt4.Qwt5 import *
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

        self.setLayout(QVBoxLayout(self))
        self.space = OWGUI.widgetBox(self)
        self.layout().setMargin(8)
        #self.layout().addWidget(self.space)

        box = OWGUI.widgetBox(self.space, "Image Size")
        if isinstance(graph, QwtPlot):
            size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"], callback = self.updateGUI)
            self.customXEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customX", "Width:", orientation = "horizontal", valueType = int)
            self.customYEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customY", "Height:", orientation = "horizontal", valueType = int)
            OWGUI.comboBoxWithCaption(self.space, self, "penWidthFactor", label = 'Factor:   ', box = " Pen width multiplication factor ",  tooltip = "Set the pen width factor for all curves in the plot\n(Useful for example when the lines in the plot look to thin)\nDefault: 1", sendSelectedValue = 1, valueType = int, items = range(1,20))
        elif isinstance(graph, QGraphicsScene):
            OWGUI.widgetLabel(box, "Image size will be set automatically.")

        #self.printButton =          OWGUI.button(self.space, self, "Print", callback = self.printPic)
        self.saveImageButton =      OWGUI.button(self.space, self, "Save Image", callback = self.saveImage)
        self.saveMatplotlibButton = OWGUI.button(self.space, self, "Save Graph as matplotlib Script", callback = self.saveToMatplotlib)
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
            ext = "PNG"      # if no format was specified, we choose png
            filename = filename + ".png"

        if not size:
            size = self.getSize()

        if isinstance(self.graph, QGraphicsScene):
            minx,maxx,miny,maxy = self.getQGraphicsSceneBoundaries()
            source = QRectF(minx, miny, maxx-minx, maxy-miny)

        painter = QPainter()
        buffer = QPixmap(size)
        painter.begin(buffer)
        painter.fillRect(buffer.rect(), QBrush(Qt.white)) # make background same color as the widget's background

        # qwt plot
        if isinstance(self.graph, QwtPlot):
            if self.penWidthFactor != 1:
                for curve in self.graph.itemList():
                    pen = curve.pen(); pen.setWidth(self.penWidthFactor*pen.width()); curve.setPen(pen)

            self.graph.print_(painter, QRect(0,0,size.width(), size.height()))

            if self.penWidthFactor != 1:
                for curve in self.graph.itemList():
                    pen = curve.pen(); pen.setWidth(pen.width()/self.penWidthFactor); curve.setPen(pen)

        # QGraphicsScene
        elif isinstance(self.graph, QGraphicsScene):
            xOff = source.left(); yOff = source.top()
            target = source.adjusted(-xOff, -yOff, -xOff, -yOff)
            self.graph.render(painter, target, source)

        buffer.save(filename, ext)

        if closeDialog:
            QDialog.accept(self)

    def saveToMatplotlib(self):
        filename = self.getFileName("graph.py","Python Script (*.py)", ".py")
        if filename:
            if isinstance(self.graph, QwtPlot):
                self.graph.saveToMatplotlib(filename, self.getSize())
            else:
                minx,maxx,miny,maxy = self.getQGraphicsSceneBoundaries()
                f = open(filename, "wt")
                f.write("from pylab import *\nfrom matplotlib.patches import Rectangle\n\n#constants\nx1 = %f; x2 = %f\ny1 = 0.0; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = 0.01\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\na = gca()\nhold(True)\n" % (minx, maxx, maxy, maxx-minx, maxy-miny))

                sortedList = [(item.z(), item) for item in self.graph.items()]
                sortedList.sort()   # sort items by z value

                for (z, item) in sortedList:
                    # a little compatibility for QT 3.3 (on Mac at least)
                    if hasattr(item, "isVisible"):
                        if not item.isVisible(): continue
                    elif not item.visible(): continue
                    if item.__class__ in [QCanvasEllipse, QCanvasLine, QCanvasPolygon and QCanvasRectangle]:
                        penc, penAlpha  = self._getColorFromObject(item.pen())
                        brushc, brushAlpha = self._getColorFromObject(item.brush())
                        penWidth = item.pen().width()

                        if   isinstance(item, QCanvasEllipse): continue
                        elif isinstance(item, QCanvasPolygon): continue
                        elif isinstance(item, QCanvasRectangle):
                            x,y,w,h = item.rect().x(), maxy-item.rect().y()-item.rect().height(), item.rect().width(), item.rect().height()
                            f.write("a.add_patch(Rectangle((%d, %d), %d, %d, edgecolor=%s, facecolor = %s, linewidth = %d, fill = %d))\n" % (x,y,w,h, penc, brushc, penWidth, type(brushc) == tuple))
                        elif isinstance(item, QCanvasLine):
                            x1,y1, x2,y2 = item.startPoint().x(), maxy-item.startPoint().y(), item.endPoint().x(), maxy-item.endPoint().y()
                            f.write("plot(%s, %s, marker = 'None', linestyle = '-', color = %s, linewidth = %d, alpha = %.3f)\n" % ([x1,x2], [y1,y2], penc, penWidth, brushAlpha))
                    elif item.__class__ == QCanvasText:
                        align = item.textFlags()
                        xalign = (align & Qt.AlignLeft and "left") or (align & Qt.AlignHCenter and "center") or (align & Qt.AlignRight and "right")
                        yalign = (align & Qt.AlignBottom and "bottom") or (align & Qt.AlignTop and "top") or (align & Qt.AlignVCenter and "center")
                        vertAlign = (yalign and ", verticalalignment = '%s'" % yalign) or ""
                        horAlign = (xalign and ", horizontalalignment = '%s'" % xalign) or ""
                        color = tuple([item.color().red()/255., item.color().green()/255., item.color().blue()/255.])
                        weight = item.font().bold() and "bold" or "normal"
                        f.write("text(%f, %f, '%s'%s%s, color = %s, name = '%s', weight = '%s', alpha = %.3f)\n" % (item.x(), maxy-item.y(), str(item.label().text()), vertAlign, horAlign, color, str(item.font().family()), weight, item.alpha()/float(255)))

                f.write("# disable grid\ngrid(False)\n\n")
                f.write("#hide axis\naxis('off')\naxis([x1, x2, y1, y2])\ngca().set_position([edgeOffset, edgeOffset, 1 - 2*edgeOffset, 1 - 2*edgeOffset])\n")
                f.write("show()")
                f.close()

            try:
                import matplotlib
            except:
                QMessageBox.information(self,'Matplotlib missing',"File was saved, but you will not be able to run it because you don't have matplotlib installed.\nYou can download matplotlib for free at matplotlib.sourceforge.net.", QMessageBox.Ok)

        QDialog.accept(self)

    # ############################################################
    # EXTRA FUNCTIONS ############################################
    def getQGraphicsSceneBoundaries(self):
        source = self.graph.itemsBoundingRect()
#        source = None
#        for item in self.graph.items():
#            if item.isVisible():
#                if source == None: source = item.boundingRect().translated(item.pos())
#                else:              source = source.united(item.boundingRect().translated(item.pos()))
#        if source == None:
#            source = QRectF(0,0,10,10)

        source.adjust(-15, -15, 15, 15)
        return source.left(), source.right(), source.top(), source.bottom()


    def getFileName(self, defaultName, mask, extension):
        fileName = str(QFileDialog.getSaveFileName(self, "Save to..", self.lastSaveDirName + defaultName, mask))
        if not fileName: return None
        if not os.path.splitext(fileName)[1][1:]: fileName = fileName + extension

        self.lastSaveDirName = os.path.split(fileName)[0] + "/"
        self.saveSettings()
        return fileName

    def getSize(self):
        if isinstance(self.graph, QGraphicsScene):
            minx,maxx,miny,maxy = self.getQGraphicsSceneBoundaries()
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
        if isinstance(obj, QBrush) and obj.style() == Qt.NoBrush: return "'none'", 1
        if isinstance(obj, QPen)   and obj.style() == Qt.NoPen: return "'none'", 1
        col = [obj.color().red(), obj.color().green(), obj.color().blue()];
        col = tuple([v/float(255) for v in col])
        return col, obj.color().alpha()/float(255)



class ColorPalette(OWBaseWidget):
    def __init__(self,parent, caption = "Color Palette", callback = None, modal  = TRUE):
        OWBaseWidget.__init__(self, None, None, caption, modal = modal)
        self.setLayout(QVBoxLayout(self))
        self.layout().setMargin(4)

        self.callback = callback
        self.counter = 1
        self.paletteNames = []
        self.colorButtonNames = []
        self.colorSchemas = []
        self.discreteColors = []
        self.selectedSchemaIndex = 0

        self.mainArea = OWGUI.widgetBox(self, spacing = 4)
        self.layout().addWidget(self.mainArea)
        self.schemaCombo = OWGUI.comboBox(self.mainArea, self, "selectedSchemaIndex", box = "Saved Profiles", callback = self.paletteSelected)

        self.hbox = OWGUI.widgetBox(self, orientation = "horizontal")
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
        self.__dict__[buttonName] = ColorButton(self, box, buttonCaption)
        self.__dict__[buttonName].setColor(QColor(initialColor))
        self.colorButtonNames.append(buttonName)
        self.counter += 1


    def createContinuousPalette(self, paletteName, boxCaption, passThroughBlack = 0, initialColor1 = QColor(Qt.white), initialColor2 = Qt.black):
        self.__dict__["buttonBox"+str(self.counter)] = OWGUI.widgetBox(self.mainArea, boxCaption)

        self.__dict__["paletteBox"+str(self.counter)] = OWGUI.widgetBox(self.__dict__["buttonBox"+str(self.counter)], orientation = "horizontal")
        self.__dict__[paletteName+"Left"]  = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)], color = QColor(initialColor1))
        #self.__dict__[paletteName+"Left"].master = self
        self.__dict__[paletteName+ "View"] = InterpolationView(self.__dict__["paletteBox"+str(self.counter)])
        self.__dict__[paletteName+"Right"] = ColorButton(self, self.__dict__["paletteBox"+str(self.counter)], color = QColor(initialColor2))
        #self.__dict__[paletteName+"Right"].master = self

        self.__dict__[paletteName+"passThroughBlack"] = passThroughBlack
        self.__dict__[paletteName+"passThroughBlackCheckbox"] = OWGUI.checkBox(self.__dict__["buttonBox"+str(self.counter)], self, paletteName+"passThroughBlack", "Pass through black", callback = self.colorSchemaChange)
        self.paletteNames.append(paletteName)
        self.counter += 1

    # #####################################################
    # DISCRETE COLOR PALETTE
    # #####################################################
    def createDiscretePalette(self, boxCaption, colorPalette = OWGraphTools.defaultRGBColors):
        hbox = OWGUI.widgetBox(self.mainArea, boxCaption, orientation = 'horizontal')
        self.discListbox = OWGUI.listBox(hbox, self)
        vbox = OWGUI.widgetBox(hbox, orientation = 'vertical')
        #self.buttLoad       = OWGUI.button(vbox, self, "D", callback = self.showPopup, tooltip="Load a predefined set of colors")
        self.buttLoad       = OWGUI.button(vbox, self, "D", tooltip="Load a predefined set of colors", callback = self.showPopup)
        self.buttLoad.setFixedSize(20, 20)
        buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attributes up")
        buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attributes down")
        buttonUPAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        buttonUPAttr.setMaximumWidth(20)
        buttonDOWNAttr.setIcon(QIcon(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        buttonDOWNAttr.setMaximumWidth(20)
        self.connect(self.discListbox, SIGNAL("itemDoubleClicked ( QListWidgetItem *)"), self.changeDiscreteColor)

        self.popupMenu = QMenu(self)
        self.popupMenu.addAction("Load default RGB palette", self.loadRGBPalette)
        self.popupMenu.addAction("Load Color Brewer palette", self.loadCBPalette)
        #self.buttLoad.setMenu(self.popupMenu)

        self.discreteColors = [QColor(r,g,b) for (r,g,b) in colorPalette]
        for ind in range(len(self.discreteColors)):
            self.discListbox.addItem(QListWidgetItem(OWTools.ColorPixmap(self.discreteColors[ind], 14), "Color %d" % (ind)))

    def changeDiscreteColor(self, item):
        ind = self.discListbox.indexFromItem(item).row()
        color = QColorDialog.getColor(self.discreteColors[ind], self)
        if color.isValid():
            self.discListBox.item(ind).setIcon(OWTools.ColorPixmap(color, 14))
            self.discreteColors[ind] = color

    def loadRGBPalette(self):
        self.discListbox.clear()
        self.discreteColors = [QColor(r,g,b) for (r,g,b) in OWGraphTools.defaultRGBColors]
        for ind in range(len(self.discreteColors)):
            self.discListbox.addItem(QListWidgetItem(OWTools.ColorPixmap(self.discreteColors[ind], 14), "Color %d" % (ind)))

    def loadCBPalette(self):
        self.discListbox.clear()
        self.discreteColors = [QColor(r,g,b) for (r,g,b) in OWGraphTools.ColorBrewerColors]
        for ind in range(len(self.discreteColors)):
            self.discListbox.addItem(QListWidgetItem(OWTools.ColorPixmap(self.discreteColors[ind], 14), "Color %d" % (ind)))

    def showPopup(self):
        point = self.buttLoad.mapToGlobal(QPoint(0, self.buttLoad.height()))
        self.popupMenu.popup(point)

    # move selected attribute in "Attribute Order" list one place up
    def moveAttrUP(self):
        if len(self.discListbox.selectedIndexes()) == 0: return
        ind = self.discListbox.selectedIndexes()[0].row()
        if ind == 0: return
        iconI = self.discListbox.item(ind-1).icon()
        iconII = self.discListbox.item(ind).icon()
        self.discListbox.item(ind-1).setIcon(iconII)
        self.discListbox.item(ind).setIcon(iconI)
        self.discListbox.setCurrentRow(ind-1)
        self.discreteColors.insert(ind-1, self.discreteColors.pop(ind))


    # move selected attribute in "Attribute Order" list one place down
    def moveAttrDOWN(self):
        if len(self.discListbox.selectedIndexes()) == 0: return
        ind = self.discListbox.selectedIndexes()[0].row()
        if ind == self.discListbox.count()-1: return
        iconI = self.discListbox.item(ind+1).icon()
        iconII = self.discListbox.item(ind).icon()
        self.discListbox.item(ind+1).setIcon(iconII)
        self.discListbox.item(ind).setIcon(iconI)
        self.discListbox.setCurrentRow(ind+1)
        self.discreteColors.insert(ind+1, self.discreteColors.pop(ind))


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
        self.schemaCombo.addItems([s[0] for s in schemas])
        self.schemaCombo.addItem("Save current palette as...")
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
            palette = self.createPalette(self.rgbToQColor(l), self.rgbToQColor(r), chk) + 5*[QColor(Qt.white).rgb()]
            self.__dict__[name+"View"].setPalette1(palette)

        self.discreteColors = [self.rgbToQColor(col) for col in discPalette]
        if self.discreteColors:
            self.discListbox.clear()
            for ind in range(len(self.discreteColors)):
                self.discListbox.addItem(QListWidgetItem(OWTools.ColorPixmap(self.discreteColors[ind], 14), "Color %d" % (ind)))

    def paletteSelected(self):
        if not self.schemaCombo.count(): return

        # if we selected "Save current palette as..." option then add another option to the list
        if self.selectedSchemaIndex == self.schemaCombo.count()-1:
            message = "Please enter a new name for the current color schema:"
            ok = FALSE
            while not ok:
                text, ok = QInputDialog.getText(self, "New Schema Name", message)
                if (ok):
                    newName = str(text)
                    oldNames = [str(self.schemaCombo.itemText(i)).lower() for i in range(self.schemaCombo.count()-1)]
                    if newName.lower() == "default":
                        ok = FALSE
                        message = "Can not change the 'Default' schema. Please enter a different name:"
                    elif newName.lower() in oldNames:
                        index = oldNames.index(newName.lower())
                        self.colorSchemas.pop(index)

                    if ok:
                        self.colorSchemas.insert(0, (newName, self.getCurrentState()))
                        self.schemaCombo.insertItem(0, newName)
                        #self.schemaCombo.setCurrentIndex(0)
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

    c = ColorPalette(None, modal = FALSE)
    c.createContinuousPalette("continuousPalette", "Continuous Palette")
    c.createDiscretePalette("Discrete Palette")
    box = c.createBox("otherColors", "Colors")
    c.createColorButton(box, "Canvas", "Canvas")
    box.addSpace(5)
    c.createColorButton(box, "Grid", "Grid")
    box.addSpace(5)
    c.createColorButton(box, "test", "ttest")
    box.addSpace(5)
    box.adjustSize()
    c.setColorSchemas()
    c.show()
    a.exec_()
    """
    c = OWChooseImageSizeDlg(None)
    c.show()
    a.exec_()
    """
import os
from OWBaseWidget import *
import OWGUI
from PyQt4.Qwt5 import *
from ColorPalette import *

class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName", "penWidthFactor"]
    def __init__(self, graph, extraButtons = []):
        OWBaseWidget.__init__(self, None, None, "Image settings", modal = TRUE, resizingEnabled = 0)

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
            self.customXEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customX", "Width: ", orientation = "horizontal", valueType = int)
            self.customYEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customY", "Height:", orientation = "horizontal", valueType = int)
            OWGUI.comboBoxWithCaption(self.space, self, "penWidthFactor", label = 'Factor:   ', box = " Pen width multiplication factor ",  tooltip = "Set the pen width factor for all curves in the plot\n(Useful for example when the lines in the plot look to thin)\nDefault: 1", sendSelectedValue = 1, valueType = int, items = range(1,20))
        elif isinstance(graph, QGraphicsScene):
            OWGUI.widgetLabel(box, "Image size will be set automatically.")

        box = OWGUI.widgetBox(self.space, 1)
        #self.printButton =          OWGUI.button(self.space, self, "Print", callback = self.printPic)
        self.saveImageButton =      OWGUI.button(box, self, "Save Image", callback = self.saveImage)
        self.saveMatplotlibButton = OWGUI.button(box, self, "Save Graph as matplotlib Script", callback = self.saveToMatplotlib)
        for (text, funct) in extraButtons:
            butt = OWGUI.button(box, self, text, callback = funct)
            self.connect(butt, SIGNAL("clicked()"), self.accept)        # also connect the button to accept so that we close the dialog
        OWGUI.button(box, self, "Cancel", callback = self.reject)

        self.resize(250,300)
        self.updateGUI()

    def saveImage(self, filename = None, size = None, closeDialog = 1):
        if not filename:
            filename = self.getFileName("graph.png", "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF)", ".png")
            if not filename: return

        (fil,ext) = os.path.splitext(filename)
        if ext.lower() not in [".bmp", ".gif", ".png"] :
            ext = ".png"                                        # if no format was specified, we choose png
        filename = fil + ext

        if isinstance(self.graph, QGraphicsScene):
            source = self.getSceneBoundingRect().adjusted(-15, -15, 15, 15)
            size = source.size()
        elif not size:
            size = self.getSize()

        painter = QPainter()
        buffer = QPixmap(int(size.width()), int(size.height()))
        painter.begin(buffer)
        painter.setRenderHint(QPainter.Antialiasing)
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
            target = QRectF(0,0, source.width(), source.height())
            self.graph.render(painter, target, source)

        buffer.save(filename)

        if closeDialog:
            QDialog.accept(self)


    def getSceneBoundingRect(self):
        source = QRectF()
        for item in self.graph.items():
            if item.isVisible(): 
                source = source.united(item.boundingRect().translated(item.pos()))
        return source

    def saveToMatplotlib(self):
        filename = self.getFileName("graph.py","Python Script (*.py)", ".py")
        if filename:
            if isinstance(self.graph, QwtPlot):
                self.graph.saveToMatplotlib(filename, self.getSize())
            else:
                rect = self.getSceneBoundingRect()
                minx, maxx, miny, maxy = rect.x(), rect.x()+rect.width(), rect.y(), rect.y()+rect.height()
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
    def getFileName(self, defaultName, mask, extension):
        fileName = str(QFileDialog.getSaveFileName(self, "Save to..", self.lastSaveDirName + defaultName, mask))
        if not fileName: return None
        if not os.path.splitext(fileName)[1][1:]: fileName = fileName + extension

        self.lastSaveDirName = os.path.split(fileName)[0] + "/"
        self.saveSettings()
        return fileName

    def getSize(self):
        if isinstance(self.graph, QGraphicsScene):
            size = self.getSceneBoundingRect().size()
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





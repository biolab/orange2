import os
from OWBaseWidget import *
import OWGUI

_have_qwt = True
try:
    from PyQt4.Qwt5 import *
except ImportError:
    _have_qwt = False

from PyQt4.QtGui import QGraphicsScene, QGraphicsView
from PyQt4.QtSvg import *
from ColorPalette import *
import OWQCanvasFuncts

class OWChooseImageSizeDlg(OWBaseWidget):
    settingsList = ["selectedSize", "customX", "customY", "lastSaveDirName", "penWidthFactor"]
    def __init__(self, graph, extraButtons = [], defaultName="graph", parent=None):
        OWBaseWidget.__init__(self, parent, None, "Image settings", modal = TRUE, resizingEnabled = 0)

        self.graph = graph
        self.selectedSize = 0
        self.customX = 400
        self.customY = 400
        self.saveAllSizes = 0
        self.penWidthFactor = 1
        self.lastSaveDirName = "./"
        self.defaultName = defaultName

        self.loadSettings()

        self.setLayout(QVBoxLayout(self))
        self.space = OWGUI.widgetBox(self)
        self.layout().setMargin(8)
        #self.layout().addWidget(self.space)

        box = OWGUI.widgetBox(self.space, "Image Size")

        global _have_qwt
        if _have_qwt and isinstance(graph, QwtPlot):
            size = OWGUI.radioButtonsInBox(box, self, "selectedSize", ["Current size", "400 x 400", "600 x 600", "800 x 800", "Custom:"], callback = self.updateGUI)
            self.customXEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customX", "Width: ", orientation = "horizontal", valueType = int)
            self.customYEdit = OWGUI.lineEdit(OWGUI.indentedBox(box), self, "customY", "Height:", orientation = "horizontal", valueType = int)
            OWGUI.comboBoxWithCaption(self.space, self, "penWidthFactor", label = 'Factor:   ', box = " Pen width multiplication factor ",  tooltip = "Set the pen width factor for all curves in the plot\n(Useful for example when the lines in the plot look to thin)\nDefault: 1", sendSelectedValue = 1, valueType = int, items = range(1,20))
        elif isinstance(graph, QGraphicsScene) or isinstance(graph, QGraphicsView):
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
            filename = self.getFileName(self.defaultName, "Portable Network Graphics (*.PNG);;Windows Bitmap (*.BMP);;Graphics Interchange Format (*.GIF);;Scalable Vector Graphics (*.SVG)", ".png")
            if not filename: return

        (fil,ext) = os.path.splitext(filename)
        if ext.lower() not in [".bmp", ".gif", ".png", ".svg"] :
            ext = ".png"                                        # if no format was specified, we choose png
        filename = fil + ext
        
        real_graph = self.graph if isinstance(self.graph, QGraphicsView) else None
        if real_graph:
            self.graph = self.graph.scene()            

        if isinstance(self.graph, QGraphicsScene):
            source = self.getSceneBoundingRect().adjusted(-15, -15, 15, 15)
            size = source.size()
        elif isinstance(self.graph, QGraphicsView):
            source = self.graph.sceneRect()
            size = source.size()
        elif not size:
            size = self.getSize()

        painter = QPainter()
        if filename.lower().endswith(".svg"):
            buffer = QSvgGenerator()
            buffer.setFileName(filename)
            buffer.setSize(QSize(int(size.width()), int(size.height())))
        else:
            buffer = QPixmap(int(size.width()), int(size.height()))
        painter.begin(buffer)
        painter.setRenderHint(QPainter.Antialiasing)
        if not filename.lower().endswith(".svg"):
            if isinstance(self.graph, QGraphicsScene) or isinstance(self.graph, QGraphicsView):
                # make background same color as the widget's background
                brush = self.graph.backgroundBrush()
                if brush.style() == Qt.NoBrush:
                    brush = QBrush(self.graph.palette().color(QPalette.Base))
                painter.fillRect(buffer.rect(), brush)
            else:
                painter.fillRect(buffer.rect(), QBrush(Qt.white))

        # qwt plot
        global _have_qwt
        if _have_qwt and isinstance(self.graph, QwtPlot):

            if self.penWidthFactor != 1:
                for curve in self.graph.itemList():
                    pen = curve.pen(); pen.setWidth(self.penWidthFactor*pen.width()); curve.setPen(pen)

            self.graph.print_(painter, QRect(0,0,size.width(), size.height()))

            if self.penWidthFactor != 1:
                for curve in self.graph.itemList():
                    pen = curve.pen(); pen.setWidth(pen.width()/self.penWidthFactor); curve.setPen(pen)

        # QGraphicsScene
        elif isinstance(self.graph, QGraphicsScene) or isinstance(self.graph, QGraphicsView):
            target = QRectF(0,0, source.width(), source.height())
            self.graph.render(painter, target, source)

        if not filename.lower().endswith(".svg"):
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
        filename = self.getFileName(self.defaultName, "Python Script (*.py)", ".py")
        if filename:
            global _have_qwt
            if _have_qwt and isinstance(self.graph, QwtPlot):
                self.graph.saveToMatplotlib(filename, self.getSize())
            else:
                rect = self.getSceneBoundingRect()
                minx, maxx, miny, maxy = rect.x(), rect.x()+rect.width(), rect.y(), rect.y()+rect.height()
                f = open(filename, "wt")
                f.write("# This Python file uses the following encoding: utf-8\n")
                f.write("from pylab import *\nfrom matplotlib.patches import Rectangle\n\n#constants\nx1 = %f; x2 = %f\ny1 = 0.0; y2 = %f\ndpi = 80\nxsize = %d\nysize = %d\nedgeOffset = 0.01\n\nfigure(facecolor = 'w', figsize = (xsize/float(dpi), ysize/float(dpi)), dpi = dpi)\na = gca()\nhold(True)\n" % (minx, maxx, maxy, maxx-minx, maxy-miny))
                
                if isinstance(self.graph, QGraphicsView):
                    scene = self.graph.scene()
                else:
                    scene = self.graph
                
                sortedList = [(item.zValue(), item) for item in scene.items()]
                sortedList.sort()   # sort items by z value

                for (z, item) in sortedList:
                    # a little compatibility for QT 3.3 (on Mac at least)
                    if hasattr(item, "isVisible"):
                        if not item.isVisible(): continue
                    elif not item.visible(): continue
                    if item.__class__ in [QGraphicsRectItem, QGraphicsLineItem]:
                        penc, penAlpha  = self._getColorFromObject(item.pen())
                        penWidth = item.pen().width()

                        if isinstance(item, QGraphicsRectItem):
                            x,y,w,h = item.rect().x(), maxy-item.rect().y()-item.rect().height(), item.rect().width(), item.rect().height()
                            brushc, brushAlpha = self._getColorFromObject(item.brush())
                            f.write("a.add_patch(Rectangle((%d, %d), %d, %d, edgecolor=%s, facecolor = %s, linewidth = %d, fill = %d))\n" % (x,y,w,h, penc, brushc, penWidth, type(brushc) == tuple))
                        elif isinstance(item, QGraphicsLineItem):
                            x1,y1, x2,y2 = item.line().x1(), maxy-item.line().y1(), item.line().x2(), maxy-item.line().y2()
                            f.write("plot(%s, %s, marker = 'None', linestyle = '-', color = %s, linewidth = %d, alpha = %.3f)\n" % ([x1,x2], [y1,y2], penc, penWidth, penAlpha))
                    elif item.__class__ in [QGraphicsTextItem, OWQCanvasFuncts.OWCanvasText]:
                        if item.__class__  == QGraphicsTextItem:
                            xalign, yalign = "left", "top"
                            x, y = item.x(), item.y()
                        else:
                            align = item.alignment
                            #xalign = (align & Qt.AlignLeft and "right") or (align & Qt.AlignRight and "left") or (align & Qt.AlignHCenter and "center")
                            #yalign = (align & Qt.AlignBottom and "top") or (align & Qt.AlignTop and "bottom") or (align & Qt.AlignVCenter and "center")
                            xalign = (align & Qt.AlignLeft and "left") or (align & Qt.AlignRight and "right") or (align & Qt.AlignHCenter and "center")
                            yalign = (align & Qt.AlignBottom and "bottom") or (align & Qt.AlignTop and "top") or (align & Qt.AlignVCenter and "center")
                            x, y = item.x, item.y
                        vertAlign = (yalign and ", verticalalignment = '%s'" % yalign) or ""
                        horAlign = (xalign and ", horizontalalignment = '%s'" % xalign) or ""
                        color = tuple([item.defaultTextColor().red()/255., item.defaultTextColor().green()/255., item.defaultTextColor().blue()/255.])
                        weight = item.font().bold() and "bold" or "normal"
                        f.write("text(%f, %f, '%s'%s%s, color = %s, name = '%s', weight = '%s', alpha = %.3f)\n" % (item.x, maxy-item.y, unicode(item.toPlainText()).encode("utf-8"), vertAlign, horAlign, color, str(item.font().family()), weight, item.defaultTextColor().alpha()/float(255)))

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
        fileName = str(QFileDialog.getSaveFileName(self, "Save to..", os.path.join(self.lastSaveDirName, defaultName), mask))
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
        global _have_qwt
        if _have_qwt and isinstance(self.graph, QwtPlot):
            self.customXEdit.setEnabled(self.selectedSize == 4)
            self.customYEdit.setEnabled(self.selectedSize == 4)

    def _getColorFromObject(self, obj):
        if isinstance(obj, QBrush) and obj.style() == Qt.NoBrush: return "'none'", 1
        if isinstance(obj, QPen)   and obj.style() == Qt.NoPen: return "'none'", 1
        col = [obj.color().red(), obj.color().green(), obj.color().blue()];
        col = tuple([v/float(255) for v in col])
        return col, obj.color().alpha()/float(255)

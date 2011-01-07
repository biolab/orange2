"""<name>Image Viewer</name>
<description>View images embeded in example table</description>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *

import OWGUI


#def setupShadow(widget):
#    if qVersion() >= "4.6":
#        ge = QGraphicsDropShadowEffect(widget)
#        ge.setBlurRadius(4)
#        ge.setOffset(QPointF(2, 2))
#        ge.setColor(QColor(0, 0, 0, 100))
#        widget.setGraphicsEffect(ge)
        

class GraphicsPixmapWidget(QGraphicsWidget):
    def __init__(self, pixmap, parent=None, scene=None):
        QGraphicsWidget.__init__(self, parent)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        self.pixmap = pixmap
        self.pixmapSize = QSizeF(100, 100)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        if scene is not None:
            scene.addItem(self)
        
        
    def sizeHint(self, which, contraint=QSizeF()):
        return self.pixmapSize
    
    
    def paint(self, painter, option, widget=0):
        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 50), 3))
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        pixmapRect = QRectF(QPointF(0,0), self.pixmapSize)
        painter.drawPixmap(pixmapRect, self.pixmap, QRectF(QPointF(0, 0), QSizeF(self.pixmap.size())))
        painter.restore()
        
        
    def setPixmapSize(self, size):
        self.pixmapSize = size
        self.updateGeometry()
         
            
class GraphicsTextWidget(QGraphicsWidget):
    def __init__(self, text, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.labelItem = QGraphicsTextItem(self)
        self.setHtml(text)
        
        self.connect(self.labelItem.document().documentLayout(),
                     SIGNAL("documentSizeChanged(QSizeF)"), self.onLayoutChanged)
        
        
    def onLayoutChanged(self, *args):
        self.updateGeometry()
        
        
    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.MinimumSize:
            return self.labelItem.boundingRect().size()
        else:
            return self.labelItem.boundingRect().size()
        
        
    def setTextWidth(self, width):
        self.labelItem.setTextWidth(width)
        
        
    def setHtml(self, text):
        self.labelItem.setHtml(text)
        
        
#    def paint(self, painter, *args):
#        painter.drawRect(self.boundingRect())
        
            
class GraphicsThumbnailWidget(QGraphicsWidget):
    def __init__(self, pixmap, title="Image", parent=None, scene=None):
        QGraphicsWidget.__init__(self, parent)
        #self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        layout = QGraphicsLinearLayout(Qt.Vertical, self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.pixmapWidget = GraphicsPixmapWidget(pixmap, self)
        self.labelWidget = GraphicsTextWidget(title, self)
        layout.addItem(self.pixmapWidget)
        layout.addItem(self.labelWidget)
        layout.setAlignment(self.pixmapWidget, Qt.AlignCenter)
        layout.setAlignment(self.labelWidget, Qt.AlignCenter)
        self.setLayout(layout)
        
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setTitle(title)
        self.setTitleWidth(150)
        self.setThumbnailSize(QSizeF(150, 150))
        
        
    def setTitle(self, text):
        self.labelWidget.setHtml('<center>' + text + '</center>')
        self.layout().invalidate()
        
        
    def setThumbnailSize(self, size):
        self.pixmapWidget.setPixmapSize(size)
        self.labelWidget.setTextWidth(max(100, size.width()))
        
        
    def setTitleWidth(self, width):
        self.labelWidget.setTextWidth(width)
        self.layout().invalidate()
        
        
    def setGeometry(self, rect):
        QGraphicsWidget.setGeometry(self, rect)


    def paint(self, painter, option, widget=0):
        contents = self.contentsRect()
        if self.isSelected():
            painter.save()
            pen = painter.pen()
            painter.setPen(QPen(QColor(125, 162, 206, 192)))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(QRectF(contents.topLeft(), self.geometry().size()), 3, 3)
            painter.restore()
            

class ThumbnailWidget(QGraphicsWidget):
    def __init__(self, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        
        
class GraphicsScene(QGraphicsScene):
    def __init__(self, *args):
        QGraphicsScene.__init__(self, *args)
        self.selectionRect = None
        
        
    def mousePressEvent(self, event):
        QGraphicsScene.mousePressEvent(self, event)
        
        
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            screenPos = event.screenPos()
            buttonDown = event.buttonDownScreenPos(Qt.LeftButton)
            if (screenPos - buttonDown).manhattanLength() > 2.0:
                self.updateSelectionRect(event)
        QGraphicsScene.mouseMoveEvent(self, event)
        
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selectionRect:
                self.removeItem(self.selectionRect)
                self.selectionRect = None
        QGraphicsScene.mouseReleaseEvent(self, event)
        
        
    def updateSelectionRect(self, event):
        pos = event.scenePos()
        buttonDownPos = event.buttonDownScenePos(Qt.LeftButton)
        rect = QRectF(pos, buttonDownPos).normalized()
        rect = rect.intersected(self.sceneRect())
        if not self.selectionRect:
            self.selectionRect = QGraphicsRectItem()
            self.selectionRect.setBrush(QColor(10, 10, 10, 20))
            self.selectionRect.setPen(QPen(QColor(200, 200, 200, 200)))
            self.addItem(self.selectionRect)
        self.selectionRect.setRect(rect)
        if event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.ShiftModifier: 
            path = self.selectionArea()
        else:
            path = QPainterPath()
        path.addRect(rect)
        self.setSelectionArea(path)
        self.emit(SIGNAL("selectionRectPointChanged(QPointF)"), pos)


class OWImageViewer(OWWidget):
    contextHandlers = {"": DomainContextHandler("", ["imageAttr", "titleAttr"])}
    settingsList = ["zoom"]
    def __init__(self, parent=None, signalManager=None, name="Image viewer"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        
        self.inputs = [("Example Table", ExampleTable, self.setData)]
        self.outputs = [("Example Table", ExampleTable)]
        
        self.imageAttr = 0
        self.titleAttr = 0
        self.zoom = 25
        self.autoCommit = False
        self.selectionChangedFlag = False
        
        # ###
        # GUI
        # ###
        
        self.loadSettings()
        
        self.imageAttrCB = OWGUI.comboBox(self.controlArea, self, "imageAttr",
                                          box="Image Filename Attribute",
                                          tooltip="Attribute with image filenames",
                                          callback=self.setupScene,
                                          addSpace=True
                                          )
        
        self.titleAttrCB = OWGUI.comboBox(self.controlArea, self, "titleAttr",
                                          box="Title Attribute",
                                          tooltip="Attribute with image title",
                                          callback=self.updateTitles,
                                          addSpace=True
                                          )
        
        OWGUI.hSlider(self.controlArea, self, "zoom",
                      box="Zoom", minValue=1, maxValue=100, step=1,
                      callback=self.updateZoom,
                      createLabel=False
                      )
        
        OWGUI.separator(self.controlArea)
        
        box = OWGUI.widgetBox(self.controlArea, "Selection")
        b = OWGUI.button(box, self, "Commit", callback=self.commit)
        cb = OWGUI.checkBox(box, self, "autoCommit", "Commit on any change",
                            tooltip="Send selections on any change",
                            callback=self.commitIf
                            )
        OWGUI.setStopper(self, b, cb, "selectionChangedFlag", callback=self.commit)
        
        OWGUI.rubber(self.controlArea)
        
        self.scene = GraphicsScene()
        self.sceneView = QGraphicsView(self.scene, self)
        self.sceneView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.sceneView.setRenderHint(QPainter.Antialiasing, True)
        self.sceneView.setRenderHint(QPainter.TextAntialiasing, True)
        self.sceneView.setFocusPolicy(Qt.WheelFocus)
        self.mainArea.layout().addWidget(self.sceneView)
        
        self.connect(self.scene, SIGNAL("selectionChanged()"), self.onSelectionChanged)
        self.connect(self.scene, SIGNAL("selectionRectPointChanged(QPointF)"), self.onSelectionRectPointChanged, Qt.QueuedConnection)
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveScene)
        self.resize(800, 600)
        
        self.sceneLayout = None
        self.selectedExamples = []
        self.hasTypeImageHint = False
        
        self.updateZoom()
        
        
    def setData(self, data):
        self.data = data
        self.closeContext("")
        self.information(0)
        self.error(0)
        if data is not None:
            self.imageAttrCB.clear()
            self.titleAttrCB.clear()
            self.allAttrs = data.domain.variables + data.domain.getmetas().values()
            self.stringAttrs = [attr for attr in self.allAttrs if attr.varType == orange.VarTypes.String]
            self.hasTypeImageHint = any("type" in attr.attributes for attr in self.stringAttrs)
            self.stringAttrs = sorted(self.stringAttrs, key=lambda  attr: 0 if "type" in attr.attributes else 1)
            icons = OWGUI.getAttributeIcons()
            for attr in self.stringAttrs:
                self.imageAttrCB.addItem(icons[attr.varType], attr.name)
            for attr in self.allAttrs:
                self.titleAttrCB.addItem(icons[attr.varType], attr.name)
            
            self.openContext("", data)
            self.imageAttr = max(min(self.imageAttr, len(self.stringAttrs) - 1), 0)
            self.titleAttr = max(min(self.titleAttr, len(self.allAttrs) - 1), 0)
            
            if self.stringAttrs:
                self.setupScene()
            else:
                self.clearScene()
        else:
            self.imageAttrCB.clear()
            self.titleAttrCB.clear()
            self.clearScene()
            
            
    def setupScene(self):
        self.clearScene()
        self.scene.blockSignals(True)
        thumbnailSize = self.zoom / 25.0 * 150.0
        self.information(0)
        self.error(0)
        if self.data:
            attr = self.stringAttrs[self.imageAttr]
            titleAttr = self.allAttrs[self.titleAttr]
            examples = [ex for ex in self.data if not ex[attr].isSpecial()]
            widget = ThumbnailWidget()
            layout = QGraphicsGridLayout()
            layout.setSpacing(10)
            widget.setLayout(layout)
            widget.setPos(10, 10)
            self.scene.addItem(widget)
            fileExistsCount = 0
            for i, ex in enumerate(examples):
                filename = self.filenameFromValue(ex[attr])
                if os.path.exists(filename):
                    fileExistsCount += 1
                title = str(ex[titleAttr])
                pixmap = self.pixmapFromFile(filename)
                thumbnail = GraphicsThumbnailWidget(pixmap, title=title, parent=widget)
                thumbnail.setToolTip(filename)
                thumbnail.setThumbnailSize(QSizeF(thumbnailSize, thumbnailSize))
                thumbnail.example = ex
                layout.addItem(thumbnail, i/5, i%5)
            widget.show()
            layout.invalidate()
            self.sceneLayout = layout
            if fileExistsCount == 0 and not "type" in attr.attributes:
                self.error(0, "No images found!\nMake sure the '%s' attribute is tagged with 'type=image'" % attr.name) 
            elif fileExistsCount < len(examples):
                self.information(0, "Only %i out of %i images found." % (fileExistsCount, len(examples)))
            
                
                
        self.scene.blockSignals(False)
        
        qApp.processEvents()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
            
    def filenameFromValue(self, value):
        variable = value.variable
        if isinstance(variable, orange.StringVariable):
            origin = variable.attributes.get("origin", "")
            name = str(value)
            return os.path.join(origin, name)
        elif isinstance(variable, URIVariable):
            return str(value)
            
    def pixmapFromFile(self, filename):
        pixmap = QPixmap(filename)
        if pixmap.isNull():
            try:
                import Image, ImageQt
                img = Image.open(filename)
#                print img.format, img.mode, img.size
#                data = img.tostring()
#                pixmap = QPixmap.loadFromData(data)
                pixmap = QPixmap.fromImage(ImageQt.ImageQt(img))
            except Exception, ex:
                print ex
        return pixmap
    
    def clearScene(self):
        self.scene.clear()
        self.sceneLayout = None
        qApp.processEvents()
        
        
    def thumbnailItems(self):
        for item in self.scene.items():
            if isinstance(item, GraphicsThumbnailWidget):
                yield item
                
    def updateZoom(self):
        self.scene.blockSignals(True)
        scale = self.zoom / 25.0 
        for item in self.thumbnailItems():
            item.setThumbnailSize(QSizeF(scale * 150, scale * 150))
            
        if self.sceneLayout:
            self.sceneLayout.activate()
        qApp.processEvents()
        self.scene.blockSignals(False)
        
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        
        
    def updateTitles(self):
        titleAttr = self.allAttrs[self.titleAttr]
        for item in self.scene.items():
            if isinstance(item, GraphicsThumbnailWidget):
                item.setTitle(str(item.example[titleAttr]))
                
        qApp.processEvents()
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        
            
    def onSelectionChanged(self):
        try:
            items = self.scene.selectedItems()
            items = [item for item in items if isinstance(item, GraphicsThumbnailWidget)]
            self.selectedExamples = [item.example for item in items]
            self.commitIf()
        except RuntimeError, err:
            pass
        
    def onSelectionRectPointChanged(self, point):
        self.sceneView.ensureVisible(QRectF(point , QSizeF(1, 1)), 5, 5)
        
    def commitIf(self):
        if self.autoCommit:
            self.commit()
        else:
            self.selectionChangedFlag = True
            
    def commit(self):
        if self.data:
            if self.selectedExamples:
                selected = orange.ExampleTable(self.data.domain, self.selectedExamples)
            else:
                selected = None
            self.send("Example Table", selected)
        else:
            self.send("Example Table", None)
        self.selectionChangedFlag = False
            
    def saveScene(self):
        from OWDlgs import OWChooseImageSizeDlg
        sizeDlg = OWChooseImageSizeDlg(self.scene, parent=self)
        sizeDlg.exec_()
        
        
if __name__ == "__main__":
    app = QApplication([])
    w = OWImageViewer()
    w.show()
#    data = orange.ExampleTable(os.path.expanduser("~/Desktop/images.tab"))
#    os.chdir(os.path.expanduser("~/Desktop/"))
    data = orange.ExampleTable(os.path.expanduser("~/Downloads/pex11_orng_sample/pex11_sample.tab"))
    os.chdir(os.path.expanduser("~/Downloads/pex11_orng_sample/"))
    w.setData(data)
    app.exec_()
    w.saveSettings()
     
        
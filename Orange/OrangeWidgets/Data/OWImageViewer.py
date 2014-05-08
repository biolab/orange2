import weakref
import logging
from xml.sax.saxutils import escape
from collections import namedtuple
from functools import partial
from itertools import izip_longest

from PyQt4.QtCore import pyqtSignal as Signal
from PyQt4.QtNetwork import (
    QNetworkAccessManager, QNetworkDiskCache, QNetworkRequest, QNetworkReply
)

from OWWidget import *
from OWItemModels import VariableListModel
from OWConcurrent import Future, FutureWatcher

import OWGUI

_log = logging.getLogger(__name__)


NAME = "Image Viewer"
DESCRIPTION = "Views images embedded in the data."
LONG_DESCRIPTION = ""
ICON = "icons/ImageViewer.svg"
PRIORITY = 4050
AUTHOR = "Ales Erjavec"
AUTHOR_EMAIL = "ales.erjavec(@at@)fri.uni-lj.si"
INPUTS = [("Data", Orange.data.Table, "setData")]
OUTPUTS = [("Data", Orange.data.Table, )]


class GraphicsPixmapWidget(QGraphicsWidget):

    def __init__(self, pixmap, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.setCacheMode(QGraphicsItem.ItemCoordinateCache)
        self._pixmap = pixmap
        self._pixmapSize = QSizeF()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setPixmap(self, pixmap):
        if self._pixmap != pixmap:
            self._pixmap = QPixmap(pixmap)
            self.updateGeometry()

    def pixmap(self):
        return QPixmap(self._pixmap)

    def setPixmapSize(self, size):
        if self._pixmapSize != size:
            self._pixmapSize = QSizeF(size)
            self.updateGeometry()

    def pixmapSize(self):
        if self._pixmapSize.isValid():
            return QSizeF(self._pixmapSize)
        else:
            return QSizeF(self._pixmap.size())

    def sizeHint(self, which, constraint=QSizeF()):
        if which == Qt.PreferredSize:
            return self.pixmapSize()
        else:
            return QGraphicsWidget.sizeHint(self, which, constraint)

    def paint(self, painter, option, widget=0):
        if self._pixmap.isNull():
            return

        rect = self.contentsRect()

        pixsize = self.pixmapSize()
        pixrect = QRectF(QPointF(0, 0), pixsize)
        pixrect.moveCenter(rect.center())

        painter.save()
        painter.setPen(QPen(QColor(0, 0, 0, 50), 3))
        painter.drawRoundedRect(pixrect, 2, 2)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        source = QRectF(QPointF(0, 0), QSizeF(self._pixmap.size()))
        painter.drawPixmap(pixrect, self._pixmap, source)
        painter.restore()


class GraphicsTextWidget(QGraphicsWidget):

    def __init__(self, text, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.labelItem = QGraphicsTextItem(self)
        self.setHtml(text)

        self.labelItem.document().documentLayout().documentSizeChanged.connect(
            self.onLayoutChanged
        )

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


class GraphicsThumbnailWidget(QGraphicsWidget):

    def __init__(self, pixmap, title="", parent=None):
        QGraphicsWidget.__init__(self, parent)

        self._title = None
        self._size = QSizeF()

        layout = QGraphicsLinearLayout(Qt.Vertical, self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setContentsMargins(0, 0, 0, 0)

        self.pixmapWidget = GraphicsPixmapWidget(pixmap, self)
        self.labelWidget = GraphicsTextWidget(title, self)

        layout.addItem(self.pixmapWidget)
        layout.addItem(self.labelWidget)

        layout.setAlignment(self.pixmapWidget, Qt.AlignCenter)
        layout.setAlignment(self.labelWidget, Qt.AlignHCenter | Qt.AlignBottom)
        self.setLayout(layout)

        self.setSizePolicy(QSizePolicy.MinimumExpanding,
                           QSizePolicy.MinimumExpanding)

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setTitle(title)
        self.setTitleWidth(100)

    def setPixmap(self, pixmap):
        self.pixmapWidget.setPixmap(pixmap)
        self._updatePixmapSize()

    def pixmap(self):
        return self.pixmapWidget.pixmap()

    def setTitle(self, title):
        if self._title != title:
            self._title = title
            self.labelWidget.setHtml(
                '<center>' + escape(title) + '</center>'
            )
            self.layout().invalidate()

    def title(self):
        return self._title

    def setThumbnailSize(self, size):
        if self._size != size:
            self._size = QSizeF(size)
            self._updatePixmapSize()
            self.labelWidget.setTextWidth(max(100, size.width()))

    def setTitleWidth(self, width):
        self.labelWidget.setTextWidth(width)
        self.layout().invalidate()

    def paint(self, painter, option, widget=0):
        contents = self.contentsRect()
        if self.isSelected():
            painter.save()
            painter.setPen(QPen(QColor(125, 162, 206, 192)))
            painter.setBrush(QBrush(QColor(217, 232, 252, 192)))
            painter.drawRoundedRect(QRectF(contents.topLeft(),
                                           self.geometry().size()), 3, 3)
            painter.restore()

    def _updatePixmapSize(self):
        pixmap = self.pixmap()
        if not pixmap.isNull() and self._size.isValid():
            pixsize = QSizeF(self.pixmap().size())
            pixsize.scale(self._size, Qt.KeepAspectRatio)
        else:
            pixsize = QSizeF()
        self.pixmapWidget.setPixmapSize(pixsize)


class ThumbnailWidget(QGraphicsWidget):

    def __init__(self, parent=None):
        QGraphicsWidget.__init__(self, parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.setContentsMargins(10, 10, 10, 10)
        layout = QGraphicsGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setLayout(layout)

    def setGeometry(self, geom):
        super(ThumbnailWidget, self).setGeometry(geom)
        self.reflow(self.size().width())

    def reflow(self, width):
        if not self.layout():
            return

        left, right, _, _ = self.getContentsMargins()
        layout = self.layout()
        width -= left + right

        hints = self._hints(Qt.PreferredSize)
        widths = [max(24, h.width()) for h in hints]
        ncol = self._fitncols(widths, layout.horizontalSpacing(), width)

        if ncol == layout.columnCount():
            return

        items = [layout.itemAt(i) for i in range(layout.count())]

        # re-add the items them back in updated positions
        # (QGraphicsGridLayout takes care of moving existing items)
        for i, item in enumerate(items):
            layout.addItem(item, i // ncol, i % ncol)

    def items(self):
        layout = self.layout()
        if layout:
            return [layout.itemAt(i) for i in range(layout.count())]
        else:
            return []

    def _hints(self, which):
        return [item.effectiveSizeHint(which) for item in self.items()]

    def _fitncols(self, widths, spacing, constraint):
        def sliced(seq, ncol):
            return [seq[i:i + ncol] for i in range(0, len(seq), ncol)]

        def flow_width(widths, spacing, ncol):
            W = sliced(widths, ncol)
            col_widths = map(max, izip_longest(*W, fillvalue=0))
            return sum(col_widths) + (ncol - 1) * spacing

        ncol_best = 1
        for ncol in range(2, len(widths)):
            w = flow_width(widths, spacing, ncol)
            if w <= constraint:
                ncol_best = ncol
            else:
                break

        return ncol_best


class GraphicsScene(QGraphicsScene):

    selectionRectPointChanged = Signal(QPointF)

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
        if event.modifiers() & Qt.ControlModifier or \
                event.modifiers() & Qt.ShiftModifier:
            path = self.selectionArea()
        else:
            path = QPainterPath()
        path.addRect(rect)
        self.setSelectionArea(path)
        self.selectionRectPointChanged.emit(pos)


_ImageItem = namedtuple(
    "_ImageItem",
    ["widget",    # GraphicsThumbnailWidget belonging to this item.
     "url",       # Composed final url.
     "future"]    # Future instance yielding an QImage
)


class OWImageViewer(OWWidget):
    contextHandlers = {
        "": DomainContextHandler("", ["imageAttr", "titleAttr"])
    }
    settingsList = ["zoom"]

    def __init__(self, parent=None, signalManager=None, name="Image viewer"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)

        self.inputs = [("Data", ExampleTable, self.setData)]
        self.outputs = [("Data", ExampleTable)]

        self.imageAttr = 0
        self.titleAttr = 0
        self.zoom = 25
        self.autoCommit = False
        self.selectionChangedFlag = False

        #
        # GUI
        #

        self.loadSettings()

        self.info = OWGUI.widgetLabel(
            OWGUI.widgetBox(self.controlArea, "Info"),
            "Waiting for input\n"
        )

        self.imageAttrCB = OWGUI.comboBox(
            self.controlArea, self, "imageAttr",
            box="Image Filename Attribute",
            tooltip="Attribute with image filenames",
            callback=[self.clearScene, self.setupScene],
            addSpace=True
        )

        self.titleAttrCB = OWGUI.comboBox(
            self.controlArea, self, "titleAttr",
            box="Title Attribute",
            tooltip="Attribute with image title",
            callback=self.updateTitles,
            addSpace=True
        )

        OWGUI.hSlider(
            self.controlArea, self, "zoom",
            box="Zoom", minValue=1, maxValue=100, step=1,
            callback=self.updateZoom,
            createLabel=False
        )

        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Selection")
        b = OWGUI.button(box, self, "Commit", callback=self.commit)
        cb = OWGUI.checkBox(
            box, self, "autoCommit", "Commit on any change",
            tooltip="Send selections on any change",
            callback=self.commitIf
        )

        OWGUI.setStopper(self, b, cb, "selectionChangedFlag",
                         callback=self.commit)

        OWGUI.rubber(self.controlArea)

        self.scene = GraphicsScene()
        self.sceneView = QGraphicsView(self.scene, self)
        self.sceneView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.sceneView.setRenderHint(QPainter.Antialiasing, True)
        self.sceneView.setRenderHint(QPainter.TextAntialiasing, True)
        self.sceneView.setFocusPolicy(Qt.WheelFocus)
        self.sceneView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.sceneView.installEventFilter(self)
        self.mainArea.layout().addWidget(self.sceneView)

        self.scene.selectionChanged.connect(self.onSelectionChanged)
        self.scene.selectionRectPointChanged.connect(
            self.onSelectionRectPointChanged, Qt.QueuedConnection
        )
        self.graphButton.clicked.connect(self.saveScene)
        self.resize(800, 600)

        self.thumbnailWidget = None
        self.sceneLayout = None
        self.selectedExamples = []

        #: List of _ImageItems
        self.items = []

        self._errcount = 0
        self._successcount = 0

        self.loader = ImageLoader(self)

    def setData(self, data):
        self.data = data
        self.closeContext("")
        self.information(0)
        self.error(0)
        self.imageAttrCB.clear()
        self.titleAttrCB.clear()
        self.clearScene()

        if data is not None:
            self.allAttrs = data.domain.variables + data.domain.getmetas().values()
            self.stringAttrs = [attr for attr in self.allAttrs
                                if isinstance(attr, Orange.feature.String)]

            self.stringAttrs = sorted(
                self.stringAttrs,
                key=lambda attr: 0 if "type" in attr.attributes else 1
            )

            self.imageAttrCB.setModel(VariableListModel(self.stringAttrs))
            self.titleAttrCB.setModel(VariableListModel(self.allAttrs))

            self.openContext("", data)

            self.imageAttr = max(min(self.imageAttr, len(self.stringAttrs) - 1), 0)
            self.titleAttr = max(min(self.titleAttr, len(self.allAttrs) - 1), 0)

            if self.stringAttrs:
                self.setupScene()
        else:
            self.info.setText("Waiting for input\n")

    def setupScene(self):
        self.information(0)
        self.error(0)
        if self.data:
            attr = self.stringAttrs[self.imageAttr]
            titleAttr = self.allAttrs[self.titleAttr]
            instances = [inst for inst in self.data
                         if not inst[attr].isSpecial()]
            widget = ThumbnailWidget()
            layout = widget.layout()

            self.scene.addItem(widget)

            for i, inst in enumerate(instances):
                url = self.urlFromValue(inst[attr])
                title = str(inst[titleAttr])

                thumbnail = GraphicsThumbnailWidget(
                    QPixmap(), title=title, parent=widget
                )

                thumbnail.setToolTip(unicode(url.toString()))
                thumbnail.instance = inst
                layout.addItem(thumbnail, i / 5, i % 5)

                if url.isValid():
                    future = self.loader.get(url)
                    watcher = FutureWatcher(future, parent=thumbnail)

                    def set_pixmap(thumb=thumbnail, future=future):
                        if future.cancelled():
                            return
                        if future.exception():
                            # Should be some generic error image.
                            pixmap = QPixmap()
                            thumb.setToolTip(unicode(thumb.toolTip()) + "\n" +
                                             str(future.exception()))
                        else:
                            pixmap = QPixmap.fromImage(future.result())

                        thumb.setPixmap(pixmap)
                        if not pixmap.isNull():
                            thumb.setThumbnailSize(self.pixmapSize(pixmap))

                        self._updateStatus(future)

                    watcher.finished.connect(set_pixmap, Qt.QueuedConnection)
                else:
                    future = None
                self.items.append(_ImageItem(thumbnail, url, future))

            widget.show()
            widget.geometryChanged.connect(self._updateSceneRect)

            self.info.setText("Retrieving...\n")
            self.thumbnailWidget = widget
            self.sceneLayout = layout

        if self.sceneLayout:
            width = (self.sceneView.width() -
                     self.sceneView.verticalScrollBar().width())
            self.thumbnailWidget.reflow(width)
            self.thumbnailWidget.setPreferredWidth(width)
            self.sceneLayout.activate()

    def filenameFromValue(self, value):
        variable = value.variable
        origin = variable.attributes.get("origin", "")
        name = str(value)
        return os.path.join(origin, name)

    def urlFromValue(self, value):
        variable = value.variable
        origin = variable.attributes.get("origin", "")
        if origin and QDir(origin).exists():
            origin = QUrl.fromLocalFile(origin)
        elif origin:
            origin = QUrl(origin)
            if not origin.scheme():
                origin.setScheme("file")
        else:
            origin = QUrl("")
        base = unicode(origin.path())
        if base.strip() and not base.endswith("/"):
            origin.setPath(base + "/")

        name = QUrl(str(value))
        url = origin.resolved(name)
        if not url.scheme():
            url.setScheme("file")
        return url

    def pixmapSize(self, pixmap):
        """
        Return the preferred pixmap size based on the current `zoom` value.
        """
        scale = 2 * self.zoom / 100.0
        size = QSizeF(pixmap.size()) * scale
        return size.expandedTo(QSizeF(16, 16))

    def clearScene(self):
        for item in self.items:
            if item.future:
                item.future._reply.close()
                item.future.cancel()

        self.items = []
        self._errcount = 0
        self._successcount = 0

        self.scene.clear()
        self.thumbnailWidget = None
        self.sceneLayout = None

    def thumbnailItems(self):
        return [item.widget for item in self.items]

    def updateZoom(self):
        for item in self.thumbnailItems():
            item.setThumbnailSize(self.pixmapSize(item.pixmap()))

        if self.thumbnailWidget:
            width = (self.sceneView.width() -
                     self.sceneView.verticalScrollBar().width())

            self.thumbnailWidget.reflow(width)
            self.thumbnailWidget.setPreferredWidth(width)

        if self.sceneLayout:
            self.sceneLayout.activate()

    def updateTitles(self):
        titleAttr = self.allAttrs[self.titleAttr]
        for item in self.items:
            item.widget.setTitle(str(item.widget.instance[titleAttr]))

    def onSelectionChanged(self):
        selected = [item.widget for item in self.items
                    if item.widget.isSelected()]
        self.selectedExamples = [item.instance for item in selected]
        self.commitIf()

    def onSelectionRectPointChanged(self, point):
        self.sceneView.ensureVisible(QRectF(point, QSizeF(1, 1)), 5, 5)

    def commitIf(self):
        if self.autoCommit:
            self.commit()
        else:
            self.selectionChangedFlag = True

    def commit(self):
        if self.data:
            if self.selectedExamples:
                selected = Orange.data.Table(self.data.domain, self.selectedExamples)
            else:
                selected = None
            self.send("Data", selected)
        else:
            self.send("Data", None)
        self.selectionChangedFlag = False

    def saveScene(self):
        from OWDlgs import OWChooseImageSizeDlg
        sizeDlg = OWChooseImageSizeDlg(self.scene, parent=self)
        sizeDlg.exec_()

    def _updateStatus(self, future):
        if future.cancelled():
            return

        if future.exception():
            self._errcount += 1
            _log.debug("Error: %r", future.exception())
        else:
            self._successcount += 1

        count = len([item for item in self.items if item.future is not None])
        self.info.setText(
            "Retrieving:\n" +
            "{} of {} images" .format(self._successcount, count))

        if self._errcount + self._successcount == count:
            if self._errcount:
                self.info.setText(
                    "Done:\n" +
                    "{} images, {} errors".format(count, self._errcount)
                )
            else:
                self.info.setText(
                    "Done:\n" +
                    "{} images".format(count)
                )
            attr = self.stringAttrs[self.imageAttr]
            if self._errcount == count and not "type" in attr.attributes:
                self.error(0,
                           "No images found! Make sure the '%s' attribute "
                           "is tagged with 'type=image'" % attr.name)

    def _updateSceneRect(self):
        self.scene.setSceneRect(self.scene.itemsBoundingRect())

    def onDeleteWidget(self):
        for item in self.items:
            item.future._reply.abort()
            item.future.cancel()

    def eventFilter(self, receiver, event):
        if receiver is self.sceneView and event.type() == QEvent.Resize \
                and self.thumbnailWidget:
            width = (self.sceneView.width() -
                     self.sceneView.verticalScrollBar().width())

            self.thumbnailWidget.reflow(width)
            self.thumbnailWidget.setPreferredWidth(width)

        return super(OWImageViewer, self).eventFilter(receiver, event)


class ImageLoader(QObject):

    #: A weakref to a QNetworkAccessManager used for image retrieval.
    #: (we can only have only one QNetworkDiskCache opened on the same
    #: directory)
    _NETMANAGER_REF = None

    def __init__(self, parent=None):
        QObject.__init__(self, parent=None)
        assert QThread.currentThread() is QApplication.instance().thread()

        netmanager = self._NETMANAGER_REF and self._NETMANAGER_REF()
        if netmanager is None:
            netmanager = QNetworkAccessManager()
            cache = QNetworkDiskCache()
            cache.setCacheDirectory(
                os.path.join(environ.widget_settings_dir,
                             __name__ + ".ImageLoader.Cache")
            )
            netmanager.setCache(cache)
            ImageLoader._NETMANAGER_REF = weakref.ref(netmanager)
        self._netmanager = netmanager

    def get(self, url):
        future = Future()
        url = url = QUrl(url)
        request = QNetworkRequest(url)
        request.setRawHeader("User-Agent", "OWImageViewer/1.0")
        request.setAttribute(
            QNetworkRequest.CacheLoadControlAttribute,
            QNetworkRequest.PreferCache
        )

        # Future yielding a QNetworkReply when finished.
        reply = self._netmanager.get(request)
        future._reply = reply

        def on_reply_ready(reply, future):
            if reply.error() == QNetworkReply.OperationCanceledError:
                # The network request itself was canceled
                future.cancel()
                return

            if reply.error() != QNetworkReply.NoError:
                # XXX Maybe convert the error into standard
                # http and urllib exceptions.
                future.set_exception(Exception(reply.errorString()))
                return

            reader = QImageReader(reply)
            image = reader.read()

            if image.isNull():
                future.set_exception(Exception(reader.errorString()))
            else:
                future.set_result(image)

        reply.finished.connect(partial(on_reply_ready, reply, future))
        return future


if __name__ == "__main__":
    app = QApplication([])
    w = OWImageViewer()
    w.show()
    data = Orange.data.Table(os.path.expanduser("~/Downloads/pex11_orng_sample/pex11_sample.tab"))
    os.chdir(os.path.expanduser("~/Downloads/pex11_orng_sample/"))
    w.setData(data)
    app.exec_()
    w.saveSettings()

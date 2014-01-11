import itertools
import unicodedata
import copy
import random


from PyQt4 import QtCore
from PyQt4.QtCore import pyqtSignal as Signal

import Orange

import OWToolbars
import OWColorPalette
from OWWidget import *
from OWGraph import *
from OWItemModels import VariableListModel, PyListModel, ModelActionsWidget


NAME = "Paint Data"
DESCRIPTION = """
Paints the data on a 2D plane. Place individual data
points or use brush to paint a larger data sets.
"""
LONG_DESCRIPTION = ""
ICON = "icons/PaintData.svg"
PRIORITY = 40
OUTPUTS = [("Data", Orange.data.Table, )]

dir = OWToolbars.dir
icon_magnet = os.path.join(dir, "magnet_64px.png")
icon_jitter = os.path.join(dir, "jitter_64px.png")
icon_brush = os.path.join(dir, "brush_64px.png")
icon_put = os.path.join(dir, "put_64px.png")
icon_select = os.path.join(dir, "select-transparent_42px.png")
icon_lasso = os.path.join(dir, "lasso-transparent_42px.png")
#icon_remove = os.path.join(dir, "remove.svg")


class PaintDataGraph(OWGraph):
    def __init__(self, *args, **kwargs):
        OWGraph.__init__(self, *args, **kwargs)
        self.toolPixmap = QPixmap()

    def setToolPixmap(self, pixmap):
        self.toolPixmap = pixmap
        self.update()

    def setData(self, data, attr1, attr2):
        """
        Set the data to display.

        :param data: data
        :param attr1: attr for X axis
        :param attr2: attr for Y axis
        """
        OWGraph.setData(self, data)
        self.data = data
        self.attr1 = attr1
        self.attr2 = attr2
        self.updateGraph()

    def updateGraph(self, dataInterval=None):
        if dataInterval:
            start, end = dataInterval
            data = self.data[start:end]
        else:
            self.removeDrawingCurves()
            data = self.data

        domain = self.data.domain
        clsValues = domain.class_var.values if domain.class_var else [0]

        palette = ColorPaletteGenerator(len(clsValues))
        for i, cls in enumerate(clsValues):
            x = [float(ex[self.attr1]) for ex in data if ex.getclass() == cls]
            y = [float(ex[self.attr2]) for ex in data if ex.getclass() == cls]
            self.addCurve("data points", xData=x, yData=y,
                          brushColor=palette[i], penColor=palette[i])
        self.replot()

    def drawCanvas(self, painter):
        OWGraph.drawCanvas(self, painter)
        pixmap = self.toolPixmap
        if not pixmap.isNull():
            painter.drawPixmap(0, 0, pixmap)


class DataTool(QObject):
    """
    A base class for data tools that operate on PaintDataGraph
    widget by installing itself as its event filter.

    """
    editing = Signal()
    editingFinished = Signal()

    cursor = Qt.ArrowCursor

    class optionsWidget(QFrame):
        """
        An options (parameters) widget for the tool (this will
        be put in the "Options" box in the main OWPaintData widget
        when this tool is selected.

        """
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool

    def __init__(self, graph, parent=None):
        QObject.__init__(self, parent)
        self.setGraph(graph)

    def setGraph(self, graph):
        """
        Install this tool to operate on ``graph``.

        If another tool is already operating on the graph it will first
        be removed.

        """
        self.graph = graph
        if graph:
            installed = getattr(graph, "_data_tool_event_filter", None)
            if installed:
                self.graph.canvas().removeEventFilter(installed)
                installed.removed()
            self.graph.canvas().setMouseTracking(True)
            self.graph.canvas().installEventFilter(self)
            self.graph._data_tool_event_filter = self
            self.graph.setToolPixmap(QPixmap())
            self.graph.setCursor(self.cursor)
            self.graph.replot()
            self.installed()

    def removed(self):
        """Called when the tool is removed from a graph.
        """
        pass

    def installed(self):
        """Called when the tool is installed on a graph.
        """
        pass

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            return self.mousePressEvent(event)
        elif event.type() == QEvent.MouseButtonRelease:
            return self.mouseReleaseEvent(event)
        elif event.type() == QEvent.MouseButtonDblClick:
            return self.mouseDoubleClickEvent(event)
        elif event.type() == QEvent.MouseMove:
            return self.mouseMoveEvent(event)
        elif event.type() == QEvent.Paint:
            return self.paintEvent(event)
        elif event.type() == QEvent.Leave:
            return self.leaveEvent(event)
        elif event.type() == QEvent.Enter:
            return self.enterEvent(event)
        return False

    # These are actually event filters (note the return values)
    def paintEvent(self, event):
        return False

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def enterEvent(self, event):
        return False

    def leaveEvent(self, event):
        return False

    def keyPressEvent(self, event):
        return False

    def transform(self, point):
        x = self.graph.transform(QwtPlot.xBottom, point.x())
        y = self.graph.transform(QwtPlot.yLeft, point.y())
        return QPoint(x, y)

    def invTransform(self, point):
        x = self.graph.invTransform(QwtPlot.xBottom, point.x())
        y = self.graph.invTransform(QwtPlot.yLeft, point.y())
        return QPointF(x, y)

    def attributes(self):
        return self.graph.attr1, self.graph.attr2

    def dataTransform(self, *args):
        pass


class GraphSelections(QObject):
    selectionRegionAdded = Signal(int, QPainterPath)
    selectionRegionUpdated = Signal(int, QPainterPath)
    selectionRegionRemoved = Signal(int, QPainterPath)

    selectionRegionMoveStarted = Signal(int, QPointF, QPainterPath)
    selectionRegionMoved = Signal(int, QPointF, QPainterPath)
    selectionRegionMoveFinished = Signal(int, QPointF, QPainterPath)
    selectionGeometryChanged = Signal()

    def __init__(self, parent, movable=True, multipleSelection=False):
        QObject.__init__(self, parent)
        self.selection = []
        self.movable = movable
        self.multipleSelection = multipleSelection

        self._moving_index, self._moving_pos, self._selection_region = \
            (-1, QPointF(), (QPointF(), QPointF()))

    def getPos(self, event):
        graph = self.parent()
        pos = event.pos()
        x = graph.invTransform(QwtPlot.xBottom, pos.x())
        y = graph.invTransform(QwtPlot.yLeft, pos.y())
        return QPointF(x, y)

    def toPath(self, region):
        path = QPainterPath()
        if isinstance(region, QRectF) or isinstance(region, QRect):
            path.addRect(region.normalized())
        elif isinstance(region, tuple):
            path.addRect(QRectF(*region).normalized())
        elif isinstance(region, list):
            path.addPolygon(QPolygonF(region + [region[0]]))
        return path

    def addSelectionRegion(self, region):
        self.selection.append(region)
        self.selectionRegionAdded.emit(
            len(self.selection) - 1, self.toPath(region))

    def setSelectionRegion(self, index, region):
        self.selection[index] = region
        self.selectionRegionUpdated.emit(index, self.toPath(region))

    def clearSelection(self):
        for i, region in enumerate(self.selection):
            self.selectionRegionRemoved.emit(i, self.toPath(region))
        self.selection = []

    def start(self, event):
        pos = self.getPos(event)
        index = self.regionAt(event)
        if index == -1 or not self.movable:
            if event.modifiers() & Qt.ControlModifier and \
                    self.multipleSelection:
                self.addSelectionRegion((pos, pos))
            else:
                self.clearSelection()
                self.addSelectionRegion((pos, pos))
            self._moving_index = -1
        else:
            self._moving_index, self._moving_pos, self._selection_region = \
                (index, pos, self.selection[index])

            self.selectionRegionMoveStarted.emit(
                index, pos, self.toPath(self.selection[index])
            )
        self.selectionGeometryChanged.emit()

    def update(self, event):
        pos = self.getPos(event)
        index = self._moving_index
        if index == -1:
            self.selection[-1] = self.selection[-1][:-1] + (pos,)
            self.selectionRegionUpdated.emit(
                len(self.selection) - 1, self.toPath(self.selection[-1]))

        else:
            diff = self._moving_pos - pos
            self.selection[index] = \
                tuple([p - diff for p in self._selection_region])
            self.selectionRegionMoved.emit(
                index, pos, self.toPath(self.selection[index])
            )
        self.selectionGeometryChanged.emit()

    def end(self, event):
        self.update(event)
        if self._moving_index != -1:
            self.selectionRegionMoveFinished.emit(
                self._moving_index, self.getPos(event),
                self.toPath(self.selection[self._moving_index])
            )
        self._moving_index = -1

    def regionAt(self, event):
        pos = self.getPos(event)
        for i, region in enumerate(self.selection):
            if self.toPath(region).contains(pos):
                return i
        return -1

    def testSelection(self, data):
        data = numpy.asarray(data)
        path = QPainterPath()
        for region in self.selection:
            path = path.united(self.toPath(region))
        def test(point):
            return path.contains(QPointF(point[0], point[1]))
        test = numpy.apply_along_axis(test, 1, data)
        return test

    def __nonzero__(self):
        return bool(self.selection)

    def __bool__(self):
        return bool(self.selection)

    def path(self):
        path = QPainterPath()
        for region in self.selection:
            path = path.united(self.toPath(region))
        return path

    def qTransform(self):
        graph = self.parent()
        invTransform = graph.invTransform
        sx = invTransform(QwtPlot.xBottom, 1) - invTransform(QwtPlot.xBottom, 0)
        sy = invTransform(QwtPlot.yLeft, 1) - invTransform(QwtPlot.yLeft, 0)
        dx = invTransform(QwtPlot.xBottom, 0)
        dy = invTransform(QwtPlot.yLeft, 0)
        return QTransform(sx, 0.0, 0.0, sy, dx, dy)


class SelectTool(DataTool):
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QHBoxLayout()
            action = QAction(
                 "Delete", self,
                 toolTip="Delete selected instances.",
                 shortcut=QKeySequence.Delete,
                 shortcutContext=Qt.WindowShortcut,
            )
            if sys.platform == "darwin":
                action.setShortcuts(
                    [QKeySequence(Qt.ControlModifier + Qt.Key_Backspace),
                     QKeySequence(QKeySequence.Delete)]
                )

            delete = QToolButton(self)
            delete.setDefaultAction(action)

            action.triggered.connect(self.tool.deleteSelected)

            layout.addWidget(delete)
            layout.addStretch(10)
            self.setLayout(layout)

    def __init__(self, graph, parent=None, graphSelection=None):
        DataTool.__init__(self, graph, parent)
        if graphSelection is None:
            self.selection = GraphSelections(graph)
        else:
            self.selection = graphSelection

        self.pen = QPen(Qt.black, 1, Qt.DashDotLine)
        self.pen.setCosmetic(True)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setCapStyle(Qt.RoundCap)

        self.selection.selectionRegionMoveStarted.connect(self.onMoveStarted)
        self.selection.selectionRegionMoved.connect(self.onMove)
        self.selection.selectionRegionMoveFinished.connect(self.onMoveFinished)
        self.selection.selectionRegionUpdated.connect(
                self.invalidateMoveSelection)

        self._validMoveSelection = False
        self._moving = None

    def setGraph(self, graph):
        DataTool.setGraph(self, graph)
        if graph and hasattr(self, "selection"):
            self.selection.setParent(graph)

    def installed(self):
        DataTool.installed(self)
        self.invalidateMoveSelection()

    def paintEvent(self, event):
        if self.selection:
            pixmap = QPixmap(self.graph.canvas().size())
            pixmap.fill(QColor(255, 255, 255, 0))
            painter = QPainter(pixmap)
            painter.setRenderHints(QPainter.Antialiasing)
            inverted, singular = self.selection.qTransform().inverted()
            painter.setPen(self.pen)

            painter.setTransform(inverted)
            for region in self.selection.selection:
                painter.drawPath(self.selection.toPath(region))
            painter.end()
            self.graph.setToolPixmap(pixmap)
        return False

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection.start(event)
            self.graph.replot()
        return True

    def mouseMoveEvent(self, event):
        index = self.selection.regionAt(event)
        if index != -1:
            self.graph.canvas().setCursor(Qt.OpenHandCursor)
        else:
            self.graph.canvas().setCursor(self.graph._cursor)

        if event.buttons() & Qt.LeftButton:
            self.selection.update(event)
            self.graph.replot()
        return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selection.end(event)
            self.graph.replot()
        return True

    def invalidateMoveSelection(self, *args):
        self._validMoveSelection = False
        self._moving = None

    def onMoveStarted(self, index, pos, path):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        if not self._validMoveSelection:
            self._moving = [(i, float(ex[attr1]), float(ex[attr2]))
                            for i, ex in enumerate(data)]
            self._moving = [(i, x, y) for i, x, y in self._moving
                            if path.contains(QPointF(x, y))]
            self._validMoveSelection = True
        self._move_anchor = pos

    def onMove(self, index, pos, path):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2

        diff = pos - self._move_anchor
        for i, x, y in self._moving:
            ex = data[i]
            ex[attr1] = x + diff.x()
            ex[attr2] = y + diff.y()
        self.graph.updateGraph()
        self.editing.emit()

    def onMoveFinished(self, index, pos, path):
        self.onMove(index, pos, path)
        diff = pos - self._move_anchor
        self._moving = [(i, x + diff.x(), y + diff.y()) \
                        for i, x, y in self._moving]

        self.editingFinished.emit()

    def deleteSelected(self, *args):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        path = self.selection.path()
        selected = [i for i, ex in enumerate(data)
                    if path.contains(QPointF(float(ex[attr1]),
                                             float(ex[attr2])))]
        if selected:
            self.editing.emit()

        for i in reversed(selected):
            del data[i]
        self.graph.updateGraph()

        if selected:
            self.editingFinished.emit()


class GraphLassoSelections(GraphSelections):
    def start(self, event):
        pos = self.getPos(event)
        index = self.regionAt(event)
        if index == -1:
            self.clearSelection()
            self.addSelectionRegion([pos])
        else:
            self._moving_index, self._moving_pos, self._selection_region = \
                index, pos, self.selection[index]

            self.selectionRegionMoveStarted.emit(
                index, pos, self.toPath(self.selection[index])
            )

        self.selectionGeometryChanged.emit()

    def update(self, event):
        pos = self.getPos(event)
        index = self._moving_index
        if index == -1:
            self.selection[-1].append(pos)
            self.selectionRegionUpdated.emit(
                len(self.selection) - 1, self.toPath(self.selection[-1])
            )
        else:
            diff = self._moving_pos - pos
            self.selection[index] = [p - diff for p in self._selection_region]
            self.selectionRegionMoved.emit(
                index, pos, self.toPath(self.selection[index])
            )

        self.selectionGeometryChanged.emit()

    def end(self, event):
        self.update(event)
        if self._moving_index != -1:
            self.selectionRegionMoveFinished.emit(
                self._moving_index, self.getPos(event),
                self.toPath(self.selection[self._moving_index])
            )
        self._moving_index = -1


class LassoTool(SelectTool):
    def __init__(self, graph, parent=None):
        SelectTool.__init__(self, graph, parent,
                            graphSelection=GraphLassoSelections(graph))


class ZoomTool(DataTool):
    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)

    def paintEvent(self, event):
        return False

    def mousePressEvent(self, event):
        return False

    def mouseMoveEvent(self, event):
        return False

    def mouseReleaseEvent(self, event):
        return False

    def mouseDoubleClickEvent(self, event):
        return False

    def keyPressEvent(self, event):
        return False


class PutInstanceTool(DataTool):
    cursor = Qt.CrossCursor

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            coord = self.invTransform(event.pos())
            val1, val2 = coord.x(), coord.y()
            attr1, attr2 = self.attributes()
            self.editing.emit()
            self.dataTransform(attr1, val1, attr2, val2)
            self.editingFinished.emit()
        return True

    def dataTransform(self, attr1, val1, attr2, val2):
        domain = self.graph.data.domain
        example = Orange.data.Instance(
            domain, [val1, val2, domain.class_var.base_value]
        )

        self.graph.data.append(example)
        self.graph.updateGraph(dataInterval=(-1, sys.maxint))


class BrushTool(DataTool):
    brushRadius = 20
    density = 5
    cursor = Qt.CrossCursor

    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QFormLayout()
            self.radiusSlider = QSlider(Qt.Horizontal, minimum=10, maximum=30,
                                        value=self.tool.brushRadius)
            self.densitySlider = QSlider(Qt.Horizontal, minimum=3, maximum=10,
                                         value=self.tool.density)

            layout.addRow("Radius", self.radiusSlider)
            layout.addRow("Density", self.densitySlider)
            self.setLayout(layout)

            self.radiusSlider.valueChanged[int].connect(
                lambda value: setattr(self.tool, "brushRadius", value)
            )

            self.densitySlider.valueChanged[int].connect(
                lambda value: setattr(self.tool, "density", value)
            )

    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.brushState = -20, -20, 0, 0

    def mousePressEvent(self, event):
        self.brushState = (event.pos().x(), event.pos().y(),
                           self.brushRadius, self.brushRadius)
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.editing.emit()
            self.dataTransform(attr1, x, rx, attr2, y, ry)

        self.graph.replot()
        return True

    def mouseMoveEvent(self, event):
        self.brushState = (event.pos().x(), event.pos().y(),
                           self.brushRadius, self.brushRadius)
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
        self.graph.replot()
        return True

    def mouseReleaseEvent(self, event):
        self.graph.replot()
        if event.button() & Qt.LeftButton:
            self.editingFinished.emit()
        return True

    def leaveEvent(self, event):
        self.graph.setToolPixmap(QPixmap())
        self.graph.replot()
        return False

    def paintEvent(self, event):
        if not self.graph.canvas().underMouse():
            self.graph.setToolPixmap(QPixmap())
            return False

        pixmap = QPixmap(self.graph.canvas().size())
        pixmap.fill(QColor(255, 255, 255, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 1))
        x, y, w, h = self.brushState
        painter.drawEllipse(QPoint(x, y), w, h)

        painter.end()

        self.graph.setToolPixmap(pixmap)
        return False

    def brushGeometry(self, point):
        coord = self.invTransform(point)
        dcoord = self.invTransform(QPoint(point.x() + self.brushRadius,
                                          point.y() + self.brushRadius))
        x, y = coord.x(), coord.y()
        rx, ry = dcoord.x() - x, -(dcoord.y() - y)
        return x, y, rx, ry

    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        new = []
        domain = self.graph.data.domain

        for i in range(self.density):
            ex = Orange.data.Instance(
                domain, [random.normalvariate(x, rx),
                         random.normalvariate(y, ry),
                         domain.class_var.base_value]
            )

            new.append(ex)
        self.graph.data.extend(new)
        self.graph.updateGraph(dataInterval=(-len(new), sys.maxint))


class MagnetTool(BrushTool):
    cursor = Qt.ArrowCursor

    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            distsq = (x1 - x) ** 2 + (y1 - y) ** 2
            dist = math.sqrt(distsq)
            attraction = self.density / 100.0
            advance = 0.005
            dx = -(x1 - x) / dist * attraction / max(distsq, rx) * advance
            dy = -(y1 - y) / dist * attraction / max(distsq, ry) * advance
            ex[attr1] = x1 + dx
            ex[attr2] = y1 + dy
        self.graph.updateGraph()


class JitterTool(BrushTool):
    cursor = Qt.ArrowCursor

    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            distsq = (x1 - x) ** 2 + (y1 - y) ** 2
            dist = math.sqrt(distsq)
            attraction = self.density / 100.0
            advance = 0.005
            dx = -(x1 - x) / dist * attraction / max(distsq, rx) * advance
            dy = -(y1 - y) / dist * attraction / max(distsq, ry) * advance
            ex[attr1] = x1 - random.normalvariate(0, dx)    # *self.density)
            ex[attr2] = y1 - random.normalvariate(0, dy)    # *self.density)
        self.graph.updateGraph()


class EnumVariableModel(PyListModel):
    def __init__(self, var, parent=None, **kwargs):
        PyListModel.__init__(self, [], parent, **kwargs)
        self.wrap(var.values)
        self.colorPalette = OWColorPalette.ColorPaletteHSV(len(self))

    def __delitem__(self, index):
        raise TypeError("Cannot delete EnumVariable value")

    def __delslice__(self, i, j):
        raise TypeError("Cannot delete EnumVariable values")

    def __setitem__(self, index, item):
        self._list[index] = str(item)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            i = index.row()
            return QVariant(self.itemQIcon(i))
        else:
            return PyListModel.data(self, index, role)

    def itemQIcon(self, i):
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(255, 255, 255, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(self.colorPalette[i])
        painter.drawEllipse(QRectF(15, 15, 39, 39))
        painter.end()
        return QIcon(pixmap)


class OWPaintData(OWWidget):
    TOOLS = [("Brush", "Create multiple instances", BrushTool, icon_brush),
             ("Put", "Put individual instances", PutInstanceTool, icon_put),
             ("Select", "Select and move instances", SelectTool, icon_select),
             ("Lasso", "Select and move instances", LassoTool, icon_lasso),
             ("Jitter", "Jitter instances", JitterTool, icon_jitter),
             ("Magnet", "Move (drag) multiple instances", MagnetTool, icon_magnet),
             ("Zoom", "Zoom", ZoomTool, OWToolbars.dlg_zoom)
             ]
    settingsList = ["commitOnChange"]

    def __init__(self, parent=None, signalManager=None, name="Data Generator"):
        OWWidget.__init__(self, parent, signalManager, name)

        self.outputs = [("Data", ExampleTable)]

        self.addClassAsMeta = False
        self.attributes = []
        self.cov = []
        self.commitOnChange = False

        self.loadSettings()

        self.variablesModel = VariableListModel(
            [Orange.feature.Continuous(name) for name in ["X", "Y"]], self,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )

        self.classVariable = Orange.feature.Discrete(
            "Class label", values=["Class 1", "Class 2"], baseValue=0
        )

        w = OWGUI.widgetBox(self.controlArea, "Class Label")
        w.layout().setSpacing(1)

        self.classValuesView = listView = QListView(
            selectionMode=QListView.SingleSelection,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored,
                                   QSizePolicy.Maximum)
        )

        self.classValuesModel = EnumVariableModel(
            self.classVariable, self,
            flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )
        self.classValuesModel.wrap(self.classVariable.values)

        listView.setModel(self.classValuesModel)
        listView.selectionModel().select(self.classValuesModel.index(0),
                                         QItemSelectionModel.ClearAndSelect)
        listView.selectionModel().selectionChanged.connect(
            self.onClassLabelSelection
        )
        w.layout().addWidget(listView)

        self.addClassLabel = QAction(
            "+", self, toolTip="Add class label"
        )
        self.addClassLabel.triggered.connect(self.addNewClassLabel)

        self.removeClassLabel = QAction(
            unicodedata.lookup("MINUS SIGN"), self,
            toolTip="Remove selected class label"
        )
        self.removeClassLabel.triggered.connect(self.removeSelectedClassLabel)

        actionsWidget = ModelActionsWidget(
            [self.addClassLabel, self.removeClassLabel], self
        )
        actionsWidget.layout().addStretch(10)
        actionsWidget.layout().setSpacing(1)

        w.layout().addWidget(actionsWidget)

        toolbox = OWGUI.widgetBox(self.controlArea, "Tools",
                                  orientation=QGridLayout())
        self.toolActions = QActionGroup(self, exclusive=True)

        for i, (name, tooltip, tool, icon) in enumerate(self.TOOLS):
            action = QAction(name, self, toolTip=tooltip, checkable=True)
            if os.path.exists(icon):
                action.setIcon(QIcon(icon))

            action.triggered[()].connect(
                lambda tool=tool: self.onToolAction(tool)
            )

            button = QToolButton(
                iconSize=QSize(24, 24),
                toolButtonStyle=Qt.ToolButtonTextUnderIcon,
                sizePolicy=QSizePolicy(QSizePolicy.MinimumExpanding,
                                       QSizePolicy.Fixed)
            )
            button.setDefaultAction(action)
            toolbox.layout().addWidget(button, i / 3, i % 3)
            self.toolActions.addAction(action)

        # TODO: Columns should have uniform widths
        for column in range(3):
            toolbox.layout().setColumnMinimumWidth(column, 10)
            toolbox.layout().setColumnStretch(column, 1)

        self.optionsLayout = QStackedLayout()
        self.toolsStackCache = {}
        OWGUI.widgetBox(self.controlArea, "Options",
                        orientation=self.optionsLayout)

        ur = OWGUI.widgetBox(self.controlArea, "")
        undo = QAction("Undo", self, toolTip="Undo action",
                       shortcut=QKeySequence.Undo)
        undo.triggered.connect(self.undo)

        redo = QAction("Redo", self, toolTip="Redo action",
                       shortcut=QKeySequence.Redo)
        redo.triggered.connect(self.redo)

        actionsWidget = ModelActionsWidget([undo, redo], self)
        actionsWidget.layout().addStretch(10)
        actionsWidget.layout().setSpacing(1)

        ur.layout().addWidget(actionsWidget)

        OWGUI.rubber(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Commit")

        cb = OWGUI.checkBox(box, self, "commitOnChange", "Commit on change",
                            tooltip="Send the data on any change.",
                            callback=self.commitIf,)
        b = OWGUI.button(box, self, "Commit",
                         callback=self.commit, default=True)
        OWGUI.setStopper(self, b, cb, "dataChangedFlag", callback=self.commit)

        self.graph = PaintDataGraph(self)
        self.graph.setAxisScale(QwtPlot.xBottom, 0.0, 1.0)
        self.graph.setAxisScale(QwtPlot.yLeft, 0.0, 1.0)
        self.graph.setAttribute(Qt.WA_Hover, True)
        self.mainArea.layout().addWidget(self.graph)

        self.currentOptionsWidget = None
        self.data = []
        self.dataChangedFlag = False
        self.domain = None

        self.onDomainChanged()
        self.toolActions.actions()[0].trigger()

        self.dataHistory = [(Orange.data.Table(self.domain),
                             ["Class 1", "Class 2"])]
        self.historyCounter = 0
        self.updateHistoryBool = True

        self.resize(800, 600)

    def addNewClassLabel(self):
        """
        Create and add a new class label
        """
        labels = ("Class %i" % i for i in itertools.count(1))
        newlabel = next(label for label in labels
                        if not label in self.classValuesModel)

        values = list(self.classValuesModel) + [newlabel]
        newclass = Orange.feature.Discrete("Class label", values=values)
        newdomain = Orange.data.Domain(self.graph.data.domain.attributes,
                                       newclass)
        newdata = Orange.data.Table(newdomain)

        newdata.extend(
            [Orange.data.Instance(
                newdomain, [ex[a] for a in ex.domain.attributes] +
                           [str(ex.getclass())])
             for ex in self.graph.data]
        )
        self.classVariable = newclass
        self.classValuesModel.wrap(self.classVariable.values)

        self.graph.data = newdata
        self.graph.updateGraph()

        newindex = self.classValuesModel.index(len(self.classValuesModel) - 1)
        self.classValuesView.selectionModel().select(
            newindex, QItemSelectionModel.ClearAndSelect
        )

        self.removeClassLabel.setEnabled(len(self.classValuesModel) > 1)

        self.updateHistory()

    def removeSelectedClassLabel(self, label=None):
        index = self.selectedClassLabelIndex()
        if index is not None and len(self.classValuesModel) > 1:
            if not label:
                label = self.classValuesModel[index]
            examples = [ex for ex in self.graph.data
                        if str(ex.getclass()) != label]

            values = [val for val in self.classValuesModel if val != label]

            newclass = Orange.feature.Discrete("Class label", values=values)
            newdomain = Orange.data.Domain(self.graph.data.domain.attributes,
                                           newclass)
            newdata = Orange.data.Table(newdomain)
            for ex in examples:
                if ex[self.classVariable] != label and \
                        ex[self.classVariable] in values:
                    values = ([ex[a] for a in ex.domain.attributes] +
                              [str(ex.getclass())])
                    newdata.append(Orange.data.Instance(newdomain, values))

            self.classVariable = newclass
            self.classValuesModel.wrap(self.classVariable.values)

            self.graph.data = newdata
            self.graph.updateGraph()

            newindex = self.classValuesModel.index(max(0, index - 1))
            self.classValuesView.selectionModel().select(
                newindex, QItemSelectionModel.ClearAndSelect
            )

            self.removeClassLabel.setEnabled(len(self.classValuesModel) > 1)

            self.updateHistory()

    def selectedClassLabelIndex(self):
        rows = self.classValuesView.selectionModel().selectedRows()
        if rows:
            return rows[0].row()
        else:
            return None

    def onClassLabelSelection(self, selected, unselected):
        index = self.selectedClassLabelIndex()
        if index is not None:
            self.classVariable.baseValue = index

    def onToolAction(self, tool):
        self.setCurrentTool(tool)

    def setCurrentTool(self, tool):
        if tool not in self.toolsStackCache:
            newtool = tool(None, self)
            option = newtool.optionsWidget(newtool, self)
            self.optionsLayout.addWidget(option)
            newtool.editing.connect(self.onDataChanged)
            newtool.editingFinished.connect(self.commitIf)
            newtool.editingFinished.connect(self.updateHistory)

            self.toolsStackCache[tool] = (newtool, option)

        self.currentTool, self.currentOptionsWidget = self.toolsStackCache[tool]
        self.optionsLayout.setCurrentWidget(self.currentOptionsWidget)
        self.currentTool.setGraph(self.graph)

    def updateHistory(self):
        if not self.updateHistoryBool:
            return
        # if we start updating from previously undone actions, we cut off redos in our history
        if not self.historyCounter == len(self.dataHistory) - 1:
            self.dataHistory = self.dataHistory[:self.historyCounter + 1]
        # append an update of labels and data
        labels = list(self.classValuesModel)
        self.dataHistory.append((copy.deepcopy(self.graph.data), labels))
        # set the size of the data history stack and remove the oldest update if we go over the size
        if len(self.dataHistory) > 100:
            self.dataHistory.pop(0)
            self.historyCounter -= 1
        self.historyCounter += 1

    def undo(self):
        # check to see if we are at the end of the stack
        if self.historyCounter > 0:
            self.historyCounter -= 1
            data, labels = self.dataHistory[self.historyCounter]
            # we dont update history when undoing
            self.updateHistoryBool = False
            # check if we only need to update labels
            if len(self.classValuesModel) > len(labels):
                diff = set(self.classValuesModel) - set(labels)
                self.removeSelectedClassLabel(label=diff.pop())
            elif len(self.classValuesModel) < len(labels):
                self.addNewClassLabel()
            # if not, update data
            else:
                self.graph.data = copy.deepcopy(data)
                self.graph.updateGraph()
            self.updateHistoryBool = True

    def redo(self):
        if self.historyCounter < len(self.dataHistory) - 1:
            self.historyCounter += 1
            data, labels = self.dataHistory[self.historyCounter]
            self.updateHistoryBool = False
            if len(self.classValuesModel) > len(labels):
                diff = set(self.classValuesModel) - set(labels)
                self.removeSelectedClassLabel(label=diff.pop())
            elif len(self.classValuesModel) < len(labels):
                self.addNewClassLabel()
            else:
                self.graph.data = copy.deepcopy(data)
                self.graph.updateGraph()
            self.updateHistoryBool = True

    def onDomainChanged(self, *args):
        if self.variablesModel:
            self.domain = Orange.data.Domain(list(self.variablesModel),
                                             self.classVariable)
            if self.data:
                self.data = Orange.data.Table(self.domain, self.data)
            else:
                self.data = Orange.data.Table(self.domain)
            self.graph.setData(self.data, 0, 1)

    def onDataChanged(self):
        self.dataChangedFlag = True

    def commitIf(self):
        if self.commitOnChange and self.dataChangedFlag:
            self.commit()
        else:
            self.dataChangedFlag = True

    def commit(self):
        data = self.graph.data
        values = set([str(ex.getclass()) for ex in data])
        if len(values) == 1:
            # Remove the useless class variable.
            domain = Orange.data.Domain(data.domain.attributes, None)
            data = Orange.data.Table(domain, data)
        self.send("Data", data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWPaintData()
    w.show()
    app.exec_()

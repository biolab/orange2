"""<name>Generate data</name>
<description>Generate artificial data sets with a simple 'Paint' like interface</description>
<contact>ales.erjavec(@at@)fri.uni-lj.si</contact>
<icon>icons/GenerateData.png</icon>
"""

import orange

from OWWidget import *
from OWGraph import *
import OWToolbars

from OWItemModels import VariableListModel, VariableDelegate, PyListModel, ModelActionsWidget
import OWColorPalette

dir = OWToolbars.dir
icon_magnet = os.path.join(dir, "magnet_64px.png")
icon_jitter = os.path.join(dir, "jitter_64px.png")
icon_brush = os.path.join(dir, "brush_64px.png")
icon_put = os.path.join(dir, "put_64px.png")
icon_select = os.path.join(dir, "select-transparent_42px.png")
icon_lasso = os.path.join(dir, "lasso-transparent_42px.png")
#icon_remove = os.path.join(dir, "remove.svg")


class DataGeneratorGraph(OWGraph):
    def setData(self, data, attr1, attr2):
        OWGraph.setData(self, data)
        self.data = data
        self.attr1 = attr1
        self.attr2 = attr2
        self.updateGraph()
        
    def updateGraph(self, dataInterval = None):
        if dataInterval:
            start, end = dataInterval
            data = self.data[start:end]
        else:
            self.removeDrawingCurves()
            data = self.data
        clsValues, hasCls = (self.data.domain.classVar.values, True) if self.data.domain.classVar else ([0], False)
        
        palette = ColorPaletteGenerator(len(clsValues))
        for i, cls in enumerate(clsValues):
            x = [float(ex[self.attr1]) for ex in data if ex.getclass() == cls]
            y = [float(ex[self.attr2]) for ex in data if ex.getclass() == cls]
            self.addCurve("data points", xData=x, yData=y, brushColor=palette[i], penColor=palette[i])
        self.replot()
        
    def drawCanvas(self, painter):
        OWGraph.drawCanvas(self, painter)
        pixmap = getattr(self, "_tool_pixmap", None)
        if pixmap:
            painter.drawPixmap(0, 0, pixmap)
        
class DataTool(QObject):
    """ A base class for data tools that operate on OWGraph
    widgets by installing itself as its event filter.
    """
    classIndex = 0
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            
    def __init__(self, graph, parent=None):
        QObject.__init__(self, parent)
        self.setGraph(graph)
        
    def setGraph(self, graph):
        self.graph = graph
        if graph:
            installed = getattr(graph,"_data_tool_event_filter", None)
            if installed:
                self.graph.canvas().removeEventFilter(installed)
            self.graph.canvas().installEventFilter(self)
            self.graph._data_tool_event_filter = self
            self.graph._tool_pixmap = None
            self.graph.replot()
        
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
        elif event.type() == QEvent.HoverLeave:
            return self.leaveEvent(event)
        elif event.type() == QEvent.HoverEnter:
            return self.enterEvent(event)
        return False  
    
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
        x, y = point.x(), point.y()
        x = self.graph.transform(QwtPlot.xBottom, x)
        y = self.graph.transform(QwtPlot.yLeft, x)
        return QPoint(x, y)
    
    def invTransform(self, point):
        x, y = point.x(), point.y()
        x = self.graph.invTransform(QwtPlot.xBottom, x)
        y = self.graph.invTransform(QwtPlot.yLeft, y)
        return QPointF(x, y)
    
    def attributes(self):
        return self.graph.attr1, self.graph.attr2
    
    def dataTransform(self, *args):
        pass
    
class GraphSelections(QObject):
    def __init__(self, parent, movable=True, multipleSelection=False):
        QObject.__init__(self, parent)
        self.selection = []
        self.movable = movable
        self.multipleSelection = multipleSelection
        
        self._moving_index, self._moving_pos, self._selection_region = -1, QPointF(), (QPointF(), QPointF())
        
    def getPos(self, event):
        graph = self.parent()
        pos = event.pos()
        x = graph.invTransform(QwtPlot.xBottom, pos.x())
        y = graph.invTransform(QwtPlot.yLeft, pos.y())
        return QPointF(x, y)
    
    def toPath(self, region):
        path = QPainterPath()
        if isinstance(region, QRectF) or isinstance(region, QRect):
            path.addRect(rect.normalized())
        elif isinstance(region, tuple):
            path.addRect(QRectF(*region).normalized())
        elif isinstance(region, list):
            path.addPolygon(QPolygonF(region + [region[0]]))
        return path
            
    
    def addSelectionRegion(self, region):
        self.selection.append(region)
        self.emit(SIGNAL("selectionRegionAdded(int, QPainterPath)"), len(self.selection) - 1, self.toPath(region))
        
    def setSelectionRegion(self, index, region):
        self.selection[index] = region
        self.emit(SIGNAL("selectionRegionUpdated(int, QPainterPath)"), index, self.toPath(region))
        
    def clearSelection(self):
        for i, region in enumerate(self.selection):
            self.emit(SIGNAL("selectionRegionRemoved(int, QPainterPath)"), i, self.toPath(region))
        self.selection = []
        
    def start(self, event):
        pos = self.getPos(event)
        index = self.regionAt(event)
        if index == -1 or not self.movable:
            if event.modifiers() & Qt.ControlModifier and self.multipleSelection: 
                self.addSelectionRegion((pos, pos))
            else:
                self.clearSelection()
                self.addSelectionRegion((pos, pos))
            self._moving_index = -1
        else:
            self._moving_index, self._moving_pos, self._selection_region = index, pos, self.selection[index]
            self.emit(SIGNAL("selectionRegionMoveStarted(int, QPointF, QPainterPath)"), index, pos, self.toPath(self.selection[index]))  
        self.emit(SIGNAL("selectionGeometryChanged()"))
    
    def update(self, event):
        pos = self.getPos(event)
        index = self._moving_index
        if index == -1:
            self.selection[-1] = self.selection[-1][:-1] + (pos,)
            self.emit(SIGNAL("selectionRegionUpdated(int, QPainterPath)"), len(self.selection) - 1 , self.toPath(self.selection[-1]))
        else:
            diff = self._moving_pos - pos
            self.selection[index] = tuple([p - diff for p in self._selection_region])
            self.emit(SIGNAL("selectionRegionMoved(int, QPointF, QPainterPath)"), index, pos, self.toPath(self.selection[index]))
            
        self.emit(SIGNAL("selectionGeometryChanged()"))
    
    def end(self, event):
        self.update(event)
        if self._moving_index != -1:
            self.emit(SIGNAL("selectionRegionMoveEnd(int)"), self._moving_index)
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
        e1 = graph.canvas().mapFrom(graph, QPoint(1, 0))
        e2 = graph.canvas().mapFrom(graph, QPoint(0, 1))
        e1x, e1y = e1.x(), e1.y()
        e2x, e2y = e2.x(), e2.y()
        sx = invTransform(QwtPlot.xBottom, 1) - invTransform(QwtPlot.xBottom, 0)
        sy = invTransform(QwtPlot.yLeft, 1) - invTransform(QwtPlot.yLeft, 0)
        dx = invTransform(QwtPlot.xBottom, 0)
        dy = invTransform(QwtPlot.yLeft, 0)
        return QTransform(sx, 0.0, 0.0, sy, dx, dy)
    
class SelectTool(DataTool):
    class optionsWidget(QFrame):
        actions = [("Move", ""),
                   ("Delete selected", "")]
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QHBoxLayout()
            delete = QToolButton(self)
            delete.pyqtConfigure(text="Delete", toolTip="Delete selected instances")
            self.connect(delete, SIGNAL("clicked()"), self.tool.deleteSelected)
            
            layout.addWidget(delete)
            self.setLayout(layout)
        
            
    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.selection = GraphSelections(graph)
        self.pen = QPen(Qt.black, 1, Qt.DashDotLine)
        self.pen.setCosmetic(True)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setCapStyle(Qt.RoundCap)
        self.connect(self.selection, SIGNAL("selectionRegionMoveStarted(int, QPointF, QPainterPath)"), self.onMoveStarted)
        self.connect(self.selection, SIGNAL("selectionRegionMoved(int, QPointF, QPainterPath)"), self.onMove)
        
        
    def setGraph(self, graph):
        DataTool.setGraph(self, graph)
        if graph and hasattr(self, "selection"):
            self.selection.setParent(graph)

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
            del painter
            self.graph._tool_pixmap = pixmap
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
        
    def onMoveStarted(self, index, pos, path):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        self._moving = [(i, float(ex[attr1]), float(ex[attr2])) for i, ex in enumerate(data)]
        self._moving = [(i, x, y) for i, x, y in self._moving if path.contains(QPointF(x, y))]
        self._moving_pos = pos
        
    def onMove(self, index, pos, path):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        
        diff = pos - self._moving_pos
        for i, x, y in self._moving:
            ex = data[i]
            ex[attr1] = x + diff.x()
            ex[attr2] = y + diff.y()
        self.graph.updateGraph()
        
    def deleteSelected(self, *args):
        data = self.graph.data
        attr1, attr2 = self.graph.attr1, self.graph.attr2
        path = self.selection.path()
        selected = [i for i, ex in enumerate(data) if path.contains(QPointF(float(ex[attr1]) , float(ex[attr2])))]
        for i in reversed(selected):
            del data[i]
        self.graph.updateGraph() 
        
class GraphLassoSelections(GraphSelections):
    def start(self, event):
        pos = self.getPos(event)
        index = self.regionAt(event)
        if index == -1:
            self.clearSelection()
            self.addSelectionRegion([pos])
        else:
            self._moving_index, self._moving_pos, self._selection_region = index, pos, self.selection[index]
            self.emit(SIGNAL("selectionRegionMoveStarted(int, QPointF, QPainterPath)"), index, pos, self.toPath(self.selection[index]))  
        self.emit(SIGNAL("selectionGeometryChanged()"))
        
    def update(self, event):
        pos = self.getPos(event)
        index = self._moving_index
        if index == -1:
            self.selection[-1].append(pos)
            self.emit(SIGNAL("selectionRegionUpdated(int, QPainterPath)"), len(self.selection) - 1 , self.toPath(self.selection[-1]))
        else:
            diff = self._moving_pos - pos
            self.selection[index] = [p - diff for p in self._selection_region]
            self.emit(SIGNAL("selectionRegionMoved(int, QPointF, QPainterPath)"), index, pos, self.toPath(self.selection[index]))
            
        self.emit(SIGNAL("selectionGeometryChanged()"))
        
    def end(self, event):
        self.update(event)
        if self._moving_index != -1:
            self.emit(SIGNAL("selectionRegionMoveEnd(int)"), self._moving_index)
        self._moving_index = -1
        
class LassoTool(SelectTool):
    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.selection = GraphLassoSelections(graph)
        self.pen = QPen(Qt.black, 1, Qt.DashDotLine)
        self.pen.setCosmetic(True)
        self.pen.setJoinStyle(Qt.RoundJoin)
        self.pen.setCapStyle(Qt.RoundCap)
        self.connect(self.selection, SIGNAL("selectionRegionMoveStarted(int, QPointF, QPainterPath)"), self.onMoveStarted)
        self.connect(self.selection, SIGNAL("selectionRegionMoved(int, QPointF, QPainterPath)"), self.onMove)
    
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
    
    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            coord = self.invTransform(event.pos())
            val1, val2 = coord.x(), coord.y()
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, val1, attr2, val2)
        return True
        
    def dataTransform(self, attr1, val1, attr2, val2):
        example = orange.Example(self.graph.data.domain)
        example[attr1] = val1
        example[attr2] = val2
        example.setclass(self.graph.data.domain.classVar(self.graph.data.domain.classVar.baseValue))
        self.graph.data.append(example)
        self.graph.updateGraph(dataInterval=(-1, sys.maxint))
        
class BrushTool(DataTool):
    brushRadius = 20
    density = 5
    
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QFormLayout()
            self.radiusSlider = QSlider(Qt.Horizontal)
            self.radiusSlider.pyqtConfigure(minimum=10, maximum=30, value=self.tool.brushRadius)
            self.densitySlider = QSlider(Qt.Horizontal)
            self.densitySlider.pyqtConfigure(minimum=3, maximum=10, value=self.tool.density)
            
            layout.addRow("Radius", self.radiusSlider)
            layout.addRow("Density", self.densitySlider)
            self.setLayout(layout)
            
            self.connect(self.radiusSlider, SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "brushRadius", value))
            
            self.connect(self.densitySlider, SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "density", value))
    
    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.brushState = -20, -20, 0, 0
    
    def mousePressEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
        self.graph.replot()
        return True
        
    def mouseMoveEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
        self.graph.replot()
        return True
    
    def mouseReleaseEvent(self, event):
        self.graph.replot()
        return True
    
    def leaveEvent(self, event):
        self.graph._tool_pixmap = None
        self.graph.replot()
        return False
        
    def paintEvent(self, event):
        pixmap = QPixmap(self.graph.canvas().size())
        pixmap.fill(QColor(255, 255, 255, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            painter.setPen(QPen(Qt.black, 1))
            x, y, w, h = self.brushState
            painter.drawEllipse(QPoint(x, y), w, h)
        except Exception, ex:
            print ex
        del painter
        self.graph._tool_pixmap = pixmap
        return False
        
    def brushGeometry(self, point):
        coord = self.invTransform(point)
        dcoord = self.invTransform(QPoint(point.x() + self.brushRadius, point.y() + self.brushRadius))
        x, y = coord.x(), coord.y()
        rx, ry = dcoord.x() - x, -(dcoord.y() - y)
        return x, y, rx, ry
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        import random
        new = []
        for i in range(self.density):
            ex = orange.Example(self.graph.data.domain)
            ex[attr1] = random.normalvariate(x, rx)
            ex[attr2] = random.normalvariate(y, ry)
            ex.setclass(self.graph.data.domain.classVar(self.graph.data.domain.classVar.baseValue))
            new.append(ex)
        self.graph.data.extend(new)
        self.graph.updateGraph(dataInterval=(-len(new), sys.maxint))
    
class MagnetTool(BrushTool):
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            distsq = (x1 - x)**2 + (y1 - y)**2
            dist = math.sqrt(distsq)
            attraction = self.density / 100.0
            advance = 0.005
            dx = -(x1 - x)/dist * attraction / max(distsq, rx) * advance
            dy = -(y1 - y)/dist * attraction / max(distsq, ry) * advance
            ex[attr1] = x1 + dx
            ex[attr2] = y1 + dy
        self.graph.updateGraph()
    
class JitterTool(BrushTool):
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        import random
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            distsq = (x1 - x)**2 + (y1 - y)**2
            dist = math.sqrt(distsq)
            attraction = self.density / 100.0
            advance = 0.005
            dx = -(x1 - x)/dist * attraction / max(distsq, rx) * advance
            dy = -(y1 - y)/dist * attraction / max(distsq, ry) * advance
            ex[attr1] = x1 - random.normalvariate(0, dx) #*self.density)
            ex[attr2] = y1 - random.normalvariate(0, dy) #*self.density)
        self.graph.updateGraph()
        
class EnumVariableModel(PyListModel):
    def __init__(self, var, parent=None, **kwargs):
        PyListModel.__init__(self, [], parent, **kwargs)
        self.wrap(var.values)
        self.colorPalette = OWColorPalette.ColorPaletteHSV(len(self))
        self.connect(self, SIGNAL("columnsInserted(QModelIndex, int, int)"), self.updateColors)
        self.connect(self, SIGNAL("columnsRemoved(QModelIndex, int, int)"), self.updateColors)

    def __delitem__(self, index):
        raise TypeErorr("Cannot delete EnumVariable value")
    
    def __delslice__(self, i, j):
        raise TypeErorr("Cannot delete EnumVariable values")
    
    def __setitem__(self, index, item):
        self._list[index] = str(item)
        
    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole:
            i = index.row()
            return QVariant(self.itemQIcon(i))
        else:
            return PyListModel.data(self, index, role)
        
    def updateColors(self, index, start, end):
        self.colorPalette = OWColorPalette.ColorPaletteHSV(len(self))
        self.emit(SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.index(0), self.index(len(self) - 1))
        
    def itemQIcon(self, i):
        pixmap = QPixmap(64, 64)
        pixmap.fill(QColor(255, 255, 255, 0))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(self.colorPalette[i])
        painter.drawEllipse(QRectF(15, 15, 39, 39))
        painter.end()
        return QIcon(pixmap)
   
class OWDataGenerator(OWWidget):
    TOOLS = [("Brush", "Create multiple instances", BrushTool,  icon_brush),
             ("Put", "Put individual instances", PutInstanceTool, icon_put),
             ("Select", "Select and move instances", SelectTool, icon_select),
             ("Lasso", "Select and move instances", LassoTool, icon_lasso),
             ("Jitter", "Jitter instances", JitterTool, icon_jitter),
             ("Magnet", "Move (drag) multiple instances", MagnetTool, icon_magnet),
             ("Zoom", "Zoom", ZoomTool, OWToolbars.dlg_zoom) #"GenerateDataZoomTool.png")
             ]
    def __init__(self, parent=None, signalManager=None, name="Data Generator"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        
        self.outputs = [("Example Table", ExampleTable)]
        
        self.addClassAsMeta = False
        self.attributes = []
        self.cov = []
        
        self.loadSettings()
        
#        box = OWGUI.widgetBox(self.controlArea, "Domain")
#        self.varListView = QListView(self)
#        self.varListView.setItemDelegate(VariableDelegate(self))
        self.variablesModel = VariableListModel([orange.FloatVariable(name) for name in ["X", "Y"]], self, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
#        self.varListView.setModel(self.variablesModel)
#        box.layout().addWidget(self.varListView)
#        
#        self.connect(self.variablesModel, SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.onDomainChanged)
        
#        self.addDomainFeatureAction = QAction("+", self)
#        self.removeDomainFeatureAction = QAction("-", self)
#        self.connect(self.addDomainFeatureAction, SIGNAL("triggered()"), self.onAddFeature)
#        self.connect(self.removeDomainFeatureAction, SIGNAL("triggered()"), self.onRemoveFeature)
        
#        self.connect(self.varListView.selectionModel(), SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.onVariableSelectionChange)
        
#        hlayout = QHBoxLayout()
#        button1 = QToolButton()
#        button1.setDefaultAction(self.addDomainFeatureAction)
#        button2 = QToolButton()
#        button2.setDefaultAction(self.removeDomainFeatureAction)
#        
#        hlayout.addWidget(button1)
##        hlayout.addWidget(button2)
#        hlayout.addStretch(10)
#        box.layout().addLayout(hlayout)
        
        self.classVariable = orange.EnumVariable("Class label", values=["Class 1", "Class 2"], baseValue=0)
        
#        cb = OWGUI.comboBox(self.controlArea, self, "classIndex", "Class Label")
        w = OWGUI.widgetBox(self.controlArea, "Class Label")
        
        self.classValuesView = listView = QListView()
        listView.setSelectionMode(QListView.SingleSelection)
        
        self.classValuesModel = EnumVariableModel(self.classVariable, self, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled| Qt.ItemIsEditable)
        self.classValuesModel.wrap(self.classVariable.values)
        
        listView.setModel(self.classValuesModel)
        listView.selectionModel().select(self.classValuesModel.index(0), QItemSelectionModel.ClearAndSelect)
        self.connect(listView.selectionModel(), SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.onClassLabelSelection)
        listView.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        w.layout().addWidget(listView)
        
        addClassLabel = QAction("+", self)
        addClassLabel.pyqtConfigure(toolTip="Add class label")#, icon=QIcon(icon_put))
        self.connect(addClassLabel, SIGNAL("triggered()"), self.addNewClassLabel)
        
        removeClassLabel = QAction("-", self)
        removeClassLabel.pyqtConfigure(toolTip="Remove class label")#, icon=QIcon(icon_remove))
        self.connect(removeClassLabel, SIGNAL("triggered()"), self.removeSelectedClassLabel)
        
        actionsWidget =  ModelActionsWidget([addClassLabel, removeClassLabel], self)
        actionsWidget.layout().addStretch(10)
#        actionsWidget.setStyleSheet("QToolButton { qproperty-iconSize: 12px }")
        
        w.layout().addWidget(actionsWidget)
#        bc.setModel(self.classValuesModel)
#        self.connect(cb, SIGNAL("currentIndexChanged(int)"), lambda base: setattr(self.classVariable, "baseValue", base))
#        cb.box.layout().addSpacing(4)
#        OWGUI.button(cb.box, self, "Add new class label", callback=self.onAddClassLabel)
        
        toolbox = OWGUI.widgetBox(self.controlArea, "Tools", orientation=QGridLayout())
        self.toolActions = QActionGroup(self)
        self.toolActions.setExclusive(True)
        for i, (name, tooltip, tool, icon) in enumerate(self.TOOLS):
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.setCheckable(True)
            if os.path.exists(icon):
                action.setIcon(QIcon(icon))
            self.connect(action, SIGNAL("triggered()"), lambda tool=tool: self.onToolAction(tool))
            button = QToolButton()
            button.setDefaultAction(action)
            button.setIconSize(QSize(24, 24))
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            toolbox.layout().addWidget(button, i / 3, i % 3)
            self.toolActions.addAction(action)
            
        for column in range(3):
            toolbox.layout().setColumnMinimumWidth(column, 10)
            toolbox.layout().setColumnStretch(column, 1)
            
        self.optionsLayout = QStackedLayout()
        self.toolsStackCache = {}
        optionsbox = OWGUI.widgetBox(self.controlArea, "Options", orientation=self.optionsLayout)
        
#        OWGUI.checkBox(self.controlArea, self, "addClassAsMeta", "Add class ids as meta attributes")
        
        OWGUI.rubber(self.controlArea)
        OWGUI.button(self.controlArea, self, "Commit", callback=self.commit)
        
        self.graph = DataGeneratorGraph(self)
        self.graph.setAxisScale(QwtPlot.xBottom, 0.0, 1.0)
        self.graph.setAxisScale(QwtPlot.yLeft, 0.0, 1.0)
        self.graph.setAttribute(Qt.WA_Hover, True)
        self.mainArea.layout().addWidget(self.graph)
        
        self.currentOptionsWidget = None
        self.data = []
        self.domain = None
        
        self.onDomainChanged()
        self.toolActions.actions()[0].trigger()
        
        self.resize(800, 600)
        
    def addNewClassLabel(self):
        i = 1
        while True:
            newlabel = "Class %i" %i
            if newlabel not in self.classValuesModel:
#                self.classValuesModel.append(newlabel)
                break
            i += 1
        values = list(self.classValuesModel) + [newlabel]
        newclass = orange.EnumVariable("Class label", values=values)
        newdomain = orange.Domain(self.graph.data.domain.attributes, newclass)
        newdata = orange.ExampleTable(newdomain)
        for ex in self.graph.data:
            newdata.append(orange.Example(newdomain, [ex[a] for a in ex.domain.attributes] + [str(ex.getclass())]))
        
        self.classVariable = newclass
        self.classValuesModel.wrap(self.classVariable.values)
        
        self.graph.data = newdata
        self.graph.updateGraph()
        
        newindex = self.classValuesModel.index(len(self.classValuesModel) - 1)
        self.classValuesView.selectionModel().select(newindex, QItemSelectionModel.ClearAndSelect)
        
    def removeSelectedClassLabel(self):
        index = self.selectedClassLabelIndex()
        if index is not None and len(self.classValuesModel) > 1:
            label = self.classValuesModel[index]
            examples = [ex for ex in self.graph.data if str(ex.getclass()) != label]
            
            values = [val for val in self.classValuesModel if val != label]
            newclass = orange.EnumVariable("Class label", values=values)
            newdomain = orange.Domain(self.graph.data.domain.attributes, newclass)
            newdata = orange.ExampleTable(newdomain)
            for ex in examples:
                if ex[self.classVariable] != label and ex[self.classVariable] in values:
                    newdata.append(orange.Example(newdomain, [ex[a] for a in ex.domain.attributes] + [str(ex.getclass())]))
                
            self.classVariable = newclass
            self.classValuesModel.wrap(self.classVariable.values)
            
            self.graph.data = newdata
            self.graph.updateGraph()
            
            newindex = self.classValuesModel.index(max(0, index - 1))
            self.classValuesView.selectionModel().select(newindex, QItemSelectionModel.ClearAndSelect)
        
    def selectedClassLabelIndex(self):
        rows = [i.row() for i in self.classValuesView.selectionModel().selectedRows()]
        if rows:
            return rows[0]
        else:
            return None
        
    def onClassLabelSelection(self, selected, unselected):
        index = self.selectedClassLabelIndex()
        if index is not None:
            self.classVariable.baseValue = index
        
    def onAddFeature(self):
        self.variablesModel.append(orange.FloatVariable("New feature"))
        self.varListView.edit(self.variablesModel.index(len(self.variablesModel) - 1))
        
    def onRemoveFeature(self):
        raise NotImplemented
  
            
    def onVariableSelectionChange(self, selected, deselected):
        self.selected
    
    def onToolAction(self, tool):
        self.setCurrentTool(tool)
        
    def setCurrentTool(self, tool):
        if tool not in self.toolsStackCache:
            newtool = tool(None, self)
            option = newtool.optionsWidget(newtool, self)
            self.optionsLayout.addWidget(option)
            self.connect(newtool, SIGNAL("dataChanged()"), self.graph.updateGraph)
            self.toolsStackCache[tool] = (newtool, option)
        
        self.currentTool, self.currentOptionsWidget = tool, option = self.toolsStackCache[tool]
        self.optionsLayout.setCurrentWidget(option)
        self.currentTool.setGraph(self.graph)
        
    def onDomainChanged(self, *args):
        if self.variablesModel:
            self.domain = orange.Domain(list(self.variablesModel), self.classVariable)
            if self.data:
                self.data = orange.ExampleTable(self.domain, self.data)
            else:
                self.data = orange.ExampleTable(self.domain)
            self.graph.setData(self.data, 0, 1)
    
    def commit(self):
        if self.addClassAsMeta:
            domain = orange.Domain(self.graph.data.domain.attributes, None)
            domain.addmeta(orange.newmetaid(), self.graph.data.domain.classVar)
            data = orange.ExampleTable(domain, self.graph.data)
        else:
            data = self.graph.data
        self.send("Example Table", self.graph.data)
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWDataGenerator()
    w.show()
    app.exec_()
        
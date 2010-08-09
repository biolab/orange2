"""<name>Data generator</name>
"""

import orange

from OWWidget import *
from OWGraph import *

from OWItemModels import VariableListModel, VariableDelegate, PyListModel

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
#        data, c, w = data.toNumpy()
        for i, cls in enumerate(clsValues):
            x = [float(ex[self.attr1]) for ex in data if ex.getclass() == cls]
            y = [float(ex[self.attr2]) for ex in data if ex.getclass() == cls]
            self.addCurve("data points", xData=x, yData=y, brushColor=palette[i], penColor=palette[i])
        self.replot()
        
class DataTool(QObject):
    """ A base class for data tools that operate on OWGraph
    widgets by installing itself as its event filter.
    """
    clusterIndex = 0
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
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            
        def setClusterListModel(self, model):
            self.cb.setModel(model)
            
        def onClusterChanged(self, index):
            self.tool.clusterIndex = index
            
    
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
        self.graph.updateGraph()
        
class BrushTool(DataTool):
    brushRadius = 20
    intensity = 5
    
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QFormLayout()
            self.radiusSlider = QSlider(Qt.Horizontal)
            self.radiusSlider.pyqtConfigure(minimum=10, maximum=30, value=self.tool.brushRadius)
            self.intensitySlider = QSlider(Qt.Horizontal)
            self.intensitySlider.pyqtConfigure(minimum=3, maximum=10, value=self.tool.intensity)
            
            layout.addRow("Radius", self.radiusSlider)
            layout.addRow("Intensity", self.intensitySlider)
            self.setLayout(layout)
            
            self.connect(self.radiusSlider, SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "brushRadius", value))
            
            self.connect(self.intensitySlider, SIGNAL("valueChanged(int)"),
                         lambda value: setattr(self.tool, "intensity", value))
        
    
    def __init__(self, graph, parent=None):
        DataTool.__init__(self, graph, parent)
        self.brushState = -20, -20, 0, 0
    
    def mousePressEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
        return True
        
    def mouseMoveEvent(self, event):
        self.brushState = event.pos().x(), event.pos().y(), self.brushRadius, self.brushRadius
        x, y, rx, ry = self.brushGeometry(event.pos())
        if event.buttons() & Qt.LeftButton:
            attr1, attr2 = self.attributes()
            self.dataTransform(attr1, x, rx, attr2, y, ry)
        self.graph.canvas().update()
        return True
    
    def mouseReleseEvent(self, event):
        return True
    
    def paintEvent(self, event):
        self.graph.canvas().paintEvent(event)
        painter = QPainter(self.graph.canvas())
        painter.setRenderHint(QPainter.Antialiasing)
        try:
            painter.setPen(QPen(Qt.black, 1))
            x, y, w, h = self.brushState
            painter.drawEllipse(QPoint(x, y), w, h)
        except Exception, ex:
            print ex
        del painter
        return True
        
    def brushGeometry(self, point):
        coord = self.invTransform(point)
        dcoord = self.invTransform(QPoint(point.x() + self.brushRadius, point.y() + self.brushRadius))
        x, y = coord.x(), coord.y()
        rx, ry = dcoord.x() - x, -(dcoord.y() - y)
        return x, y, rx, ry
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        import random
        new = []
        for i in range(self.intensity):
            ex = orange.Example(self.graph.data.domain)
            ex[attr1] = random.normalvariate(x, rx)
            ex[attr2] = random.normalvariate(y, ry)
            ex.setclass(self.graph.data.domain.classVar(self.graph.data.domain.classVar.baseValue))
            new.append(ex)
        self.graph.data.extend(new)
        self.graph.updateGraph(dataInterval=(-len(new), -1))
        
        
    
class AddClusterTool(DataTool):
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            layout = QVBoxLayout()
            self.listView = QListView()
            layout.addWidget(self.listView)
            
    
    def __init__(self, graph, parent=None):
        DataTool.__init__(graph, parent)
        
    def mousePressEvent(self, event):
        coords = self.invTransform(event.pos())
        covMatrix = self.getCov()
        return False
    
    def paintEvent(self, event):
        self.graph.canvas().paintEvent(event)
        for cluster in self.clusters:
            name, cov = cluster
            x, y = numpy.array([1.0, 0.0]), numpy.array([0.0, 1.0])
            x = numpy.multiply(x, cov)
            y = numpy.multiply(y, cov)
            
    
class MagnetTool(BrushTool):
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        import random
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            dist = ((x1 - x)/rx)**2 + ((y1 - y)/ry)**2
            alpha =  lambda d: min(1.0 / d, 1)
            dx = numpy.sign(x - x1) * alpha(dist)
            dy = numpy.sign(y - y1) * alpha(dist)
            ex[attr1] = x1 + dx * self.intensity
            ex[attr2] = y1 + dy * self.intensity
        self.graph.updateGraph()
        
class MultiplyTool(DataTool):
    multiplier = 1.0
    class optionsWidget(QFrame):
        def __init__(self, tool, parent=None):
            QFrame.__init__(self, parent)
            self.tool = tool
            
    
class JitterTool(BrushTool):
    
    def dataTransform(self, attr1, x, rx, attr2, y, ry):
        import random
        for ex in self.graph.data:
            x1, y1 = float(ex[attr1]), float(ex[attr2])
            dist = ((x1 - x)/rx)**2 + ((y1 - y)/ry)**2
            alpha =  lambda d: min(1.0 / d, 1)
            dx = numpy.sign(x - x1) * alpha(dist)
            dy = numpy.sign(y - y1) * alpha(dist) 
            ex[attr1] = x1 - random.normalvariate(0, dx*self.intensity)
            ex[attr2] = y1 - random.normalvariate(0, dy*self.intensity)
        self.graph.updateGraph()
        
class OWDataGenerator(OWWidget):
    TOOLS = [("Zoom", "Zoom", ZoomTool),
             ("Put", "Put individual instances", PutInstanceTool),
             ("Brush", "Create multiple instances", BrushTool),
#             ("Add cluster", "Create a new cluster", AddClusterTool),
             ("Magnet", "Move (drag) multiple instances", MagnetTool),
             ("Jitter", "Jitter instances", JitterTool),
#             ("Multiply","Multiply instances", MultiplyTool)
             ]
    def __init__(self, parent=None, signalManager=None, name="Data Generator"):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        
        self.outputs = [("Example Table", ExampleTable)]
        
        self.addClustersAsMeta = True
        self.attributes = []
        self.cov = []
        
        self.loadSettings()
        
        box = OWGUI.widgetBox(self.controlArea, "Domain")
        self.varListView = QListView(self)
        self.varListView.setItemDelegate(VariableDelegate(self))
        self.variablesModel = VariableListModel([orange.FloatVariable(name) for name in ["X", "Y"]], self, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
        self.varListView.setModel(self.variablesModel)
        box.layout().addWidget(self.varListView)
        
        self.connect(self.variablesModel, SIGNAL("dataChanged(QModelIndex, QModelIndex)"), self.onDomainChanged)
        
        self.addDomainFeatureAction = QAction("+", self)
        self.removeDomainFeatureAction = QAction("-", self)
        self.connect(self.addDomainFeatureAction, SIGNAL("triggered()"), self.onAddFeature)
        self.connect(self.removeDomainFeatureAction, SIGNAL("triggered()"), self.onRemoveFeature)
        
        hlayout = QHBoxLayout()
        button1 = QToolButton()
        button1.setDefaultAction(self.addDomainFeatureAction)
        button2 = QToolButton()
        button2.setDefaultAction(self.removeDomainFeatureAction)
        
        hlayout.addWidget(button1)
#        hlayout.addWidget(button2)
        hlayout.addStretch(10)
        box.layout().addLayout(hlayout)
        
        self.clusterVariable = orange.EnumVariable("Cluster", values=["Cluster 1", "Cluster 2"], baseValue=0)
        
        cb = OWGUI.comboBox(self.controlArea, self, "clusterIndex", "Cluster")
        self.clusterValuesModel = PyListModel([], self)
        self.clusterValuesModel.wrap(self.clusterVariable.values)
        cb.setModel(self.clusterValuesModel)
        self.connect(cb, SIGNAL("currentIndexChanged(int)"), lambda base: setattr(self.clusterVariable, "baseValue", base))
        
        
        toolbox = OWGUI.widgetBox(self.controlArea, "Tools", orientation=QGridLayout())
        self.toolActions = QActionGroup(self)
        self.toolActions.setExclusive(True)
        for i, (name, tooltip, tool) in enumerate(self.TOOLS):
            action = QAction(name, self)
            action.setToolTip(tooltip)
            action.setCheckable(True)
            self.connect(action, SIGNAL("triggered()"), lambda tool=tool: self.onToolAction(tool))
            button = QToolButton()
            button.setDefaultAction(action)
            toolbox.layout().addWidget(button, i / 2, i % 2)
            self.toolActions.addAction(action)
            
        self.optionsLayout = QStackedLayout()
        self.toolsStackCache = {}
        optionsbox = OWGUI.widgetBox(self.controlArea, "Options", orientation=self.optionsLayout)
        
        OWGUI.checkBox(self.controlArea, self, "addClustersAsMeta", "Add cluster ids as meta attributes")
        OWGUI.button(self.controlArea, self, "Commit", callback=self.commit)
        
        self.graph = DataGeneratorGraph(self)
        self.mainArea.layout().addWidget(self.graph)
        
        self.currentOptionsWidget = None
        self.data = []
        self.domain = None
        
        self.onDomainChanged()
#        self.setCurrentTool(PutInstanceTool)
        self.toolActions.actions()[0].trigger()
        
        self.resize(800, 600)
        
    def onAddFeature(self):
        self.variablesModel.append(orange.FloatVariable("New feature"))
        self.varListView.edit(self.variablesModel.index(len(self.variablesModel) - 1))
        
    def onRemoveFeature(self):
        pass
    
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
            self.domain = orange.Domain(list(self.variablesModel), self.clusterVariable)
            if self.data:
                self.data = orange.ExampleTable(self.domain, self.data)
            else:
                self.data = orange.ExampleTable(self.domain)
            print list(self.data)
            self.graph.setData(self.data, 0, 1)
        
    def commit(self):
        if self.addClustersAsMeta:
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
        
        
        
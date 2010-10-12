import orange, orngTree, OWGUI, OWColorPalette
from OWWidget import *

from PyQt4.QtCore import *
from PyQt4.QtGui import *

DefDroppletRadiust=7
DefNodeWidth=30
DefNodeHeight=20
DefDroppletBrush=QBrush(Qt.darkGray)

class graph_node(object):
    def __init__(self, *args, **kwargs):
        self.edges = kwargs.get("edges", set())
        
    def graph_edges(self):
        return self.edges
    
    def graph_add_edge(self, edge):
        self.edges.add(edge)
        
    def __iter__(self):
        for edge in self.edges:
            yield edge.node2
            
    def graph_nodes(self, type=1):
        pass
            
class graph_edge(object):
        
    def __init__(self, node1=None, node2=None, type=1):
        self.node1 = node1
        self.node2 = node2
        self.type = type
        node1.graph_add_edge(self)
        node2.graph_add_edge(self)
        
class GraphicsDroplet(QGraphicsEllipseItem):
    
    def __init__(self, *args):
        QGraphicsEllipseItem.__init__(self, *args)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setBrush(QBrush(Qt.gray))
        
    def hoverEnterEvent(self, event):
        QGraphicsEllipseItem.hoverEnterEvent(self, event)
        self.setBrush(QBrush(QColor(100, 100, 100)))
        self.update()
        
    def hoverLeaveEvent(self, event):
        QGraphicsEllipseItem.hoverLeaveEvent(self, event)
        self.setBrush(QBrush(QColor(200, 200, 200)))
        self.update()
        
    def mousePressEvent(self, event):
        QGraphicsEllipseItem.mousePressEvent(self, event)
        self.parentItem().setOpen(not self.parentItem().isOpen)
        if self.scene():
            self.scene().fixPos()
        
class TextTreeNode(QGraphicsTextItem, graph_node):
    
    borderRadius = pyqtProperty("int",
                lambda self: getattr(self, "_borderRadius", 0),
                lambda self, val: (setattr(self, "_borderRadius", val), self.update()) and None,
                doc="Rounded rect's border radius"
                )
    backgroundBrush = pyqtProperty("QBrush",
                lambda self: getattr(self, "_backgroundBrush", getattr(self.scene(), "defaultItemBrush", Qt.NoBrush)),
                lambda self, brush: (setattr(self, "_backgroundBrush", brush), self.update()) and None,
                doc="Background brush"
                )
    truncateText = pyqtProperty("bool",
                lambda self: getattr(self, "_truncateText", False),
                lambda self, val: (setattr(self, "_truncateText", val), self.updateContents()),
                doc="Truncate text")
    
    def __init__(self, tree, parent, *args, **kwargs):
        QGraphicsTextItem.__init__(self, *args)
        graph_node.__init__(self, **kwargs)
        self.tree = tree
        self.parent = parent
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.droplet = GraphicsDroplet(-5, 0, 10, 10, self, self.scene())
        
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        
        self.connect(self.document(), SIGNAL("contentsChanged()"), self.updateContents)
        self.isOpen = True
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        
    def setHtml(self, html):
        if qVersion() < "4.5":
            html = html.replace("<hr>", "<hr width=200>") #bug in Qt4.4 (need width = 200)
        return QGraphicsTextItem.setHtml(self, "<body>" + html + "</body>") 
    
    def updateContents(self):
        if getattr(self, "_rect", QRectF()).isValid() and not self.truncateText:
            self.setTextWidth(self._rect.width())
        else:
            self.setTextWidth(-1)
            self.setTextWidth(self.document().idealWidth())
        self.droplet.setPos(self.rect().center().x(), self.rect().height())
        self.droplet.setVisible(bool(self.branches))
        
    def setRect(self, rect):
        self.prepareGeometryChange()
        rect = QRectF() if rect is None else rect
        self._rect = rect
        self.updateContents()
        self.update()
        
    def shape(self):
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path
    
    def rect(self):
        if self.truncateText and getattr(self, "_rect", QRectF()).isValid():
            return self._rect
        else:
            return QRectF(QPointF(0,0), self.document().size()) | getattr(self, "_rect", QRectF(0, 0, 1, 1))
        
    def boundingRect(self):
        if self.truncateText and getattr(self, "_rect", QRectF()).isValid():
            return self._rect
        else:
            return QGraphicsTextItem.boundingRect(self)
    
    @property  
    def branches(self):
        return [edge.node2 for edge in self.graph_edges() if edge.node1 is self]
    
    def paint(self, painter, option, widget=0):
        painter.save()
        painter.setBrush(self.backgroundBrush)
        rect = self.rect()
        painter.drawRoundedRect(rect, self.borderRadius, self.borderRadius)
        painter.restore()
        painter.setClipRect(rect)
        return QGraphicsTextItem.paint(self, painter, option, widget)
        
def graph_traverse_bf(nodes, level=None, test=None):
    visited = set()
    queue = list(nodes)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            yield node
            visited.add(node)
            if not test or test(node):
                queue.extend(list(node))
                
class GraphicsNode(TextTreeNode):
    def setOpen(self, open, level=1):
        self.isOpen = open
        for node in graph_traverse_bf(self, test=lambda node: node.isOpen):
            if node is not self:
                node.setVisible(open)
               
    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.updateEdge()
        elif change == QGraphicsItem.ItemVisibleHasChanged:
            self.updateEdge()
            
        return TextTreeNode.itemChange(self, change, value)
    
    def updateEdge(self):
        for edge in self.edges:
            if edge.node1 is self:
                QTimer.singleShot(0, edge.updateEnds)
            elif edge.node2 is self:
                edge.setVisible(self.isVisible())
                
    def edgeInPoint(self, edge):
        return edge.mapFromItem(self, QPointF(self.rect().center().x(), self.rect().y()))

    def edgeOutPoint(self, edge):
        return edge.mapFromItem(self.droplet, self.droplet.rect().center())
    
    def paint(self, painter, option, widget=0):
        if self.isSelected():
            option.state = option.state.__xor__(QStyle.State_Selected)
        if self.isSelected():
            rect = self.rect()
            painter.save()
#            painter.setBrush(QBrush(QColor(100, 0, 255, 100)))
            painter.setBrush(QBrush(QColor(125, 162, 206, 192)))
            painter.drawRoundedRect(rect.adjusted(-4, -4, 4, 4), self.borderRadius, self.borderRadius)
            painter.restore()
        TextTreeNode.paint(self, painter, option, widget)
        
    def boundingRect(self):
        return TextTreeNode.boundingRect(self).adjusted(-5, -5, 5, 5)
            
    def mousePressEvent(self, event):
        return TextTreeNode.mousePressEvent(self, event)
    
class GraphicsEdge(QGraphicsLineItem, graph_edge):
    def __init__(self, *args, **kwargs):
        QGraphicsLineItem.__init__(self, *args)
        graph_edge.__init__(self, **kwargs)
        self.setZValue(-30)
        
    def updateEnds(self):
        try:
            self.prepareGeometryChange()
            self.setLine(QLineF(self.node1.edgeOutPoint(self), self.node2.edgeInPoint(self)))
        except RuntimeError: # this gets called through QTimer.singleShot and might already be deleted by Qt 
            pass 

class TreeGraphicsView(QGraphicsView):
    def __init__(self, master, scene, *args):
        QGraphicsView.__init__(self, scene, *args)
#        try:
#            import PyQt4.QtOpenGL as gl
#            fmt = gl.QGLFormat()
#            fmt.setSampleBuffers(True)
#            fmt.setSamples(32)
#            print fmt.sampleBuffers()
#            self.setViewport(gl.QGLWidget(fmt, self))
#        except Exception, ex:
#            print ex
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.TextAntialiasing)
        self.setRenderHint(QPainter.HighQualityAntialiasing)

    def resizeEvent(self, event):
        QGraphicsView.resizeEvent(self, event)
        self.emit(SIGNAL("resized(QSize)"), self.size())

class TreeGraphicsScene(QGraphicsScene):
    def __init__(self, master, *args):
        QGraphicsScene.__init__(self, *args)
        self.HSpacing=10
        self.VSpacing=10
        self.master=master
        self.nodeList=[]
        self.edgeList=[]

    def fixPos(self, node=None, x=10, y=10):
        self.gx=x
        self.gy=y
        if not node:
            if self.nodes():
                node = [node for node in self.nodes() if not node.parent][0]
            else:
                return
        if not x or not y: x, y= self.HSpacing, self.VSpacing
        self._fixPos(node,x,y)
        
        self.setSceneRect(reduce(QRectF.united, [item.sceneBoundingRect() for item in self.items() if item.isVisible()], QRectF(0, 0, 10, 10)).adjusted(0, 0, 100, 100))
#        print self.sceneRect()
        self.update()
        
    def _fixPos(self, node, x, y):
        ox=x
        
        def bRect(node):
            return node.boundingRect() | node.childrenBoundingRect()
        if node.branches and node.isOpen:
            for n in node.branches:
                (x,ry)=self._fixPos(n,x,y+self.VSpacing + bRect(node).height())
            x=(node.branches[0].pos().x() + node.branches[-1].pos().x())/2
#            print x,y
            node.setPos(x,y)
            for e in node.edges:
                e.updateEnds()
        else:
#            print self.gx, y
            node.setPos(self.gx,y)
            self.gx+=self.HSpacing + bRect(node).width()
            x+=self.HSpacing + bRect(node).width()
            self.gy=max([y,self.gy])

        return (x,y)

    def mouseMoveEvent(self,event):
        return QGraphicsScene.mouseMoveEvent(self, event)

    def mousePressEvent(self, event):
        return QGraphicsScene.mousePressEvent(self, event)
    
    def edges(self):
        return [item for item in self.items() if isinstance(item, graph_edge)]
    
    def nodes(self):
        return [item for item in self.items() if isinstance(item, graph_node)]     

class TreeNavigator(QGraphicsView):

    def __init__(self, masterView, *args):
        QGraphicsView.__init__(self)
        self.masterView = masterView
        self.setScene(self.masterView.scene())
        self.connect(self.scene(), SIGNAL("sceneRectChanged(QRectF)"), self.updateSceneRect)
        self.connect(self.masterView, SIGNAL("resized(QSize)"), self.updateView)
        self.setRenderHint(QPainter.Antialiasing)

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.masterView.centerOn(self.mapToScene(event.pos()))
            self.updateView()
        return QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.masterView.centerOn(self.mapToScene(event.pos()))
            self.updateView()
        return QGraphicsView.mouseMoveEvent(self, event)

    def resizeEvent(self, event):
        QGraphicsView.resizeEvent(self, event)
        self.updateView()
#
    def resizeView(self):
        self.updateView()

    def updateSceneRect(self, rect):
        QGraphicsView.updateSceneRect(self, rect)
        self.updateView()
        
    def updateView(self, *args):
        if self.scene():
            self.fitInView(self.scene().sceneRect())

    def paintEvent(self, event):
        QGraphicsView.paintEvent(self, event)
        painter = QPainter(self.viewport())
        painter.setBrush(QColor(100, 100, 100, 100))
        painter.setRenderHints(self.renderHints())
        painter.drawPolygon(self.viewPolygon())
        
    def viewPolygon(self):
        return self.mapFromScene(self.masterView.mapToScene(self.masterView.viewport().rect()))


class OWTreeViewer2D(OWWidget):
    settingsList = ["ZoomAutoRefresh", "AutoArrange", "ToolTipsEnabled",
                    "Zoom", "VSpacing", "HSpacing", "MaxTreeDepth", "MaxTreeDepthB",
                    "LineWidth", "LineWidthMethod",
                    "MaxNodeWidth", "LimitNodeWidth", "NodeInfo", "NodeColorMethod",
                    "TruncateText"]

    def __init__(self, parent=None, signalManager = None, name='TreeViewer2D'):
        OWWidget.__init__(self, parent, signalManager, name, wantGraph=True)
        self.root = None
        self.selectedNode = None

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]

        #set default settings
        self.ZoomAutoRefresh = 0
        self.AutoArrange = 0
        self.ToolTipsEnabled = 1
        self.MaxTreeDepth = 5; self.MaxTreeDepthB = 0
        self.LineWidth = 5; self.LineWidthMethod = 0
        self.NodeSize = 5
        self.MaxNodeWidth = 150
        self.LimitNodeWidth = True
        self.NodeInfo = [0, 1]

        self.NodeColorMethod = 0
        self.Zoom = 5
        self.VSpacing = 5; self.HSpacing = 5
        self.TruncateText = 1
        
        self.loadSettings()
        self.NodeInfo.sort()

        # GUI definition
        self.tabs = OWGUI.tabWidget(self.controlArea)

        # GENERAL TAB
        GeneralTab = OWGUI.createTabPage(self.tabs, "General")
        TreeTab = OWGUI.createTabPage(self.tabs, "Tree")
        NodeTab = OWGUI.createTabPage(self.tabs, "Node")

        OWGUI.hSlider(GeneralTab, self, 'Zoom', box='Zoom', minValue=1, maxValue=10, step=1,
                      callback=self.toggleZoomSlider, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'VSpacing', box='Vertical spacing', minValue=1, maxValue=10, step=1,
                      callback=self.toggleVSpacing, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'HSpacing', box='Horizontal spacing', minValue=1, maxValue=10, step=1,
                      callback=self.toggleHSpacing, ticks=1)

        OWGUI.checkBox(GeneralTab, self, 'ToolTipsEnabled', 'Show node tool tips',
                       tooltip='When mouse over the node show tool tip', callback=self.updateNodeToolTips)
#        OWGUI.checkBox(GeneralTab, self, 'TruncateText', 'Truncate text to fit margins',
#                       tooltip='Truncate any text to fit the node width',
#                       callback=self.toggleTruncateText)

        self.infBox = OWGUI.widgetBox(GeneralTab, 'Tree Size', sizePolicy = QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        self.infoa = OWGUI.widgetLabel(self.infBox, 'No tree.')
        self.infob = OWGUI.widgetLabel(self.infBox, " ")

        # TREE TAB
        OWGUI.checkWithSpin(TreeTab, self, 'Max tree depth:', 1, 20, 'MaxTreeDepthB', "MaxTreeDepth",
                            tooltip='Defines the depth of the tree displayed',
                            checkCallback=self.toggleTreeDepth,
                            spinCallback=self.toggleTreeDepth)
        OWGUI.spin(TreeTab, self, 'LineWidth', min=1, max=10, step=1, label='Max line width:',
                   tooltip='Defines max width of the edges that connect tree nodes',
                   callback=self.toggleLineWidth)
        OWGUI.radioButtonsInBox(TreeTab, self,  'LineWidthMethod',
                                ['No dependency', 'Root node', 'Parent node'],
                                box='Reference for Line Width',
                                tooltips=['All edges are of the same width',
                                          'Line width is relative to number of cases in root node',
                                          'Line width is relative to number of cases in parent node'],
                                callback=self.toggleLineWidth)

        # NODE TAB
#        # Node size options

        cb, sb = OWGUI.checkWithSpin(NodeTab, self, "Max node width:", 50, 200, "LimitNodeWidth", "MaxNodeWidth",
                                     tooltip="Limit the width of tree nodes",
                                     checkCallback=self.toggleNodeSize,
                                     spinCallback=self.toggleNodeSize,
                                     step=10)
        b = OWGUI.checkBox(OWGUI.indentedBox(NodeTab), self, "TruncateText", "Truncate text", callback=self.toggleTruncateText)
        cb.disables.append(b)
        cb.makeConsistent() 
#        w = OWGUI.widgetBox(NodeTab, orientation="horizontal")
#        cb = OWGUI.checkBox(w, self, "LimitNodeWidth", "Max node width", callback=self.toggleNodeSize)
#        sl = OWGUI.hSlider(w, self, 'MaxNodeWidth', #box='Max node width',
#                      minValue=50, maxValue=200, step=10,
#                      callback=self.toggleNodeSize, ticks=50)
#        cb.disables.append(sl)
#        cb.makeConsistent()

        # Node information
        grid = QGridLayout()
        
        navButton = OWGUI.button(self.controlArea, self, "Navigator", self.toggleNavigator, debuggingEnabled = 0, addToLayout=False)
#        findbox = OWGUI.widgetBox(self.controlArea, orientation = "horizontal")
        self.centerRootButton=OWGUI.button(self.controlArea, self, "Find Root", addToLayout=False,
                                           callback=lambda :self.rootNode and \
                                           self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y()))
        self.centerNodeButton=OWGUI.button(self.controlArea, self, "Find Selected", addToLayout=False,
                                           callback=lambda :self.selectedNode and \
                                           self.sceneView.centerOn(self.selectedNode.scenePos()))
        grid.addWidget(navButton, 0, 0, 1, 2)
        grid.addWidget(self.centerRootButton, 1, 0)
        grid.addWidget(self.centerNodeButton, 1, 1)
        self.leftWidgetPart.layout().insertLayout(1, grid)
        
        self.NodeTab=NodeTab
        self.TreeTab=TreeTab
        self.GeneralTab=GeneralTab
        OWGUI.rubber(GeneralTab)
        OWGUI.rubber(TreeTab)
#        OWGUI.rubber(NodeTab)
        self.rootNode=None
        self.tree=None
        self.resize(800, 500)
        
        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveGraph)

    def sendReport(self):
        from PyQt4.QtSvg import QSvgGenerator
        self.reportSection("Tree")
        urlfn, filefn = self.getUniqueImageName(ext=".svg")
        svg = QSvgGenerator()
        svg.setFileName(filefn)
        ssize = self.scene.sceneRect().size()
        w, h = ssize.width(), ssize.height()
        fact = 600/w
        svg.setSize(QSize(600, h*fact))
        painter = QPainter()
        painter.begin(svg)
        self.scene.render(painter)
        painter.end()
        
#        buffer = QPixmap(QSize(600, h*fact))
#        painter.begin(buffer)
#        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
#        self.scene.render(painter)
#        painter.end()
#        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))
        from OWDlgs import OWChooseImageSizeDlg
        self.reportImage(OWChooseImageSizeDlg(self.scene).saveImage)
        self.reportRaw('<!--browsercode<br/>(Click <a href="%s">here</a> to view or download this image in a scalable vector format)-->' % urlfn)
        #self.reportObject(self.svg_type, urlfn, width="600", height=str(h*fact))

    def toggleZoomSlider(self):
        k = 0.0028 * (self.Zoom ** 2) + 0.2583 * self.Zoom + 1.1389
        self.sceneView.setTransform(QTransform().scale(k/2, k/2))
        self.scene.update()

    def toggleVSpacing(self):
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleHSpacing(self):
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleTreeDepth(self):
        self.walkupdate(self.rootNode)
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleLineWidth(self):
        for edge in self.scene.edges():
            if self.LineWidthMethod==0:
                width=self.LineWidth
            elif self.LineWidthMethod == 1:
                width = (edge.node2.tree.distribution.cases/self.tree.distribution.cases) * self.LineWidth
            elif self.LineWidthMethod == 2:
                width = (edge.node2.tree.distribution.cases/edge.node1.tree.distribution.cases) * self.LineWidth

            edge.setPen(QPen(Qt.gray, width, Qt.SolidLine, Qt.RoundCap))
        self.scene.update()
        
    def toggleNodeSize(self):
        pass
    
    def toggleTruncateText(self):
        for n in self.scene.nodes():
            n.truncateText = self.TruncateText
        self.scene.fixPos(self.rootNode, 10, 10)

    def toggleNavigator(self):
        self.navWidget.setHidden(not self.navWidget.isHidden())

    def activateLoadedSettings(self):
        if not self.tree:
            return
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()
        self.toggleTreeDepth()
        self.toggleLineWidth()
#        self.toggleNodeSize()

    def ctree(self, tree=None):
        self.clear()
        if not tree:
            self.centerRootButton.setDisabled(1)
            self.centerNodeButton.setDisabled(0)
            self.infoa.setText('No tree.')
            self.infob.setText('')
            self.tree=None
            self.rootNode = None
        else:
            self.tree=tree.tree
            self.infoa.setText('Number of nodes: ' + str(orngTree.countNodes(tree)))
            self.infob.setText('Number of leaves: ' + str(orngTree.countLeaves(tree)))
            if hasattr(self.scene, "colorPalette"):
                self.scene.colorPalette.setNumberOfColors(len(self.tree.distribution))
#            self.scene.setDataModel(GraphicsTree(self.tree))
            self.rootNode=self.walkcreate(self.tree, None)
#            self.scene.addItem(self.rootNode)
            self.scene.fixPos(self.rootNode,self.HSpacing,self.VSpacing)
            self.activateLoadedSettings()
            self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y())
            self.updateNodeToolTips()
            self.centerRootButton.setDisabled(0)
            self.centerNodeButton.setDisabled(1)

        self.scene.update()

    def walkcreate(self, tree, parent=None, level=0):
        node = GraphicsNode(tree, parent, None, self.scene)
        if parent:
            parent.graph_add_edge(GraphicsEdge(None, self.scene, node1=parent, node2=node))
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1)
        return node

    def walkupdate(self, node, level=0):
        if not node: return
        if self.MaxTreeDepthB and self.MaxTreeDepth<=level+1:
            node.setOpen(False)
            return
        else:
            node.setOpen(True,1)
        for n in node.branches:
            self.walkupdate(n,level+1)

    def clear(self):
        self.tree=None
        self.scene.clear()

    def updateNodeToolTips(self):
        
        for node in self.scene.nodes():
            node.setToolTip(self.nodeToolTip(node) if self.ToolTipsEnabled else "")
            
    def nodeToolTip(self, tree):
        return "tree node"
    
    def rescaleTree(self):
        NodeHeight = DefNodeHeight
        NodeWidth = DefNodeWidth * ((self.NodeSize -1) * (1.5 / 9.0) + 0.5)
        k = 1.0
        self.scene.VSpacing=int(NodeHeight*k*(0.3+self.VSpacing*0.15))
        self.scene.HSpacing=int(NodeWidth*k*(0.3+self.HSpacing*0.20))
        for r in self.scene.nodeList:
            r.setRect(r.rect().x(), r.rect().y(), int(NodeWidth*k), int(NodeHeight*k))
        
        self.scene.fixPos() #self.rootNode, 10, 10)

    def updateSelection(self):
        self.selectedNode = (self.scene.selectedItems() + [None])[0]
        self.centerNodeButton.setDisabled(not self.selectedNode)
        self.send("Examples", self.selectedNode.tree.examples if self.selectedNode else None)

    def saveGraph(self, fileName = None):
        from OWDlgs import OWChooseImageSizeDlg
        dlg = OWChooseImageSizeDlg(self.scene, [("Save as Dot Tree File (.dot)", self.saveDot)], parent=self)
        dlg.exec_()
        
    def saveDot(self, filename=None):
        if filename==None:
            filename = str(QFileDialog.getSaveFileName(self, "Save to ...", "tree.dot", "Dot Tree File (.DOT)"))
            if not filename:
                return
        orngTree.printDot(self.tree, filename)
        
class OWDefTreeViewer2D(OWTreeViewer2D):
    def __init__(self, parent=None, signalManager = None, name='DefTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)
        self.settingsList=self.settingsList+["ShowPie"]

        self.scene = TreeGraphicsScene(self)
        self.sceneView = TreeGraphicsView(self, self.scene, self.mainArea)
        self.mainArea.layout().addWidget(self.sceneView)
#        self.scene.setSceneRect(0,0,800,800)
        self.navWidget = QWidget(None)
        self.navWidget.setLayout(QVBoxLayout(self.navWidget))
        scene = TreeGraphicsScene(self.navWidget)
        self.treeNav = TreeNavigator(self.sceneView)
#        self.treeNav.setScene(scene)
        self.navWidget.layout().addWidget(self.treeNav)
#        self.sceneView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
#        OWGUI.button(self.TreeTab,self,"Navigator",self.toggleNavigator)

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDefTreeViewer2D()

    #data = orange.ExampleTable('../../doc/datasets/voting.tab')
    data = orange.ExampleTable(r"..//doc//datasets//zoo.tab")
    data = orange.ExampleTable(r"..//doc//datasets//iris.tab")
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.activateLoadedSettings()
    ow.ctree(None)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_()
    ow.saveSettings()


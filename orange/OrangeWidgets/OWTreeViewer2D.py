import orange, orngTree, OWGUI, OWColorPalette
from OWWidget import *

from PyQt4.QtCore import *
from PyQt4.QtGui import *

DefDroppletRadiust=7
DefNodeWidth=30
DefNodeHeight=20
DefDroppletBrush=QBrush(Qt.darkGray)


class GraphicsTextContainer(QGraphicsRectItem):
    def __init__(self, *args):
        QGraphicsRectItem.__init__(self, *args)
        self.textObj=[]
        self.spliterObj=[]
        self.sceneObj=[]
        self.lines=[]
        self.spliterLines=[]
        self.textLines=[]
        self.lineObj=[]
        self.textOffset=0
        self.lineSpacing=2
        self.setBrush(QBrush(Qt.white))
        self.isShown=False

    def setZValue(self, z):
        QGraphicsRectItem.setZValue(self,z)
        for o in self.textObj+self.spliterObj:
            o.setZValue(z+1)


    def setRect(self, x, y, w, h):
        QGraphicsRectItem.setRect(self, x, y, w, h)
        for s in self.spliterObj:
            s.setLine(0,0,w-1,0)

    def setPos(self, x, y):
        QGraphicsRectItem.setPos(self, x, y)

    def setFont(self, font, rearange=True):
        for r in self.textObj:
            r.setFont(font)
        if rearange:
            self.reArangeText()

    def setBrush(self, brush):
        QGraphicsRectItem.setBrush(self, brush)
        (h,s,v,a) = brush.color().getHsv()
        for t in self.textObj:
            (th,ts,tv,ta)=t.brush().color().getHsv()
            if (v<tv+200 and tv==0) or (v-tv and tv!=0) or s<ts:
                t.setBrush(QBrush(Qt.white))
            else:
                t.setBrush(QBrush(Qt.black))

    def show(self):
        self.isShown=True
        QGraphicsRectItem.show(self)
##        for o in self.textObj+self.sceneObj+self.spliterObj:
##            o.show()

    def hide(self):
        self.isShown=False
        QGraphicsRectItem.hide(self)
##        for o in self.textObj+self.sceneObj+self.spliterObj:
##            o.hide()

    def addTextLine(self,text=None, color=None, fitSquare=True):
        self.lines.append((text,color))
        if text!=None:
            t = QGraphicsSimpleTextItem(text, self, self.scene())
            t.setBrush(QBrush(color or Qt.black))
            t.setZValue(self.zValue()+1)
            t.setPos(1,self.textOffset)
            self.textObj.append(t)
            self.textOffset+=t.boundingRect().height()+self.lineSpacing
            self.textLines.append((text, color))
        else:
            t = QGraphicsLineItem(self, self.scene())
            t.setZValue(self.zValue()+1)
            t.setPen(QPen(color or Qt.black))
            t.setLine(0,0,self.rect().width()-1,0)
            t.setPos(0,self.textOffset+self.lineSpacing)
            self.textOffset+=self.lineSpacing*2
            self.spliterObj.append(t)
            self.spliterLines.append((text, color))
        self.lineObj.append(t)
        if fitSquare:
            self.fitSquare()
        if self.isShown:
            self.show()

    def truncateText(self, trunc=True):
        if trunc:
            for r in self.textObj:
                if r.boundingRect().width()>self.rect().width():
                    t=str(r.text())
                    cw=float(r.boundingRect().width())/len(t)
                    r.setText(t[:-int((r.boundingRect().width()-self.rect().width())/cw) or -1])
        else:
            for i in range(len(self.textObj)):
                self.textObj[i].setText(self.textLines[i][0])

    def reArangeText(self, fitSquare=True, startOffset=0):
        self.textOffset=startOffset
        for line in self.lineObj:
            if isinstance(line, QGraphicsSimpleTextItem):
                line.setPos(self.rect().x()+1, self.rect().y()+self.textOffset)
                self.textOffset+=line.boundingRect().height()+self.lineSpacing
            else:
                line.setPos(self.rect().x(),self.rect().y()+self.textOffset)
                self.textOffset+=self.lineSpacing
        if fitSquare:
            self.fitSquare()

    def setText(self, ind, text, fitSquare=True):
        if ind <len(self.textObj):
            self.textObj[ind].setText(text)
            self.textLines[ind]=(text, self.textLines[ind][1])
            if fitSquare:
                self.fitSquare()

    def fitSquare(self):
        w=max([t.boundingRect().width() for t in self.textObj]+[2])
        h=self.textOffset+2
        GraphicsTextContainer.setRect(self, 0, 0, w+2, h)

class GraphicsNode(GraphicsTextContainer):
    def __init__(self, tree, parent=None, *args):
        GraphicsTextContainer.__init__(self, None, *args)
        self.tree=tree
        self.parent=parent
        self.isRoot=False
        self.nodeList=[]
        self.isOpen=True
        self.isSelected=False
        self.dropplet = QGraphicsEllipseItem(self, self.scene())
        self.dropplet.setBrush(DefDroppletBrush)
        self.dropplet.setZValue(self.zValue()-1)
        self.dropplet.setStartAngle(180*16)
        self.dropplet.setSpanAngle(180*16)
        self.dropplet.node=self
        self.parentEdge=QGraphicsLineItem(self, self.scene())
        self.parentEdge.setPen(QPen())
        self.parentEdge.setZValue(self.zValue()-1)
        if not parent:
            self.parentEdge.hide()
        self.sceneObj+=[self.dropplet]
        self.selectionSquare=[]
        if parent:
            parent.insertNode(self)
        else:
            self.scene().nodeList.append(self)
            self.dropplet.show()

        self.setZValue(-20)

    def insertNode(self, node):
        self.nodeList.append(node)
        self.scene().nodeList.append(node)
##        node.parent=self
##        node.setParentItem(self)
        self.dropplet.show()

    def takeNode(self, node):
        self.nodeList.remove(node)
        self.scene().nodeList.remove(node)

    def setEdgeWidth(self, width=2):
        self.parentEdge.setPen(QPen(Qt.gray,width))

    def setOpen(self, open, level=-1):
        self.isOpen=open
        self.show()
        if level==0 and open:
            self.isOpen=False
            for n in self.nodeList:
                n.hideSubtree()
        elif open and level>0:
            for n in self.nodeList:
                n.setOpen(open,level-1)
        elif open and level<0:
            for n in self.nodeList:
                n.showSubtree()
        elif not open:
            for n in self.nodeList:
                n.hideSubtree()

    def setRect(self, x, y, w,h):
        GraphicsTextContainer.setRect(self, x, y, w, h)
##        self.dropplet.setRect(self.rect().x()+self.rect().width()/2,self.rect().y()+self.rect().height(), w/4,w/4)
        self.dropplet.setRect(3*w/8, (8*h-w)/8, w/4,w/4)
        if self.isSelected:
            self.setSelectionBox()

    def setZValue(self, z):
        GraphicsTextContainer.setZValue(self,z)
        self.dropplet.setZValue(z-1)
        self.parentEdge.setZValue(z-2)

    def show(self):
        GraphicsTextContainer.show(self)
        if self.parent:
            self.parentEdge.show()
        if not self.nodeList:
            self.dropplet.hide()
        else:
            self.dropplet.show()

    def hideSubtree(self):
        self.hide()
        if self.isSelected:
            self.setSelected(False)
        for n in self.nodeList:
            n.hideSubtree()

    def showSubtree(self):
        self.show()
        if self.isOpen:
            for n in self.nodeList:
                n.showSubtree()

    def updateEdge(self):
##        if self.parent!=self.scene() and self.parentEdge:
        if self.parent and self.parentEdge:
            droppletCenter=self.mapFromItem(self.parent.dropplet, self.parent.dropplet.rect().center())
            center=self.rect().center()
            self.parentEdge.setLine(self.rect().width()/2, 0,
                        droppletCenter.x(), droppletCenter.y())

    def setSelectionBox(self):
        self.isSelected=True
        if self.selectionSquare:
             self.selectionSquare = sl = self.selectionSquare
        else:
##             self.selectionSquare = sl = [QGraphicsLineItem(self, self.scene()) for i in range(8)]
             self.selectionSquare = sl = [QGraphicsPathItem(self, self.scene()) for i in range(8)]
             self.sceneObj.extend(self.selectionSquare)
        for line in sl:
            line.setZValue(-5)
##            line.setPos(self.x(),self.y())
            line.setPen(QPen(QColor(0, 0, 150), 3))
        xleft = -3.0; xright = self.rect().width() + 2.0
        yup = -3.0; ydown = self.rect().height() + 2.0
        xspan = self.rect().width() / 4.0; yspan = self.rect().height() / 4.0
        path = QPainterPath(QPointF(xleft + xspan, yup))
        path.lineTo(xleft, yup)
        path.lineTo(xleft, yup + yspan)
##        sl[0].setLine(xleft, yup, xleft + xspan, yup)
        sl[0].setPath(path)
        
        path = QPainterPath(QPointF(xleft, ydown - yspan))
        path.lineTo(xleft, ydown)
        path.lineTo(xleft + xspan, ydown)
        sl[1].setPath(path)

        path = QPainterPath(QPointF(xright - xspan, ydown))
        path.lineTo(xright, ydown)
        path.lineTo(xright, ydown - yspan)
        sl[2].setPath(path)

        path = QPainterPath(QPointF(xright, yup + yspan))
        path.lineTo(xright, yup)
        path.lineTo(xright - xspan, yup)
        sl[4].setPath(path)
        
        if self.isShown:
            self.show()

    def removeSelectionBox(self):
        self.isSelected=False
        for l in self.selectionSquare:
            if l.scene():
                l.scene().removeItem(l)
            self.sceneObj.remove(l)
        self.selectionSquare=[]

class TreeGraphicsView(QGraphicsView):
    def __init__(self, master, scene, *args):
        apply(QGraphicsView.__init__,(self,scene)+args)
        self.master=master
        self.dropplet=None
        self.selectedNode=None
        self.navigator=None
        self.viewport().setMouseTracking(True)
        self.setRenderHint(QPainter.Antialiasing)
#        self.setRenderHint(QPainter.TextAntialiasing)
#        self.setRenderHint(QPainter.HighQualityAntialiasing)
        
    def sizeHint(self):
        return QSize(200,200)

    def setNavigator(self, nav):
        self.navigator=nav

    def resizeEvent(self, event):
        QGraphicsView.resizeEvent(self, event)
        if self.navigator:
            self.navigator.resizeView()

    def updateDropplet(self, dropplet=None):
        if dropplet==self.dropplet:
            return
        if self.dropplet:
            self.dropplet.setBrush(DefDroppletBrush)
        self.dropplet=dropplet
        if self.dropplet:
            self.dropplet.setBrush(QBrush(Qt.black))
        self.scene().update()

    def updateSelection(self, node=None):
        if not node or node==self.selectedNode:
            if self.selectedNode:
                self.selectedNode.removeSelectionBox()
            self.selectedNode=None
            self.master.updateSelection(None)
        else:
            if self.selectedNode:
                self.selectedNode.removeSelectionBox()
            self.selectedNode=node
            self.selectedNode.setSelectionBox()
            self.master.updateSelection(self.selectedNode)

class TreeGraphicsScene(QGraphicsScene):
    def __init__(self, master, *args):
        apply(QGraphicsScene.__init__,(self,)+args)
        self.HSpacing=10
        self.VSpacing=10
        self.dropplet=None
        self.selectedNode=None
        self.master=master
        self.nodeList=[]
        self.edgeList=[]

    def drawItems(self, painter, items, options, widget=None):
        items = [(item.zValue(), item, opt) for item, opt in zip(items, options)]
        items.sort()
        for z, item, opt in items:
            painter.save()
            painter.setMatrix(item.sceneMatrix(), True)
            item.paint(painter, opt, widget)
            painter.restore()

    def insertNode(self, node, parent=None):
        if parent==None:
            self.clear()
            self.nodeList=[node]
            self.edgeList=[]
            node.isRoot=True
            node.parent=self
        else:
            parent.insertNode(node)

    def clear(self):
        for n in self.nodeList:
            if n.scene():
                n.scene().removeItem(n)
        self.nodeList=[]
        for item in self.items():
            self.removeItem(item)

    def fixPos(self, node=None, x=10, y=10):
        self.gx=x
        self.gy=y
        if not node:
            if self.nodeList == []: return        # don't know if this is ok
            node=self.nodeList[0]
        if not x or not y: x, y= self.HSpacing, self.VSpacing
        self._fixPos(node,x,y)
        
        self.setSceneRect(reduce(QRectF.united, [item.sceneBoundingRect() for item in self.nodeList if node.isVisible()], QRectF(0, 0, 10, 10)).adjusted(0, 0, 100, 100))
        self.update()
        
    def _fixPos(self, node, x, y):
        ox=x
        if node.nodeList and node.isOpen:
            for n in node.nodeList:
                (x,ry)=self._fixPos(n,x,y+self.VSpacing+node.rect().height())
            x=(node.nodeList[0].pos().x()+node.nodeList[-1].pos().x())/2
            node.setPos(x,y)
            for n in node.nodeList:
                n.updateEdge()
        else:
            node.setPos(self.gx,y)
            self.gx+=self.HSpacing + node.rect().width()
            x+=self.HSpacing+node.rect().width()
            self.gy=max([y,self.gy])

        return (x,y)

    def mouseMoveEvent(self,event):
        obj=self.items(event.scenePos())
        obj=filter(lambda a:a.zValue()==-21 or a.zValue()==-20,obj)
        if not obj:
            self.updateDropplet()
        elif obj[0].__class__ == QGraphicsEllipseItem:
            self.updateDropplet(obj[0])
        else:
            self.updateDropplet()

    def mousePressEvent(self, event):
        if self.dropplet:
            self.dropplet.node.setOpen(not self.dropplet.node.isOpen,
                (event.button()==Qt.RightButton and 1) or -1)
            self.fixPos()
        else:
            obj=self.items(event.scenePos())
            obj=filter(lambda a:a.zValue()==-20, obj)
            if obj and isinstance(obj[0], QGraphicsRectItem):
                self.updateSelection(obj[0])
            else:
                self.updateSelection(None)
        self.update()

    def updateDropplet(self, dropplet=None):
        if dropplet==self.dropplet:
            return
        if self.dropplet:
            self.dropplet.setBrush(DefDroppletBrush)
        self.dropplet=dropplet
        if self.dropplet:
            self.dropplet.setBrush(QBrush(Qt.black))
        self.update()

    def updateSelection(self, node=None):
        if not node or node==self.selectedNode:
            if self.selectedNode:
                self.selectedNode.removeSelectionBox()
            self.selectedNode=None
            self.master.updateSelection(None)
        else:
            if self.selectedNode:
                self.selectedNode.removeSelectionBox()
            self.selectedNode=node
            self.selectedNode.setSelectionBox()
            self.master.updateSelection(self.selectedNode)        

class TreeNavigator(QGraphicsView):

    def __init__(self, masterView, *args):
        QGraphicsView.__init__(self)
        self.masterView = masterView
        self.setScene(self.masterView.scene())
        self.connect(self.scene(), SIGNAL("sceneRectChanged(QRectF)"), self.updateSceneRect)
        self.setRenderHint(QPainter.Antialiasing)
#        self.setInteractive(False)

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
        
    def updateView(self):
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
                    "NodeSize", "NodeInfo", "NodeColorMethod",
                    "TruncateText"]

    def __init__(self, parent=None, signalManager = None, name='TreeViewer2D'):
        OWWidget.__init__(self, parent, signalManager, name)
#        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.root = None
        self.selectedNode = None

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]

        # some globaly used variables (get rid!)
        #self.TargetClassIndex = 0

        #set default settings
        self.ZoomAutoRefresh = 0
        self.AutoArrange = 0
        self.ToolTipsEnabled = 1
        self.MaxTreeDepth = 5; self.MaxTreeDepthB = 0
        self.LineWidth = 5; self.LineWidthMethod = 0
        self.NodeSize = 5
        self.NodeInfo = [0, 1]

        self.NodeColorMethod = 0
        self.Zoom = 5
        self.VSpacing = 5; self.HSpacing = 5
        self.TruncateText = 1

        self.loadSettings()
        self.NodeInfo.sort()
        self.scaleSizes()

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

        # OWGUI.checkBox(GeneralTab, self, 'ZoomAutoRefresh', 'Auto refresh after zoom',
        #                tooltip='Refresh after change of zoom setting?')
        # OWGUI.checkBox(GeneralTab, self, 'AutoArrange', 'Auto arrange',
        #                tooltip='Auto arrange the position of the nodes\nafter any change of nodes visibility')
        OWGUI.checkBox(GeneralTab, self, 'ToolTipsEnabled', 'Show node tool tips',
                       tooltip='When mouse over the node show tool tip', callback=self.updateNodeToolTips)
        OWGUI.checkBox(GeneralTab, self, 'TruncateText', 'Truncate text to fit margins',
                       tooltip='Truncate any text to fit the node width',
                       callback=self.toggleTruncateText)

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
        # Node size options
        OWGUI.hSlider(NodeTab, self, 'NodeSize', box='Node width',
                      minValue=1, maxValue=10, step=1,
                      callback=self.toggleNodeSize, ticks=1)

        # Node information
        OWGUI.button(self.controlArea, self, "Navigator", self.toggleNavigator, debuggingEnabled = 0)
        findbox = OWGUI.widgetBox(self.controlArea, orientation = "horizontal")
        self.centerRootButton=OWGUI.button(findbox, self, "Find Root",
                                           callback=lambda :self.rootNode and self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y()))
        self.centerNodeButton=OWGUI.button(findbox, self, "Find Selected",
                                           callback=lambda :self.selectedNode and \
                                     self.sceneView.centerOn(self.selectedNode.scenePos()))
        self.NodeTab=NodeTab
        self.TreeTab=TreeTab
        self.GeneralTab=GeneralTab
        OWGUI.rubber(GeneralTab)
        OWGUI.rubber(TreeTab)
        OWGUI.rubber(NodeTab)
        self.rootNode=None
        self.tree=None
        self.resize(800, 500)

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
        
        buffer = QPixmap(QSize(600, h*fact))
        painter.begin(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        self.scene.render(painter)
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))
        self.reportRaw('<!--browsercode<br/>(Click <a href="%s">here</a> to view or download this image in a scalable vector format)-->' % urlfn)
        #self.reportObject(self.svg_type, urlfn, width="600", height=str(h*fact))
        
        
    def scaleSizes(self):
        pass

    def toggleZoomSlider(self):
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleVSpacing(self):
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleHSpacing(self):
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()

    def toggleTruncateText(self):
        for n in self.scene.nodeList:
           n.truncateText(self.TruncateText)
        self.scene.update()

    def toggleTreeDepth(self):
        self.walkupdate(self.rootNode)
        self.scene.fixPos(self.rootNode,10,10)
        rect = reduce(QRectF.united, [item.sceneBoundingRect() for item in self.scene.nodeList if item.isVisible()], QRectF(0, 0, 10, 10))
        self.scene.setSceneRect(rect.adjusted(0, 0, 10, 10)) #self.scene.itemsBoundingRect().united(QRectF(0,0,1,1)))
#        print rect
        self.scene.update()

    def toggleLineWidth(self):
        for n in self.scene.nodeList:
            if self.LineWidthMethod==0:
                width=self.LineWidth
            elif self.LineWidthMethod == 1:
                width = (n.tree.distribution.cases/self.tree.distribution.cases) * self.LineWidth
            elif self.LineWidthMethod == 2:
                width = (n.tree.distribution.cases/((n.parent and \
                                    n.parent.tree.distribution.cases) or n.tree.distribution.cases)) * self.LineWidth
            n.setEdgeWidth(width)
        self.scene.update()

    def toggleNodeSize(self):
        self.rescaleTree()

    def toggleNavigator(self):
        self.navWidget.setHidden(not self.navWidget.isHidden())

    def activateLoadedSettings(self):
        if not self.tree:
            return
        self.rescaleTree()
        self.scene.fixPos(self.rootNode,10,10)
        self.scene.update()
        self.toggleTruncateText()
        self.toggleTreeDepth()
        self.toggleLineWidth()
        self.toggleNodeSize()

    def ctree(self, tree=None):
        self.clear()
        if not tree:
            self.centerRootButton.setDisabled(1)
            self.centerNodeButton.setDisabled(0)
            self.infoa.setText('No tree.')
            self.infob.setText('')
            self.tree=None
        else:
            self.tree=tree.tree
            self.infoa.setText('Number of nodes: ' + str(orngTree.countNodes(tree)))
            self.infob.setText('Number of leaves: ' + str(orngTree.countLeaves(tree)))
            if hasattr(self.scene, "colorPalette"):
                self.scene.colorPalette.setNumberOfColors(len(self.tree.distribution))
#            self.scene.setDataModel(GraphicsTree(self.tree))
            self.rootNode=self.walkcreate(self.tree, None)
            self.scene.addItem(self.rootNode)
            self.scene.fixPos(self.rootNode,self.HSpacing,self.VSpacing)
            self.activateLoadedSettings()
            self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y())
            self.updateNodeToolTips()
            self.centerRootButton.setDisabled(0)
            self.centerNodeButton.setDisabled(1)

        self.scene.update()

    def walkcreate(self, tree, parent=None, level=0):
        node = GraphicsNode(tree, parent, self.scene)
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
        for n in node.nodeList:
            self.walkupdate(n,level+1)

    def clear(self):
        self.tree=None
        self.scene.clear()
#        self.scene.setSceneRect(QRectF())
#        self.treeNav.scene().clear()

    def updateNodeToolTips(self):
        for node in self.scene.nodeList:
            node.setToolTip(self.nodeToolTip(node) if self.ToolTipsEnabled else "")
            
    def nodeToolTip(self, tree):
        return "tree node"
    
    def rescaleTree(self):
        NodeHeight = DefNodeHeight #* (self.NodeSize / 5.0 + 2.0 / 5.0)
        NodeWidth = DefNodeWidth * ((self.NodeSize -1) * (1.5 / 9.0) + 0.5)
        k = 0.0028 * (self.Zoom ** 2) + 0.2583 * self.Zoom + 1.1389
        self.scene.VSpacing=int(NodeHeight*k*(0.3+self.VSpacing*0.15))
        self.scene.HSpacing=int(NodeWidth*k*(0.3+self.HSpacing*0.20))
        for r in self.scene.nodeList:
            r.setRect(r.rect().x(), r.rect().y(), int(NodeWidth*k), int(NodeHeight*k))
        
        self.scene.fixPos() #self.rootNode, 10, 10)

    def updateSelection(self, node=None):
        self.selectedNode=node
        if node:
            self.centerNodeButton.setDisabled(0)
            self.send("Examples", node.tree.examples)
        else:
            self.centerNodeButton.setDisabled(1)
            self.send("Examples", None)

    def saveGraph(self, fileName = None):
        from OWDlgs import OWChooseImageSizeDlg
        dlg = OWChooseImageSizeDlg(self.scene, [("Save as Dot Tree File (.dot)", self.saveDot)])
        dlg.exec_()
        
    def saveDot(self, filename=None):
        if filename==None:
            filename = str(QFileDialog.getSaveFileName(None, "Save to ...", "tree.dot", "Dot Tree File (.DOT)"))
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
        self.sceneView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
#        OWGUI.button(self.TreeTab,self,"Navigator",self.toggleNavigator)

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDefTreeViewer2D()

    #data = orange.ExampleTable('../../doc/datasets/voting.tab')
    data = orange.ExampleTable(r"..//doc//datasets//zoo.tab")
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.activateLoadedSettings()
    ow.ctree(None)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_()
    ow.saveSettings()


import orange, orngTree, OWGUI, OWColorPalette
from OWWidget import *

from PyQt4.QtCore import *
from PyQt4.QtGui import *

DefDroppletRadiust=7
DefNodeWidth=30
DefNodeHeight=20
ExpectedBubbleWidth=200
ExpectedBubbleHeight=400
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

#    def setCanvas(self, canvas):
#        QGraphicsRectItem.setCanvas(self, canvas)
#        for o in self.textObj+self.canvasObj+self.spliterObj:
#            o.setCanvas(canvas)

#    def setPos(self, x, y):
#        dx, dy=x-self.x(), y-self.y()
#        QGraphicsRectItem.setPos(self, x, y)
#        for o in self.textObj+self.canvasObj+self.spliterObj:
#            o.moveBy(dx, dy)

    def setZValue(self, z):
        QGraphicsRectItem.setZValue(self,z)
        for o in self.textObj+self.spliterObj:
            o.setZValue(z+1)

#    def setSize(self, w, h):
#        QGraphicsRectItem.setSize(self, w, h)
#        for s in self.spliterObj:
#            s.setLine(0,0,w-1,0)

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
#        print w, h
##        GraphicsTextContainer.setRect(self, self.x(), self.y(), w+2, h)
##        self.setRect(self.rect().setSize(QSizeF(w+2, h)))

class GraphicsBubbleInfo(GraphicsTextContainer):
    def __init__(self, node, pos, *args):
        GraphicsTextContainer.__init__(self, None, *args)
        self.shadow = QGraphicsRectItem(self, self.scene())
        self.shadow.setBrush(QBrush(Qt.darkGray))
        self.sceneObj.append(self.shadow)
        self.setPos(5, 5)

    def setZValue(self, z):
        GraphicsTextContainer.setZValue(self,z)
        self.shadow.setZValue(z-1)

    def setRect(self, x, y, w, h):
        GraphicsTextContainer.setRect(self, x, y, w, h)
        self.shadow.setRect(0, 0, w, h)

    def fitSquare(self):
        w=max([t.boundingRect().width() for t in self.textObj]+[2])
        h=self.textOffset+2
#        print w, h
        GraphicsBubbleInfo.setRect(self, 0, 0, w+2, h)


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

#    def setCanvas(self, canvas):
#        CanvasTextContainer.setCanvas(self, canvas)
#        self.parentEdge.setCanvas(canvas)

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

##    def setPos(self, x, y):
##        GraphicsTextContainer.setPos(self, x, y)
##        dx,dy=x-self.x(), y-self.y()
##        self.dropplet.moveBy(dx,dy)

    def show(self):
        GraphicsTextContainer.show(self)
        if self.parent:
            self.parentEdge.show()
        if not self.nodeList:
            self.dropplet.hide()
        else:
            self.dropplet.show()

##    def hide(self):
##        GraphicsTextContainer.hide(self)
##        self.parentEdge.hide()

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
             self.selectionSquare = sl=self.selectionSquare
        else:
             self.selectionSquare = sl = [QGraphicsLineItem(self, self.scene()) for i in range(8)]
             self.sceneObj.extend(self.selectionSquare)
        for line in sl:
            line.setZValue(-5)
##            line.setPos(self.x(),self.y())
            line.setPen(QPen(QColor(0, 0, 150), 3))
        xleft = -3; xright = self.rect().width() + 2
        yup = -3; ydown = self.rect().height() + 2
        xspan = self.rect().width() / 4; yspan = self.rect().height() / 4
        sl[0].setLine(xleft, yup, xleft + xspan, yup)
        sl[1].setLine(xleft, yup-1, xleft, yup + yspan)
        sl[2].setLine(xright, yup, xright - xspan, yup)
        sl[3].setLine(xright, yup-1, xright, yup + yspan)
        sl[4].setLine(xleft, ydown, xleft + xspan, ydown)
        sl[5].setLine(xleft, ydown+2, xleft, ydown - yspan)
        sl[6].setLine(xright, ydown, xright - xspan, ydown)
        sl[7].setLine(xright, ydown+2, xright, ydown - yspan)
        if self.isShown:
            self.show()

    def removeSelectionBox(self):
        self.isSelected=False
        for l in self.selectionSquare:
            l.scene().removeItem(l)
            self.sceneObj.remove(l)
        self.selectionSquare=[]


def bubbleConstructor(node=None, pos =None, scene=None):
    return GraphicsBubbleInfo(node, pos, scene)

class TreeGraphicsView(QGraphicsView):
    def __init__(self, master, scene, *args):
        apply(QGraphicsView.__init__,(self,scene)+args)
        self.master=master
        self.dropplet=None
        self.selectedNode=None
        self.bubble=None
        self.bubbleNode=None
        self.navigator=None
        self.viewport().setMouseTracking(True)
        self.bubbleConstructor=bubbleConstructor
#        self.setRenderHint(QPainter.Antialiasing, 0)
#        self.setRenderHint(QPainter.TextAntialiasing, 0)
#        self.setRenderHint(QPainter.HighQualityAntialiasing, 0)
        
    def sizeHint(self):
        return QSize(200,200)

    def setNavigator(self, nav):
        self.navigator=nav
        self.master.connect(self.scene(),SIGNAL("resized()"),self.navigator.resizeScene)
        self.master.connect(self, SIGNAL("contentsMoving(int,int)"),self.navigator.moveView)

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

    def updateBubble(self, node=None, pos=QPoint(0,0)):
        if self.bubbleNode==node and self.bubble:
            self.bubble.setPos(pos.x()+5,pos.y()+5)
            self.bubble.show()
        elif node:
            if self.bubble:
                self.bubble.scene().removeItem(self.bubble)
            self.bubbleNode=node
            self.bubble=self.bubbleConstructor(node, pos, self.scene())
            self.bubble.setPos(pos.x()+5,pos.y()+5)
            self.bubble.setZValue(50)
            self.bubble.show()
        elif self.bubble:
            self.bubble.scene().removeItem(self.bubble)
            self.bubble=self.bubbleNode=None
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

##    def contentsMouseMoveEvent(self,event):
##        obj=self.scene().collisions(event.pos())        # TO DO
##        obj=filter(lambda a:a.zValue()==-21 or a.zValue()==-20,obj)
##        if not obj:
##            self.updateDropplet()
##            self.updateBubble()
##        elif isinstance(obj[0], QGraphicsRectItem) and self.master.NodeBubblesEnabled:
##            self.updateBubble(obj[0],event.pos())
##            self.updateDropplet()
##        elif obj[0].__class__ == QGraphicsEllipseItem:
##            self.updateDropplet(obj[0])
##        else:
##            self.updateDropplet()
##            self.updateBubble()
##
##    def contentsMousePressEvent(self, event):
##        if self.dropplet:
##            self.dropplet.node.setOpen(not self.dropplet.node.isOpen,
##                (event.button()==QEvent.RightButton and 1) or -1)
##            self.scene().fixPos()
##        else:
##            obj=self.scene().collisions(event.pos())        # to do
##            obj=filter(lambda a:a.zValue()==-20, obj)
##            if obj and isinstance(obj[0], QGraphicsRectItem):
##                self.updateSelection(obj[0])
##        self.scene().update()


class TreeGraphicsScene(QGraphicsScene):
    def __init__(self, master, *args):
        apply(QGraphicsScene.__init__,(self,)+args)
        self.HSpacing=10
        self.VSpacing=10
        self.dropplet=None
        self.selectedNode=None
        self.bubble=None
        self.bubbleNode=None
        self.master=master
        self.bubbleConstructor=bubbleConstructor
        self.nodeList=[]
        self.edgeList=[]

    def drawItems(self, painter, items, options, widget=None):
        items = [(item.zValue(), item, opt) for item, opt in zip(items, options)]
        items.sort()
        for z, item, opt in items:
##            print item
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

    def fixPos(self, node=None, x=10, y=10):
        self.gx=x
        self.gy=y
        if not node:
            if self.nodeList == []: return        # don't know if this is ok
            node=self.nodeList[0]
        if not x or not y: x, y= self.HSpacing, self.VSpacing
        self._fixPos(node,x,y)
        self.setSceneRect(0,0,self.gx+ExpectedBubbleWidth, self.gy+ExpectedBubbleHeight)
        
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
            self.updateBubble()
        elif isinstance(obj[0], QGraphicsRectItem) and self.master.NodeBubblesEnabled:
            self.updateBubble(obj[0],event.scenePos())
            self.updateDropplet()
        elif obj[0].__class__ == QGraphicsEllipseItem:
            self.updateDropplet(obj[0])
        else:
            self.updateDropplet()
            self.updateBubble()

    def mousePressEvent(self, event):
        if self.dropplet:
            self.dropplet.node.setOpen(not self.dropplet.node.isOpen,
                (event.button()==Qt.RightButton and 1) or -1)
            self.fixPos()
        else:
            obj=self.items(event.scenePos())        # to do
            obj=filter(lambda a:a.zValue()==-20, obj)
            if obj and isinstance(obj[0], QGraphicsRectItem):
                self.updateSelection(obj[0])
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

    def updateBubble(self, node=None, pos=QPointF(0,0)):
        if self.bubbleNode==node and self.bubble:
            self.bubble.setPos(pos.x()+5,pos.y()+5)
            self.bubble.show()
        elif node:
            if self.bubble:
                self.removeItem(self.bubble)
            self.bubbleNode=node
            self.bubble=self.bubbleConstructor(node, pos, self)
            self.bubble.setPos(pos.x()+5,pos.y()+5)
            self.bubble.setZValue(50)
            self.bubble.show()
        elif self.bubble:
            self.removeItem(self.bubble)
            self.bubble=self.bubbleNode=None
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

class TreeNavigator(TreeGraphicsView):
    class NavigatorNode(GraphicsNode):
        def __init__(self,masterNode,*args):
            GraphicsNode.__init__(self, *args)
            self.masterNode=masterNode
        def leech(self):
            self.isOpen=self.masterNode.isOpen
            self.setBrush(self.masterNode.brush())

    def __init__(self, masterView, *args):
        apply(TreeGraphicsView.__init__,(self,)+args)
        self.myScene = self.scene()
        self.masterView=masterView
        self.scene().setSceneRect(0,0,self.width(),self.height())
        self.rootNode=None
        self.updateRatio()
        self.viewRect=QGraphicsRectItem(None, self.scene())
        self.viewRect.setRect(0,0, self.rx*self.masterView.width(), self.ry*self.masterView.height())
        self.viewRect.show()
        self.viewRect.setBrush(QBrush(Qt.lightGray))
        self.viewRect.setZValue(-10)
        self.buttonPressed=False
        self.bubbleConstructor=self.myBubbleConstructor
        self.isShown=False

    def leech(self):
        if not self.isShown:
            return
        self.updateRatio()
        if self.rootNode!=self.masterView.scene().nodeList[0]: #display a new tree
            for item in self.scene().items():
                self.scene().removeItem(item)
            self.rootNode=self.walkcreate(self.masterView.scene().nodeList[0], None)
            self.walkupdate(self.rootNode)
        else:
            self.walkupdate(self.rootNode)
        self.scene().update()


    def walkcreate(self, masterNode, parent):
        node=self.NavigatorNode(masterNode, masterNode.tree, parent, self.scene())
        node.setZValue(0)
        for n in masterNode.nodeList:
            self.walkcreate(n,node)
        return node

    def walkupdate(self, node):
        node.leech()
        if node.masterNode.isShown:
            node.show()
            node.setPos(self.rx*node.masterNode.x(), self.ry*node.masterNode.y())
            node.setRect(0, 0, self.rx*node.masterNode.rect().width(), self.ry*node.masterNode.rect().height())
            for n in node.nodeList:
                self.walkupdate(n)
                n.updateEdge()
        else:
            node.hideSubtree()

    def mousePressEvent(self, event):
##        if self.scene().sceneRect().contains(event.pos()):
        self.masterView.centerOn(event.pos().x()/self.rx, event.pos().y()/self.ry)
        self.buttonPressed=True

    def mouseReleaseEvent(self, event):
        self.buttonPressed=False

    def mouseMoveEvent(self, event):
        if self.buttonPressed:
            self.masterView.centerOn(event.pos().x()/self.rx, event.pos().y()/self.ry)
            self.updateBubble(None)
        else:
            obj=self.scene().items(self.mapToScene(event.pos()))        # to do
            if obj and obj[0].__class__==self.NavigatorNode:
                self.updateBubble(obj[0], self.mapToScene(event.pos()))
            else:
                self.updateBubble()

    def resizeEvent(self, event):
        self.scene().setSceneRect(0,0,event.size().width(), event.size().height())
        self.leech()
        self.updateView()
        self.scene().update()

    def resizeView(self):
        self.updateRatio()
        self.updateView()
        self.scene().update()

    def resizeScene(self):
        self.updateRatio()
        if self.rootNode:
            self.leech()
            self.updateView()
            self.scene().update()

    def moveView(self, x,y):
        self.updateRatio()
        self.viewRect.setRect(x*self.rx, y*self.ry, self.masterView.width()*self.rx, self.masterView.height()*self.ry)
        self.scene().update()

    def updateRatio(self):
        self.rx=float(self.scene().width())/float(self.masterView.scene().width())
        self.ry=float(self.scene().height())/float(self.masterView.scene().height())
        #print "Ratio: ", self.rx, self.ry

    def updateView(self):
        pos=self.masterView.mapFromScene(0, 0)
##        self.viewRect.setRect(self.masterView.sceneRect().x()*self.rx, self.masterView.sceneRect().y()*self.ry, self.masterView.sceneRect().width()*self.rx, self.masterView.sceneRect().height()*self.ry)
        self.viewRect.setRect(-pos.x()*self.rx, -pos.y()*self.ry, self.masterView.width()*self.rx, self.masterView.height()*self.ry)
        #print "UpdateView:", -pos.x()*self.rx, -pos.y()*self.ry

    def myBubbleConstructor(self, node, pos, scene):
        return self.masterView.scene().bubbleConstructor(node.masterNode, pos, scene)


class OWTreeViewer2D(OWWidget):
    settingsList = ["ZoomAutoRefresh", "AutoArrange", "NodeBubblesEnabled",
                    "Zoom", "VSpacing", "HSpacing", "MaxTreeDepth", "MaxTreeDepthB",
                    "LineWidth", "LineWidthMethod",
                    "NodeSize", "NodeInfo", "NodeColorMethod",
                    "TruncateText"]

    def __init__(self, parent=None, signalManager = None, name='TreeViewer2D'):
        OWWidget.__init__(self, parent, signalManager, name)
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.root = None
        self.selectedNode = None

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.ctree)]
        self.outputs = [("Examples", ExampleTable)]

        # some globaly used variables (get rid!)
        #self.TargetClassIndex = 0

        #set default settings
        self.ZoomAutoRefresh = 0
        self.AutoArrange = 0
        self.NodeBubblesEnabled = 1
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
        OWGUI.checkBox(GeneralTab, self, 'NodeBubblesEnabled', 'Node bubbles',
                       tooltip='When mouse over the node show info bubble')
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
        pass

    def toggleNavigator(self):
        if self.navWidget.isVisible():
            self.navWidget.hide()
            self.treeNav.isShown=False
        else:
            self.navWidget.show()
            self.treeNav.isShown=True    # just so it knows it is shown
            self.treeNav.leech()

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
        self.treeNav.leech()
        #self.toggleNavigator()

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
            self.ClassColors = OWColorPalette.ColorPaletteHSV(len(self.tree.distribution))
            self.rootNode=self.walkcreate(self.tree, None)
            self.scene.addItem(self.rootNode)
            self.scene.fixPos(self.rootNode,self.HSpacing,self.VSpacing)
            self.activateLoadedSettings()
            self.sceneView.centerOn(self.rootNode.x(), self.rootNode.y())
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
        self.treeNav.scene().clear()

    def rescaleTree(self):
        k = 0.0028 * (self.Zoom ** 2) + 0.2583 * self.Zoom + 1.1389
        self.scene.VSpacing=int(DefNodeHeight*k*(0.3+self.VSpacing*0.15))
        self.scene.HSpacing=int(DefNodeWidth*k*(0.3+self.HSpacing*0.20))
        for r in self.scene.nodeList:
            r.setRect(r.rect().x(), r.rect().y(), int(DefNodeWidth*k), int(DefNodeHeight*k))

    def updateSelection(self, node=None):
        self.selectedNode=node
        if node:
            self.centerNodeButton.setDisabled(0)
            self.send("Examples", node.tree.examples)
        else:
            self.centerNodeButton.setDisabled(1)
            self.send("Examples", None)

    def saveGraph(self, fileName = None):
        if not fileName:
            qfileName = QFileDialog.getSaveFileName(None, "Save to..", "tree.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)\nDot Tree File(.DOT)")
            fileName = str(qfileName)
            if not fileName:
                return

        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        if ext=="DOT":
            orngTree.printDot(self.tree, fileName)
            return
        dSize= self.scene.sceneRect().size()
        buffer = QPixmap(dSize.width(),dSize.height()) # any size can do, now using the window size
        painter = QPainter(buffer)

        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        self.scene.render(painter)
        painter.end()
        buffer.save(fileName, ext)
        
class OWDefTreeViewer2D(OWTreeViewer2D):
    def __init__(self, parent=None, signalManager = None, name='DefTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)
        self.settingsList=self.settingsList+["ShowPie"]

        self.scene = TreeGraphicsScene(self)
        self.sceneView = TreeGraphicsView(self, self.scene, self.mainArea)
        self.mainArea.layout().addWidget(self.sceneView)
        self.scene.setSceneRect(0,0,800,800)
        self.navWidget = QWidget(None)
        self.navWidget.setLayout(QVBoxLayout(self.navWidget))
        scene = TreeGraphicsScene(self.navWidget)
        self.treeNav = TreeNavigator(self.sceneView, self, scene, self.navWidget)
        self.treeNav.setScene(scene)
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


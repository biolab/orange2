import orange, orngTree, qt, OWGUI, OWGraphTools
from OWWidget import *
from qtcanvas import *

DefDroppletRadiust=7
DefNodeWidth=30
DefNodeHeight=20
ExpectedBubbleWidth=200
ExpectedBubbleHeight=400
DefDroppletBrush=QBrush(Qt.darkGray)

class CanvasTextContainer(QCanvasRectangle):
    def __init__(self, *args):
        QCanvasRectangle.__init__(self, *args)
        self.textObj=[]
        self.spliterObj=[]
        self.canvasObj=[]
        self.lines=[]
        self.spliterLines=[]
        self.textLines=[]
        self.lineObj=[]
        self.textOffset=0
        self.lineSpacing=2
        self.setBrush(QBrush(Qt.white))
        self.isShown=False

    def setCanvas(self, canvas):
        QCanvasRectangle.setCanvas(self, canvas)
        for o in self.textObj+self.canvasObj+self.spliterObj:
            o.setCanvas(canvas)

    def move(self, x, y):
        dx, dy=x-self.x(), y-self.y()
        QCanvasRectangle.move(self, x, y)
        for o in self.textObj+self.canvasObj+self.spliterObj:
            o.moveBy(dx, dy)

    def setZ(self, z):
        QCanvasRectangle.setZ(self,z)
        for o in self.textObj+self.spliterObj:
            o.setZ(z+1)

    def setSize(self, w, h):
        QCanvasRectangle.setSize(self, w, h)
        for s in self.spliterObj:
            s.setPoints(0,0,w-1,0)

    def setFont(self, font, rearange=True):
        for r in self.textObj:
            r.setFont(font)
        if rearange:
            self.reArangeText()

    def setBrush(self, brush):
        QCanvasRectangle.setBrush(self, brush)
        (h, s,v)=brush.color().hsv()
        for t in self.textObj:
            (th,ts,tv)=t.color().hsv()
            if (v<tv+200 and tv==0) or (v-tv and tv!=0) or s<ts:
                t.setColor(Qt.white)
            else:
                t.setColor(Qt.black)

    def show(self):
        self.isShown=True
        QCanvasRectangle.show(self)
        for o in self.textObj+self.canvasObj+self.spliterObj:
            o.show()

    def hide(self):
        self.isShown=False
        QCanvasRectangle.hide(self)
        for o in self.textObj+self.canvasObj+self.spliterObj:
            o.hide()

    def addTextLine(self,text=None, color=None, fitSquare=True):
        self.lines.append((text,color))
        if text!=None:
            t=QCanvasText(text,self.canvas())
            t.setColor(color or Qt.black)
            t.setZ(self.z()+1)
            t.move(self.x()+1,self.y()+self.textOffset)
            self.textObj.append(t)
            self.textOffset+=t.boundingRect().height()+self.lineSpacing
            self.textLines.append((text, color))
        else:
            t=QCanvasLine(self.canvas())
            t.setZ(self.z()+1)
            t.setPen(QPen(color or Qt.black))
            t.setPoints(0,0,self.width()-1,0)
            t.move(self.x(),self.y()+self.textOffset+self.lineSpacing)
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
                if r.boundingRect().width()>self.width():
                    t=str(r.text())
                    cw=float(r.boundingRect().width())/len(t)
                    r.setText(t[:-int((r.boundingRect().width()-self.width())/cw) or -1])
        else:
            for i in range(len(self.textObj)):
                self.textObj[i].setText(self.textLines[i][0])

    def reArangeText(self, fitSquare=True, startOffset=0):
        self.textOffset=startOffset
        for line in self.lineObj:
            if isinstance(line, QCanvasText):
                line.move(self.x()+1, self.y()+self.textOffset)
                self.textOffset+=line.boundingRect().height()+self.lineSpacing
            else:
                line.move(self.x(),self.y()+self.textOffset)
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
        CanvasTextContainer.setSize(self, w+2, h)

class CanvasBubbleInfo(CanvasTextContainer):
    def __init__(self, node, pos, *args):
        CanvasTextContainer.__init__(self, *args)
        self.shadow=QCanvasRectangle(self.canvas())
        self.shadow.setBrush(QBrush(Qt.darkGray))
        self.shadow.move(5,5)
        self.canvasObj.append(self.shadow)

    def setZ(self, z):
        CanvasTextContainer.setZ(self,z)
        self.shadow.setZ(z-1)

    def setSize(self, w, h):
        CanvasTextContainer.setSize(self, w, h)
        self.shadow.setSize(w,h)

    def fitSquare(self):
        w=max([t.boundingRect().width() for t in self.textObj]+[2])
        h=self.textOffset+2
        CanvasBubbleInfo.setSize(self, w+2, h)


class CanvasNode(CanvasTextContainer):
    def __init__(self, tree, parent=None, *args):
        CanvasTextContainer.__init__(self, *args)
        self.tree=tree
        self.parent=parent
        self.isRoot=False
        self.nodeList=[]
        self.isOpen=True
        self.isSelected=False
        self.dropplet=QCanvasEllipse(self.canvas())
        self.dropplet.setBrush(DefDroppletBrush)
        self.dropplet.setZ(self.z()-1)
        self.dropplet.node=self
        self.parentEdge=QCanvasLine(self.canvas())
        self.parentEdge.setPen(QPen())
        self.parentEdge.setZ(self.z()-1)
        self.canvasObj+=[self.dropplet]
        self.selectionSquare=[]
        if parent:
            parent.insertNode(self)

        self.setZ(-20)

    def insertNode(self, node):
        self.nodeList.append(node)
        self.canvas().nodeList.append(node)
        node.parent=self
        self.dropplet.show()

    def takeNode(self, node):
        self.nodeList.remove(node)
        self.canvas().nodeList.remove(node)

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

    def setCanvas(self, canvas):
        CanvasTextContainer.setCanvas(self, canvas)
        self.parentEdge.setCanvas(canvas)

    def setSize(self, w,h):
        CanvasTextContainer.setSize(self,w,h)
        self.dropplet.setSize(w/4,w/4)
        self.dropplet.move(self.rect().x()+self.width()/2,self.rect().y()+self.height())
        if self.isSelected:
            self.setSelectionBox()

    def setZ(self, z):
        CanvasTextContainer.setZ(self,z)
        self.dropplet.setZ(z-1)
        self.parentEdge.setZ(z-2)

    def move(self, x, y):
        CanvasTextContainer.move(self, x, y)
        dx,dy=x-self.x(), y-self.y()
        self.dropplet.moveBy(dx,dy)

    def show(self):
        CanvasTextContainer.show(self)
        self.parentEdge.show()
        if not self.nodeList:
            self.dropplet.hide()
        else:
            self.dropplet.show()

    def hide(self):
        CanvasTextContainer.hide(self)
        self.parentEdge.hide()

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
        if self.parent!=self.canvas and self.parentEdge:
            self.parentEdge.setPoints(self.x()+self.width()/2,self.y()+3,
                        self.parent.x()+self.parent.width()/2,self.parent.y()+ \
                        self.parent.height())

    def setSelectionBox(self):
        self.isSelected=True
        if self.selectionSquare:
             self.selectionSquare = sl=self.selectionSquare
        else:
             self.selectionSquare = sl = [QCanvasLine(self.canvas()) for i in range(8)]
             self.canvasObj.extend(self.selectionSquare)
        for line in sl:
            line.setZ(-5)
            line.move(self.x(),self.y())
            line.setPen(QPen(QColor(0, 0, 150), 3))
        xleft = -3; xright = self.width() + 2
        yup = -3; ydown = self.height() + 2
        xspan = self.width() / 4; yspan = self.height() / 4
        sl[0].setPoints(xleft, yup, xleft + xspan, yup)
        sl[1].setPoints(xleft, yup-1, xleft, yup + yspan)
        sl[2].setPoints(xright, yup, xright - xspan, yup)
        sl[3].setPoints(xright, yup-1, xright, yup + yspan)
        sl[4].setPoints(xleft, ydown, xleft + xspan, ydown)
        sl[5].setPoints(xleft, ydown+2, xleft, ydown - yspan)
        sl[6].setPoints(xright, ydown, xright - xspan, ydown)
        sl[7].setPoints(xright, ydown+2, xright, ydown - yspan)
        if self.isShown:
            self.show()

    def removeSelectionBox(self):
        self.isSelected=False
        for l in self.selectionSquare:
            l.setCanvas(None)
            self.canvasObj.remove(l)
        self.selectionSquare=[]



def bubbleConstructor(node=None, pos =None, canvas=None):
    return CanvasBubbleInfo(node, pos, canvas)

class TreeCanvasView(QCanvasView):
    def __init__(self, master, canvas, *args):
        apply(QCanvasView.__init__,(self,canvas)+args)
        self.master=master
        self.dropplet=None
        self.selectedNode=None
        self.bubble=None
        self.bubbleNode=None
        self.navigator=None
        self.viewport().setMouseTracking(True)
        self.bubbleConstructor=bubbleConstructor

    def setNavigator(self, nav):
        self.navigator=nav
        self.master.connect(self.canvas(),SIGNAL("resized()"),self.navigator.resizeCanvas)
        self.master.connect(self ,SIGNAL("contentsMoving(int,int)"),self.navigator.moveView)

    def resizeEvent(self, event):
        QCanvasView.resizeEvent(self, event)
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
        self.canvas().update()

    def updateBubble(self, node=None, pos=QPoint(0,0)):
        if self.bubbleNode==node and self.bubble:
            self.bubble.move(pos.x()+5,pos.y()+5)
            self.bubble.show()
        elif node:
            if self.bubble:
                self.bubble.setCanvas(None)
            self.bubbleNode=node
            self.bubble=self.bubbleConstructor(node, pos, self.canvas())
            self.bubble.move(pos.x()+5,pos.y()+5)
            self.bubble.setZ(50)
            self.bubble.show()
        elif self.bubble:
            self.bubble.setCanvas(None)
            self.bubble=self.bubbleNode=None
        self.canvas().update()

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

    def contentsMouseMoveEvent(self,event):
        obj=self.canvas().collisions(event.pos())
        obj=filter(lambda a:a.z()==-21 or a.z()==-20,obj)
        if not obj:
            self.updateDropplet()
            self.updateBubble()
        elif isinstance(obj[0], QCanvasRectangle) and self.master.NodeBubblesEnabled:
            self.updateBubble(obj[0],event.pos())
            self.updateDropplet()
        elif obj[0].__class__==QCanvasEllipse:
            self.updateDropplet(obj[0])
        else:
            self.updateDropplet()
            self.updateBubble()

    def contentsMousePressEvent(self, event):
        if self.dropplet:
            self.dropplet.node.setOpen(not self.dropplet.node.isOpen,
                (event.button()==QEvent.RightButton and 1) or -1)
            self.canvas().fixPos()
        else:
            obj=self.canvas().collisions(event.pos())
            obj=filter(lambda a:a.z()==-20, obj)
            if obj and isinstance(obj[0], QCanvasRectangle):
                self.updateSelection(obj[0])
        self.canvas().update()


class TreeCanvas(QCanvas):
    def __init__(self, parent, *args):
        apply(QCanvas.__init__,(self,)+args)
        self.HSpacing=10
        self.VSpacing=10
        self.parent=parent
        self.nodeList=[]
        self.edgeList=[]

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
            n.setCanvas(None)
        self.nodeList=[]

    def fixPos(self, node=None, x=10, y=10):
        self.gx=x
        self.gy=y
        if not node: node=self.nodeList[0]
        if not x or not y: x, y= self.HSpacing, self.VSpacing
        self._fixPos(node,x,y)
        self.resize(self.gx+ExpectedBubbleWidth, self.gy+ExpectedBubbleHeight)

    def _fixPos(self, node, x, y):
        ox=x
        if node.nodeList and node.isOpen:
            for n in node.nodeList:
                (x,ry)=self._fixPos(n,x,y+self.VSpacing+node.height())
            x=(node.nodeList[0].rect().x()+node.nodeList[-1].rect().x())/2
            node.move(x,y)
            for n in node.nodeList:
                n.updateEdge()
        else:
            node.move(self.gx,y)
            self.gx+=self.HSpacing+node.width()
            x+=self.HSpacing+node.width()
            self.gy=max([y,self.gy])

        return (x,y)

class TreeNavigator(TreeCanvasView):
    class NavigatorNode(CanvasNode):
        def __init__(self,masterNode,*args):
            CanvasNode.__init__(self, *args)
            self.masterNode=masterNode
        def leech(self):
            self.isOpen=self.masterNode.isOpen
            self.setBrush(self.masterNode.brush())

    def __init__(self, masterView, *args):
        apply(TreeCanvasView.__init__,(self,)+args)
        self.myCanvas=self.canvas()
        self.masterView=masterView
        self.canvas().resize(self.width(),self.height())
        self.rootNode=None
        self.updateRatio()
        self.viewRect=QCanvasRectangle(self.canvas())
        self.viewRect.setSize(self.rx*self.masterView.width(), self.ry*self.masterView.height())
        self.viewRect.show()
        self.viewRect.setBrush(QBrush(Qt.lightGray))
        self.viewRect.move(50,50)
        self.viewRect.setZ(-10)
        self.buttonPressed=False
        self.bubbleConstructor=self.myBubbleConstructor
        self.isShown=False

    def leech(self):
        if not self.isShown:
            return
        self.updateRatio()
        if self.rootNode!=self.masterView.canvas().nodeList[0]: #display a new tree
            self.canvas().clear()
            self.rootNode=self.walkcreate(self.masterView.canvas().nodeList[0], None)
            self.walkupdate(self.rootNode)
        else:
            self.walkupdate(self.rootNode)
        self.canvas().update()


    def walkcreate(self, masterNode, parent):
        node=self.NavigatorNode(masterNode, masterNode.tree, parent or self.canvas(), self.canvas())
        node.setZ(0)
        for n in masterNode.nodeList:
            self.walkcreate(n,node)
        return node

    def walkupdate(self, node):
        node.leech()
        if node.masterNode.isShown:
            node.show()
            node.move(self.rx*node.masterNode.x(), self.ry*node.masterNode.y())
            node.setSize(self.rx*node.masterNode.width(), self.ry*node.masterNode.height())
            for n in node.nodeList:
                self.walkupdate(n)
                n.updateEdge()
        else:
            node.hideSubtree()

    def contentsMousePressEvent(self, event):
        if self.canvas().onCanvas(event.pos()):
            self.masterView.center(event.pos().x()/self.rx,event.pos().y()/self.ry)
        self.buttonPressed=True

    def contentsMouseReleaseEvent(self, event):
        self.buttonPressed=False

    def contentsMouseMoveEvent(self, event):
        if self.buttonPressed:
            self.masterView.center(event.pos().x()/self.rx, event.pos().y()/self.ry)
            self.updateBubble(None)
        else:
            obj=self.canvas().collisions(event.pos())
            if obj and obj[0].__class__==self.NavigatorNode:
                self.updateBubble(obj[0], event.pos())
            else:
                self.updateBubble()

    def viewportResizeEvent(self, event):
        self.canvas().resize(event.size().width(), event.size().height())
        self.leech()
        self.updateView()
        self.canvas().update()

    def resizeView(self):
        self.updateRatio()
        self.updateView()
        self.canvas().update()

    def resizeCanvas(self):
        self.updateRatio()
        if self.rootNode:
            self.leech()
            self.updateView()
            self.canvas().update()

    def moveView(self, x,y):
        self.updateRatio()
        self.viewRect.move(x*self.rx, y*self.ry)
        self.viewRect.setSize(self.masterView.width()*self.rx, self.masterView.height()*self.ry)
        self.canvas().update()

    def updateRatio(self):
        self.rx=float(self.canvas().width())/float(self.masterView.canvas().width())
        self.ry=float(self.canvas().height())/float(self.masterView.canvas().height())

    def updateView(self):
        self.viewRect.move(self.masterView.contentsX()*self.rx, self.masterView.contentsY()*self.ry)
        self.viewRect.setSize(self.masterView.width()*self.rx, self.masterView.height()*self.ry)

    def myBubbleConstructor(self, node, pos, canvas):
        return self.masterView.bubbleConstructor(node.masterNode, pos, canvas)


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
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')

        # GENERAL TAB
        GeneralTab = QVGroupBox(self)

        OWGUI.hSlider(GeneralTab, self, 'Zoom', box='Zoom', minValue=1, maxValue=10, step=1,
                      callback=self.toggleZoomSlider, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'VSpacing', box='Vertical Spacing', minValue=1, maxValue=10, step=1,
                      callback=self.toggleVSpacing, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'HSpacing', box='Horizontal Spacing', minValue=1, maxValue=10, step=1,
                      callback=self.toggleHSpacing, ticks=1)

        # OWGUI.checkBox(GeneralTab, self, 'ZoomAutoRefresh', 'Auto Refresh After Zoom',
        #                tooltip='Refresh after change of zoom setting?')
        # OWGUI.checkBox(GeneralTab, self, 'AutoArrange', 'Auto Arrange',
        #                tooltip='Auto arrange the position of the nodes\nafter any change of nodes visibility')
        OWGUI.checkBox(GeneralTab, self, 'NodeBubblesEnabled', 'Node bubbles',
                       tooltip='When mouse over the node show info bubble')
        OWGUI.checkBox(GeneralTab, self, 'TruncateText', 'Truncate text to fit margins',
                       tooltip='Truncate any text to fit the node width',
                       callback=self.toggleTruncateText)
        

        OWGUI.hSlider(GeneralTab, self, 'Zoom', box='Zoom', minValue=1, maxValue=10, step=1, callback=self.toggleZoomSlider, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'VSpacing', box='Vertical Spacing', minValue=1, maxValue=10, step=1, callback=self.toggleVSpacing, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'HSpacing', box='Horizontal Spacing', minValue=1, maxValue=10, step=1, callback=self.toggleHSpacing, ticks=1)

        #OWGUI.checkBox(GeneralTab, self, 'ZoomAutoRefresh', 'Auto Refresh After Zoom', tooltip='Refresh after change of zoom setting?')
        #OWGUI.checkBox(GeneralTab, self, 'AutoArrange', 'Auto Arrange', tooltip='Auto arrange the position of the nodes\nafter any change of nodes visibility')
        OWGUI.checkBox(GeneralTab, self, 'NodeBubblesEnabled', 'Node Bubbles', tooltip='When mouse over the node show info bubble')
        OWGUI.checkBox(GeneralTab, self, 'TruncateText', 'Truncate Text To Fit Margins', tooltip='Truncate any text to fit the node width', callback=self.toggleTruncateText)

        self.infBox = QVGroupBox(GeneralTab)
        self.infBox.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        self.infBox.setTitle('Tree Size')
	self.infoa = QLabel('No tree.', self.infBox)
        self.infob = QLabel('', self.infBox)

        self.tabs.insertTab(GeneralTab, "General")

        # TREE TAB
        TreeTab = QVGroupBox(self)
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
        self.tabs.insertTab(TreeTab, "Tree")

        # NODE TAB
        NodeTab = QVGroupBox(self)
        # Node size options
        OWGUI.hSlider(NodeTab, self, 'NodeSize', box='Node Width',
                      minValue=1, maxValue=10, step=1,
                      callback=self.toggleNodeSize, ticks=1)

        # Node information
        OWGUI.button(self.controlArea, self, "Navigator", self.toggleNavigator, debuggingEnabled = 0)
        findbox = QHBox(self.controlArea)
        self.centerRootButton=OWGUI.button(findbox, self, "Find Root",
                                           callback=lambda :self.rootNode and self.canvasView.center(self.rootNode.x(), self.rootNode.y()))
        self.centerNodeButton=OWGUI.button(findbox, self, "Find Selected",
                                           callback=lambda :self.canvasView.selectedNode and \
                                     self.canvasView.center(self.canvasView.selectedNode.x(),
                                                            self.canvasView.selectedNode.y()))
        self.tabs.insertTab(NodeTab,"Node")
        self.NodeTab=NodeTab
        self.TreeTab=TreeTab
        self.GeneralTab=GeneralTab
        self.rootNode=None
        self.tree=None

    def scaleSizes(self):
        pass

    def toggleZoomSlider(self):
        self.rescaleTree()
        self.canvas.fixPos(self.rootNode,10,10)
        self.canvas.update()

    def toggleVSpacing(self):
        self.rescaleTree()
        self.canvas.fixPos(self.rootNode,10,10)
        self.canvas.update()

    def toggleHSpacing(self):
        self.rescaleTree()
        self.canvas.fixPos(self.rootNode,10,10)
        self.canvas.update()

    def toggleTruncateText(self):
        for n in self.canvas.nodeList:
           n.truncateText(self.TruncateText)
        self.canvas.update()

    def toggleTreeDepth(self):
        self.walkupdate(self.rootNode)
        self.canvas.fixPos(self.rootNode,10,10)
        self.canvas.update()

    def toggleLineWidth(self):
        for n in self.canvas.nodeList:
            if self.LineWidthMethod==0:
                width=self.LineWidth
            elif self.LineWidthMethod == 1:
                width = (n.tree.distribution.cases/self.tree.distribution.cases) * self.LineWidth
            elif self.LineWidthMethod == 2:
                width = (n.tree.distribution.cases/((n.parent!=self.canvas and \
                                    n.parent.tree.distribution.cases) or n.tree.distribution.cases)) * self.LineWidth
            n.setEdgeWidth(width)
        self.canvas.update()

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
        self.canvas.fixPos(self.rootNode,10,10)
        self.canvas.update()
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
            self.ClassColors=OWGraphTools.ColorPaletteHSV(len(self.tree.distribution))
            self.rootNode=self.walkcreate(self.tree, None)
            self.canvas.fixPos(self.rootNode,self.HSpacing,self.VSpacing)
            self.activateLoadedSettings()
            self.canvasView.center(self.rootNode.x(), self.rootNode.y())
            self.centerRootButton.setDisabled(0)
            self.centerNodeButton.setDisabled(1)

        self.canvas.update()

    def walkcreate(self, tree, parent=None, level=0):
        node=CanvasNode(tree, parent or self.canvas, self.canvas)
        if tree.branches:
            for i in range(len(tree.branches)):
                if tree.branches[i]:
                    self.walkcreate(tree.branches[i],node,level+1)
        return node

    def walkupdate(self, node, level=0):
        if self.MaxTreeDepthB and self.MaxTreeDepth<=level+1:
            node.setOpen(False)
            return
        else:
            node.setOpen(True,1)
        for n in node.nodeList:
            self.walkupdate(n,level+1)

    def clear(self):
        self.tree=None
        self.canvas.clear()
        self.treeNav.canvas().clear()

    def rescaleTree(self):
        k = 0.0028 * (self.Zoom ** 2) + 0.2583 * self.Zoom + 1.1389
        self.canvas.VSpacing=int(DefNodeHeight*k*(0.3+self.VSpacing*0.15))
        self.canvas.HSpacing=int(DefNodeWidth*k*(0.3+self.HSpacing*0.20))
        for r in self.canvas.nodeList:
            r.setSize(int(DefNodeWidth*k), int(DefNodeHeight*k))

    def updateSelection(self, node=None):
        self.selectedNode=node
        if node:
            self.centerNodeButton.setDisabled(0)
            self.send("Examples", node.tree.examples)
        else:
            self.centerNodeButton.setDisabled(1)
            self.send("Examples", None)

class OWDefTreeViewer2D(OWTreeViewer2D):
    def __init__(self, parent=None, signalManager = None, name='DefTreeViewer2D'):
        OWTreeViewer2D.__init__(self, parent, signalManager, name)
        self.settingsList=self.settingsList+["ShowPie"]

        self.canvas=TreeCanvas(self)
        self.canvasView=TreeCanvasView(self, self.canvas, self.mainArea, "CView")
        layout=QVBoxLayout(self.mainArea)
        layout.addWidget(self.canvasView)
        self.canvas.resize(800,800)
        self.navWidget=QWidget(None, "Navigator")
        self.navWidget.lay=QVBoxLayout(self.navWidget)
        canvas=TreeCanvas(self.navWidget)
        self.treeNav=TreeNavigator(self.canvasView,self,canvas,self.navWidget, "Nav")
        self.treeNav.setCanvas(canvas)
        self.navWidget.lay.addWidget(self.treeNav)
        self.canvasView.setNavigator(self.treeNav)
        self.navWidget.resize(400,400)
#        OWGUI.button(self.TreeTab,self,"Navigator",self.toggleNavigator)

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWDefTreeViewer2D()
    a.setMainWidget(ow)

    data = orange.ExampleTable('../../doc/datasets/voting.tab')
    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.activateLoadedSettings()
    ow.ctree(None)
    ow.ctree(tree)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()
    ow.saveSettings()


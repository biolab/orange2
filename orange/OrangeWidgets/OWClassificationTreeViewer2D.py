"""
<name>Classification Tree Viewer 2D</name>
<description>Classification Tree</description>
<category>Classification</category>
<icon>icons/ClassificationTreeViewer2D.png</icon>
<priority>2110</priority>
"""

## viewportToContents

import copy, time
import orange, orngTree
import OWGUI
#from string import *
from qt import *
from qtcanvas import *
from OWWidget import *
from qwt import *

ScreenSize = [640, 480]

PopupBubble = 2
PopupSubtree = 3
PopupFocus = 4
PopupRedraw = 5

RefreshBubble = False

# Resolucija canvasa v pixlih
# Uporabljeno za lovljenje mouse eventom
CanvasResolution = 3

# radius of a dropplet at the bottom of node with hidden childreen
DroppletRadious_Default = 7

# pricakovana velikost bubbla, da ge je videti tudi za robna vozlisca 
ExpectedBubbleWidth_Default = 200
ExpectedBubbleHeigth_Default = 200

# Zacetne nastavitve pisave
TextFont = 'Cyrillic'

BodyColor_Default = QColor(255, 225, 10)
BodyCasesColor_Default = QColor(0, 0, 128)

BubbleWidth_Default = 100
BubbleHeight_Default = 200

# BARVE
# barve razredov

class PopupWindow(QWidget):
    def __init__(self, *args):
        apply(QWidget.__init__, (self,) + args)
        self.setGeometry((ScreenSize[0]-300)/2, (ScreenSize[1]-100)/2, 300, 100)
        self.Label = QLabel(self, 'Just some label')
        self.Label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.Label.setFont(QFont(TextFont, 16))
        self.Label.resize(self.size())

    def setText(self, text):
        self.Label.setText(text)

# set items of text boxes (used in definition of node body and in bubbles)
def setTextSeparator(container, canvas):
    s = QCanvasLine(canvas)
    s.isSeparator = 1
    container.append(s)

def setTextLine(container, canvas, text, color=None, separator=0, font=None, truncated=0):
    line = QCanvasText(canvas)
    line.setText(text)
    line.isSeparator = 0
    if color:
        line.setColor(color)
    line.text = text  # this stores the full text of the label
    container.append(line)
    if separator: setTextSeparator(container, canvas)
##    elif truncated and 0:
##        if font:
##            line.setFont(font)
##        w = line.boundingRect().width()
##        if w > truncated:
##            for i in range(3): # this is approximite, as we compute the average char width and
##                nchars = int(len(text) * truncated / w)  # based on this reduce string
##                text = text[:nchars]                     # as the line may still be too long, we iterate
##                line.setText(text)
##                w = line.boundingRect().width()
##                if w < truncated: break
    return line

class BubbleInfo(QCanvasRectangle):
    def __init__(self, *args):
        apply(QCanvasRectangle.__init__, (self,) + args)
        self.canvas = args[0]
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setZ(60)
        self.bubbleShadow = QCanvasRectangle(self.canvas)
        self.bubbleShadow.setBrush(QBrush(QColor(128, 128, 128)))
        self.bubbleShadow.setPen(QPen(QColor(128, 128, 128)))
        self.bubbleShadow.setZ(50)
        
    # set the text container plus some separation lines
    def setText(self, master, node):
        self.text = []      # stores text lines of the bubble

        # rule that led to the node (IF ... THEN ...)
        if len(node.rule):
            text = 'IF ' + reduce(lambda x,y: x + ' AND\n    ' + y, node.rule) + '\n'
        else:
            text = ''
        classes = map(lambda i,c=master.classes: c[i], node.majClass)
        text += 'THEN ' + reduce(lambda x,y: x + ' OR\n           ' + y, classes)
        setTextLine(self.text, self.canvas, text, separator=1)
        
        # instances
        text = 'Instances: %d (%5.1f%s)' % (node.distribution.cases, 100. * node.distribution.cases / master.root.distribution.cases, '%')
        setTextLine(self.text, self.canvas, text, separator=1)
        
        # distribution
        distItems = node.distribution.items()
        for i in range(len(distItems)):
            dist = distItems[i]
            if dist[1] > 0:     # we show the class distribution only if non-zero instances
                text = "%s: %d (%5.1f%s)" % (master.classes[i], dist[1], 100. * dist[1] / node.distribution.cases, '%')
                setTextLine(self.text, self.canvas, text, color = master.ClassColors[i])

        setTextSeparator(self.text, self.canvas)

        # partition attribute info (or leaf)
        if node.branches:
            text = 'Partition on: ' + node.branchSelector.classVar.name
        else:
            text = '(leaf)'
        setTextLine(self.text, self.canvas, text)

        for line in self.text:
            line.setZ(70)

    def moveit(self, x, y):
        self.showit()
        self.move(x, y)
        self.bubbleShadow.move(x+5, y+5)
        distance = 0
        sizeX = 0
        sizeY = 0

        # set position of text and separators
        for text in self.text:
            if text.isSeparator:
                distance += 2
                text.setX(x)
                text.setY(y + distance)
                distance += 2
            else:
                text.setX(x + 2)
                text.setY(y + distance)
                sizeX = max(text.boundingRect().width(), sizeX)
                distance += text.boundingRect().height()

        # set the size of the bubble border
        sizeX += 5
        sizeY = distance + 5

        #dolocim dolzino crt
        for line in self.text:
            if line.isSeparator:
                line.setPoints(line.startPoint().x(), line.startPoint().y(), line.startPoint().x() + sizeX - 1, line.startPoint().y())
        self.setSize(sizeX , sizeY)
        self.bubbleShadow.setSize(sizeX, sizeY)

    def showit(self):
        self.show()
        self.bubbleShadow.show()
        for text in self.text:
            text.show()

    def hideit(self):
        self.hide()
        self.bubbleShadow.hide()
        for text in self.text:
            text.hide()


# self.doc.canvasDlg.ctrlPressed == 0

class MyCanvasView(QCanvasView):
    def __init__(self, *args):
        apply(QCanvasView.__init__,(self,) + args)
        #self.doc = doc
        self.canvas = args[0]
        self.viewport().setMouseTracking(True)
        
        (self.x, self.y) = (0,0) # coordinates of the last event
        self.dropplet = None     # last dropplet changed in color
        # Kazalec na trenutno odprti bubble
        self.bubbleShown = None
        # Vozlisce, na katerem trenutno izvajamo akcije
        self.selectedNode = None
        # Ali se bubble prikazuje?
        self.bubbleIsShown = True
        # Popup izbirni menu
        self.popup = QPopupMenu(self)
        self.popup.setCheckable(True)
        self.popup.insertItem("Show subtree", self.showSubtree, 0, PopupSubtree)
        self.popup.setItemChecked(PopupSubtree, True)
        self.popup.insertItem("Bubble enabled", self.showBubble, 0, PopupBubble)
        self.popup.setItemChecked(PopupBubble, True)

    def infoSet(self, master = None):
        self.master = master

    def coordInNode(self, node, x, y):
        return (node.x <= x) and ((node.x + self.master.BodyWidth + self.master.PieWidth/2) >= x) and (node.y <= y) and (node.y + self.master.BodyHeight >= y)

    # Skrije ali prikaze poddrevo trenutnega vozlisca
    def showSubtree(self):
        if self.selectedNode != None:
            if self.popup.isItemChecked(PopupSubtree) == True:
                TreeWalk_setVisible(self.selectedNode, False)
                self.popup.setItemChecked(PopupSubtree, False)
            else:
                TreeWalk_setVisible(self.selectedNode, True, True)
                self.popup.setItemChecked(PopupSubtree, True)
            self.canvas.update()
        self.viewport().setMouseTracking(self.bubbleIsShown)

    # Skrije ali prikaze bubble s podatki
    def showBubble(self):
        if self.popup.isItemChecked(PopupBubble) == True:
            self.bubbleIsShown = False
        else:
            self.bubbleIsShown = True
        self.viewport().setMouseTracking(self.bubbleIsShown)
        self.popup.setItemChecked(PopupBubble, self.bubbleIsShown)

    def contentsMousePressEvent(self, event):
        self.viewport().setMouseTracking(self.bubbleIsShown)
        if event.button() == QEvent.LeftButton:
            #if self.doc.canvasDlg.ctrlPressed:
            #    print 'CONTROL CRTL'
            items = filter(lambda ci: ci.z()<15 and ci.z()>5, self.canvas.collisions(event.pos()))
            if len(items)==0: # nothing significant was clicked on
                self.master.deselectNode() # deselect node (if it was selected)
                return
            node = items[0].node
            if items[0].z() == 10: # click on a node
                self.master.selectNode(node)
            else: # click on a dropplet
                if node.borderline:  # should open node's siblings, they are new borderline
                    node.borderline = 0
                    for sibling in node.branches:
                        self.master.visibleToBorderline(sibling)
                else:
                    node.borderline = 1
                    for sibling in node.branches:
                        self.master.hideTree(sibling)
                    if self.master.selectedNode and not self.master.selectedNode.visible:
                        self.master.deselectNode()
                if self.master.AutoArrange:
                    self.master.rescaleTree()
                self.canvas.update()

        elif event.button() == QEvent.RightButton:
            for node in master.nodes:
                if node.visible and self.coordInNode(node, event.x(), event.y()):
                    self.bubbleIsShown = self.viewport().hasMouseTracking()
                    self.viewport().setMouseTracking(False)
                    if self.bubbleShown != None:
                        self.bubbleShown.hideit()
                    self.selectedNode = node
                    self.popup.setItemChecked(PopupBubble, self.bubbleIsShown)
                    self.popup.popup(event.globalPos())
        self.canvas.update()

    def contentsMouseMoveEvent(self, event):
##        if abs(self.x - event.x()) <= CanvasResolution and abs(self.y - event.y()) <= CanvasResolution:
##            return
        items = filter(lambda ci: ci.z()<15 and ci.z()>5, self.canvas.collisions(event.pos()))
        if len(items) == 0: # mouse over nothing special
            if self.bubbleShown:
                self.bubbleShown.hideit()
                self.canvas.update()
            if self.dropplet:
                self.dropplet.setBrush(QBrush(Qt.gray))
                self.dropplet = None
                self.canvas.update()
        elif items[0].z() == 10: # mose over the node
            if not self.master.NodeBubblesEnabled:
                return
            node = items[0].node
            self.x = event.x()
            self.y = event.y()
            if node.bubble == None:  # create the bubble for the node if it does not exist yet
                if self.bubbleShown: # an old bubble is still on (mouse moved fast)
                    self.bubbleShown.hideit()
                node.bubble = BubbleInfo(self.canvas)
                node.bubble.setText(self.master, node)
                self.bubbleShown = node.bubble
                node.bubble.showit()
            elif self.bubbleShown: # a buble is visible
                if self.bubbleShown <> node.bubble: # but it is not the right one (mouse moved fast)
                    self.bubbleShown.hideit()
                    self.bubbleShown = node.bubble
                    self.bubbleShown.showit()

            node.bubble.moveit(event.x()+20, event.y()+20)
            self.canvas.update()
        else: # mouse over the dropplet
            dropplet = items[0].node.dropplet
            if dropplet <> self.dropplet:
                if self.dropplet:
                    self.dropplet.setBrush(QBrush(Qt.gray))
                self.dropplet = dropplet
                dropplet.setBrush(QBrush(Qt.black))
                self.canvas.update()


##############################################################################
# main class

PieWidth_Default = 15
PieHeight_Default = 15
BodyWidth_Default = 30
BodyHeight_Default = 20
AttributeBoundBoxHeight_Default = 6
AttributeTextSize_Default = AttributeBoundBoxHeight_Default * 0.65

import warnings
warnings.filterwarnings("ignore", ".*TreeNode.*", orange.AttributeWarning)

class OWClassificationTreeViewer2D(OWWidget):	
    settingsList = ["ZoomAutoRefresh", "AutoArrange", "NodeBubblesEnabled",
                    "Zoom", "VSpacing", "HSpacing", "MaxTreeDepth", "MaxTreeDepthB",
                    "LineWidth", "LineWidthMethod",
                    "NodeSize", "NodeInfo", "NodeInfoSorted", "NodeColorMethod",
                    "ShowPies", "TruncateText"]

    def __init__(self, parent=None, name='ClassificationTreeViewer2D'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.canvas = None
        self.root = None
        self.selectedNode = None
        OWWidget.__init__(self, parent, name, 'A graphical 2D view of a classification tree.', FALSE, FALSE) 
        
        self.addInput("target")
        self.addInput("ctree")

        # some globaly used variables (get rid!)
        self.TargetClassIndex = 0

        #set default settings
        self.ZoomAutoRefresh = 0
        self.AutoArrange = 0
        self.NodeBubblesEnabled = 1
        self.MaxTreeDepth = 5; self.MaxTreeDepthB = 0
        self.LineWidth = 5; self.LineWidthMethod = 0
        self.NodeSize = 5
        self.NodeInfo = []
        
        self.NodeColorMethod = 0
        self.ShowPies = 1
        self.Zoom = 5
        self.VSpacing = 5; self.HSpacing = 5
        self.TruncateText = 1

        self.loadSettings()
        self.NodeInfoSorted = copy.copy(self.NodeInfo)
        self.NodeInfoSorted.sort()
        self.scaleSizes()

        # GUI definition
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')

        # GENERAL TAB
        GeneralTab = QVGroupBox(self)

        OWGUI.hSlider(GeneralTab, self, 'Zoom', box='Zoom', minValue=1, maxValue=10, step=1, callback=self.toggleZoomSlider, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'VSpacing', box='Vertical Spacing', minValue=1, maxValue=10, step=1, callback=self.toggleVSpacing, ticks=1)
        OWGUI.hSlider(GeneralTab, self, 'HSpacing', box='Horizontal Spacing', minValue=1, maxValue=10, step=1, callback=self.toggleHSpacing, ticks=1)

        OWGUI.checkOnly(GeneralTab, self, 'Auto Refresh After Zoom', 'ZoomAutoRefresh', tooltip='Refresh after change of zoom setting?')
        OWGUI.checkOnly(GeneralTab, self, 'Auto Arrange', 'AutoArrange', tooltip='Auto arrange the position of the nodes\nafter any change of nodes visibility')
        OWGUI.checkOnly(GeneralTab, self, 'Node Bubbles', 'NodeBubblesEnabled', tooltip='When mouse over the node show info bubble')
        OWGUI.checkOnly(GeneralTab, self, 'Truncate Text To Fit Margins', 'TruncateText', tooltip='Truncate any text to fit the node width', callback=self.toggleTruncateText)
        
        self.tabs.insertTab(GeneralTab, "General")

        # TREE TAB
        TreeTab = QVGroupBox(self)
        OWGUI.checkWithSpin(TreeTab, self, 'Max Tree Depth:', 1, 20, 'MaxTreeDepthB', "MaxTreeDepth", tooltip='Defines the depth of the tree displayed', checkCallback=self.toogleTreeDepth, spinCallback=self.toogleTreeDepth)
        OWGUI.labelWithSpin(TreeTab, self, 'Max Line Width:', min=1, max=10, value='LineWidth', step = 1, tooltip='Defines max width of the edges that connect tree nodes', callback=self.toggleLineWidth)
        OWGUI.radioButtonsInBox(TreeTab, self, 'Baseline for Line Width', ['No Dependency', 'Root Node', 'Parent Node'], 'LineWidthMethod',
                                tooltips=['All edges are of the same width', 'Line width is relative to number of cases in root node', 'Line width is relative to number of cases in parent node'],
                                callback=self.toggleLineWidth)
        self.tabs.insertTab(TreeTab, "Tree")

        # NODE TAB
        NodeTab = QVGroupBox(self)
        # Node size options
        # OWGUI.radioButtonsInBox(NodeTab, self, 'Node Size', ['Small', 'Medium', 'Big'], 'NodeSize')
        OWGUI.hSlider(NodeTab, self, 'NodeSize', box='Node Width', minValue=1, maxValue=10, step=1, callback=self.toggleNodeSize, ticks=1)
        # Node information
        nodeInfoBox = QVButtonGroup("Show Info On", NodeTab)
        nodeInfoButtons = ['Majority Class', 'Majority Class Probability', 'Target Class Probability', 'Number of Instances']
        self.NodeInfoW = []; self.dummy = 0
        for i in range(len(nodeInfoButtons)):
            self.dummy = i in self.NodeInfo
            w = OWGUI.checkOnly(nodeInfoBox, self, nodeInfoButtons[i], 'dummy', callback=self.setNodeInfo, getwidget=1, id=i)
            self.NodeInfoW.append(w)
        
        OWGUI.radioButtonsInBox(NodeTab, self, 'Node Color', ['Default', 'Instances in Node', 'Majority Class Probability', 'Target Class Probability', 'Target Class Distribution'],
                                'NodeColorMethod',
                                tooltips=['Use the default color for all nodes in the tree.', 'The node color saturation of the color depends on the \nnumber of instances in node (lighter, fewer instances)', 'The saturation of depends on the \nprobability of the prevailing class',
                                          'The saturation of depends on the \nprobability of the target clas', 'Shows the distribution of instances with target class (100% at the root)\nthrough the nodes of the tree'],
                                callback=self.toggleNodeColor)
        # pies
        pieBox = QVButtonGroup("Pies",NodeTab)
        OWGUI.checkOnly(pieBox, self, 'Show Pies', 'ShowPies', tooltip='Show pie graph with class distribution?', callback=self.togglePies)

        self.tabs.insertTab(NodeTab, "Node")
        
        # refresh button at the bottom
        RefreshButton=QPushButton('Refresh', self.controlArea)
        #OWGUI.button(self.controlArea, self, 'Test', self.test)
        #self.connect(RefreshButton, SIGNAL('clicked()'), self.refresh)

        self.Popup = PopupWindow(None)
        self.setMouseTracking(True)

        self.canvasView = None
        self.layout=QVBoxLayout(self.mainArea)

    def test(self):
        pass

    ##########################################################################
    # callbacks (rutines called after some GUI event, like click on a button)

    def setNodeInfo(self, widget=None, id=None):
        if widget.isChecked():
            if len(self.NodeInfo) == 2:
                self.NodeInfoW[self.NodeInfo[0]].setChecked(0)
            self.NodeInfo.append(id)
        else:
            self.NodeInfo.remove(id)
        self.NodeInfoSorted = copy.copy(self.NodeInfo)
        self.NodeInfoSorted.sort()

        if self.root:
            self.setTreeParam()
            self.truncateTreeText()
            self.canvas.update()

    def toggleZoomSlider(self):
        self.scaleSizes()
        if self.ZoomAutoRefresh and self.canvasView and self.root:
            self.rescaleTree()
            self.canvas.update()

    def toggleVSpacing(self):
        if self.canvasView and self.root:
            self.rescaleTreeY()
            self.canvas.update()

    def toggleHSpacing(self):
        if self.canvasView and self.root:
            self.rescaleTreeX()
            self.canvas.update()

    def toggleNodeSize(self):
        self.truncateTreeText()
        pass

    def togglePies(self):
        if self.root:
            self.showTreePies()
            self.canvas.update()

    def toggleLineWidth(self):
        if self.root:
            self.setTreeEdges(self.root)
            self.canvas.update()

    def toggleNodeColor(self):
        self.setColorOfNodes()
        self.canvas.update()

    def toogleTreeDepth(self):
        if self.root:
            self.setTreeVisibility()
            if self.AutoArrange:
                self.rescaleTree()
            self.showTree()
            self.canvas.update()

    def toggleTruncateText(self):
        self.truncateTreeText()
        self.canvas.update()

    ##########################################################################
    # handling of input signals
    import time
    def ctree(self, tree):
        self.tree = tree
        self.root = tree.tree
        self.classes = [x[0] for x in self.root.distribution.items()]
        self.numInstances = self.root.distribution.cases
        
        if self.canvas != None:
            for i in self.canvas.allItems():
                i.setCanvas(None)
        else:
            self.canvas = QCanvas(200, 200)
            self.canvasView = MyCanvasView(self.canvas, self.mainArea)
            self.canvasView.infoSet(master = self)
            self.layout.add(self.canvasView)

        self.ClassColors = []
        for i in range(len(self.classes)):
            newColor = QColor()
            newColor.setHsv(i*360/len(self.classes), 255, 255)
            self.ClassColors.append(newColor)

        # Annotate Orange tree and show it
        self.Popup.setText('Drawing tree, please wait...')
        self.Popup.show()
        self.buildTree2D()
        self.Popup.hide()

    def target(self, target):
        self.TargetClassIndex = target
        self.refresh()

    ##########################################################################
    # definition of a tree

    def buildTree2D(self):
        # Annotate tree nodes (with attributes specific to 2D drawing)
        self.nodes = []
        self.annotateTree(self.root)
        self.constructNodeSelection()
        self.setTreeVisibility()
        
        self.rescaleTree()
        self.setTreeEdges(self.root)
        self.setColorOfNodes()
        self.showTree()

    # selectedSortAttribute = node.branchSelector.classVar.name
    # selectedAttributeVal =  node.branchDescriptions
    # numCases = node.distribution.cases
    def annotateTree(self, node, prevAttributeVal=None, prevAttributeName=None, rule=[], depth=0):
        self.nodes.append(node)

        node.prevAttributeName = prevAttributeName
        node.prevAttributeVal = prevAttributeVal
        node.bubble = None
        node.inverted = 0
        node.depth = depth + 1
        node.borderline = node.branches and self.MaxTreeDepthB and (node.depth == self.MaxTreeDepth)

        crule = copy.copy(rule)
        if prevAttributeName <> None:
            crule.append(prevAttributeName + ' = ' + prevAttributeVal)
        node.rule = crule
        
        # finds indices of classes for which instances is in majority
        majClassInstances = max(node.distribution)
        node.majClass = filter(lambda i, m=majClassInstances: node.distribution[i]==m, range(len(node.distribution)))

        # info ['Majority Class', 'Majority Class Probability', 'Target Class Probability', 'Number of Instances']
        node.majClassName = node.distribution.items()[node.majClass[0]][0]
        node.majClassProb = max(node.distribution) / node.distribution.cases
        node.targetClassProb = node.distribution[self.TargetClassIndex] / node.distribution.cases
        node.info = [node.majClassName, "%5.3f" % node.majClassProb, "%5.3f" % node.targetClassProb, node.distribution.cases]

        if node.branches:
            node.name = node.branchSelector.classVar.name
        else:
            node.name = node.majClassName
        
        self.drawNode(node)
        
        # recurse on childreen
        if node.treesize() > 1:
            for i in range(len(node.branches)):
                self.annotateTree(node.branches[i], node.branchDescriptions[i], node.branchSelector.classVar.name, crule, depth+1)

    def setTreeVisibility(self):
        for node in self.nodes:
            node.visible = (not self.MaxTreeDepthB) or (node.depth <= self.MaxTreeDepth)
            node.borderline = node.branches and self.MaxTreeDepthB and (node.depth == self.MaxTreeDepth)
        if self.selectedNode and not self.selectedNode.visible:
            self.deselectNode()


    ##########################################################################
    # CONSTRUCTION OF VISUAL ELEMENTS OF THE TREE
    
    def constructNodeBody(self, node):
        node.body = QCanvasRectangle(self.canvas)
        node.body.node = node # used for event handling (canvas returns an object mouse is on)
        node.body.setBrush(QBrush(BodyColor_Default))
        node.body.setZ(10)
        if node.branches:
            node.dropplet = QCanvasEllipse(self.canvas)
            node.dropplet.node = node
            node.dropplet.setBrush(QBrush(Qt.gray))
            node.dropplet.setZ(9)

    def setNodeParam(self ,node):
        for i in range(2):
            if i < len(self.NodeInfoSorted):
                node.parameter[i].setText(str(node.info[self.NodeInfoSorted[i]]))
                node.Texts[i+1].text = str(node.info[self.NodeInfoSorted[i]])
            else:
                node.parameter[i].setText('')
                node.Texts[i+1].text = ''

    def setTreeParam(self):
        for node in self.nodes:
            self.setNodeParam(node)

    def constructNodeText(self, node):
        node.Texts = []
        setTextLine(node.Texts, self.canvas, node.prevAttributeVal)
        node.parameter = [None] * 2
        for i in range(2):
            node.parameter[i] = setTextLine(node.Texts, self.canvas, '')  # this is a bit wastefull
        self.setNodeParam(node)
        
        # separator
        setTextSeparator(node.Texts, self.canvas)
        
        # the name of the partition attribute
        setTextLine(node.Texts, self.canvas, node.name)
        
        for t in node.Texts:
            if not t.isSeparator:
                t.setFont(self.nodeFont) # this is also a bit wastefull (sets font for second time for some)
            t.setZ(30)

    def constructInEdge(self, node):
        node.edge = QCanvasLine(self.canvas)
        node.edge.setZ(3)

    def constructNodePie(self, node):
        node.pieFrame = QCanvasEllipse(self.canvas)
        node.pieFrame.setBrush(QBrush(Qt.black))
        node.pieFrame.setZ(50)
        
        # construction of sections of pie
        node.pies=[]
        startAngle = 0
        i = 0
        for count in node.distribution:
            if count > 0:
                sizeAngle = (count / node.distribution.cases) * 360
                pie = QCanvasEllipse(self.canvas)
                pie.setAngles(startAngle*16, sizeAngle*16)
                pie.setBrush(QBrush(self.ClassColors[i]))
                pie.setZ(51)
                node.pies.append(pie)
                startAngle += sizeAngle
            i += 1

    def drawNode(self, node):
        self.constructNodeBody(node)
        self.constructNodeText(node)
        self.constructInEdge(node)
        self.constructNodePie(node)

    ##########################################################################
    # SET PROPERTIES OF NODES AND EDGES (LIKE COLORS, LINE WIDTHS)...

    # light: 100 is neutral, 150 is 50% lighter
    def setNodeInfoColor(self, node, color):
        qc = QColor(color)
        for i in range(2):
            node.parameter[i].setColor(qc)
        node.Texts[3].setPen(QPen(color, 1))
        node.Texts[4].setColor(qc)

    def checkInvertColor(self, node, reference, light):
        if light <= reference: # darker
            if not node.inverted:
                self.setNodeInfoColor(node, Qt.white)
                node.inverted = 1
        else:
            if node.inverted:
                self.setNodeInfoColor(node, Qt.black)
                node.inverted = 0

    def setColorOfNodes(self):
        for node in self.nodes:
            body = node.body
            if self.NodeColorMethod == 0:   # default
                body.setBrush(QBrush(BodyColor_Default))
            elif self.NodeColorMethod == 1: # instances in node
                light = 400 - 300*node.distribution.cases/self.numInstances
                body.setBrush(QBrush(BodyCasesColor_Default.light(light)))
                self.checkInvertColor(node, 200, light)
            elif self.NodeColorMethod == 2: # majority class probability
                body.setBrush(QBrush(self.ClassColors[node.majClass[0]].light(400 - 300 * node.majClassProb)))
            elif self.NodeColorMethod == 3: # target class probability
                body.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(400 - 300*node.targetClassProb)))
            elif self.NodeColorMethod == 4: # target class distribution
                body.setBrush(QBrush(self.ClassColors[self.TargetClassIndex].light(400 - 300*node.distribution[self.TargetClassIndex]/self.root.distribution[self.TargetClassIndex])))
            if self.NodeColorMethod <> 1:
                if node.inverted:
                    self.setNodeInfoColor(node, Qt.black)
                    node.inverted = 0

    def setTreeEdges(self, node):
        if node.branches:
            for sibling in node.branches:
                if self.LineWidthMethod == 0:
                    width = self.LineWidth
                elif self.LineWidthMethod == 1:
                    width = (sibling.distribution.cases/self.root.distribution.cases) * self.LineWidth
                elif self.LineWidthMethod == 2:
                    width = (sibling.distribution.cases/node.distribution.cases) * self.LineWidth
                sibling.edge.setPen(QPen(Qt.gray, width))
                self.setTreeEdges(sibling)

    ##########################################################################
    # SET THE SIZE AND POSITION OF GRAPHICAL ELEMENTS

    def scaleSizes(self): # zz1
        # we got the following from: k(1)=1.4, k(5)=2.5, k(10)=4
        k = 0.0028 * (self.Zoom ** 2) + 0.2583 * self.Zoom + 1.1389
        self.BodyWidth = BodyWidth_Default * k
        self.BodyHeight = BodyHeight_Default * k
        self.PieWidth = PieWidth_Default * k
        self.PieHeight = PieHeight_Default * k
        self.AttributeBoundBoxHeight = AttributeBoundBoxHeight_Default * k
        self.DroppletRadious = DroppletRadious_Default * k
        self.AttributeTextSize = self.AttributeBoundBoxHeight * 0.65
        
        self.nodeWidth = self.BodyWidth + self.PieWidth/2
        self.xNodeToNode = self.nodeWidth * (1.0 + self.HSpacing * 0.15)
        self.nodeHeight = self.BodyHeight
        self.yNodeToNode = self.nodeHeight * (1.3 + self.VSpacing * 0.2)
        self.nodeFont = QFont(TextFont, self.AttributeTextSize)

    def setSizes(self):
        for node in self.nodes:
            node.body.setSize(self.BodyWidth, self.BodyHeight)
            node.pieFrame.setSize(self.PieWidth + 2, self.PieHeight + 2)
            for pie in node.pies:
                pie.setSize(self.PieWidth, self.PieHeight)
            for text in node.Texts:
                if not text.isSeparator:
                    text.setFont(QFont(TextFont, self.AttributeTextSize))
            if node.branches:
                node.dropplet.setSize(self.DroppletRadious, self.DroppletRadious)
        self.setNodeSelectionSize()

    def setNodePositionX(self, node, x):
        node.x = x
        node.body.setX(x)
        for text in node.Texts:
            text.setX(x + 3)
        node.Texts[3].setX(x)
        node.Texts[3].setPoints(0, 0, self.BodyWidth-1, 0)
        for pie in [node.pieFrame] + node.pies:
            pie.setX(x + self.BodyWidth)
        if node.branches:
            node.dropplet.setX(x+self.BodyWidth/2)
    
    def setNodePositionY(self, node, y):
        node.y = y
        node.body.setY(y)
        node.Texts[0].setY(y-self.AttributeBoundBoxHeight)
        node.Texts[1].setY(y+1)
        node.Texts[2].setY(y+1+self.AttributeBoundBoxHeight)
        node.Texts[4].setY(y+2+self.AttributeBoundBoxHeight*2)
        node.Texts[3].setY(y+1+self.AttributeBoundBoxHeight*2)
        for pie in [node.pieFrame] + node.pies:
            pie.setY(y + self.BodyHeight/2)
        if node.branches:
            node.dropplet.setY(y+self.BodyHeight)

    def setNodePosition(self, node, x, y):
        self.setNodePositionX(node, x)
        self.setNodePositionY(node, y)
    
    def setTreePositions(self, node, level=0):
        y = 10 + level * self.yNodeToNode
        if (not node.branches) or (self.AutoArrange and node.borderline): # finish with last visible node
            x = self.offset
            self.offset = self.offset + self.xNodeToNode
        else:
            for child in node.branches:
                self.setTreePositions(child, level + 1)
            x = (node.branches[0].x+node.branches[-1].x) * 0.5 #- self.nodeWidth
        self.setNodePosition(node, x, y)
        self.maxX = max(self.maxX, x + self.nodeWidth)
        self.maxY = max(self.maxY, y + self.nodeHeight)

    def setTreePositionsX(self, node, level=0):
        if (not node.branches) or (self.AutoArrange and node.borderline): # finish with last visible node
            x = self.offset
            self.offset = self.offset + self.xNodeToNode
        else:
            for child in node.branches:
                self.setTreePositions(child, level + 1)
            x = (node.branches[0].x+node.branches[-1].x) * 0.5 #- self.nodeWidth
        self.setNodePositionX(node, x)
        self.maxX = max(self.maxX, x + self.nodeWidth)

    def setTreePositionsY(self, node, level=0):
        y = 10 + level * self.yNodeToNode
        self.setNodePositionY(node, y)
        self.maxY = max(self.maxY, y + self.nodeHeight)
        if not((not node.branches) or (self.AutoArrange and node.borderline)):
            for child in node.branches:
                self.setTreePositionsY(child, level + 1)

    # set position of outgoing edges
    def setEdgesPositions(self, node):
        if node.branches and not (self.AutoArrange and node.borderline):
            [x1, y1] = [node.x + self.BodyWidth/2, node.y + self.BodyHeight]  # CCC -5
            for sibling in node.branches:
                [x2, y2] = [sibling.x + self.BodyWidth/2, sibling.y + 5]
                sibling.edge.setPoints(x1, y1, x2, y2)
                self.setEdgesPositions(sibling)

    def rescaleTree(self):
        self.maxX = 0; self.maxY = 0; self.offset = 10
        self.setSizes()
        self.setTreePositions(self.root)
        if self.selectedNode:
            self.setSelectedNodePosition()
        self.setEdgesPositions(self.root)
        self.truncateTreeText()
        self.canvas.resize(self.maxX + ExpectedBubbleWidth_Default, self.maxY + ExpectedBubbleHeigth_Default)

    def rescaleTreeX(self):
        self.maxX = 0; self.offset = 10
        self.scaleSizes()
        self.setTreePositionsX(self.root)
        if self.selectedNode:
            self.setSelectedNodePosition()
        self.setEdgesPositions(self.root)
        self.canvas.resize(self.maxX + ExpectedBubbleWidth_Default, self.maxY + ExpectedBubbleHeigth_Default)

    def rescaleTreeY(self):
        self.maxY = 0
        self.scaleSizes()
        self.setTreePositionsY(self.root)
        if self.selectedNode:
            self.setSelectedNodePosition()
        self.setEdgesPositions(self.root)
        self.canvas.resize(self.maxX + ExpectedBubbleWidth_Default, self.maxY + ExpectedBubbleHeigth_Default)
        
    def truncateTreeText(self):
        maxTextWidth = self.TruncateText and (self.BodyWidth - 10)
        # assumes that the font of the text is set appropriately
        for node in self.nodes:
                for i in [1,2,4]:
                    text = node.Texts[i]
                    label = text.text
                    text.setText(label)
                    if self.TruncateText:
                        w = text.boundingRect().width()
                        if w > maxTextWidth:
                            for i in range(3): # this is approximite, as we compute the average char width and
                                nchars = int(len(label) * maxTextWidth / w)  # based on this reduce string
                                label = label[:nchars]                     # as the line may still be too long, we iterate
                                text.setText(label)
                                w = text.boundingRect().width()
                                if w < maxTextWidth: break

    ##########################################################################
    # HANDLE SELECTION OF THE NODE

    def constructNodeSelection(self): ## zzz
        sl = [QCanvasLine(self.canvas) for i in range(8)]
        for line in sl:
            line.setZ(50)
            line.setPen(QPen(QColor(0, 0, 150), 3))
        self.selectionSquare = sl

    def setNodeSelectionSize(self):
        sl = self.selectionSquare
        xleft = -3; xright = self.BodyWidth + 2
        yup = -3; ydown = self.BodyHeight + 2
        xspan = self.BodyWidth / 4; yspan = self.BodyHeight / 4
        sl[0].setPoints(xleft, yup, xleft + xspan, yup)
        sl[1].setPoints(xleft, yup-1, xleft, yup + yspan)
        sl[2].setPoints(xright, yup, xright - xspan, yup)
        sl[3].setPoints(xright, yup-1, xright, yup + yspan)
        sl[4].setPoints(xleft, ydown, xleft + xspan, ydown)
        sl[5].setPoints(xleft, ydown+2, xleft, ydown - yspan)
        sl[6].setPoints(xright, ydown, xright - xspan, ydown)
        sl[7].setPoints(xright, ydown+2, xright, ydown - yspan)

    def setSelectedNodePosition(self):
        (x, y) = (self.selectedNode.x, self.selectedNode.y)
        for line in self.selectionSquare:
            line.setX(x); line.setY(y)
            line.show()

    def deselectNode(self):
        if self.selectedNode:
            self.selectedNode = None
            for line in self.selectionSquare:
                line.hide()

    def selectNode(self, node): # zzz
        if node == self.selectedNode:
            self.deselectNode()
        else:
            self.selectedNode = node
            self.setSelectedNodePosition()
        self.canvas.update()

    ##########################################################################
    # SHOW / HIDE AND VISIBILITY OF NODES

    def showTreePies(self):
        for node in self.nodes:
            if node.visible and self.ShowPies:
                node.pieFrame.show()
                for pie in node.pies:
                    pie.show()
            else:
                node.pieFrame.hide()
                for pie in node.pies:
                    pie.hide()

    def showNode(self, node):
        node.visible = 1
        node.body.show()
        for text in node.Texts:
            text.show()
        node.edge.show()
        if node.branches:
            node.dropplet.show()
        if self.ShowPies:
            node.pieFrame.show()
            for pie in node.pies:
                pie.show()

    def hideNode(self, node):
        node.visible = 0
        node.body.hide()
        for text in node.Texts:
            text.hide()
        node.edge.hide()
        if node.branches:
            node.dropplet.hide()
        node.pieFrame.hide()
        for pie in node.pies:
            pie.hide()

    def showTree(self):
        for node in self.nodes:
            if node.visible:
                self.showNode(node)
            else:
                self.hideNode(node)

    def hideTree(self, node):
        self.hideNode(node)
        if node.branches:
            for sibling in node.branches:
                self.hideTree(sibling)

    def visibleToBorderline(self, node):
        self.showNode(node)
        if not node.borderline and node.branches:
            for sibling in node.branches:
                self.visibleToBorderline(sibling)

##################################################################################################
# TreeWalk constructs a visual definition of the tree that is a collection of the
# visual defines for nodes

if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWClassificationTreeViewer2D()
    a.setMainWidget(ow)

    data = orange.ExampleTable('voting')
    tree = orange.TreeLearner(data)
    ow.ctree(tree)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()
    ow.saveSettings()

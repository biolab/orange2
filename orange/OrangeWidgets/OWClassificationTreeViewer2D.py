"""
<name>Classification Tree Viewer 2D</name>
<description>Classification Tree</description>
<category>Classification</category>
<icon>icons/ClassificationTreeViewer2D.png</icon>
<priority>2110</priority>
"""

import sys, math
import orange, orngTree
from string import *
from qt import *
from qtcanvas import *
from OWWidget import *
from OWClassificationTreeViewer2DOptions import *

#sirina crke v pikslih, pomembno za resize bubbla
FontWidth_Default = 9

ZoomFactor_Default = 1
scalingFactor = ZoomFactor_Default / 100.0

ScreenSize = [640, 480]

PopupZupan = 1
PopupBubble = 2
PopupSubtree = 3
PopupFocus = 4
PopupRedraw = 5

RefreshBubble = False

# Resolucija canvasa v pixlih
# Uporabljeno za lovljenje mouse eventom
CanvasResolution = 3

# Zacetna povecava
ZoomFactor_Default = 1

# Najvecja velikost canvasa
# !!! Uporabnik lahko to povozi z novo nastavitvijo v okencu
MaxCanvasWidth = 3000

# Radij kroga pod vozliscem, iz katerega izhajajo povezave
ChildsCircleRadius_Default = 3

# pricakovana velikost bubbla, da ge je videti tudi za robna vozlisca 
ExpectedBubbleWidth_Default = 200
ExpectedBubbleHeigth_Default = 300

# Zacetne nastavitve pisave
TextFont_Default = 'Cyrillic'
AttributeTextSize_Default = 5
ClassTextSize_Default = 5

# Initial size of pie
PieWidth_Default = 15
PieHeight_Default = 15
# Initial size of body rectangle
BodyWidth_Default = 30
BodyHeight_Default = 20

BodyColor_Default = [255, 225, 10]

global BubbleWidth_Default
BubbleWidth_Default = 100
BubbleHeight_Default = 200
BubbleColor_Default = [135, 165, 223]
BubbleShadowColor_Default = [128, 128, 128]
BodyCasesColor_Default = [0, 0, 128]

# BARVE
# Barve povezav
LineColors = []
LineColors.append(QColor(117,12,136))
LineColors.append(QColor(196,3,50))
LineColors.append(QColor(50,39,123))
LineColors.append(QColor(101,63,18))
LineColors.append(QColor(112,3,196))
LineColors.append(QColor(30,86,64))
LineColors.append(QColor(10,125,121))
LineColors.append(QColor(7,131,82))
LineColors.append(QColor(96,111,176))
LineColors.append(QColor(17,129,125))
LineColors.append(QColor(95,86,76))
LineColors.append(QColor(25,4,188))
LineColors.append(QColor(87,39,123))
LineColors.append(QColor(119,20,45))
LineColors.append(QColor(215,122,0))
LineColors.append(QColor(0,133,206))
LineColors.append(QColor(6,86,135))
LineColors.append(QColor(206,0,133))
LineColors.append(QColor(7,186,15))
LineColors.append(QColor(255,0,0))

# barve razredov
ClassColors = []
ClassColors.append(QColor(247,91,36))
ClassColors.append(QColor(74,200,136))
ClassColors.append(QColor(36,247,244))
ClassColors.append(QColor(247,184,36))
ClassColors.append(QColor(36,247,155))
ClassColors.append(QColor(36,132,247))
ClassColors.append(QColor(138,247,36))
ClassColors.append(QColor(135,36,247))
ClassColors.append(QColor(214,36,247))
ClassColors.append(QColor(247,36,182))
ClassColors.append(QColor(236,247,36))
ClassColors.append(QColor(139,74,200))
ClassColors.append(QColor(36,83,247))
ClassColors.append(QColor(83,74,200))
ClassColors.append(QColor(200,74,90))
ClassColors.append(QColor(192,74,200))
ClassColors.append(QColor(36,129,192))
ClassColors.append(QColor(74,155,200))
ClassColors.append(QColor(171,159,81))
ClassColors.append(QColor(171,81,141))

ClassColors.append(QColor(247,91,36))
ClassColors.append(QColor(74,200,136))
ClassColors.append(QColor(36,247,244))
ClassColors.append(QColor(247,184,36))
ClassColors.append(QColor(36,247,155))
ClassColors.append(QColor(36,132,247))
ClassColors.append(QColor(138,247,36))
ClassColors.append(QColor(135,36,247))
ClassColors.append(QColor(214,36,247))
ClassColors.append(QColor(247,36,182))
ClassColors.append(QColor(236,247,36))
ClassColors.append(QColor(139,74,200))
ClassColors.append(QColor(36,83,247))
ClassColors.append(QColor(83,74,200))
ClassColors.append(QColor(200,74,90))
ClassColors.append(QColor(192,74,200))
ClassColors.append(QColor(36,129,192))
ClassColors.append(QColor(74,155,200))
ClassColors.append(QColor(171,159,81))
ClassColors.append(QColor(171,81,141))

# Variables to control body rectangle size
BodyWidth = BodyWidth_Default
BodyHeight = BodyHeight_Default
# Variable to control pie size
PieWidth = PieWidth_Default
PieHeight = PieHeight_Default

# Spremenljivka za advanced2 algoritem, ki hrani stevilo
# vozlisc na posameznem nivoju
levelNodesCount = []

# stevilo ze obdelanih vozlisc na posameznem nivoju
levelNodesDrawnCount = []

# Veljavno samo BottomUp algoritem
# stevilo zasedenih trakov na posameznem nivoju (za EnhancedBottomUp algoritem)
StripesUsed = []
# Ali naj se uposteva pravilo ne-prekrivanja vozlisc
AllowCollision = 1
# Konec samo za BottomUp


#Spremenljivka z razredom Visual2D
v2d = None

#globalna spremenljivka, ki doloca ali so pite prikazane
BodyPiesShow = 0

class PopupWindow(QWidget):
    def __init__(self, *args):
        apply(QWidget.__init__, (self,) + args)
        self.setGeometry((ScreenSize[0]-300)/2, (ScreenSize[1]-100)/2, 300, 100)
        self.Label = QLabel(self, 'Just some label')
        self.Label.setAlignment(Qt.AlignVCenter | Qt.AlignHCenter)
        self.Label.setFont(QFont(TextFont_Default, 16))
        self.Label.resize(self.size())

    def setText(self, text):
        self.Label.setText(text)

class BubbleInfo(QCanvasRectangle):
    def __init__(self, *args):
        apply(QCanvasRectangle.__init__, (self,) + args)
        self.canvas = args[0]
        self.setBrush(QBrush(QColor(255, 255, 255)))
        self.setZ(6)
        self.bubbleShadow = QCanvasRectangle(self.canvas)
        self.bubbleShadow.setBrush(QBrush(QColor(128, 128, 128)))
        self.bubbleShadow.setPen(QPen(QColor(128, 128, 128)))
        self.bubbleShadow.setZ(5)
        # crte, ki razdelijo bubble na dele
        self.sepLine0=QCanvasLine(self.canvas)
        self.sepLine1=QCanvasLine(self.canvas)
        self.sepLine2=QCanvasLine(self.canvas)
        self.sepLine2_1=QCanvasLine(self.canvas)
        self.sepLine3=QCanvasLine(self.canvas)		

    # Doloci besedila znotraj bubbla
    # Parametri:
    #   Names ... imena razredov
    #   Values ... vrednosti razredov
    #   allCases ... stevilo vseh primerov v vozliscu
    #   conditionalPath ... pot dosedanjih izbir za razvrscanje
    #   selectedSortAttributeName ... izbran atribut za sortiranje
    def setText(self, Names, Values, allCases, conditionalPath, selectedSortAttributeName, AttributePath, MajorityClass, TargetClassProbbability):
        global AttributeList
        
        # Array vseh tekstov v bubblu
        self.newclassText = []

        #pot, po kateri pridemo v vozlisce (po vrednostih atributov)
        #AttributePath je dolzine 1 pri ROOT 
        if len(AttributePath) > 1:
            Text = QCanvasText(self.canvas)
            Text.setText('IF '+AttributePath[0]+' AND')
            Text.setZ(7)
            self.newclassText.append(Text)
            for i in range(1, len(AttributePath) - 1):
                Text = QCanvasText(self.canvas)
                Text.setText('    '+AttributePath[i]+' AND')
                Text.setZ(7)
                self.newclassText.append(Text)
            Text = QCanvasText(self.canvas)
            Text.setText('    '+AttributePath[len(AttributePath) - 1])
            Text.setZ(7)
            self.newclassText.append(Text)
        #ce je sin od korena (root)	
        if len(AttributePath) == 1:
            Text = QCanvasText(self.canvas)
            Text.setText('IF '+AttributePath[len(AttributePath) - 1])
            Text.setZ(7)
            self.newclassText.append(Text)

        #==izpis prevladujocih (Majority) razredov		
        if len(MajorityClass) > 1:
            Text = QCanvasText(self.canvas)
            Text.setText('THEN ' + MajorityClass[0] + ' AND')
            Text.setZ(7)
            self.newclassText.append(Text)
            for i in range(1, len(MajorityClass) - 1):
                Text = QCanvasText(self.canvas)
                Text.setText('           '+MajorityClass[i] + ' AND')
                Text.setZ(7)
                self.newclassText.append(Text)
            Text = QCanvasText(self.canvas)
            Text.setText('           '+MajorityClass[len(MajorityClass) - 1])
            Text.setZ(7)
            self.newclassText.append(Text)	
        else:	
            Text = QCanvasText(self.canvas)
            Text.setText('THEN ' + MajorityClass[0])
            Text.setZ(7)
            self.newclassText.append(Text)
                    
                    
        #==konec izpisa prevladujocih (Majority) razredov				
        
        self.newclassText.append(self.sepLine0)
        self.sepLine0.setZ(7)
                
        # Izpis stevila vseh primerov v vozliscu
        Text = QCanvasText(self.canvas)
        Text.setZ(7)
        Text.setText('Instances: %d' % (allCases))
        self.newclassText.append(Text)
        self.newclassText.append(self.sepLine1)
        self.sepLine1.setZ(7)
    
        #ime prejsnjega atributa
        Text = QCanvasText(self.canvas)
        Text.setZ(7)
        self.newclassText.append(Text)
        # ce je koren drevesa
        if conditionalPath == ' = ':
            Text.setText('ROOT')
        else:
            Text.setText('%s' % (conditionalPath))

        self.newclassText.append(self.sepLine2)
        self.sepLine2.setZ(7)


        #verjetnost target classa
        #Text = QCanvasText(self.canvas)
        #Text.setText('P(Target): %2.1f' % (TargetClassProbbability) + '%')
        #Text.setZ(7)
        #self.newclassText.append(Text)
        
        #self.newclassText.append(self.sepLine2_1)
        #self.sepLine2_1.setZ(7)
        
        for i in range(0, len(Names)):
            # Ce je stevilo primerov v razredu 0, razreda ne izpisujemo
            if Values[i] > 0:
                Text = QCanvasText(self.canvas)
                Text.setText(Names[i] + ': ' + str(Values[i]) + ' (' + '%2.1f' % (Values[i]/allCases*100) + '% )')
                Text.setColor(ClassColors[i])
                Text.setZ(7)
                self.newclassText.append(Text)

        self.newclassText.append(self.sepLine3)
        self.sepLine3.setZ(7)

        #ime atributa po katerem se deli naprej
        Text = QCanvasText(self.canvas)
        if selectedSortAttributeName == 'None':
            selectedSortAttributeName = '(leaf)'
        if selectedSortAttributeName == '(leaf)':
            Text.setText('(leaf)')
        else:
            Text.setText('Split on: ' + selectedSortAttributeName)            
        Text.setZ(7)
        self.newclassText.append(Text)	

    def moveit(self, x, y):
        self.showit()
        self.move(x, y)
        self.bubbleShadow.move(x+5, y+5)
        distance = 0
        sizeX = 0
        sizeY = 0

        counter = 0
        for text in self.newclassText:
            #pozicioniram 'separate lines'
            if text == self.sepLine0 or text == self.sepLine1 or text == self.sepLine2 or text == self.sepLine2_1 or text == self.sepLine3:
                distance += 2
                text.setX(x)
                text.setY(y + distance)
            else:
                text.setX(x + 2)
                text.setY(y + distance)
                if text.boundingRect().width() > sizeX:
                    sizeX = text.boundingRect().width()
                distance += text.boundingRect().height()
            counter = 1

        # prilagodim velikost popup bubbla na st. primerov in dolzino textov
        # +5 je dobljeno izkustveno
        sizeX = sizeX + 5
        sizeY = distance + 5

        #dolocim dolzino crt
        self.sepLine1.setPoints(self.sepLine1.startPoint().x(), self.sepLine1.startPoint().y(), self.sepLine1.startPoint().x() + sizeX - 1, self.sepLine1.startPoint().y())
        self.sepLine2.setPoints(self.sepLine1.startPoint().x(), self.sepLine1.startPoint().y(), self.sepLine1.startPoint().x() + sizeX - 1, self.sepLine1.startPoint().y())
        self.sepLine2_1.setPoints(self.sepLine1.startPoint().x(), self.sepLine1.startPoint().y(), self.sepLine1.startPoint().x() + sizeX - 1, self.sepLine1.startPoint().y())
        self.sepLine3.setPoints(self.sepLine1.startPoint().x(), self.sepLine1.startPoint().y(), self.sepLine1.startPoint().x() + sizeX - 1, self.sepLine1.startPoint().y())
        self.sepLine0.setPoints(self.sepLine1.startPoint().x(), self.sepLine1.startPoint().y(), self.sepLine1.startPoint().x() + sizeX - 1, self.sepLine1.startPoint().y())
        self.setSize(sizeX , sizeY)
        self.bubbleShadow.setSize(sizeX, sizeY)

    def showit(self):
        self.show()
        self.bubbleShadow.show()
        for text in self.newclassText:
            text.show()

    def hideit(self):
        self.hide()
        self.bubbleShadow.hide()
        for text in self.newclassText:
            text.hide()

class MyCanvasView(QCanvasView):
    def __init__(self, *args):
        apply(QCanvasView.__init__,(self,)+args)
        self.canvas = args[0]
        self.viewport().setMouseTracking(True)
        # Koordinate zadnjega miskinega pohandlanega eventa
        self.x = 0
        self.y = 0
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
#		self.popup.insertSeparator()

    def tralala(self):
        self.viewport().setMouseTracking(self.bubbleIsShown)
    
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
        global NodeChilds
        self.viewport().setMouseTracking(self.bubbleIsShown)
        if event.button() == QEvent.LeftButton:
            for node in NodeChilds:
                if node.visible == True and node.posInNode(event.x(), event.y()) == True:
                    TreeWalk_setVisible(node, not node.childsVisible)
                    self.popup.setItemChecked(PopupSubtree, False)
                    break
        elif event.button() == QEvent.RightButton:
            for node in NodeChilds:
                if node.visible == True and node.posInNode(event.x(), event.y()) == True:
                    self.bubbleIsShown = self.viewport().hasMouseTracking()
                    self.viewport().setMouseTracking(False)
                    if self.bubbleShown != None:
                        self.bubbleShown.hideit()
                    self.selectedNode = node
                    self.popup.setItemChecked(PopupBubble, self.bubbleIsShown)
                    self.popup.popup(event.globalPos())
        self.canvas.update()

    def contentsMouseMoveEvent(self, event):
        global RefreshBubble
        
        
        handled = False

        if abs(self.x - event.x()) > CanvasResolution or abs(self.y - event.y()) > CanvasResolution:
            self.x = event.x()
            self.y = event.y()
            if self.bubbleShown != None:
                self.bubbleShown.hideit()
            for node in NodeChilds:
                if node.visible == True and node.posInNode(event.x(), event.y()) == True:
                    if node.bubble == None:
                        node.bubble = BubbleInfo(self.canvas)
                        node.BubbleRefresh = False
                        allCases = 0
                        for case in node.classesCounts:
                            allCases += case
                        
                        if node.conditionalPath != '':
                            conditionalPath = node.conditionalPath + ' AND\n'
                        else:
                            conditionalPath = ''
                        if node.prevSortAttributeValue != '' and node.prevSortAttributeValue[0] in ["<", ">"]:
                            conditionalPath += node.prevSortAttributeName + ' ' + node.prevSortAttributeValue
                        else:
                            conditionalPath += node.prevSortAttributeName + ' = ' + node.prevSortAttributeValue	
                        node.bubble.setText(node.classesNames, node.classesCounts, allCases, conditionalPath, node.selectedSortAttribute, node.AttributePath, node.MajorityClass, node.TargetClassProbbability)
                        node.bubble.moveit(event.x()+20, event.y()+20)
                        node.bubble.showit()
                        self.bubbleShown = node.bubble
                    elif node.bubble.visible() == False:
                        node.bubble.moveit(event.x()+20, event.y()+20)
                        self.bubbleShown = node.bubble
                        self.bubbleShown.showit()
                    handled = True
                    break
        self.canvas.update()

class OWClassificationTreeViewer2D(OWWidget):	

    settingsList = ["CanvasAutoFit", "ZoomAutoRefresh", "OptionsAutoRefresh", "InitialTreeDepth","NodeSizeBig","NodeSizeMedium","NodeSizeSmall","StandardAlgorithm","AdvancedAlgorithm","Advanced2Algorithm","BottomUpAlgorithm","BottomUpMode", "AllowCollision", "InitialLineWidth","LineOnRoot","LineOnNode","LineEqual","BodyColorDefault","BodyColorCases","BodyColorMajority","BodyColorTarget","MajorityClass","TargetClass","TargetClassProbability","Instances","BodyColorTargetRelative","ShowPies"]

    def __init__(self, parent=None, name='2D Tree Viewer'):
        global BodyPiesShow, AllowCollision
                
        self.lastZoom = 0
        self.Tree2D = None
        
        OWWidget.__init__(self, parent, name, 'A graphical 2D view of a classification tree.', TRUE, FALSE) 

        global v2d
        v2d = self
        
        self.addInput("target")
        self.addInput("ctree")


        # === SETTINGS ===
        #set default settings
        self.AllowCollision = AllowCollision
        self.CanvasAutoFit = 1
        self.ZoomAutoRefresh = 0
        self.OptionsAutoRefresh = 0
        self.InitialTreeDepth = 5
        self.NodeSizeBig = 0
        self.NodeSizeMedium = 0
        self.NodeSizeSmall = 1
        self.StandardAlgorithm = 1
        self.AdvancedAlgorithm = 0
        self.Advanced2Algorithm = 0
        self.BottomUpAlgorithm = 0
        self.BottomUpMode = AllowCollision
        self.InitialLineWidth = 10
        self.LineOnRoot = 1
        self.LineOnNode = 0
        self.LineEqual = 0
        self.BodyColorDefault = 1
        self.BodyColorCases = 0
        self.BodyColorMajority = 0
        self.BodyColorTarget = 0
        self.MajorityClass = 1
        self.TargetClass = 0
        self.TargetClassProbability = 0
        self.Instances = 0
        self.BodyColorTargetRelative = 0
        self.ShowPies = 1
        
            #load settings
        self.loadSettings()

        # Trenutna povecava
        self.zoom = ZoomFactor_Default * 1.0

        # === START LeftBox ===
        LeftBox=QVBox(self.controlArea)
        # === END LeftBox ===
        
        # === START TopBox ===
        TopBox=QVBox(LeftBox)
        TopBox.setSpacing(5)

        # Labela zoom
        self.ZoomLabel = QLabel(TopBox)
        self.ZoomLabel.setText('Zoom: '+str(self.zoom))

        # Slider za zoom
        self.slider = QSlider(0, 40, 5, self.zoom, self.Horizontal, TopBox, "Povecava")
        self.slider.setTickmarks(QSlider.Below)
        self.connect(self.slider, SIGNAL("valueChanged(int)"), self.zoomSliderSocket)

        # Dolocanje najvecje velikosti canvasa
        self.CanvasBox = QButtonGroup(1, Qt.Horizontal, "Input canvas size", TopBox)
        self.CanvasWidthBox = QLineEdit('0', self.CanvasBox)
        self.CanvasAutoPredict = QCheckBox("Canvas auto fit", self.CanvasBox)
        QToolTip.add(self.CanvasAutoPredict, "Try to predict sufficient canvas width")

        # Gumb za refresh
        RefreshButton=QPushButton('Refresh', TopBox)
        self.connect(RefreshButton, SIGNAL('clicked()'), self.refresh)

        #MatrixButton = QPushButton('test', TopBox)
        #self.connect(MatrixButton, SIGNAL('clicked()'), self.test)

        # Gumb za matrix
#		MatrixButton = QPushButton('Refresh', TopBox)
#		self.connect(MatrixButton, SIGNAL('clicked()'), self.qwmatrix)

        # === END TopBox ===

        #1. parameter, ki je izpisan v Body-ju
        self.BodyParameter1 = None
        #2. parameter, ki je izpisan v Body-ju
        self.BodyParameter2 = None


        #settings test
        self.options=VisualTreeOptions()
        
        self.options.TreeDepthBox.setText(QString(str(self.InitialTreeDepth)))
        self.options.LineWidthSelectBox.setText(QString(str(self.InitialLineWidth)))

        if self.CanvasAutoFit == 1:
            self.CanvasAutoPredict.setChecked(True)
        else:
            self.CanvasAutoPredict.setChecked(False)
        if self.ZoomAutoRefresh == 1:
            self.options.ZoomAutoRefresh.setChecked(True)
        else:
            self.options.ZoomAutoRefresh.setChecked(False)
        if self.OptionsAutoRefresh == 1:
            self.options.OptionsAutoRefresh.setChecked(True)
        else:
            self.options.OptionsAutoRefresh.setChecked(False)

        if self.NodeSizeBig == 1:
            self.options.NodeSizeBig.setOn(True)
            self.options.NodeSizeMedium.setOn(False)
            self.options.NodeSizeSmall.setOn(False)
        elif self.NodeSizeMedium == 1:
            self.options.NodeSizeBig.setOn(False)
            self.options.NodeSizeMedium.setOn(True)
            self.options.NodeSizeSmall.setOn(False)
        elif self.NodeSizeSmall == 1:
            self.options.NodeSizeBig.setOn(False)
            self.options.NodeSizeMedium.setOn(False)
            self.options.NodeSizeSmall.setOn(True)

        if self.BodyColorDefault == 1:
            self.options.NodeBodyColorDefault.setOn(True)
            self.options.NodeBodyColorCases.setOn(False)
            self.options.NodeBodyColorMajorityClass.setOn(False)
            self.options.NodeBodyColorTargetClass.setOn(False)
            self.options.NodeBodyColorTargetClassRelative.setOn(False)
        elif self.BodyColorCases == 1:
            self.options.NodeBodyColorDefault.setOn(False)
            self.options.NodeBodyColorCases.setOn(True)
            self.options.NodeBodyColorMajorityClass.setOn(False)
            self.options.NodeBodyColorTargetClass.setOn(False)
            self.options.NodeBodyColorTargetClassRelative.setOn(False)
        elif self.BodyColorMajority == 1:
            self.options.NodeBodyColorDefault.setOn(False)
            self.options.NodeBodyColorCases.setOn(False)
            self.options.NodeBodyColorMajorityClass.setOn(True)
            self.options.NodeBodyColorTargetClass.setOn(False)
            self.options.NodeBodyColorTargetClassRelative.setOn(False)
        elif self.BodyColorTarget == 1:
            self.options.NodeBodyColorDefault.setOn(False)
            self.options.NodeBodyColorCases.setOn(False)
            self.options.NodeBodyColorMajorityClass.setOn(False)
            self.options.NodeBodyColorTargetClass.setOn(True)
            self.options.NodeBodyColorTargetClassRelative.setOn(False)
        elif self.BodyColorTargetRelative == 1:
            self.options.NodeBodyColorDefault.setOn(False)
            self.options.NodeBodyColorCases.setOn(False)
            self.options.NodeBodyColorMajorityClass.setOn(False)
            self.options.NodeBodyColorTargetClass.setOn(False)
            self.options.NodeBodyColorTargetClassRelative.setOn(True)
        
        if self.StandardAlgorithm == 1:
            self.options.StandardAlgorithm.setOn(True)
            self.options.AdvancedAlgorithm.setOn(False)
            self.options.Advanced2Algorithm.setOn(False)
            self.options.BottomUpAlgorithm.setOn(False)
        elif self.AdvancedAlgorithm == 1:
            self.options.StandardAlgorithm.setOn(False)
            self.options.AdvancedAlgorithm.setOn(True)
            self.options.Advanced2Algorithm.setOn(False)
            self.options.BottomUpAlgorithm.setOn(False)
        elif self.Advanced2Algorithm == 1:
            self.options.StandardAlgorithm.setOn(False)
            self.options.AdvancedAlgorithm.setOn(False)
            self.options.Advanced2Algorithm.setOn(True)
            self.options.BottomUpAlgorithm.setOn(False)
        elif self.BottomUpAlgorithm == 1:
            self.options.StandardAlgorithm.setOn(False)
            self.options.AdvancedAlgorithm.setOn(False)
            self.options.Advanced2Algorithm.setOn(False)
            self.options.BottomUpAlgorithm.setOn(True)
        if self.BottomUpMode == 1:
            self.options.BottomUpMode.setChecked(True)
            AllowCollision = 1
        else:
            self.options.BottomUpMode.setChecked(False)
            AllowCollision = 0

        if self.LineOnRoot == 1:
            self.options.LineOnRoot.setOn(True)
            self.options.LineOnNode.setOn(False)
            self.options.LineEqual.setOn(False)
        elif self.LineOnNode == 1:
            self.options.LineOnRoot.setOn(False)
            self.options.LineOnNode.setOn(True)
            self.options.LineEqual.setOn(False)
        elif self.LineEqual == 1:
            self.options.LineOnRoot.setOn(False)
            self.options.LineOnNode.setOn(False)
            self.options.LineEqual.setOn(True)
            
        
        counter = 0
        if self.MajorityClass == 1 and counter < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Majority'
            else:	
                self.BodyParameter2 = 'Majority'
            self.options.MajorityClass.setChecked(True)	
            counter += 1
        if self.TargetClass == 1 and counter < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Target'
            else:	
                self.BodyParameter2 = 'Target'
            self.options.TargetClass.setChecked(True)	
            counter += 1	
        if self.TargetClassProbability == 1 and counter < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'TargetClassProbbability'
            else:	
                self.BodyParameter2 = 'TargetClassProbbability'
            self.options.TargetClassProbbability.setChecked(True)	
            counter += 1	
        if self.Instances == 1 and counter < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Instances'
            else:	
                self.BodyParameter2 = 'Instances'
            self.options.NumOfInstances.setChecked(True)	
            counter += 1	

        if self.ShowPies == 1:
            global BodyPiesShow
            BodyPiesShow = 1
            self.options.PieShowCheckBox.setChecked(True)
        else:
            global BodyPiesShow
            BodyPiesShow = 0
            self.options.PieShowCheckBox.setChecked(False)


        self.connect(self.settingsButton,SIGNAL("clicked()"),self.options.show)

        #connect GUI controls of options in options dialog to settings
        self.connect(self.CanvasAutoPredict, SIGNAL("stateChanged(int)"), self.setCanvasAutoFit) 
        self.connect(self.options.ZoomAutoRefresh, SIGNAL("stateChanged(int)"), self.setZoomAutoRefresh) 
        self.connect(self.options.OptionsAutoRefresh, SIGNAL("stateChanged(int)"), self.setOptionsAutoRefresh) 
        self.connect(self.options.TreeDepthBox, SIGNAL("textChanged(const QString &)"), self.setNewInitialTreeDepth) 
        self.connect(self.options.NodeSizeBig, SIGNAL("stateChanged(int)"), self.setNodeSizeGroup) 
        self.connect(self.options.NodeSizeMedium, SIGNAL("stateChanged(int)"), self.setNodeSizeGroup) 
        self.connect(self.options.NodeSizeSmall, SIGNAL("stateChanged(int)"), self.setNodeSizeGroup) 
        self.connect(self.options.NodeBodyColorDefault, SIGNAL("stateChanged(int)"), self.setNodeBodyColorSetting) 
        self.connect(self.options.NodeBodyColorCases, SIGNAL("stateChanged(int)"), self.setNodeBodyColorSetting) 
        self.connect(self.options.NodeBodyColorMajorityClass, SIGNAL("stateChanged(int)"), self.setNodeBodyColorSetting) 
        self.connect(self.options.NodeBodyColorTargetClass, SIGNAL("stateChanged(int)"), self.setNodeBodyColorSetting)
        self.connect(self.options.NodeBodyColorTargetClassRelative, SIGNAL("stateChanged(int)"), self.setNodeBodyColorSetting)
        self.connect(self.options.StandardAlgorithm, SIGNAL("stateChanged(int)"), self.setAlgorithm)
        self.connect(self.options.AdvancedAlgorithm, SIGNAL("stateChanged(int)"), self.setAlgorithm)
        self.connect(self.options.Advanced2Algorithm, SIGNAL("stateChanged(int)"), self.setAlgorithm)
        self.connect(self.options.BottomUpAlgorithm, SIGNAL("stateChanged(int)"), self.setAlgorithm)
        self.connect(self.options.BottomUpMode, SIGNAL("stateChanged(int)"), self.setAllowCollision) 
        self.connect(self.options.ok,SIGNAL("clicked()"),self.setText)
        self.connect(self.options.LineWidthSelectBox,SIGNAL("textChanged(const QString &)"),self.setNewInitialLineWidth)
        self.connect(self.options.LineOnRoot,SIGNAL("stateChanged(int)"),self.setLinesWidthRelativity)
        self.connect(self.options.LineOnNode,SIGNAL("stateChanged(int)"),self.setLinesWidthRelativity)
        self.connect(self.options.LineEqual,SIGNAL("stateChanged(int)"),self.setLinesWidthRelativity)
        self.connect(self.options.MajorityClass,SIGNAL("stateChanged(int)"),self.setText)
        self.connect(self.options.TargetClass,SIGNAL("stateChanged(int)"),self.setText)
        self.connect(self.options.TargetClassProbbability,SIGNAL("stateChanged(int)"),self.setText)
        self.connect(self.options.NumOfInstances,SIGNAL("stateChanged(int)"),self.setText)
        self.connect(self.options.PieShowCheckBox,SIGNAL("stateChanged(int)"),self.ShowPiesInBody)
        self.connect(self.options.TreeDepthBox, SIGNAL("textChanged(const QString &)"),self.refreshMode)
        self.connect(self.options.NodeSizeBig, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.NodeSizeMedium, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.NodeSizeSmall, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.NodeBodyColorDefault, SIGNAL("stateChanged(int)"), self.refreshMode) 
        self.connect(self.options.NodeBodyColorCases, SIGNAL("stateChanged(int)"), self.refreshMode) 
        self.connect(self.options.NodeBodyColorMajorityClass, SIGNAL("stateChanged(int)"), self.refreshMode) 
        self.connect(self.options.NodeBodyColorTargetClass, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.NodeBodyColorTargetClassRelative, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.StandardAlgorithm, SIGNAL("stateChanged(int)"), self.refreshMode)		
        self.connect(self.options.AdvancedAlgorithm, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.Advanced2Algorithm, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.BottomUpAlgorithm, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.BottomUpMode, SIGNAL("stateChanged(int)"), self.refreshMode)
        self.connect(self.options.ok,SIGNAL("clicked()"),self.refresh)
        self.connect(self.options.LineWidthSelectBox,SIGNAL("textChanged(const QString &)"),self.refreshMode)
        self.connect(self.options.LineOnRoot,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.LineOnNode,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.LineEqual,SIGNAL("stateChanged(int)"),self.refreshMode)		
        self.connect(self.options.MajorityClass,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.TargetClass,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.TargetClassProbbability,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.NumOfInstances,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.PieShowCheckBox,SIGNAL("stateChanged(int)"),self.refreshMode)
        self.connect(self.options.ok,SIGNAL("clicked()"),self.refreshModeOk)
        # === END SETTINGS ===


        # === START MiddleBox ===
        MiddleBox=QVBox(LeftBox)
        # === END MiddleBox ===	 

        # === START BottomBox ===
        BottomBox=QVBox(LeftBox)
        BottomBox.resize(100, 80)
        BottomBox.setSpacing(5)

#		self.ZoomMatrix = self.canvasView.worldMatrix()
        self.counter = 0
        self.Popup = PopupWindow(None)
        self.setMouseTracking(True)

        self.canvasView = None
        self.layout=QVBoxLayout(self.mainArea)

    #refresh ob kliku na ok ?
    def refreshModeOk(self):
        if self.OptionsAutoRefresh == 0:
            self.refresh()

    #ali naj se refresha ob kliku na ok, ali pa ob vsaki spremembi
    def refreshMode(self):
        if self.OptionsAutoRefresh == 1:
            self.refresh()

    def ctree(self, tree):
        self.tree = tree

        if self.canvasView != None:
            self.canvasView.destroy()
            self.layout.deleteAllItems()

        # Canvas
        self.canvas = QCanvas(200, 200)

        # CanvasView
        self.canvasView = MyCanvasView(self.canvas, self.mainArea)
        self.layout.add(self.canvasView)

        # Zgeneriramo orange drevo
        self.Popup.setText('Drawing tree, please wait...')
        self.Popup.show()
        self.Tree2D = Tree2D(self.canvas, self.tree)
        try:
            maxCanvasWidth = int(str(self.CanvasWidthBox.text()))
        except ValueError:
            maxCanvasWidth = 0
        try:
            maxTreeDepth = int(str(self.options.TreeDepthBox.text()))
        except ValueError:
            maxTreeDepth = int(InitialTreeDepth)
            maxTreeDepth = 1
            self.InitialTreeDepth = 1
        try:
            maxLineWidth = int(str(self.options.LineWidthSelectBox.text()))
        except ValueError:
            #maxLineWidth = int(InitialLineWidth)	
            maxLineWidth = 1
            self.InitialLineWidth = 1
        classesNames = self.Tree2D.Build2D(self.options.AlgorithmBox.id(self.options.AlgorithmBox.selected()),self.options.LinesBox.id(self.options.LinesBox.selected()), maxCanvasWidth, maxTreeDepth, self.canvasView.width(), maxLineWidth)

        if self.Tree2D.mostX < self.canvasView.width() and self.Tree2D.mostX != 0:
            factor = int(self.canvasView.width() / self.Tree2D.mostX * 10) - 10
            self.slider.setValue(factor)
        self.Popup.hide()
# refresh()  se klice ze v target() funkciji, ki se ravno tako sprozi ob vsaki
# spremembi vhodnih podatkov
#		self.refresh()

    def test(self):
        pixmap = QPixmap()
        widget = self.canvasView.viewport()
        newpixmap = pixmap.grabWidget(widget)
        imageRead = newpixmap.convertToImage()

        width = imageRead.width()
        height = imageRead.height()
        imageWrite = QImage(width, height, 32)
        piHeight = math.pi/height
        height2 = height/2
        piWidth = math.pi/width
        width2 = width/2
        for i in range(height):
            for j in range(width):
                if j > (width/2):
                    if i > (height/2):
                        imageWrite.setPixel(j, i, imageRead.pixel(width-math.sin(piWidth*j)*width2, height-math.sin(piHeight*i)*height2))
                    else:
                        imageWrite.setPixel(j, i, imageRead.pixel(width-math.sin(piWidth*j)*width2, math.sin(piHeight*i)*height2))
                else:
                    if i > (height/2):
                        imageWrite.setPixel(j, i, imageRead.pixel(math.sin(piWidth*j)*width2, height-math.sin(piHeight*i)*height2))
                    else:
                        imageWrite.setPixel(j, i, imageRead.pixel(math.sin(piWidth*j)*width2, math.sin(piHeight*i)*height2))
        pixmap.convertFromImage(imageWrite)
        self.newwidget = QWidget()
        self.newwidget.setBackgroundPixmap(pixmap)
        self.newwidget.show()


    def getComboBoxItem(self):
        global RefreshBubble
        if(TargetClassIndex != self.options.combobox.currentItem()):
            #osveziti moram bubble
            RefreshBubble = True
        #vpisem index v globalno spremenljivko TargetClassIndex 
        setTargetClassIndex(self.options.combobox.currentItem())

    def zoomSliderSocket(self, socket):
        self.zoom = 1 + self.slider.value()/10.0
        self.ZoomLabel.setText('Zoom: '+str(self.zoom))
        if self.options.ZoomAutoRefresh.isChecked() and self.Tree2D != None:
            self.refresh()

# Dela samo v Qt 3.x
    def qwmatrix(self):
        return
        scalingFactor = 0.9
        shearFactor = 1.0
        ZoomMatrix = QWMatrix()
        ZoomMatrix.scale(shearFactor, shearFactor)
        self.canvasView.setWorldMatrix(ZoomMatrix)

    def refresh(self):
        global BodyWidth,BodyWidth_Default,BodyHeight,BodyHeight_Default,PieWidth_Default,PieWidth,PieHeight_Default,PieHeight

        if self.canvasView == None:
            return
        self.zoom = 1 + self.slider.value()/10.0
        self.Popup.setText('Redrawing, please wait...')
        self.Popup.show()
        if self.CanvasAutoFit == 1:
            maxCanvasWidth = 0
        else:
            try:
                maxCanvasWidth = int(str(self.CanvasWidthBox.text()))
            except ValueError:
                maxCanvasWidth = 0

        try:
            maxTreeDepth = int(str(self.options.TreeDepthBox.text()))
        except ValueError:
            #maxTreeDepth = int(InitialTreeDepth)
            maxTreeDepth = 1
            self.InitialTreeDepth = 1
        try:
            maxLineWidth = int(str(self.options.LineWidthSelectBox.text()))
        except ValueError:
            #maxLineWidth = int(InitialLineWidth)
            maxLineWidth = 1
            self.InitialLineWidth = 1

        if self.NodeSizeMedium == 1:
            BodyWidth = BodyWidth_Default * 1.5
            BodyHeight = BodyHeight_Default * 1.5
            PieWidth = PieWidth_Default * 1.5
            PieHeight = PieHeight_Default * 1.5
        elif self.NodeSizeBig == 1:
            BodyWidth = BodyWidth_Default * 2
            BodyHeight = BodyHeight_Default * 2
            PieWidth = PieWidth_Default * 2
            PieHeight = PieHeight_Default * 2
        elif self.NodeSizeSmall == 1:
            BodyWidth = BodyWidth_Default
            BodyHeight = BodyHeight_Default
            PieWidth = PieWidth_Default
            PieHeight = PieHeight_Default

        self.Tree2D.NodesRecalculate(self.zoom, self.options.AlgorithmBox.id(self.options.AlgorithmBox.selected()), maxCanvasWidth, maxTreeDepth, self.options.LinesBox.id(self.options.LinesBox.selected()), maxLineWidth, self.BodyParameter1, self.BodyParameter2)

        if self.CanvasAutoFit == 1 or maxCanvasWidth == 0:
            self.CanvasWidthBox.setText(str(self.canvas.width()))
        else:
            self.CanvasWidthBox.setText(str(maxCanvasWidth))	
        self.CanvasWidthBox.setEdited(False)

        self.canvas.update()
        self.Popup.hide()

    def setCanvasAutoFit(self):
        if self.CanvasAutoPredict.isChecked():
            self.CanvasAutoFit = 1
        else:
            self.CanvasAutoFit = 0

    def setZoomAutoRefresh(self):
        if self.options.ZoomAutoRefresh.isChecked():
            self.ZoomAutoRefresh = 1
        else:
            self.ZoomAutoRefresh = 0

    def setOptionsAutoRefresh(self):
        if self.options.OptionsAutoRefresh.isChecked():
            self.OptionsAutoRefresh = 1
        else:
            self.OptionsAutoRefresh = 0
    
    def setNewInitialTreeDepth(self):
        if self.options.TreeDepthBox.text().length() != 0:
            try:
                self.InitialTreeDepth = int(str(self.options.TreeDepthBox.text()))
            except ValueError:
                self.InitialTreeDepth = 5
    
    def setNodeSizeGroup(self):
        if self.options.NodeSizeBig.isOn():
                self.NodeSizeBig = 1
                self.NodeSizeMedium = 0
                self.NodeSizeSmall = 0
        elif self.options.NodeSizeMedium.isOn():
                self.NodeSizeBig = 0
                self.NodeSizeMedium = 1
                self.NodeSizeSmall = 0
        elif self.options.NodeSizeSmall.isOn():
                self.NodeSizeBig = 0
                self.NodeSizeMedium = 0
                self.NodeSizeSmall = 1

    #uredi izpis zelenih 'podatkov' v body-ju (najvec 2)		        
    def setText(self):
        self.BodyParameter1 = None
        self.BodyParameter2 = None
        countSelectedParameters = 0
        
        if self.options.MajorityClass.isChecked() and countSelectedParameters < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Majority'
            else:	
                self.BodyParameter2 = 'Majority'
            countSelectedParameters += 1
            self.MajorityClass = 1
        else:
            self.options.MajorityClass.setChecked(False)
            self.MajorityClass = 0
            
        if self.options.TargetClass.isChecked() and countSelectedParameters < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Target'
            else:	
                self.BodyParameter2 = 'Target'
            countSelectedParameters += 1
            self.TargetClass = 1
        else:
            self.options.TargetClass.setChecked(False)
            self.TargetClass = 0
            
        if self.options.TargetClassProbbability.isChecked() and countSelectedParameters < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'TargetClassProbbability'
            else:	
                self.BodyParameter2 = 'TargetClassProbbability'
            countSelectedParameters += 1
            self.TargetClassProbability = 1
        else:
            self.options.TargetClassProbbability.setChecked(False)
            self.TargetClassProbability = 0
        
        if self.options.NumOfInstances.isChecked() and countSelectedParameters < 2:
            if self.BodyParameter1 == None:
                self.BodyParameter1 = 'Instances'
            else:	
                self.BodyParameter2 = 'Instances'
            countSelectedParameters += 1
            self.Instances = 1
        else:
            self.options.NumOfInstances.setChecked(False)
            self.Instances = 0
        
    def setNodeBodyColorSetting(self):
        if self.options.NodeBodyColorDefault.isOn():
            self.BodyColorDefault = 1
            self.BodyColorCases = 0
            self.BodyColorMajority = 0
            self.BodyColorTarget = 0
            self.BodyColorTargetRelative = 0
        elif self.options.NodeBodyColorCases.isOn():
            self.BodyColorDefault = 0
            self.BodyColorCases = 1
            self.BodyColorMajority = 0
            self.BodyColorTarget = 0
            self.BodyColorTargetRelative = 0
        elif self.options.NodeBodyColorMajorityClass.isOn():
            self.BodyColorDefault = 0
            self.BodyColorCases = 0
            self.BodyColorMajority = 1
            self.BodyColorTarget = 0
            self.BodyColorTargetRelative = 0
        elif self.options.NodeBodyColorTargetClass.isOn():
            self.BodyColorDefault = 0
            self.BodyColorCases = 0
            self.BodyColorMajority = 0
            self.BodyColorTarget = 1
            self.BodyColorTargetRelative = 0
        elif self.options.NodeBodyColorTargetClassRelative.isOn():
            self.BodyColorDefault = 0
            self.BodyColorCases = 0
            self.BodyColorMajority = 0
            self.BodyColorTarget = 0
            self.BodyColorTargetRelative = 1
    
    
    def ShowPiesInBody(self):
        if self.options.PieShowCheckBox.isChecked():
            global BodyPiesShow
            BodyPiesShow = 1
            self.ShowPie = 1
        else:	
            global BodyPiesShow
            BodyPiesShow = 0
            self.ShowPie = 0
            

    def setAlgorithm(self):
        if self.options.StandardAlgorithm.isOn():
            self.StandardAlgorith = 1
            self.AdvancedAlgorith = 0
            self.Advanced2Algorith = 0
            self.BottomUpAlgorith = 0
        if self.options.AdvancedAlgorithm.isOn():
            self.StandardAlgorith = 0
            self.AdvancedAlgorith = 1
            self.Advanced2Algorith = 0
            self.BottomUpAlgorith = 0
        if self.options.Advanced2Algorithm.isOn():
            self.StandardAlgorith = 0
            self.AdvancedAlgorith = 0
            self.Advanced2Algorith = 1
            self.BottomUpAlgorith = 0
        if self.options.BottomUpAlgorithm.isOn():
            self.StandardAlgorith = 0
            self.AdvancedAlgorith = 0
            self.Advanced2Algorith = 0
            self.BottomUpAlgorith = 1

    def setAllowCollision(self):
        global AllowCollision
        if self.options.BottomUpMode.isChecked():
            self.BottomUpMod = 1
            AllowCollision = 0
        else:
            self.BottomUpMod = 0
            AllowCollision = 1

    #dolocim debelino crte
    def setNewInitialLineWidth(self):
        if self.options.LineWidthSelectBox.text().length() != 0:
            try:
                self.InitialLineWidth = int(str(self.options.LineWidthSelectBox.text()))
            except ValueError:
                self.InitialLineWidth = 10

    #za debelino crt glede na koren ali vozlisce
    def setLinesWidthRelativity(self):
        if self.options.LineOnRoot.isOn():
            self.LineOnRoot = 1
            self.LineOnNode = 0
            self.LineEqual = 0
        elif self.options.LineOnNode.isOn():
            self.LineOnRoot = 0
            self.LineOnNode = 1
            self.LineEqual = 0
        elif self.options.LineEqual.isOn():
            self.LineOnRoot = 0
            self.LineOnNode = 0
            self.LineEqual = 1

    # poberem ime target classa iz OWOutcome (sprejmem signal target)
    def target(self, target):
        setTargetClassIndex(target)
        self.refresh()

# lista imen atributov
AttributeList = []

#index v classesNames kjer je Tizbrani ciljni razred (Target class)
TargetClassIndex = 0

class Tree2D:
    def __init__(self,canvas,tree):
        self.canvas = canvas
        self.tree = tree
        self.numAllCases = 0
        self.mostX = 0
        self.mostY = 0
        self.maxTreeDepth = 5
        self.AttributeList = []
        #self.AttributeColorsList = []

    def Build2D(self, DrawAlghoritem, LinesRelativity,maxCanvasWidth, maxTreeDepth, canvasViewWidth, maxLineWidth):
        global NodeChilds
        global NodeChilds
        global AttributeList
        global AttributeColorsList
        global classesNames

        #maximalna debelina crte
        self.maxLineWidth = maxLineWidth
        
        self.numAllCases = self.tree.tree.distribution.cases
        LevelCountInit(self.tree.treesize()/2 + 1)

        #zapomnim si imena vseh razredov v drevesu
        classesNames = self.tree.tree.distribution.items()

        # zapolnim tabelo barv atributov z barvami
        #GetAttributeColorsList(self.tree)

        #prepisem tabelo atributov (se ne vem, ce mi je treba prepisati to tabelo)
        AttributeLists = self.tree.domain
        for i in range(0,len(AttributeLists)):
            AttributeList.append(AttributeLists[i].name)

        self.canvas.resize(2000, 2000)

        # Sestavimo naso drevesno strukturo
        TreeWalkValue = TreeWalk(self.tree.tree, 'root', 0, self.canvas,'', '', [])    # ######
        self.root = TreeWalkValue[0]
        NodeChilds.append(self.root)

        # Neveljavne zapise izlocimo
        levelCount = LevelCountGet()
        self.levelCount = [1] + levelCount[0:levelCount.index(0)]
        
#		self.NodesReposition(self.root, 0, self.canvas.width(), 0, 0)
        self.NodesRecalculate(1, DrawAlghoritem,maxCanvasWidth, maxTreeDepth, LinesRelativity)

        #vrnem imena razredov
        return classesNames

    #izracuna podatke o target classu in jih posreduje node-u
    def calculateTargetClass(self, node):
        global TargetClassIndex
        node.TargetClassProbbability = 0
        name = classesNames[TargetClassIndex][0]
        index = node.classesNames.index(name)
        #racunam verjetnost target classa in jo vpisem v node
        TotalCassesInNode = sum(node.classesCounts)
        node.TargetClassProbbability = (100 * node.classesCounts[index] / TotalCassesInNode) 	
    
    def writeBodyParam(self ,node, ParamNum, BodyParameter):
        #izberem kateri parameter naj pisem
        if ParamNum ==1:
            parameter = node.parameter1
        else:
            parameter = node.parameter2
        
        if BodyParameter == 'Majority':
            parameter.setText(node.MajorityClass[0])
        elif BodyParameter == 'Target':
            name = classesNames[TargetClassIndex][0]
            parameter.setText(name)
        elif BodyParameter == 'TargetClassProbbability':
            parameter.setText('%2.1f' % (node.TargetClassProbbability) + '%')
        elif BodyParameter == 'Instances':
            parameter.setText('%d'%(node.numCases))
        parameter.setZ(2)
        parameter.show()

    # width je sirina traku, ki nam je na voljo
    # slide je sirina odmika od levega roba do zacetka izrisa tega poddrevesa
    # displace je sirina odmika od zacetka tega traku (torej slide) do izrisa trenutnega vozlisca
    def __NodesRecalculate_Standard(self, node, width, displace, slide, level=0, LinesRelativity=1, BodyParameter1=None, BodyParameter2=None):
        global TargetClassIndex

        #izracuna ciljni razred
        self.calculateTargetClass(node)
        
        
        if BodyParameter1 != None:
            self.writeBodyParam(node , 1, BodyParameter1)
            if BodyParameter2 != None:
                self.writeBodyParam(node, 2, BodyParameter2)
            else:
                node.parameter2.setText(None)	
        else:
            node.parameter1.setText(None)
            node.parameter2.setText(None)
                        
        if self.maxTreeDepth > 0:
            if level == (self.maxTreeDepth-1):
                node.setVisible(True)
                node.childsVisible = False
            elif level >= self.maxTreeDepth:
                node.setVisible(False)				
            else:
                node.setVisible(True)
                node.childsVisible = False				

        positionX = slide + displace + width/2
        positionY = 10 + level * BodyHeight * 4 * self.factor
        node.factor = self.factor
        node.setPosition(positionX, positionY)
        tempMostX = positionX + (BodyWidth + PieWidth/2)*self.factor
        if tempMostX > self.mostX:
            self.mostX = tempMostX
        tempMostY = positionY + BodyHeight*self.factor
        if tempMostY > self.mostY:
            self.mostY = tempMostY

        node.setSize()

        node.parameter1.setColor(QColor(0,0,0));
        node.parameter2.setColor(QColor(0,0,0));
        node.SelectedAttName.setColor(QColor(0,0,0));

        if v2d.options.NodeBodyColorDefault.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(QColor(BodyColor_Default[0], BodyColor_Default[1], BodyColor_Default[2])))
        elif v2d.options.NodeBodyColorCases.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (300*node.numCases)/self.numAllCases < 150:
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))
                    else:
                        node.parameter1.setColor(QColor(255,255,255));
                        node.parameter2.setColor(QColor(255,255,255));
                        node.SelectedAttName.setColor(QColor(255,255,255));			
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))	
        elif v2d.options.NodeBodyColorMajorityClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    i=0
                    for ClassName in node.classesNames:
                        if ClassName == node.MajorityClass[0]:
                            body.setBrush(QBrush(ClassColors[i].light(250 - 1.5*node.MajorityClassProbabbility)))
                        i+=1
        elif v2d.options.NodeBodyColorTargetClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - node.TargetClassProbbability)))
        elif v2d.options.NodeBodyColorTargetClassRelative.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (self.root.classesCounts[TargetClassIndex] != 0):
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - (100*node.classesCounts[TargetClassIndex])/self.root.classesCounts[TargetClassIndex])))
                    else:
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200)))
                        #print "No target class in the tree"
                
        bottom=node.BottomCenterPosition()
        slide=slide+displace
        if node.childsNum > 0:
            newWidth = width/node.childsNum
        else:
            return
        i = 0
        for child in node.childs:
            self.__NodesRecalculate_Standard(child, newWidth,i*newWidth, slide, level + 1,  LinesRelativity, BodyParameter1, BodyParameter2)
            #debelina crte glede na koren ali trenutno vozlisce
            if LinesRelativity == 0:
                LineWidth = (child.numCases/self.numAllCases)*self.maxLineWidth
            elif LinesRelativity == 1:
                LineWidth = (child.numCases/node.numCases)*self.maxLineWidth
            elif LinesRelativity == 2:	
                LineWidth = self.maxLineWidth
            child.setLine(bottom[0], bottom[1], Qt.black, LineWidth)
            i += 1
        
        

    def NodesRecalculate(self, factor=1, DrawAlghoritem=0,maxCanvasWidth=0, maxTreeDepth=0, LinesRelativity=1, maxLineWidth=10, BodyParameter1=None, BodyParameter2=None):
        global StripesUsed
        
        self.mostX = 0
        self.factor = factor
        self.maxTreeDepth = maxTreeDepth
        self.maxLineWidth = maxLineWidth

        
        canvasSizeY = factor * len(self.levelCount)*BodyHeight*4*factor

        if DrawAlghoritem == 0:
            if maxCanvasWidth > 0:
                canvasSizeX = maxCanvasWidth
            else:
                canvasSizeX = min(MaxCanvasWidth, factor * self.root.numberLeaves * 2.0 * (BodyWidth + 10)) # 10 je fiksna luknja med enim in drugim
            self.__NodesRecalculate_Standard(self.root, canvasSizeX,10, 0, 0, LinesRelativity, BodyParameter1, BodyParameter2)
        elif DrawAlghoritem == 1:
            if maxCanvasWidth > 0:
                canvasSizeX = maxCanvasWidth
            else:
                canvasSizeX = min(MaxCanvasWidth, factor * self.root.numberLeaves * (BodyWidth+PieWidth/2.0+10)) # 10 je fiksna luknja med enim in drugim
            self.__NodesRecalculate_Advanced(self.root, canvasSizeX, 10, 0, LinesRelativity, BodyParameter1, BodyParameter2)
        elif DrawAlghoritem == 2:
            if maxCanvasWidth > 0:
                canvasSizeX = maxCanvasWidth
            else:
                canvasSizeX = min(MaxCanvasWidth, factor * self.root.numberLeaves * 2.0 * (BodyWidth + 10)) # 10 je fiksna luknja med enim in drugim
            self.__NodesRecalculate_Advanced2(self.root, canvasSizeX, 0, LinesRelativity, BodyParameter1, BodyParameter2)
        elif DrawAlghoritem == 3:
            if maxCanvasWidth > 0:
                canvasSizeX = maxCanvasWidth
            else:
                canvasSizeX = min(MaxCanvasWidth, factor * self.root.numberLeaves * 2.0 * (BodyWidth + 10)) # 10 je fiksna luknja med enim in drugim
            stripeWidth = canvasSizeX / orngTree.countLeaves(self.tree)
            StripesUsed = []
            offset = int(self.root.childsNum/2)
            self.__NodesRecalculate_EnhancedBottomUp(self.root, stripeWidth, 0, offset, LinesRelativity, BodyParameter1, BodyParameter2)

        self.canvas.resize(self.mostX + ExpectedBubbleWidth_Default, self.mostY + ExpectedBubbleHeigth_Default)
        self.canvas.update()

    def __NodesRecalculate_Advanced(self, node, canvasWidth,displace, level=0, LinesRelativity=1, BodyParameter1=None, BodyParameter2=None):
        global TargetClassIndex
        
        if BodyParameter1 != None:
            self.writeBodyParam(node , 1, BodyParameter1)
            if BodyParameter2 != None:
                self.writeBodyParam(node, 2, BodyParameter2)
            else:
                node.parameter2.setText(None)	
        else:
            node.parameter1.setText(None)
            node.parameter2.setText(None)
        
        if self.maxTreeDepth > 0:
            if level == (self.maxTreeDepth-1):
                node.setVisible(True)
                node.childsVisible = False
            elif level >= self.maxTreeDepth:
                node.setVisible(False)
            else:
                node.setVisible(True)

        #ce je potrebno izracunati target class verjetnosti		
        #if TargetClassIndex > 0:
        self.calculateTargetClass(node)
        
        if node.childsNum > 0:
            positionX = displace + canvasWidth/2
        else:
            positionX = displace
        positionY = 10 + level * BodyHeight * 4 * self.factor

        node.factor = self.factor
        node.setPosition(positionX, positionY)
        node.setSize()

        node.parameter1.setColor(QColor(0,0,0));
        node.parameter2.setColor(QColor(0,0,0));
        node.SelectedAttName.setColor(QColor(0,0,0));

        if v2d.options.NodeBodyColorDefault.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(QColor(BodyColor_Default[0], BodyColor_Default[1], BodyColor_Default[2])))
        elif v2d.options.NodeBodyColorCases.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (300*node.numCases)/self.numAllCases < 150:
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))
                    else:
                        node.parameter1.setColor(QColor(255,255,255));
                        node.parameter2.setColor(QColor(255,255,255));
                        node.SelectedAttName.setColor(QColor(255,255,255));			
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))	
        elif v2d.options.NodeBodyColorMajorityClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    i=0
                    for ClassName in node.classesNames:
                        if ClassName == node.MajorityClass[0]:
                            body.setBrush(QBrush(ClassColors[i].light(250 - 1.5*node.MajorityClassProbabbility)))
                        i+=1
        elif v2d.options.NodeBodyColorTargetClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - node.TargetClassProbbability)))
        elif v2d.options.NodeBodyColorTargetClassRelative.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (self.root.classesCounts[TargetClassIndex] != 0):
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - (100*node.classesCounts[TargetClassIndex])/self.root.classesCounts[TargetClassIndex])))
                    else:
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200)))
                        #print "No target class in the tree"

        bottom = node.BottomCenterPosition()
        tempMostX = positionX + (BodyWidth + PieWidth/2)*self.factor
        if tempMostX > self.mostX:
            self.mostX = tempMostX
        tempMostY = positionY + BodyHeight*self.factor
        if tempMostY > self.mostY:
            self.mostY = tempMostY

        newDisplace = displace
        slideWidth = canvasWidth/node.numberLeaves
        for child in node.childs:
            newCanvasWidth = slideWidth*child.numberLeaves
            self.__NodesRecalculate_Advanced(child, newCanvasWidth, newDisplace, level + 1, LinesRelativity, BodyParameter1, BodyParameter2)
            newDisplace += newCanvasWidth
            #debelina crte glede na koren ali trenutno vozlisce
            if LinesRelativity == 0:
                LineWidth = (child.numCases/self.numAllCases)*self.maxLineWidth
            elif LinesRelativity == 1:
                LineWidth = (child.numCases/node.numCases)*self.maxLineWidth
            elif LinesRelativity == 2:
                LineWidth = self.maxLineWidth
            child.setLine(bottom[0], bottom[1], Qt.black, LineWidth)
            
    def __NodesRecalculate_EnhancedBottomUp(self, node, stripeWidth, level=0, startingStripe=0, LinesRelativity=1, BodyParameter1=None, BodyParameter2=None):
        global TargetClassIndex, StripesUsed
        
        if BodyParameter1 != None:
            self.writeBodyParam(node , 1, BodyParameter1)
            if BodyParameter2 != None:
                self.writeBodyParam(node, 2, BodyParameter2)
            else:
                node.parameter2.setText(None)	
        else:
            node.parameter1.setText(None)
            node.parameter2.setText(None)
        
        if self.maxTreeDepth > 0:
            if level == (self.maxTreeDepth-1):
                node.setVisible(True)
                node.childsVisible = False
            elif level >= self.maxTreeDepth:
                node.setVisible(False)
            else:
                node.setVisible(True)

        #ce je potrebno izracunati target class verjetnosti		
        #if TargetClassIndex > 0:
        self.calculateTargetClass(node)
        

        node.parameter1.setColor(QColor(0,0,0));
        node.parameter2.setColor(QColor(0,0,0));
        node.SelectedAttName.setColor(QColor(0,0,0));


        if v2d.options.NodeBodyColorDefault.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(QColor(BodyColor_Default[0], BodyColor_Default[1], BodyColor_Default[2])))
        elif v2d.options.NodeBodyColorCases.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (300*node.numCases)/self.numAllCases < 150:
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))
                    else:
                        node.parameter1.setColor(QColor(255,255,255));
                        node.parameter2.setColor(QColor(255,255,255));
                        node.SelectedAttName.setColor(QColor(255,255,255));			
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))	
        elif v2d.options.NodeBodyColorMajorityClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    i=0
                    for ClassName in node.classesNames:
                        if ClassName == node.MajorityClass[0]:
                            body.setBrush(QBrush(ClassColors[i].light(250 - 1.5*node.MajorityClassProbabbility)))
                        i+=1
        elif v2d.options.NodeBodyColorTargetClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - node.TargetClassProbbability)))
        elif v2d.options.NodeBodyColorTargetClassRelative.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (self.root.classesCounts[TargetClassIndex] != 0):
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - (100*node.classesCounts[TargetClassIndex])/self.root.classesCounts[TargetClassIndex])))
                    else:
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200)))
                        #print "No target class in the tree"


        # pozicioniranje
        node.factor = self.factor
        node.setSize()
        if len(StripesUsed) <= level:
            StripesUsed.append(0)
        AvailableStripe = StripesUsed[level]
        if startingStripe > AvailableStripe:
            AvailableStripe = startingStripe
        positionY = 10 + level * BodyHeight * 4 * self.factor
        if node.childsNum == 0:
            positionX = stripeWidth * (AvailableStripe + 0.5)
            AssignedStripe = AvailableStripe + 0.5
            # Omejitev, ki ne dopusca sekanja grafov, vendar postane drevo manj kompaktno (tudi spodaj)
            if (positionX + BodyWidth*1.5) > (int(AssignedStripe + 1) * stripeWidth):
                StripesUsed[level] = round(AssignedStripe + AllowCollision)
            else:
                StripesUsed[level] = round(AssignedStripe)
            node.setPosition(positionX - BodyWidth/2, positionY)
        else:
            minStripe = 100000000
            maxStripe = 0
            for child in node.childs:
                offset = AvailableStripe - int(node.childsNum / 2)
                if offset < 0:
                    offset = 0
                AssignedStripe = self.__NodesRecalculate_EnhancedBottomUp(child, stripeWidth, level + 1, offset, LinesRelativity, BodyParameter1, BodyParameter2)
                if AssignedStripe < minStripe:
                    minStripe = AssignedStripe
                if AssignedStripe > maxStripe:
                    maxStripe = AssignedStripe
                
                    
            AssignedStripe = (maxStripe + minStripe) / 2.0
            positionX = AssignedStripe * stripeWidth
            tempNumber = divmod(AssignedStripe, 1)
            if tempNumber[1] <= 0.5:
                StripesUsed[level] = round(AssignedStripe + 0.5)
            # Omejitev, ki ne dopusca sekanja grafov, vendar postane drevo manj kompaktno
            elif (positionX + BodyWidth*1.5) > (int(AssignedStripe + 1) * stripeWidth):
                StripesUsed[level] = round(AssignedStripe + AllowCollision)
            else:
                StripesUsed[level] = round(AssignedStripe)
            node.setPosition(positionX - BodyWidth/2, positionY)
            
            bottom = node.BottomCenterPosition()
            for child in node.childs:
                #debelina crte glede na koren ali trenutno vozlisce
                if LinesRelativity == 0:
                    LineWidth = (child.numCases/self.numAllCases)*self.maxLineWidth
                elif LinesRelativity == 1:
                    LineWidth = (child.numCases/node.numCases)*self.maxLineWidth
                elif LinesRelativity == 2:
                    LineWidth = self.maxLineWidth			
                child.setLine(bottom[0], bottom[1], Qt.black, LineWidth)
            
            


        # Ali narisano vozlisce gleda cez canvas? Popravimo virtualno velikost canvasa (self.mostX, self.mostY)
        tempMostX = positionX + (BodyWidth + PieWidth/2)*self.factor
        if tempMostX > self.mostX:
            self.mostX = tempMostX
        tempMostY = positionY + BodyHeight*self.factor
        if tempMostY > self.mostY:
            self.mostY = tempMostY
        
        return AssignedStripe


    # width je sirina traku, ki nam je na voljo
    def __NodesRecalculate_Advanced2(self, node, width, level=0, LinesRelativity=1, BodyParameter1=None, BodyParameter2=None):
        global TargetClassIndex
        #izracuna ciljni razred
        self.calculateTargetClass(node)
        
        if BodyParameter1 != None:
            self.writeBodyParam(node , 1, BodyParameter1)
            if BodyParameter2 != None:
                self.writeBodyParam(node, 2, BodyParameter2)
            else:
                node.parameter2.setText(None)	
        else:
            node.parameter1.setText(None)
            node.parameter2.setText(None)
                        
        if self.maxTreeDepth > 0:
            if level == (self.maxTreeDepth-1):
                node.setVisible(True)
                node.childsVisible = False
            elif level >= self.maxTreeDepth:
                node.setVisible(False)
            else:
                node.setVisible(True)

        # stevilo ze obdelanih vozlisc na posameznem nivoju
        global levelNodesDrawnCount
        if level == 0:
            levelNodesDrawnCount = []
            for i in range(len(levelNodesCount)):
                levelNodesDrawnCount.append(0)
        # stevilo ze narisanih vozlisc na tem nivoju
        drawnNodesCount = levelNodesDrawnCount[level]
        # stevilo ze narisanih vozlisc na tem nivoju + 1 pomnozimo s sirino traku, ki
        # je namenjena za posamezno vozlisce na tem nivoju
        positionX = (drawnNodesCount+1)*(width/(1+levelNodesCount[level]))
        levelNodesDrawnCount[level] += 1

        positionY = 10 + level * BodyHeight * 4 * self.factor
        node.factor = self.factor
        node.setPosition(positionX, positionY)
        tempMostX = positionX + (BodyWidth + PieWidth/2)*self.factor
        if tempMostX > self.mostX:
            self.mostX = tempMostX
        tempMostY = positionY + BodyHeight*self.factor
        if tempMostY > self.mostY:
            self.mostY = tempMostY

        node.setSize()

        node.parameter1.setColor(QColor(0,0,0));
        node.parameter2.setColor(QColor(0,0,0));
        node.SelectedAttName.setColor(QColor(0,0,0));
        
        if v2d.options.NodeBodyColorDefault.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(QColor(BodyColor_Default[0], BodyColor_Default[1], BodyColor_Default[2])))
        elif v2d.options.NodeBodyColorCases.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (300*node.numCases)/self.numAllCases < 150:
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))
                    else:
                        node.parameter1.setColor(QColor(255,255,255));
                        node.parameter2.setColor(QColor(255,255,255));
                        node.SelectedAttName.setColor(QColor(255,255,255));			
                        body.setBrush(QBrush(QColor(BodyCasesColor_Default[0], BodyCasesColor_Default[1], BodyCasesColor_Default[2]).light(400 - (300*node.numCases)/self.numAllCases)))	
        elif v2d.options.NodeBodyColorMajorityClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    i=0
                    for ClassName in node.classesNames:
                        if ClassName == node.MajorityClass[0]:
                            body.setBrush(QBrush(ClassColors[i].light(250 - 1.5*node.MajorityClassProbabbility)))
                        i+=1
        elif v2d.options.NodeBodyColorTargetClass.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - node.TargetClassProbbability)))
        elif v2d.options.NodeBodyColorTargetClassRelative.isOn():
            for body in node.Bodies:
                if (body.rtti()==5):
                    if (self.root.classesCounts[TargetClassIndex] != 0):
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200 - (100*node.classesCounts[TargetClassIndex])/self.root.classesCounts[TargetClassIndex])))
                    else:
                        body.setBrush(QBrush(ClassColors[TargetClassIndex].light(200)))
                        #print "No target class in the tree"

        bottom = node.BottomCenterPosition()

        if node.childsNum <= 0:
            return
        i = 0
        for child in node.childs:
            self.__NodesRecalculate_Advanced2(child, width, level + 1,  LinesRelativity, BodyParameter1, BodyParameter2)
            #debelina crte glede na koren ali trenutno vozlisce
            if LinesRelativity == 0:
                LineWidth = (child.numCases/self.numAllCases)*self.maxLineWidth
            elif LinesRelativity == 1:
                LineWidth = (child.numCases/node.numCases)*self.maxLineWidth
            elif LinesRelativity == 2:	
                LineWidth = self.maxLineWidth
            child.setLine(bottom[0], bottom[1], Qt.black, LineWidth)
            i += 1

# barve za istolezne atribute v AttributeList
AttributeColorsList = []

# Za posamezen nivo v drevesu nam pove stevilo vozlisc
levelCount = []
NodeChilds = []
# seznam imen razredov, ki so v vozliscih
classesNames = []

class TreeNode:
    def __init__(self, name, childs, classes, cases, parentCanvas, numberLeaves, sortAttribute, prevSortAttributeValue, prevSortAttributeName, AttributePath):
        self.name = name
        self.childs = childs
        self.childsNum = len(childs)
        self.x = 0
        self.y = 0
        self.factor = 1
        self.visible = True
        self.childsVisible = True
        self.numCases = cases
        self.numberLeaves = numberLeaves
        #poglavitni razred
        self.MajorityClass = []
        #verjetnost poglavitnega razreda
        self.MajorityClassProbabbility = 0
        #verjetnost Ciljnega razreda
        self.TargetClassProbbability = 0
        #seznam vrednosti atributov, po katerih se pride v to vozlisce
        self.AttributePath = []
        for i in AttributePath:
            self.AttributePath.append(i)
        #atribut, po katerem se drevo veji naprej
        self.selectedSortAttribute = sortAttribute
        
        # Izris dodatnih podatkov o vozliscu
        self.bubble = None
        
        # prejsnji atributi
        self.conditionalPath = ''
        #ime ter vrednost prejsnjega atributa
        self.prevSortAttributeValue = prevSortAttributeValue
        self.prevSortAttributeName = prevSortAttributeName
        
        self.classesNames = []
        self.classesCounts = []
        maxClass = 0;
        for curclass in classes:
            self.classesNames.append(curclass[0])
            self.classesCounts.append(curclass[1])
            #ce jih je vec Majority razredov(enak delez)	
            if (maxClass == curclass[1]) != 0:
                self.MajorityClass.append(curclass[0])
            #iscem prevladujoc (majority) razred v node-u
            if maxClass < curclass[1]:
                maxClass = curclass[1]				
                #ce dobim boljsega, prejsnje izbrisem
                for i in range(0, len(self.MajorityClass)):
                        self.MajorityClass.pop()
                self.MajorityClass.append(curclass[0])
        #zapomnim si verjetnost majorityClass-a
        self.MajorityClassProbabbility = int(100 * maxClass/self.numCases)		
        
#		self.Node2D = Node2D(0, 0, self.prevSortAttributeValue, self.classesNames, self.classesCounts, parentCanvas, sortAttribute , self.AttributePath)# ''
            
        ######
        self.parentCanvas = parentCanvas
        self.BodyWidth = BodyWidth
        self.BodyHeight = BodyHeight
        self.ChildsCircle = None
        
        #prvi parameter, ki se izpise v bubble
        self.parameter1 = QCanvasText(self.parentCanvas)
        self.parameter1.hide()
        #drugi parameter, ki se izpise v bubble
        self.parameter2 = QCanvasText(self.parentCanvas)
        self.parameter2.hide()
        
        #crta pred imenom atributa
        self.SelectedAttLine = QCanvasLine(self.parentCanvas)
        #ime izbranega atributa
        self.SelectedAttName = QCanvasText(sortAttribute , self.parentCanvas)
        #nastavim dolzino imena izbranega atributa, da pade v body, ce je predolg
        if self.SelectedAttName.boundingRect().width()  >= self.BodyWidth:
            FontWidth = self.SelectedAttName.boundingRect().width() / self.SelectedAttName.text().length()
            NumOfFontsToFit = self.BodyWidth / FontWidth 
            SelectedAttName = self.SelectedAttName.text()
            SelectedAttName.truncate(NumOfFontsToFit)
            self.SelectedAttName.setText(SelectedAttName)

        self.Draw()
        if self.childsNum > 0:
            self.HasChilds()


    def setPosition(self, x, y):
        self.x = x
        self.y = y
        for body in self.Bodies:
            body.setX(x)
            body.setY(y)
        
        #crta nad imenom izbranega atributa
        self.SelectedAttLine.setPoints(self.x, self.y + self.factor * 2 * BodyHeight/3, self.x + self.factor * BodyWidth,self.y +  self.factor * 2 * BodyHeight/3)
        self.SelectedAttLine.setZ(2)
        
        #ime prejsnjega atributa
        self.SelectedAttName.setX(self.x + 2)
        self.SelectedAttName.setY(self.y + self.factor * 2 * BodyHeight/3 - 2)  #
        self.SelectedAttName.setFont(QFont(TextFont_Default, self.factor * AttributeTextSize_Default))
        self.SelectedAttName.setZ(2)
        
        PieDistanceX = self.factor * BodyWidth
        PieDistanceY = self.factor * BodyHeight/2
        for pie in [self.BigPie] + self.Pies:
            pie.setX(x + PieDistanceX)
            pie.setY(y + PieDistanceY)

        AttributeTextSize = AttributeTextSize_Default*self.factor
        lastY = lastAttValPosY = y - AttributeTextSize  - self.factor +2 #lastAttValPos je zgornji rob noda zaradi vrednosti prejsnjega atributa


        displace = BodyHeight/7 * self.factor	# presledek med prvim in drugim napison v nodu	
        for text in self.Texts:
            text.setX(x + 3)
            text.setY(lastY - 5)
            lastY += AttributeTextSize + displace #+ AttributeTextSize/2 

        self.Texts[0].setY(lastAttValPosY - 5)  #premaknem vrednost prejsnjega atributa malo gor
        if self.ChildsCircle != None:
            self.ChildsCircle.setX(x+BodyWidth*self.factor/2)
            self.ChildsCircle.setY(y+BodyHeight*self.factor)


    def setSize(self):
        self.BodyWidth = self.factor * BodyWidth
        self.BodyHeight = self.factor * BodyHeight
        

        for body in self.Bodies:
            body.setSize(self.BodyWidth, self.BodyHeight)
        self.PieWidth = self.factor * PieWidth
        self.PieHeight = self.factor * PieHeight
        self.BigPie.setSize(self.PieWidth + 2, self.PieHeight + 2)
        for pie in self.Pies:
            pie.setSize(self.PieWidth, self.PieHeight)
        FontSize = int(AttributeTextSize_Default*self.factor)
        for text in self.Texts:
            text.setFont(QFont(TextFont_Default, FontSize))
        if self.ChildsCircle != None:
            self.ChildsCircle.setSize(ChildsCircleRadius_Default*self.factor, ChildsCircleRadius_Default*self.factor)
            

    def setLine(self, x, y, color, weight):
        endPoint = self.TopCenterPosition()
        self.Line.setPoints(endPoint[0], endPoint[1], x, y)
        self.Line.setPen(QPen(Qt.black,weight))
        
        
    def BottomCenterPosition(self):
        return [self.x+self.BodyWidth/2, self.y+self.BodyHeight]

    def TopCenterPosition(self):
        return [self.x+self.BodyWidth/2, self.y]
    
    def setVisible(self, value):
        global BodyPiesShow
        self.visible = value
        if value == False and self.bubble != None:
            self.bubble.hideit();
        self.BigPie.setVisible(value)
        for body in self.Bodies:
            body.setVisible(value)
        if BodyPiesShow == 0:
            for pie in self.Pies:
                pie.setVisible(False)
            self.BigPie.setVisible(False)	
        else:
            if value == True:
                for pie in self.Pies:
                    pie.setVisible(True)
                self.BigPie.setVisible(True)
            else:	
                for pie in self.Pies:
                    pie.setVisible(False)
                self.BigPie.setVisible(False)
        
        for text in self.Texts:
            text.setVisible(value)
        self.Line.setVisible(value)
        if self.ChildsCircle != None:
            self.ChildsCircle.setVisible(value)
        self.SelectedAttName.setVisible(value)
        self.SelectedAttLine.setVisible(value)
    
    def posInNode(self, positionX, positionY):
        if (self.x <= positionX) and ((self.x + self.factor*(BodyWidth+PieWidth/2)) >= positionX) and (self.y <= positionY) and (self.y + self.factor*BodyHeight >= positionY):
            return True
        else:
            return False
            
    def HasChilds(self):
        if self.ChildsCircle == None:
            self.ChildsCircle = QCanvasEllipse(self.parentCanvas)
            self.ChildsCircle.setBrush(QBrush(Qt.black))
            self.ChildsCircle.setZ(0)
            self.ChildsCircle.show()
            
    def TextConstruct(self):
        self.Texts = []
        
        numClasses = len(self.classesNames)
        
        PrevAttributeTitle = QCanvasText(self.parentCanvas)
        PrevAttributeTitle.setText(self.prevSortAttributeValue)
        PrevAttributeTitle.setFont(QFont(TextFont_Default, AttributeTextSize_Default))
        PrevAttributeTitle.setX(self.x + 3)
        PrevAttributeTitle.setY(self.y)
        PrevAttributeTitle.setZ(3)  
        PrevAttributeTitle.show()
        self.Texts.append(PrevAttributeTitle)

        numClasses = len(self.classesNames)

        ClassTextSize = ClassTextSize_Default
        if  (numClasses * ClassTextSize_Default) > (BodyHeight - 1):
            ClassTextSize = (BodyHeight - numClasses - 1) / (numClasses + 1) # ta enka je zaradi previous attribute

        distance = self.y
        
        
        self.Texts.append(self.parameter1)
        self.Texts.append(self.parameter2)

    def LineConstruct(self):
        self.Line = QCanvasLine(self.parentCanvas)
        self.Line.show()

    def BodyConstruct(self):
        self.Bodies = []

        Body = QCanvasRectangle(self.parentCanvas)
        Body.setBrush(QBrush(QColor(BodyColor_Default[0], BodyColor_Default[1], BodyColor_Default[2])))
        Body.setZ(1)
        Body.setSize(BodyWidth, BodyHeight)
        Body.setX(self.x)
        Body.setY(self.y)
        Body.show()
        
        self.Bodies.append(Body)

    def PieConstruct(self):
        x = self.x + BodyWidth
        y = self.y + BodyHeight/2

        # Obrobna pita
        pie = QCanvasEllipse(PieWidth+2, PieHeight+2, self.parentCanvas)
        pie.setBrush(QBrush(Qt.black))
        pie.setX(x)
        pie.setY(y)
        pie.setZ(3)
        pie.show()
        self.BigPie = pie

        # Pite ki sestavljajo verjetnosti
        self.Pies=[]
        startAngle = 0
        i = 0
        Sum = sum(self.classesCounts) * 1.0
        for count in self.classesCounts:
            if count > 0:
                sizeAngle = (count / Sum) * 360
                pie = QCanvasEllipse(PieWidth, PieHeight, startAngle*16, sizeAngle*16, self.parentCanvas)
                pie.setBrush(QBrush(ClassColors[i]))
                pie.setX(x)
                pie.setY(y)
                pie.setZ(4) #3
                pie.show()
                self.Pies.append(pie)
                startAngle = startAngle + sizeAngle
            i = i + 1

    def Draw(self):
        self.BodyConstruct()
        self.PieConstruct()
        self.TextConstruct()
        self.LineConstruct()

# Zgradimo drevo vizualnih elementov
def TreeWalk(tree, name, count, parentCanvas, prevAttributeVal, prevSortAttributeName, AttributePath, level=1):
    global NodeChilds
    if tree != None:
    
        # Shranjujemo stevilo vozlisc na posameznem nivoju
        global levelNodesCount
        if level == 1:
            levelNodesCount = []
        if len(levelNodesCount) < level:
            levelNodesCount.append(1)
        else:
            levelNodesCount[level-1] += 1

        #pot po kateri pridemo v trenutno vozlisce (po vrednostih atributov)
        if prevSortAttributeName != '':
            AttributePath.append(prevSortAttributeName + ' = ' + prevAttributeVal)
        childs = []
        if tree.treesize() > 1:
            selectedAttributeVal = tree.branchDescriptions #
            selectedSortAttribute = tree.branchSelector.classVar.name
            numberLeaves = 0
            branchesCount = len(tree.branches)
            for i in range(0, branchesCount):
                returnValue = TreeWalk(tree.branches[branchesCount-1-i], tree.branchDescriptions[branchesCount-1-i], count+1, parentCanvas, selectedAttributeVal[branchesCount-1-i], selectedSortAttribute, AttributePath, level+1)
                if returnValue != None:
                    childs.append(returnValue[0])
                    NodeChilds.append(returnValue[0])
                    numberLeaves += returnValue[1]
                    #brisem zadnji element iz liste AttributePath, ko se vracam
                    AttributePath.pop()
        else:
            selectedSortAttribute = '(leaf)'
            branchesCount = 0
            numberLeaves = 1
        levelCount[count] = levelCount[count] + branchesCount

        return [TreeNode(name, childs, tree.distribution.items(), tree.distribution.cases, parentCanvas, numberLeaves,selectedSortAttribute, prevAttributeVal,  prevSortAttributeName, AttributePath), numberLeaves]

def TreeWalk_setVisible(rootnode, value, applyToAll=False):
    if len(rootnode.childs) > 0:
        for node in rootnode.childs:
            node.setVisible(value)
            if applyToAll == True:
                TreeWalk_setVisible(node, value, True)
            else:
                TreeWalk_setVisible(node, False)
            if value == True:
                node.childsVisible = False
    rootnode.childsVisible = value

def LevelCountInit(value):
    for i in range(0, value):
        levelCount.append(0)

def LevelCountGet():
    return levelCount

def sum(values):
    sum = 0
    for value in values:
        sum = sum + value
    return sum

#vpisem izbrani index(combobox (visualTreeOptions)) v  TargetClassIndex (Ta metoda ni v nobenem razredu!!!)
def setTargetClassIndex(index):
    global TargetClassIndex
    TargetClassIndex = index
    
def main(args):
    global ScreenSize
    
    a=QApplication(sys.argv)
    v2d=OWClassificationTreeViewer2D()
    a.setMainWidget(v2d)

#	ScreenSize = [QApplication.desktop().rect().width(), QApplication.desktop().rect().height()]

    v2d.setGeometry((ScreenSize[0]-600)/2, (ScreenSize[1]-600)/2, 600, 600)
    v2d.show()

    data = orange.ExampleTable('test')

    tree = orange.TreeLearner(data, storeExamples = 1)
    v2d.ctree(tree)

    a.exec_loop()
    v2d.saveSettings()

if __name__=="__main__":
    main(sys.argv)


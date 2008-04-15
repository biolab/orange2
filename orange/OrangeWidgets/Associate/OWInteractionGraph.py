"""
<name>Interaction Graph</name>
<description>Interaction graph construction and viewer.</description>
<icon>icons/InteractionGraph.png</icon>
<contact>Aleks Jakulin</contact>
<priority>4000</priority>
"""
# InteractionGraph.py
#
#

from OWWidget import *
from qt import *
from qtcanvas import *
import orngInteract
import statc
import os
from re import *
from math import floor, ceil
from orngCI import FeatureByCartesianProduct
import OWGUI

class IntGraphView(QCanvasView):
    def __init__(self, parent, name, *args):
        apply(QCanvasView.__init__,(self,) + args)
        self.parent = parent
        self.name = name
        self.connect(self, SIGNAL("contentsMoving(int,int)"), self.contentsMoving)

    # mouse button was pressed
    def contentsMousePressEvent(self, ev):
        self.parent.mousePressed(self.name, ev)

    def contentsMoving(self, x,y):
        self.parent.contentsMoving(x,y)


###########################################################################################
##### WIDGET : Interaction graph
###########################################################################################
class OWInteractionGraph(OWWidget):
    settingsList = ["onlyImportantInteractions"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Interaction graph")

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable), ("Attribute Pair", list), ("Selected Attributes List", list)]


        #set default settings
        self.originalData = None
        self.data = None
        self.dataSize = 1
        self.rest = None
        self.interactionMatrix = None
        self.rectIndices = {}   # QRect rectangles
        self.rectNames   = {}   # info about rectangle names (attributes)
        self.lines = []         # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])
        self.interactionRects = []
        self.rectItems = []
        self.viewXPos = 0       # next two variables are used at setting tooltip position
        self.viewYPos = 0       # inside canvasView

        self.onlyImportantInteractions = 1
        self.mergeAttributes = 0

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.splitCanvas = QSplitter(self.mainArea)

        self.canvasL = QCanvas(2000, 2000)
        self.canvasViewL = IntGraphView(self, "interactions", self.canvasL, self.splitCanvas)
        self.canvasViewL.show()

        self.canvasR = QCanvas(2000,2000)
        self.canvasViewR = IntGraphView(self, "graph", self.canvasR, self.splitCanvas)
        self.canvasViewR.show()


        #GUI
        #add controls to self.controlArea widget
        self.shownAttribsGroup = QVGroupBox(self.space)
        self.addRemoveGroup = QHButtonGroup(self.space)
        self.hiddenAttribsGroup = QVGroupBox(self.space)
        self.shownAttribsGroup.setTitle("Selected attributes")
        self.hiddenAttribsGroup.setTitle("Unselected attributes")

        self.shownAttribsLB = QListBox(self.shownAttribsGroup)
        self.shownAttribsLB.setSelectionMode(QListBox.Extended)

        self.hiddenAttribsLB = QListBox(self.hiddenAttribsGroup)
        self.hiddenAttribsLB.setSelectionMode(QListBox.Extended)

        self.attrAddButton = QPushButton("Add attr.", self.addRemoveGroup)
        self.attrRemoveButton = QPushButton("Remove attr.", self.addRemoveGroup)

        OWGUI.separator(self.space)

        self.mergeAttributesCB = QCheckBox('Merge attributes', self.space)
        self.importantInteractionsCB = QCheckBox('Show only important interactions', self.space)
        QToolTip.add(self.mergeAttributesCB, "Enable or disable attribute merging. If enabled, you can merge \ntwo attributes with right mouse click inside attribute rectangle.\nMerged attribute is then built as cartesian product of corresponding attribute pair\nand added to the list of possible attributes")

        OWGUI.separator(self.space)

        self.selectionButton = QPushButton("Show selection", self.space)
        QToolTip.add(self.selectionButton, "Sends 'selection' signal to any successor visualization widgets.\nThis signal contains a list of selected attributes to visualize.")

        self.saveLCanvas = QPushButton("Save left canvas", self.space)
        self.saveRCanvas = QPushButton("Save right canvas", self.space)
        self.connect(self.saveLCanvas, SIGNAL("clicked()"), self.saveToFileLCanvas)
        self.connect(self.saveRCanvas, SIGNAL("clicked()"), self.saveToFileRCanvas)

        #connect controls to appropriate functions
        self.connect(self.attrAddButton, SIGNAL("clicked()"), self.addAttributeClick)
        self.connect(self.attrRemoveButton, SIGNAL("clicked()"), self.removeAttributeClick)
        self.connect(self.selectionButton, SIGNAL("clicked()"), self.selectionClick)
        self.connect(self.mergeAttributesCB, SIGNAL("toggled(bool)"), self.mergeAttributesEvent)
        self.connect(self.importantInteractionsCB, SIGNAL("toggled(bool)"), self.showImportantInteractions)

        #self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        #self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)
        self.activateLoadedSettings()

    def mergeAttributesEvent(self, b):
        self.mergeAttributes = b
        if b == 0:
            self.updateNewData(self.originalData)


    def showImportantInteractions(self, b):
        self.onlyImportantInteractions = b
        self.showInteractionRects(self.data)

    def activateLoadedSettings(self):
        self.importantInteractionsCB.setChecked(self.onlyImportantInteractions)

    # did we click inside the rect rectangle
    def clickInside(self, rect, point):
        x = point.x()
        y = point.y()

        if rect.left() > x: return 0
        if rect.right() < x: return 0
        if rect.top() > y: return 0
        if rect.bottom() < y: return 0

        return 1

    # if we clicked on edge label send "wiew" signal, if clicked inside rectangle select/unselect attribute
    def mousePressed(self, name, ev):
        if ev.button() == QMouseEvent.LeftButton and name == "graph":
            for name in self.rectNames:
                clicked = self.clickInside(self.rectNames[name].rect(), ev.pos())
                if clicked == 1:
                    self._setAttrVisible(name, not self.getAttrVisible(name))
                    self.showInteractionRects(self.data)
                    self.canvasR.update()
                    return
            for (attr1, attr2, rect) in self.lines:
                clicked = self.clickInside(rect.rect(), ev.pos())
                if clicked == 1:
                    self.send("Attribute Pair", [attr1, attr2])
                    return
        elif ev.button() == QMouseEvent.LeftButton and name == "interactions":
            self.rest = None
            for (rect1, rect2, rect3, nbrect, text1, text2, tooltipRect, tooltipText) in self.interactionRects:
                if self.clickInside(tooltipRect, ev.pos()) == 1:
                    self.send("Attribute Pair", [str(text1.text()), str(text2.text())])

        elif ev.button() == QMouseEvent.RightButton and name == "interactions":
            if not self.mergeAttributes == 1: return

            found = 0; i = 0
            while not found and i < len(self.interactionRects):
                (rect1, rect2, rect3, nbrect, text1, text2, tooltipRect, tooltipText) = self.interactionRects[i]
                if self.clickInside(tooltipRect, ev.pos()) == 1:
                    attr1 = str(text1.text()); attr2 = str(text2.text())
                    found = 1
                i+=1
            if not found: return

            data = self.interactionMatrix.discData
            (cart, profit) = FeatureByCartesianProduct(data, [data.domain[attr1], data.domain[attr2]])
            if cart in data.domain: return  # if this attribute already in domain return

            for attr in data.domain:
                if cart.name == attr.name:
                    print "Attribute pair already in the domain"
                    return

            tempData = data.select(list(data.domain) + [cart])
            dd = orange.DomainDistributions(tempData)
            vals = []
            for i in range(len(cart.values)):
                if dd[cart][i] != 0.0:
                    vals.append(cart.values[i])

            newVar = orange.EnumVariable(cart.name, values = vals)
            newData = data.select(list(data.domain) + [newVar])
            for i in range(len(newData)):
                newData[i][newVar] = tempData[i][cart]

            #rest = newData.select({cart.name:todoList})

            #print "intervals = %d, non clear values = %d" % (len(cart.values), len(todoList))
            #print "entropy left = %f" % (float(len(rest)) / float(self.dataSize))
            self.updateNewData(newData)


    # we catch mouse release event so that we can send the "view" signal
    def onMouseReleased(self, e):
        for i in range(len(self.graphs)):
            if self.graphs[i].blankClick == 1:
                (attr1, attr2, className, string) = self.graphParameters[i]
                self.send("Attribute Pair", [attr1, attr2])
                self.graphs[i].blankClick = 0

    # click on selection button
    def selectionClick(self):
        if self.data == None: return
        l = []
        for i in range(self.shownAttribsLB.count()):
            l.append(str(self.shownAttribsLB.text(i)))
        self.send("Selected Attributes List", l)

    def resizeEvent(self, e):
        if hasattr(self, "splitCanvas"):
            self.splitCanvas.resize(self.mainArea.size())


    # receive new data and update all fields
    def setData(self, data):
        self.warning([0,1])

        self.originalData = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        if not self.originalData:
            return

        self.originalData = orange.Preprocessor_dropMissing(self.originalData)

        if len(self.originalData) != len(data):
            self.warning(0, "Examples with missing values were removed. Keeping %d of %d examples." % (len(data), len(self.originalData)))
        if self.originalData.domain.hasContinuousAttributes():
            self.warning(1, "Continuous attributes were discretized using entropy discretization.")

        self.dataSize = len(self.originalData)

        self.updateNewData(self.originalData)

    def updateNewData(self, data):
        self.data = data
        self.interactionMatrix = orngInteract.InteractionMatrix(data, dependencies_too=1)

        # save discretized data and repair invalid names
        for attr in self.interactionMatrix.discData.domain.attributes:
            attr.name = attr.name.replace("ED_","")
            attr.name = attr.name.replace("D_","")
            attr.name = attr.name.replace("M_","")

        self.interactionList = []
        entropy = self.interactionMatrix.entropy
        if entropy == 0.0: return

        ################################
        # create a sorted list of total information
        for ((val,(val2, attrIndex1, attrIndex2))) in self.interactionMatrix.list:
            gain1 = self.interactionMatrix.gains[attrIndex1] / entropy
            gain2 = self.interactionMatrix.gains[attrIndex2] / entropy
            total = (val/entropy) + gain1 + gain2
            self.interactionList.append((total, (gain1, gain2, attrIndex1, attrIndex2)))
        self.interactionList.sort()
        self.interactionList.reverse()

        f = open('interaction.dot','w')
        self.interactionMatrix.exportGraph(f, significant_digits=3,positive_int=8,negative_int=8,absolute_int=0,url=1)
        f.flush()
        f.close()

        # execute dot and save otuput to pipes
        (pipePngOut, pipePngIn) = os.popen2("dot interaction.dot -Tpng", "b")
        (pipePlainOut, pipePlainIn) = os.popen2("dot interaction.dot -Tismap", "t")

        textPng = pipePngIn.read()
        textPlainList = pipePlainIn.readlines()
        pipePngIn.close()
        pipePlainIn.close()
        pipePngOut.close()
        pipePlainOut.close()
        os.remove('interaction.dot')

        # if the output from the pipe was empty, then the software isn't installed correctly
        if len(textPng) == 0:
            print "-----------------------------"
            print "Error. This widget needs graphviz software package installed. You can find it on the internet."
            print "-----------------------------"
            return

        # create a picture
        pixmap = QPixmap()
        pixmap.loadFromData(textPng)
        canvasPixmap = QCanvasPixmap(pixmap, QPoint(0,0))
        width = canvasPixmap.width()
        height = canvasPixmap.height()

        # hide all rects
        for rectInd in self.rectIndices.keys():
            self.rectIndices[rectInd].hide()

        self.canvasR.setTiles(pixmap, 1, 1, width, height)
        self.canvasR.resize(width, height)

        self.rectIndices = {}       # QRect rectangles
        self.rectNames   = {}       # info about rectangle names (attributes)
        self.lines = []             # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])


        self.parseGraphData(data, textPlainList, width, height)
        self.initLists(data)   # add all attributes found in .dot file to shown list
        self.showInteractionRects(data) # use interaction matrix to fill the left canvas with rectangles

        self.canvasL.update()
        self.canvasR.update()

        self.send("Examples", data)


    #########################################
    # do we want to show interactions between attrIndex1 and attrIndex2
    def showInteractionPair(self, attrIndex1, attrIndex2):
        attrName1 = self.data.domain[attrIndex1].name
        attrName2 = self.data.domain[attrIndex2].name

        if self.mergeAttributes == 1:
            if self.getAttrVisible(attrName1) == 0 or self.getAttrVisible(attrName2) == 0: return 0
            list1 = attrName1.split("-")
            list2 = attrName2.split("-")
            for item in list1:
                if item in list2: return 0
            for item in list2:
                if item in list1: return 0
            #return 1

        if self.getAttrVisible(attrName1) == 0 or self.getAttrVisible(attrName2) == 0: return 0
        if self.onlyImportantInteractions == 1:
            for (attr1, attr2, rect) in self.lines:
                if (attr1 == attrName1 and attr2 == attrName2) or (attr1 == attrName2 and attr2 == attrName1): return 1
            return 0
        return 1

    #########################################
    # show interactions between attributes in left canvas
    def showInteractionRects(self, data):
        if self.interactionMatrix == None: return
        if self.data == None : return

        ################################
        # hide all interaction rectangles
        for (rect1, rect2, rect3, nbrect, text1, text2, tooltipRect, tooltipText) in self.interactionRects:
            rect1.hide()
            rect2.hide()
            rect3.hide()
            nbrect.hide()
            text1.hide()
            text2.hide()
            QToolTip.remove(self.canvasViewL, tooltipRect)
        self.interactionRects = []

        for item in self.rectItems:
            item.hide()
        self.rectItems = []

        ################################
        # get max width of the attribute text
        xOff = 0
        for ((total, (gain1, gain2, attrIndex1, attrIndex2))) in self.interactionList:
            if not self.showInteractionPair(attrIndex1, attrIndex2): continue
            if gain1 > gain2: text = QCanvasText(data.domain[attrIndex1].name, self.canvasL)
            else:             text = QCanvasText(data.domain[attrIndex2].name, self.canvasL)
            rect = text.boundingRect()
            if xOff < rect.width():
                xOff = rect.width()

        xOff += 10;  yOff = 40
        index = 0
        xscale = 300;  yscale = 200
        maxWidth = xOff + xscale + 10;  maxHeight = 0
        rectHeight = yscale * 0.1    # height of the rectangle will be 1/10 of max width

        ################################
        # print scale
        line = QCanvasRectangle(xOff, yOff - 4, xscale, 1, self.canvasL)
        line.show()
        tick1 = QCanvasRectangle(xOff, yOff-10, 1, 6, self.canvasL);              tick1.show()
        tick2 = QCanvasRectangle(xOff + (xscale/2), yOff-10, 1, 6, self.canvasL); tick2.show()
        tick3 = QCanvasRectangle(xOff + xscale-1, yOff-10, 1, 6,  self.canvasL);  tick3.show()
        self.rectItems = [line, tick1, tick2, tick3]
        for i in range(10):
            tick = QCanvasRectangle(xOff + xscale * (float(i)/10.0), yOff-8, 1, 5, self.canvasL);
            tick.show()
            self.rectItems.append(tick)

        text1 = QCanvasText("0%", self.canvasL);   text1.setTextFlags(Qt.AlignHCenter); text1.move(xOff, yOff - 23); text1.show()
        text2 = QCanvasText("50%", self.canvasL);  text2.setTextFlags(Qt.AlignHCenter); text2.move(xOff + xscale/2, yOff - 23); text2.show()
        text3 = QCanvasText("100%", self.canvasL); text3.setTextFlags(Qt.AlignHCenter); text3.move(xOff + xscale, yOff - 23); text3.show()
        text4 = QCanvasText("Class entropy removed", self.canvasL); text4.setTextFlags(Qt.AlignHCenter); text4.move(xOff + xscale/2, yOff - 36); text4.show()
        self.rectItems.append(text1); self.rectItems.append(text2); self.rectItems.append(text3); self.rectItems.append(text4)

        ################################
        #create rectangles
        for ((total, (gain1, gain2, attrIndex1, attrIndex2))) in self.interactionList:
            if not self.showInteractionPair(attrIndex1, attrIndex2): continue

            interaction = (total - gain1 - gain2)
            atts = (max(attrIndex1, attrIndex2), min(attrIndex1, attrIndex2))
            #nbgain = self.interactionMatrix.ig[atts[0]][atts[1]] + self.interactionMatrix.gains[atts[0]] + self.interactionMatrix.gains[atts[1]]
            nbgain = self.interactionMatrix.gains[atts[0]] + self.interactionMatrix.gains[atts[1]]
            nbgain -= self.interactionMatrix.corr[(atts[1],atts[0])]
            rectsYOff = yOff + 3 + index * yscale * 0.15

            # swap if gain1 < gain2
            if gain1 < gain2:
                ind = attrIndex1; attrIndex1 = attrIndex2; attrIndex2 = ind
                ga = gain1; gain1 = gain2;  gain2 = ga

            x1 = round(xOff)
            if interaction < 0:
                x2 = floor(xOff + xscale*(gain1+interaction))
                x3 = ceil(xOff + xscale*gain1)
            else:
                x2 = floor(xOff + xscale*gain1)
                x3 = ceil(xOff + xscale*(total-gain2))
            x4 = ceil(xOff + xscale*total)

            # compute nbgain position
            nb_x1 = min(xOff, floor(xOff + 0.5*xscale*nbgain))
            nb_x2 = max(xOff, floor(xOff + 0.5*xscale*nbgain))
            nbrect = QCanvasRectangle(nb_x1, rectsYOff-3, nb_x2-nb_x1+1, 2, self.canvasL)


            rect2 = QCanvasRectangle(x2, rectsYOff,   x3-x2+1, rectHeight, self.canvasL)
            rect1 = QCanvasRectangle(x1, rectsYOff, x2-x1+1, rectHeight, self.canvasL)

            rect3 = QCanvasRectangle(x3, rectsYOff, x4-x3, rectHeight, self.canvasL)
            if interaction < 0.0:
                #color = QColor(255, 128, 128)
                color = QColor(200, 0, 0)
                style = Qt.DiagCrossPattern
            else:
                color = QColor(Qt.green)
                style = Qt.Dense5Pattern

            brush1 = QBrush(Qt.blue); brush1.setStyle(Qt.BDiagPattern)
            brush2 = QBrush(color);   brush2.setStyle(style)
            brush3 = QBrush(Qt.blue); brush3.setStyle(Qt.FDiagPattern)

            rect1.setBrush(brush1); rect1.setPen(QPen(QColor(Qt.blue)))
            rect2.setBrush(brush2); rect2.setPen(QPen(color))
            rect3.setBrush(brush3); rect3.setPen(QPen(QColor(Qt.blue)))
            rect1.show(); rect2.show();  rect3.show(); nbrect.show()

            # create text labels
            text1 = QCanvasText(data.domain[attrIndex1].name, self.canvasL)
            text2 = QCanvasText(data.domain[attrIndex2].name, self.canvasL)
            text1.setTextFlags(Qt.AlignRight)
            text2.setTextFlags(Qt.AlignLeft)
            text1.move(xOff - 5, rectsYOff + 3)
            text2.move(xOff + xscale*total + 5, rectsYOff + 3)

            text1.show()
            text2.show()

            tooltipRect = QRect(x1-self.viewXPos, rectsYOff-self.viewYPos, x4-x1, rectHeight)
            tooltipText = "%s : <b>%.1f%%</b><br>%s : <b>%.1f%%</b><br>Interaction : <b>%.1f%%</b><br>Total entropy removed: <b>%.1f%%</b>" %(data.domain[attrIndex1].name, gain1*100, data.domain[attrIndex2].name, gain2*100, interaction*100, total*100)
            QToolTip.add(self.canvasViewL, tooltipRect, tooltipText)

            # compute line width
            rect = text2.boundingRect()
            lineWidth = xOff + xscale*total + 5 + rect.width() + 10
            if  lineWidth > maxWidth:
                maxWidth = lineWidth

            if rectsYOff + rectHeight + 10 > maxHeight:
                maxHeight = rectsYOff + rectHeight + 10

            self.interactionRects.append((rect1, rect2, rect3, nbrect, text1, text2, QRect(x1, rectsYOff, x4-x1, rectHeight), tooltipText))
            index += 1

        # resizing of the left canvas to update width
        self.canvasViewL.setMaximumSize(QSize(maxWidth + 30, max(2000, maxHeight)))
        self.canvasViewL.setMinimumWidth(maxWidth + 10)
        self.canvasL.resize(maxWidth + 10, maxHeight)
        self.canvasViewL.setMinimumWidth(0)

        self.canvasL.update()

    #########################################
    # if we scrolled in the left canvas then we have to update tooltip positions
    def contentsMoving(self, x,y):
        for (rect1, rect2, rect3, nbrect, text1, text2, rect, tooltipText) in self.interactionRects:
            oldrect = QRect(rect.left()-self.viewXPos, rect.top()-self.viewYPos, rect.width(), rect.height())
            QToolTip.remove(self.canvasViewL, oldrect)
            newrect = QRect(rect.left()-x, rect.top()-y, rect.width(), rect.height())
            QToolTip.add(self.canvasViewL, newrect, tooltipText)

        self.viewXPos = x
        self.viewYPos = y

    #########################################
    # parse info from plain file. picWidth and picHeight are sizes in pixels
    def parseGraphData(self, data, textPlainList, picWidth, picHeight):
        scale = 0
        w = 1; h = 1
        for line in textPlainList:
            if line[:9] == "rectangle":
                list = line.split()
                topLeftRectStr = list[1]
                bottomRightRectStr = list[2]
                attrIndex = list[3]

                isAttribute = 0     # does rectangle represent attribute
                if attrIndex.find("-") < 0:
                    isAttribute = 1

                topLeftRectStr = topLeftRectStr.replace("(","")
                bottomRightRectStr = bottomRightRectStr.replace("(","")
                topLeftRectStr = topLeftRectStr.replace(")","")
                bottomRightRectStr = bottomRightRectStr.replace(")","")

                topLeftRectList = topLeftRectStr.split(",")
                bottomRightRectList = bottomRightRectStr.split(",")
                xLeft = int(topLeftRectList[0])
                yTop = int(topLeftRectList[1])
                width = int(bottomRightRectList[0]) - xLeft
                height = int(bottomRightRectList[1]) - yTop

                rect = QCanvasRectangle(xLeft+2, yTop+2, width, height, self.canvasR)
                pen = QPen(Qt.blue)
                pen.setWidth(4)
                rect.setPen(pen)
                rect.hide()

                if isAttribute == 1:
                    name = data.domain[int(attrIndex)].name
                    self.rectIndices[int(attrIndex)] = rect
                    self.rectNames[name] = rect
                else:
                    attrs = attrIndex.split("-")
                    attr1 = data.domain[int(attrs[0])].name
                    attr2 = data.domain[int(attrs[1])].name
                    pen.setStyle(Qt.NoPen)
                    rect.setPen(pen)
                    self.lines.append((attr1, attr2, rect))

    ##################################################
    # initialize lists for shown and hidden attributes
    def initLists(self, data):
        self.shownAttribsLB.clear()
        self.hiddenAttribsLB.clear()

        if data == None: return

        for key in self.rectNames.keys():
            self._setAttrVisible(key, 1)


    #################################################
    ### showing and hiding attributes
    #################################################
    def _showAttribute(self, name):
        self.shownAttribsLB.insertItem(name)    # add to shown

        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from hidden
            if str(self.hiddenAttribsLB.text(i)) == name:
                self.hiddenAttribsLB.removeItem(i)

    def _hideAttribute(self, name):
        self.hiddenAttribsLB.insertItem(name)    # add to hidden

        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from shown
            if str(self.shownAttribsLB.text(i)) == name:
                self.shownAttribsLB.removeItem(i)

    ##########
    # add attribute to showList or hideList and show or hide its rectangle
    def _setAttrVisible(self, name, visible = 1):
        if visible == 1:
            if name in self.rectNames.keys(): self.rectNames[name].show();
            self._showAttribute(name)
        else:
            if name in self.rectNames.keys(): self.rectNames[name].hide();
            self._hideAttribute(name)

    def getAttrVisible(self, name):
        for i in range(self.hiddenAttribsLB.count()):
            if str(self.hiddenAttribsLB.text(i)) == name: return 0

        if self.mergeAttributes == 1:
            names = name.split("-")
            for i in range(self.hiddenAttribsLB.count()):
                if str(self.hiddenAttribsLB.text(i)) in names: return 0

        return 1

    #################################################
    # event processing
    #################################################
    def addAttributeClick(self):
        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.isSelected(i):
                name = str(self.hiddenAttribsLB.text(i))
                self._setAttrVisible(name, 1)
        self.showInteractionRects(self.data)
        self.canvasL.update()
        self.canvasR.update()

    def removeAttributeClick(self):
        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.isSelected(i):
                name = str(self.shownAttribsLB.text(i))
                self._setAttrVisible(name, 0)
        self.showInteractionRects(self.data)
        self.canvasL.update()
        self.canvasR.update()

    ##################################################
    # SAVING GRAPHS
    ##################################################
    def saveToFileLCanvas(self):
        self.saveCanvasToFile(self.canvasViewL, self.canvasL.size())

    def saveToFileRCanvas(self):
        self.saveCanvasToFile(self.canvasViewR, self.canvasR.size())

    def saveCanvasToFile(self, canvas, size):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()

        buffer = QPixmap(size) # any size can do, now using the window size
        painter = QPainter(buffer)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255))) # make background same color as the widget's background
        canvas.drawContents(painter, 0,0, size.width(), size.height())
        painter.end()
        buffer.save(fileName, ext)


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWInteractionGraph()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings
    ow.saveSettings()

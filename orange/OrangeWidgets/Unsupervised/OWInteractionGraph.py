"""
<name>Interaction Graph</name>
<description>Interaction graph construction and viewer.</description>
<icon>icons/InteractionGraph.png</icon>
<contact>Aleks Jakulin</contact>
<priority>2200</priority>
"""
# InteractionGraph.py
#
#
from OWWidget import *
import orngInteract, OWQCanvasFuncts
import statc
import os
from re import *
from math import floor, ceil
from orngCI import FeatureByCartesianProduct
import OWGUI, OWDlgs

class IntGraphView(QGraphicsView):
    def __init__(self, parent, name, *args):
        QGraphicsView.__init__(self, *args)
        self.parent = parent
        self.name = name

    def sizeHint(self):
        return QSize(200,200)

    # mouse button was pressed
    def mousePressEvent(self, ev):
        scenePos = self.mapToScene(ev.pos())
        pos = QPoint(int(scenePos.x()), int(scenePos.y()))
        ev = QMouseEvent(ev.type(), pos, ev.button(), ev.buttons(), ev.modifiers())
        self.parent.mousePressed(self.name, ev)
        


###########################################################################################
##### WIDGET : Interaction graph
###########################################################################################
class OWInteractionGraph(OWWidget):
    settingsList = ["onlyImportantInteractions"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Interaction graph", noReport=True)

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

        self.shownAttributes = []
        self.selectedShown = []
        self.hiddenAttributes = []
        self.selectedHidden = []
        self.mergeAttributes = 0
        self.onlyImportantInteractions = 0

        #load settings
        self.loadSettings()

        # add a settings dialog and initialize its values
        self.splitter = QSplitter(Qt.Horizontal, self.mainArea)
        self.mainArea.layout().addWidget(self.splitter)

        self.canvasL = QGraphicsScene()
        self.canvasViewL = IntGraphView(self, "interactions", self.canvasL, self)
        self.splitter.addWidget(self.canvasViewL)
        self.canvasViewL.setAlignment(Qt.AlignLeft | Qt.AlignTop)

        self.canvasR = QGraphicsScene()
        self.canvasViewR = IntGraphView(self, "graph", self.canvasR, self)
        self.splitter.addWidget(self.canvasViewR)

        #GUI
        #add controls to self.controlArea widget
        self.shownAttribsGroup = OWGUI.widgetBox(self.controlArea, "Selected Attributes")
        self.addRemoveGroup = OWGUI.widgetBox(self.controlArea, 1, orientation = "horizontal" )
        self.hiddenAttribsGroup = OWGUI.widgetBox(self.controlArea, "Unselected Attributes", addSpace=True)

        self.shownAttribsLB = OWGUI.listBox(self.shownAttribsGroup, self, "selectedShown", "shownAttributes", selectionMode = QListWidget.ExtendedSelection)

        self.attrAddButton =    OWGUI.button(self.addRemoveGroup, self, "", callback = self.addAttributeClick, tooltip="Add (show) selected attributes")
        self.attrAddButton.setIcon(QIcon(os.path.join(self.widgetDir, "icons/Dlg_up3.png")))
        self.attrRemoveButton = OWGUI.button(self.addRemoveGroup, self, "", callback = self.removeAttributeClick, tooltip="Remove (hide) selected attributes")
        self.attrRemoveButton.setIcon(QIcon(os.path.join(self.widgetDir, "icons/Dlg_down3.png")))

        self.hiddenAttribsLB = OWGUI.listBox(self.hiddenAttribsGroup, self, "selectedHidden", "hiddenAttributes", selectionMode = QListWidget.ExtendedSelection)

        settingsBox = OWGUI.widgetBox(self.controlArea, "Settings", addSpace=True)
        self.mergeAttributesCB = OWGUI.checkBox(settingsBox, self, "mergeAttributes", 'Merge attributes', callback = self.mergeAttributesEvent, tooltip = "Enable or disable attribute merging. If enabled, you can merge \ntwo attributes with a right mouse click inside interaction rectangles in the left graph.\nA merged attribute is then created as a cartesian product of corresponding attributes \nand added to the list of attributes.")
        self.importantInteractionsCB = OWGUI.checkBox(settingsBox, self, "onlyImportantInteractions", 'Show only important interactions', callback = self.showImportantInteractions)

        self.selectionButton = OWGUI.button(self.controlArea, self, "Show selection", callback = self.selectionClick, tooltip = "Sends 'selection' signal to any successor visualization widgets.\nThis signal contains a list of selected attributes to visualize.")

        import sip
        sip.delete(self.buttonBackground.layout())
        
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        self.saveLCanvas = OWGUI.button(None, self, "Save left canvas", callback = self.saveToFileLCanvas, debuggingEnabled = 0, addToLayout=False)
        self.saveRCanvas = OWGUI.button(None, self, "Save right canvas", callback = self.saveToFileRCanvas, debuggingEnabled = 0, addToLayout=False)
        
        if OWReport.report:
            self.reportButton = OWGUI.button(None, self, "&Report", self.reportAndFinish, debuggingEnabled=0, addToLayout=False)

        layout.addWidget(self.saveLCanvas, 0, 0)
        layout.addWidget(self.saveRCanvas, 0, 1)
        
        if hasattr(self, "reportButton"):
            layout.addWidget(self.reportButton, 1, 0, 1, 2)
        self.buttonBackground.setLayout(layout)
        self.buttonBackground.show()

    def mergeAttributesEvent(self):
        self.updateNewData(self.originalData)

    def showImportantInteractions(self):
        self.showInteractionRects(self.data)

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
        pos = QPointF(ev.pos())
        if ev.button() == Qt.LeftButton and name == "graph":
            for name in self.rectNames:
                if self.rectNames[name].contains(pos):
                    self._setAttrVisible(name, not self.getAttrVisible(name))
                    self.showInteractionRects(self.data)
                    #self.canvasR.update()
                    return
            for (attr1, attr2, rect) in self.lines:
                if rect.contains(pos):
                    self.send("Attribute Pair", [attr1, attr2])
                    return
        elif ev.button() == Qt.LeftButton and name == "interactions":
            self.rest = None
            for rects in self.interactionRects:
                if 1 in [item.contains(pos) for item in rects]:
                    (rect1, rect2, rect3, nbrect, text1, text2) = rects
                    self.send("Attribute Pair", [str(text1.toPlainText()), str(text2.toPlainText())])

        elif ev.button() == Qt.RightButton and name == "interactions" and self.mergeAttributes:
            found = 0
            for rects in self.interactionRects:
                (rect1, rect2, rect3, nbrect, text1, text2) = rects
                if 1 in [item.contains(pos) for item in rects]:
                    attr1 = str(text1.toPlainText()); attr2 = str(text2.toPlainText())
                    found = 1
                    break
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


    # click on selection button
    def selectionClick(self):
        if self.data == None: return
        l = [str(self.shownAttribsLB.item(i).text()) for i in range(self.shownAttribsLB.count())]
        self.send("Selected Attributes List", l)


    # receive new data and update all fields
    def setData(self, data):
        self.warning([0,1])
        self.error(0)

        self.originalData = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None
        if not self.originalData:
            return

        self.originalData = orange.Preprocessor_dropMissing(self.originalData)
            
        if len(self.originalData) != len(data):
            self.warning(0, "Examples with missing values were removed. Keeping %d of %d examples." % (len(self.originalData), len(data)))
        if len(self.originalData) == 0:
            self.error(0, "All examples from the data set were removed!")
            return
        if self.originalData.domain.hasContinuousAttributes():
            self.warning(1, "Continuous attributes were discretized using entropy discretization.")

        self.dataSize = len(self.originalData)

        self.updateNewData(self.originalData)

    def updateNewData(self, data):
        if not data:
            return
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

        import tempfile, os
        fhandle, fname = tempfile.mkstemp('dot')
        os.close(fhandle)
        f = file(fname, "wt")
        self.interactionMatrix.exportGraph(f, significant_digits=3,positive_int=8,negative_int=8,absolute_int=0,url=1)
        f.close()
        del f

        # execute dot
        import subprocess
        try:
            textPng = subprocess.Popen(["dot", fname, "-Tpng"], stdout=subprocess.PIPE).communicate()[0]
            textPlainList = subprocess.Popen(["dot", fname, "-Tismap"], stdout=subprocess.PIPE).communicate()[0].splitlines()
        except OSError, ex:
            textPng = ""
            textPlainList = []
        
        os.remove(fname)

        # if the output from the pipe was empty, then the software isn't installed correctly
        if len(textPng) == 0:
            self.error(0, "This widget needs 'graphviz' software package. You can freely download it from the web.")

        # create a picture
        pixmap = QPixmap()
        pixmap.loadFromData(textPng)
        if hasattr(self, "canvasPixmap"):
            self.canvasPixmap.setPixmap(pixmap)
        else:
            self.canvasPixmap = self.canvasR.addPixmap(pixmap)
            
        width = self.canvasPixmap.pixmap().width()
        height = self.canvasPixmap.pixmap().height()
        
        self.canvasR.setSceneRect(QRectF(0, 0, width, height).adjusted(-10, -10, 10, 10))
        #self.canvasR.resize(width, height)

        # hide all rects
        for rects in self.rectIndices.values():
            rects.hide()

        self.rectIndices = {}       # QRect rectangles
        self.rectNames   = {}       # info about rectangle names (attributes)
        self.lines = []             # dict of form (rectName1, rectName2):(labelQPoint, [p1QPoint, p2QPoint, ...])

        self.parseGraphData(data, textPlainList, width, height)
        self.initLists(data)   # add all attributes found in .dot file to shown list
        self.showInteractionRects(data) # use interaction matrix to fill the left canvas with rectangles

        #self.canvasL.update()
        #self.canvasR.update()

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
        for (rect1, rect2, rect3, nbrect, text1, text2) in self.interactionRects:
            for item in [rect1, rect2, rect3, nbrect, text1, text2]:
                self.canvasL.removeItem(item)
        self.interactionRects = []

        for item in self.rectItems:
            self.canvasL.removeItem(item)
        self.rectItems = []

        ################################
        # get max width of the attribute text
        xOff = 0
        for ((total, (gain1, gain2, attrIndex1, attrIndex2))) in self.interactionList:
            if not self.showInteractionPair(attrIndex1, attrIndex2): continue
            if gain1 > gain2: text = OWQCanvasFuncts.OWCanvasText(self.canvasL, data.domain[attrIndex1].name, show = 0)
            else:             text = OWQCanvasFuncts.OWCanvasText(self.canvasL, data.domain[attrIndex2].name, show = 0)
            xOff = max(xOff, text.boundingRect().width())

        xOff += 10;  yOff = 40
        index = 0
        xscale = 300;  yscale = 200
        maxWidth = xOff + xscale + 10;  maxHeight = 0
        rectHeight = yscale * 0.1    # height of the rectangle will be 1/10 of max width

        ################################
        # print scale
        line = OWQCanvasFuncts.OWCanvasLine(self.canvasL, xOff, yOff - 4, xOff+xscale, yOff-4)
        tick1 = OWQCanvasFuncts.OWCanvasLine(self.canvasL, xOff, yOff-10, xOff, yOff-4)
        tick2 = OWQCanvasFuncts.OWCanvasLine(self.canvasL, xOff + (xscale/2), yOff-10, xOff + (xscale/2), yOff-4)
        tick3 = OWQCanvasFuncts.OWCanvasLine(self.canvasL, xOff + xscale-1, yOff-10, xOff + xscale-1, yOff-4)
        self.rectItems = [line, tick1, tick2, tick3]
        for i in range(10):
            x = xOff + xscale * (float(i)/10.0)
            tick = OWQCanvasFuncts.OWCanvasLine(self.canvasL, x, yOff-8, x, yOff-4)
            self.rectItems.append(tick)

        text1 = OWQCanvasFuncts.OWCanvasText(self.canvasL, "0%", x = xOff, y = yOff - 23, alignment = Qt.AlignHCenter)
        text2 = OWQCanvasFuncts.OWCanvasText(self.canvasL, "50%", x = xOff + xscale/2, y = yOff - 23, alignment = Qt.AlignHCenter)
        text3 = OWQCanvasFuncts.OWCanvasText(self.canvasL, "100%", x = xOff + xscale, y = yOff - 23, alignment = Qt.AlignHCenter)
        text4 = OWQCanvasFuncts.OWCanvasText(self.canvasL, "Class entropy removed", x = xOff + xscale/2, y = yOff - 36, alignment = Qt.AlignHCenter)
        self.rectItems += [text1, text2, text3, text4]

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
                x2 = xOff + xscale*(gain1+interaction)
                x3 = xOff + xscale*gain1
            else:
                x2 = xOff + xscale*gain1
                x3 = xOff + xscale*(total-gain2)
            x4 = xOff + xscale*total

            # compute nbgain position
            nb_x1 = min(xOff, xOff + 0.5*xscale*nbgain)
            nb_x2 = max(xOff, xOff + 0.5*xscale*nbgain)

            tooltipText = "%s : <b>%.1f%%</b><br>%s : <b>%.1f%%</b><br>Interaction : <b>%.1f%%</b><br>Total entropy removed: <b>%.1f%%</b>" %(data.domain[attrIndex1].name, gain1*100, data.domain[attrIndex2].name, gain2*100, interaction*100, total*100)

            nbrect = OWQCanvasFuncts.OWCanvasRectangle(self.canvasL, nb_x1, rectsYOff-4, nb_x2-nb_x1+1, 2, brushColor = Qt.black)
            rect2 = OWQCanvasFuncts.OWCanvasRectangle(self.canvasL, x2, rectsYOff,   x3-x2+1, rectHeight, tooltip = tooltipText)
            rect1 = OWQCanvasFuncts.OWCanvasRectangle(self.canvasL, x1, rectsYOff, x2-x1+1, rectHeight, tooltip = tooltipText)
            rect3 = OWQCanvasFuncts.OWCanvasRectangle(self.canvasL, x3, rectsYOff, x4-x3, rectHeight, tooltip = tooltipText)

            if interaction < 0.0:
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

            # create text labels
            text1 = OWQCanvasFuncts.OWCanvasText(self.canvasL, data.domain[attrIndex1].name, x = xOff - 5, y = rectsYOff + 3, alignment = Qt.AlignRight)
            text2 = OWQCanvasFuncts.OWCanvasText(self.canvasL, data.domain[attrIndex2].name, x = xOff + xscale*total + 5, y = rectsYOff + 3, alignment = Qt.AlignLeft)

            # compute line width
            rect = text2.boundingRect()
            lineWidth = xOff + xscale*total + 5 + rect.width() + 10
            if  lineWidth > maxWidth:
                maxWidth = lineWidth

            if rectsYOff + rectHeight + 10 > maxHeight:
                maxHeight = rectsYOff + rectHeight + 10

            self.interactionRects.append((rect1, rect2, rect3, nbrect, text1, text2))
            index += 1

        # resizing of the left canvas to update width
        #self.canvasViewL.setMaximumSize(QSize(maxWidth + 30, max(2000, maxHeight)))
        #self.canvasL.resize(maxWidth + 10, maxHeight)
        #self.canvasViewL.setMinimumWidth(0)
        #self.canvasL.update()


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
                isAttribute = "-" not in attrIndex     # does rectangle represent attribute

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

                rect = OWQCanvasFuncts.OWCanvasRectangle(self.canvasR, xLeft, yTop, width, height, penColor = Qt.blue, penWidth = 4, show = 0)

                if isAttribute == 1:
                    name = data.domain[int(attrIndex)].name
                    self.rectIndices[int(attrIndex)] = rect
                    self.rectNames[name] = rect
                else:
                    attrs = attrIndex.split("-")
                    attr1 = data.domain[int(attrs[0])].name
                    attr2 = data.domain[int(attrs[1])].name
                    rect.setPen(QPen(Qt.NoPen))
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
        self.shownAttribsLB.addItem(name)    # add to shown

        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from hidden
            if str(self.hiddenAttribsLB.item(i).text()) == name:
                self.hiddenAttribsLB.takeItem(i)

    def _hideAttribute(self, name):
        self.hiddenAttribsLB.addItem(name)    # add to hidden

        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):        # remove from shown
            if str(self.shownAttribsLB.item(i).text()) == name:
                self.shownAttribsLB.takeItem(i)

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
            if str(self.hiddenAttribsLB.item(i).text()) == name: return 0

        if self.mergeAttributes == 1:
            names = name.split("-")
            for i in range(self.hiddenAttribsLB.count()):
                if str(self.hiddenAttribsLB.item(i).text()) in names: return 0

        return 1

    #################################################
    # event processing
    #################################################
    def addAttributeClick(self):
        count = self.hiddenAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.hiddenAttribsLB.item(i).isSelected():
                name = str(self.hiddenAttribsLB.item(i).text())
                self._setAttrVisible(name, 1)
        self.showInteractionRects(self.data)
        self.canvasL.update()
        self.canvasR.update()

    def removeAttributeClick(self):
        count = self.shownAttribsLB.count()
        for i in range(count-1, -1, -1):
            if self.shownAttribsLB.item(i).isSelected():
                name = str(self.shownAttribsLB.item(i).text())
                self._setAttrVisible(name, 0)
        self.showInteractionRects(self.data)
        self.canvasL.update()
        self.canvasR.update()

    ##################################################
    # SAVING GRAPHS
    ##################################################
    def saveToFileLCanvas(self):
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.canvasL, parent=self)
        sizeDlg.exec_()


    def saveToFileRCanvas(self):
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.canvasR, parent=self)
        sizeDlg.exec_()

    def _sendReport(self):
        self.reportData(self.data)
        self.reportSettings("", [("Selected attributes", ", ".join(str(self.shownAttribsLB.item(i).text()) for i in range(self.shownAttribsLB.count())))])
        self.reportSection("Pairwise interaction bars")
        self.reportImage(OWDlgs.OWChooseImageSizeDlg(self.canvasL).saveImage)
        self.reportSection("Interaction graph")
        self.reportImage(OWDlgs.OWChooseImageSizeDlg(self.canvasR).saveImage)
                            

#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWInteractionGraph()
    ow.show()
    ow.setData(data = orange.ExampleTable("../../doc/datasets/monks-1_learn.tab"))
    a.exec_()

    #save settings
    ow.saveSettings()

"""
<name>Mosaic Display</name>
<description>Shows a mosaic display.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/MosaicDisplay.png</icon>
<priority>4100</priority>
"""
# OWMosaicDisplay.py
#

from OWWidget import *
from qtcanvas import *
import OWGUI
from OWMosaicOptimization import *
from math import sqrt, floor, ceil, pow
import operator
from orngScaleData import getVariableValuesSorted, getVariableValueIndices
from OWQCanvasFuncts import *
from OWGraphTools import *
import OWDlgs
from orngVisFuncts import permutations
from copy import copy
from OWGraphTools import ColorBrewerColors

PEARSON = 0
CLASS_DISTRIBUTION = 1

BOTTOM = 0
LEFT = 1
TOP = 2
RIGHT = 3

class SelectionRectangle(QCanvasRectangle):
    def rtti(self):
        return 123

class MosaicCanvasView(QCanvasView):
    def __init__(self, widget, *args):
        apply(QCanvasView.__init__,(self,) + args)
        self.widget = widget
        self.bMouseDown = False
        self.mouseDownPosition = QPoint(0,0)
        self.tempRect = None

    # mouse button was pressed
    def contentsMousePressEvent(self, ev):
        self.mouseDownPosition = QPoint(ev.pos().x(), ev.pos().y())
        self.bMouseDown = True
        self.contentsMouseMoveEvent(ev)

    # mouse button was pressed and mouse is moving ######################
    def contentsMouseMoveEvent(self, ev):
        if ev.button() == Qt.RightButton:
            return

        if self.tempRect:
            self.tempRect.setCanvas(None)
            self.tempRect = None

        if self.bMouseDown:
            rect = QRect(min(self.mouseDownPosition.x(), ev.pos().x()), min (self.mouseDownPosition.y(), ev.pos().y()), abs(self.mouseDownPosition.x() - ev.pos().x()), abs(self.mouseDownPosition.y() - ev.pos().y()))
            self.tempRect = SelectionRectangle(rect, self.canvas())
            self.tempRect.show()
            self.canvas().update()


    # mouse button was released #########################################
    def contentsMouseReleaseEvent(self, ev):
        self.bMouseDown = False

        if ev.button() == Qt.RightButton:
            self.widget.removeLastSelection()
            return

        if self.tempRect:
            self.widget.addSelection(self.tempRect.rect())
            self.tempRect.hide()
            self.tempRect.setCanvas(None)
            self.tempRect = None
            self.canvas().update()


class OWMosaicDisplay(OWWidget):
    settingsList = ["horizontalDistribution", "showAprioriDistributionLines", "showAprioriDistributionBoxes",
                    "horizontalDistribution", "useBoxes", "interiorColoring", "boxSize", "colorSettings", "selectedSchemaIndex", "cellspace",
                    "showSubsetDataBoxes", "removeUnusedValues"]

    contextHandlers = {"": DomainContextHandler("", ["attr1", "attr2", "attr3", "attr4", "manualAttributeValuesDict"], loadImperfect = 0)}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Mosaic display", TRUE, TRUE)

        #set default settings
        self.data = None
        self.unprocessedSubsetData = None
        self.subsetData = None
        self.tooltips = []
        self.names = []     # class values

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Example Subset", ExampleTable, self.setSubsetData)]
        self.outputs = [("Selected Examples", ExampleTable), ("Learner", orange.Learner)]

        #load settings
        self.colorSettings = None
        self.selectedSchemaIndex = 0
        self.interiorColoring = 0
        self.cellspace = 4
        self.showAprioriDistributionBoxes = 1
        self.useBoxes = 1
        self.showSubsetDataBoxes = 1
        self.horizontalDistribution = 0
        self.showAprioriDistributionLines = 0
        self.boxSize = 5
        self.exploreAttrPermutations = 0
        self.attr1 = ""
        self.attr2 = ""
        self.attr3 = ""
        self.attr4 = ""

        self.attributeNameOffset = 30
        self.attributeValueOffset = 15
        self.residuals = [] # residual values if the residuals are visualized
        self.aprioriDistributions = []
        self.colorPalette = None
        self.permutationDict = {}
        self.manualAttributeValuesDict = {}
        self.conditionalDict = None
        self.conditionalSubsetDict = None
        self.activeRule = None
        self.removeUnusedValues = 0

        self.selectionRectangle = None
        self.selectionConditionsHistorically = []
        self.selectionConditions = []

        # color paletes for visualizing pearsons residuals
        #self.blueColors = [QColor(255, 255, 255), QColor(117, 149, 255), QColor(38, 43, 232), QColor(1,5,173)]
        self.blueColors = [QColor(255, 255, 255), QColor(210, 210, 255), QColor(110, 110, 255), QColor(0,0,255)]
        self.redColors = [QColor(255, 255, 255), QColor(255, 200, 200), QColor(255, 100, 100), QColor(255, 0, 0)]

        self.loadSettings()

        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        self.GeneralTab = QVGroupBox(self)
        self.SettingsTab = QVGroupBox(self)
        self.tabs.insertTab(self.GeneralTab, "Main")
        self.tabs.insertTab(self.SettingsTab, "Settings")

        self.box = QVBoxLayout(self.mainArea)
        self.canvas = QCanvas(2000, 2000)
        self.canvasView = MosaicCanvasView(self, self.canvas, self.mainArea)
        self.box.addWidget(self.canvasView)
        self.canvasView.show()
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)

        #GUI
        #add controls to self.controlArea widget
        self.controlArea.setMinimumWidth(235)

        texts = ["1st Attribute", "2nd Attribute", "3rd Attribute", "4th Attribute"]
        for i in range(1,5):
            box = OWGUI.widgetBox(self.GeneralTab, texts[i-1], orientation = "horizontal")
            box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
            combo = OWGUI.comboBox(box, self, "attr" + str(i), None, callback = self.updateGraphAndPermList, sendSelectedValue = 1, valueType = str)

            butt = OWGUI.button(box, self, "", callback = self.orderAttributeValues, tooltip = "Change the order of attribute values", debuggingEnabled = 0)
            butt.setMaximumWidth(26); butt.setMaximumHeight(26); butt.setMinimumWidth(24); butt.setMinimumHeight(26)
            butt.setToggleButton(1)
            butt.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_sort.png")))

            setattr(self, "sort"+str(i), butt)
            setattr(self, "attr" + str(i)+ "Combo", combo)

        self.optimizationDlg = OWMosaicOptimization(self, self.signalManager)

        optimizationButtons = OWGUI.widgetBox(self.GeneralTab, "Dialogs", orientation = "horizontal")
        optimizationButtons.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed))
        OWGUI.button(optimizationButtons, self, "VizRank", callback = self.optimizationDlg.reshow, debuggingEnabled = 0, tooltip = "Find attribute combinations that will separate different classes as clearly as possible.")

        self.box7 = OWGUI.collapsableWidgetBox(self.GeneralTab, "Explore Attribute Permutations", self, "exploreAttrPermutations", callback = self.permutationListToggle)

        self.permutationList = QListBox(self.box7)
        self.connect(self.permutationList, SIGNAL("selectionChanged()"), self.setSelectedPermutation)
        self.permutationList.hide()

        # ######################
        # SETTINGS TAB
        # ######################
        box5 = OWGUI.widgetBox(self.SettingsTab, "Colors in cells represent...")
        OWGUI.comboBox(box5, self, "interiorColoring", None, items = ["Standardized (Pearson) residuals", "Class distribution"], callback = self.updateGraph)
        box5.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        OWGUI.comboBoxWithCaption(self.SettingsTab, self, "cellspace", "Minimum cell distance: ", box = "Visual settings", items = range(1,11), callback = self.updateGraph, sendSelectedValue = 1, valueType = int, tooltip = "What is the minimum distance between two rectangles in the plot?")

        self.box6 = OWGUI.widgetBox(self.SettingsTab, "Cell distribution settings")
        OWGUI.comboBox(self.box6, self, 'horizontalDistribution', items = ["Show Distribution Vertically", "Show Distribution Horizontally"], tooltip = "Do you wish to see class distribution drawn horizontally or vertically?", callback = self.updateGraph)
        OWGUI.checkBox(self.box6, self, 'showAprioriDistributionLines', 'Show apriori distribution with lines', callback = self.updateGraph, tooltip = "Show the lines that represent the apriori class distribution")

        self.box8 = OWGUI.widgetBox(self.SettingsTab, "Subboxes in Cells")
        OWGUI.spin(self.box8, self, 'boxSize', 1, 15, 1, '', "Subbox Size (pixels): ", orientation = "horizontal", callback = self.updateGraph)
        OWGUI.checkBox(self.box8, self, 'showSubsetDataBoxes', 'Show class distribution of subset data', callback = self.updateGraph, tooltip = "Show small boxes at right (or bottom) edge of cells to represent class distribution of examples from example subset input.")
        OWGUI.checkBox(self.box8, self, 'useBoxes', 'Use subboxes on left to show...', callback = self.updateGraph, tooltip = "Show small boxes at left (or top) edge of cells to represent additional information.")
        indBox = OWGUI.indentedBox(self.box8)
        OWGUI.comboBox(indBox, self, 'showAprioriDistributionBoxes', items = ["Expected class distribution", "Apriori class distribution"], tooltip = "Show additional boxes for each mosaic cell representing:\n - expected class distribution (assuming independence between attributes)\n - apriori class distribution (based on all examples).", callback = self.updateGraph)

        OWGUI.checkBox(self.SettingsTab, self, "removeUnusedValues", "Remove unused values", box = "Attributes", tooltip = "Do you want to remove unused attribute values?\nThis setting will not be considered until new data is received.")

        hbox = OWGUI.widgetBox(self.SettingsTab, "Colors", orientation = "horizontal")
        OWGUI.button(hbox, self, "Set Colors", self.setColors, tooltip = "Set the color palette for class values", debuggingEnabled = 0)
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        self.box6.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))

        self.connect(self.graphButton, SIGNAL("clicked()"), self.saveToFileCanvas)
        self.icons = self.createAttributeIconDict()
        self.resize(750, 550)

        self.activateLoadedSettings()
        dlg = self.createColorDialog()
        self.colorPalette = dlg.getDiscretePalette()

        from OWGraphTools import ColorBrewerColors, defaultRGBColors
        #self.selectionColorPalette = [QColor(*col) for col in ColorBrewerColors[::-1]]
        self.selectionColorPalette = [QColor(*col) for col in defaultRGBColors]

        self.VizRankLearner = MosaicTreeLearner(self.optimizationDlg)
        self.send("Learner", self.VizRankLearner)

        # this is needed so that the tabs are wide enough!
        self.safeProcessEvents()
        self.tabs.updateGeometry()
        self.wdChildDialogs = [self.optimizationDlg]        # used when running widget debugging

    def permutationListToggle(self):
        if self.exploreAttrPermutations:
            self.updateGraphAndPermList()

    def setSelectedPermutation(self):
        if self.permutationList.count() > 0 and self.bestPlacements and self.permutationList.currentItem() < len(self.bestPlacements):
            self.removeAllSelections()
            index = self.permutationList.currentItem()
            val, attrList, valueOrder = self.bestPlacements[index]
            if len(attrList) > 0: self.attr1 = attrList[0]
            if len(attrList) > 1: self.attr2 = attrList[1]
            if len(attrList) > 2: self.attr3 = attrList[2]
            if len(attrList) > 3: self.attr4 = attrList[3]
            self.updateGraph(customValueOrderDict = dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))]))

    def orderAttributeValues(self):
        attr = None
        if self.sort1.isOn():   attr = self.attr1
        elif self.sort2.isOn(): attr = self.attr2
        elif self.sort3.isOn(): attr = self.attr3
        elif self.sort4.isOn(): attr = self.attr4

        if self.data and attr  != "" and attr != "(None)":
            dlg = SortAttributeValuesDlg(self, self.manualAttributeValuesDict.get(attr, None) or getVariableValuesSorted(self.data, attr))
            if dlg.exec_loop() == QDialog.Accepted:
                self.manualAttributeValuesDict[attr] = [str(dlg.attributeList.text(i)) for i in range(dlg.attributeList.count())]

        for control in [self.sort1, self.sort2, self.sort3, self.sort4]:
            control.setOn(0)
        self.updateGraph()

    # initialize combo boxes with discrete attributes
    def initCombos(self, data):
        self.attr1Combo.clear(); self.attr2Combo.clear(); self.attr3Combo.clear(); self.attr4Combo.clear()

        if data == None: return

        self.attr2Combo.insertItem("(None)")
        self.attr3Combo.insertItem("(None)")
        self.attr4Combo.insertItem("(None)")

        for attr in data.domain:
            if attr.varType == orange.VarTypes.Discrete:
                for combo in [self.attr1Combo, self.attr2Combo, self.attr3Combo, self.attr4Combo]:
                    combo.insertItem(self.icons[orange.VarTypes.Discrete], attr.name)

        if self.attr1Combo.count() > 0:
            self.attr1 = str(self.attr1Combo.text(0))
            self.attr2 = str(self.attr2Combo.text(0 + 2*(self.attr2Combo.count() > 2)))
        self.attr3 = str(self.attr3Combo.text(0))
        self.attr4 = str(self.attr4Combo.text(0))

    #  when we resize the widget, we have to redraw the data
    def resizeEvent(self, e):
        OWWidget.resizeEvent(self,e)
        self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.updateGraph()

    # ------------- SIGNALS --------------------------
    # # DATA signal - receive new data and update all fields
    def setData(self, data):
        self.closeContext()
        self.data = None
        self.bestPlacements = None
        self.manualAttributeValuesDict = {}
        self.attributeValuesDict = {}
        self.information([0,1,2])

        self.data = self.optimizationDlg.setData(data, self.removeUnusedValues)

        if self.data:
            if data.domain.hasContinuousAttributes():
                self.information(0, "Continuous attributes were discretized using entropy discretization.")
            if data.domain.classVar and data.hasMissingClasses():
                self.information(1, "Examples with missing classes were removed.")
            if self.removeUnusedValues and len(data) != len(self.data):
                self.information(2, "Unused attribute values were removed.")

            if self.data.domain.classVar and self.data.domain.classVar.varType == orange.VarTypes.Discrete:
                self.interiorColoring = CLASS_DISTRIBUTION
            else:
                self.interiorColoring = PEARSON

        self.initCombos(self.data)
        self.openContext("", self.data)

        if data and self.unprocessedSubsetData:        # if we first received subset data we now have to call setSubsetData to process it
            self.setSubsetData(self.unprocessedSubsetData)
            self.unprocessedSubsetData = None

    def setSubsetData(self, data):
        if not self.data:
            self.unprocessedSubsetData = data
            self.warning(10)
        else:
            try:
                self.subsetData = data.select(self.data.domain)
                self.warning(10)
            except:
                self.subsetData = None
                self.warning(10, data and "'Examples' and 'Example Subset' data do not have copatible domains. Unable to draw 'Example Subset' data." or "")


    # this is called by OWBaseWidget after setData and setSubsetData are called. this way the graph is updated only once
    def handleNewSignals(self):
        self.updateGraphAndPermList()

    # ------------------------------------------------

    def setShownAttributes(self, attrList, **args):
        if not attrList: return
        self.attr1 = attrList[0]

        if len(attrList) > 1: self.attr2 = attrList[1]
        else: self.attr2 = "(None)"

        if len(attrList) > 2: self.attr3 = attrList[2]
        else: self.attr3 = "(None)"

        if len(attrList) > 3: self.attr4 = attrList[3]
        else: self.attr4 = "(None)"

        self.attributeValuesDict = args.get("customValueOrderDict", None)
        self.updateGraphAndPermList()

    def getShownAttributeList(self):
        attrList = [self.attr1, self.attr2, self.attr3, self.attr4]
        while "(None)" in attrList: attrList.remove("(None)")
        while "" in attrList:       attrList.remove("")
        return attrList

    def updateGraphAndPermList(self, **args):
        self.removeAllSelections()
        self.permutationList.clear()

        if self.exploreAttrPermutations:
            attrList = self.getShownAttributeList()
            if not getattr(self, "bestPlacements", []) or 0 in [attr in self.bestPlacements[0][1] for attr in attrList]:        # we might have bestPlacements for a different set of attributes
                self.setStatusBarText("Evaluating different attribute permutations. You can stop evaluation by opening VizRank dialog and pressing 'Stop optimization' button.")
                self.bestPlacements = self.optimizationDlg.optimizeCurrentAttributeOrder(attrList, updateGraph = 0)
                self.setStatusBarText("")

            if self.bestPlacements:
                for (val, attrs, order) in self.bestPlacements:
                    self.permutationList.insertItem("%.2f - %s" % (val, attrs))
                attrList, valueOrder = self.bestPlacements[0][1], self.bestPlacements[0][2]
                self.attributeValuesDict = dict([(attrList[i], tuple(valueOrder[i])) for i in range(len(attrList))])

        self.updateGraph(**args)

    # ############################################################################
    # updateGraph - gets called every time the graph has to be updated
    def updateGraph(self, data = -1, subsetData = -1, attrList = -1, **args):
        # do we want to erase previous diagram?
        if args.get("erasePrevious", 1):
            for item in self.canvas.allItems():
                if not isinstance(item, SelectionRectangle):
                    item.setCanvas(None)    # remove all canvas items, except SelectionCurves
            for (rect, tip) in self.tooltips:
                QToolTip.remove(self.canvasView, rect)
            self.names = []
            self.tooltips = []

        if data == -1:
            data = self.data

        if subsetData == -1:
            subsetData = self.subsetData

        if attrList == -1:
            attrList = [self.attr1, self.attr2, self.attr3, self.attr4]

        if data == None : return

        while "(None)" in attrList: attrList.remove("(None)")
        while "" in attrList:       attrList.remove("")
        if attrList == []:
            return

        selectList = attrList
        if data.domain.classVar:
            data = data.select(attrList + [data.domain.classVar])
        else:
            data = data.select(attrList)
        data = orange.Preprocessor_dropMissing(data)

        if len(data) == 0:
            self.warning(5, "There are no examples with valid values for currently visualized attributes. Unable to visualize.")
            return
        else:
            self.warning(5)

        self.aprioriDistributions = []
        if self.interiorColoring == PEARSON:
            self.aprioriDistributions = [orange.Distribution(attr, data) for attr in attrList]

        if args.get("positions"):
            xOff, yOff, squareSize = args.get("positions")
        else:
            # get the maximum width of rectangle
            xOff = 50
            width = 50
            if len(attrList) > 1:
                text = QCanvasText(attrList[1], self.canvas);
                font = text.font(); font.setBold(1); text.setFont(font)
                width = text.boundingRect().right() - text.boundingRect().left() + 30 + 20
                xOff = width
                if len(attrList) == 4:
                    text = QCanvasText(attrList[3], self.canvas);
                    font = text.font(); font.setBold(1); text.setFont(font)
                    width += text.boundingRect().right() - text.boundingRect().left() + 30 + 20

            # get the maximum height of rectangle
            height = 90
            yOff = 40
            squareSize = min(self.canvasView.size().width() - width - 20, self.canvasView.size().height() - height - 20)

        if squareSize < 0: return    # canvas is too small to draw rectangles

        self.legend = {}        # dictionary that tells us, for what attributes did we already show the legend
        for attr in attrList:
            self.legend[attr] = 0

        self.drawnSides = dict([(0,0),(1,0),(2,0),(3,0)])
        self.drawPositions = {}

        if not getattr(self, "attributeValuesDict", None):
            self.attributeValuesDict = self.manualAttributeValuesDict

        # compute distributions
        self.conditionalDict = self.optimizationDlg.getConditionalDistributions(data, attrList)
        self.conditionalDict[""] = len(data)
        self.conditionalSubsetDict = None
        if subsetData:
            self.conditionalSubsetDict = self.optimizationDlg.getConditionalDistributions(subsetData, attrList)
            self.conditionalSubsetDict[""] = len(subsetData)

        # draw rectangles
        self.DrawData(attrList, (xOff, xOff+squareSize), (yOff, yOff+squareSize), 0, "", len(attrList), **args)
        if args.get("drawLegend", 1):
            self.DrawLegend(data, (xOff, xOff+squareSize), (yOff, yOff+squareSize)) # draw class legend

        if args.get("drillUpdateSelection", 1):
            self.optimizationDlg.mtUpdateState()

        self.canvas.update()

    # ############################################################################
    # ############################################################################

    ##  DRAW DATA - draw rectangles for attributes in attrList inside rect (x0,x1), (y0,y1)
    def DrawData(self, attrList, (x0, x1), (y0, y1), side, condition, totalAttrs, usedAttrs = [], usedVals = [], attrVals = "", **args):
        if self.conditionalDict[attrVals] == 0:
            self.addRect(x0, x1, y0, y1, "", usedAttrs, usedVals, attrVals = attrVals)
            self.DrawText(side, attrList[0], (x0, x1), (y0, y1), totalAttrs, usedAttrs, usedVals, attrVals)  # store coordinates for later drawing of labels
            return

        attr = attrList[0]
        edge = len(attrList) * self.cellspace  # how much smaller rectangles do we draw
        values = self.attributeValuesDict.get(attr, None) or getVariableValuesSorted(self.data, attr)
        if side%2: values = values[::-1]        # reverse names if necessary

        if side%2 == 0:                                     # we are drawing on the x axis
            whole = max(0, (x1-x0)-edge*(len(values)-1))  # we remove the space needed for separating different attr. values
            if whole == 0: edge = (x1-x0)/float(len(values)-1)
        else:                                               # we are drawing on the y axis
            whole = max(0, (y1-y0)-edge*(len(values)-1))
            if whole == 0: edge = (y1-y0)/float(len(values)-1)

        if attrVals == "": counts = [self.conditionalDict[val] for val in values]
        else:              counts = [self.conditionalDict[attrVals + "-" + val] for val in values]
        total = sum(counts)

        # if we are visualizing the third attribute and the first attribute has the last value, we have to reverse the order in which the boxes will be drawn
        # otherwise, if the last cell, nearest to the labels of the fourth attribute, is empty, we wouldn't be able to position the labels
        valRange = range(len(values))
        if len(attrList + usedAttrs) == 4 and len(usedAttrs) == 2:
            attr1Values = self.attributeValuesDict.get(usedAttrs[0], None) or getVariableValuesSorted(self.data, usedAttrs[0])
            if usedVals[0] == attr1Values[-1]:
                valRange = valRange[::-1]

        for i in valRange:
            start = i*edge + whole * float(sum(counts[:i])/float(total))
            end   = i*edge + whole * float(sum(counts[:i+1])/float(total))
            val = values[i]
            htmlVal = getHtmlCompatibleString(val)
            if attrVals != "": newAttrVals = attrVals + "-" + val
            else:              newAttrVals = val

            if side % 2 == 0:   # if we are moving horizontally
                if len(attrList) == 1:  self.addRect(x0+start, x0+end, y0, y1, condition + 4*"&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", usedAttrs + [attr], usedVals + [val], newAttrVals, **args)
                else:                   self.DrawData(attrList[1:], (x0+start, x0+end), (y0, y1), side +1, condition + 4*"&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", totalAttrs, usedAttrs + [attr], usedVals + [val], newAttrVals, **args)
            else:
                if len(attrList) == 1:  self.addRect(x0, x1, y0+start, y0+end, condition + 4*"&nbsp;" + attr + ": <b> " + htmlVal + "</b><br>", usedAttrs + [attr], usedVals + [val], newAttrVals, **args)
                else:                   self.DrawData(attrList[1:], (x0, x1), (y0+start, y0+end), side +1, condition + 4*"&nbsp;" + attr + ": <b>" + htmlVal + "</b><br>", totalAttrs, usedAttrs + [attr], usedVals + [val], newAttrVals, **args)

        self.DrawText(side, attrList[0], (x0, x1), (y0, y1), totalAttrs, usedAttrs, usedVals, attrVals)


    ######################################################################
    ## DRAW TEXT - draw legend for all attributes in attrList and their possible values
    def DrawText(self, side, attr, (x0, x1), (y0, y1), totalAttrs, usedAttrs, usedVals, attrVals):
        if self.drawnSides[side]: return

        # the text on the right will be drawn when we are processing visualization of the last value of the first attribute
        if side == RIGHT:
            attr1Values = self.attributeValuesDict.get(usedAttrs[0], None) or getVariableValuesSorted(self.data, usedAttrs[0])
            if usedVals[0] != attr1Values[-1]:
                return

        if not self.conditionalDict[attrVals]:
            if not self.drawPositions.has_key(side): self.drawPositions[side] = (x0, x1, y0, y1)
            return
        else:
            if self.drawPositions.has_key(side): (x0, x1, y0, y1) = self.drawPositions[side]        # restore the positions where we have to draw the attribute values and attribute name

        self.drawnSides[side] = 1

        values = self.attributeValuesDict.get(attr, None) or getVariableValuesSorted(self.data, attr)
        if side % 2:  values = values[::-1]

        width  = x1-x0 - (side % 2 == 0) * self.cellspace*(totalAttrs-side)*(len(values)-1)
        height = y1-y0 - (side % 2 == 1) * self.cellspace*(totalAttrs-side)*(len(values)-1)

        #calculate position of first attribute
        if side == 0:    OWCanvasText(self.canvas, attr, x0+(x1-x0)/2, y1 + self.attributeNameOffset, Qt.AlignCenter, bold = 1)
        elif side == 1:  OWCanvasText(self.canvas, attr, x0 - self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignRight + Qt.AlignVCenter, bold = 1)
        elif side == 2:  OWCanvasText(self.canvas, attr, x0+(x1-x0)/2, y0 - self.attributeNameOffset, Qt.AlignCenter, bold = 1)
        else:            OWCanvasText(self.canvas, attr, x1 + self.attributeNameOffset, y0+(y1-y0)/2, Qt.AlignLeft + Qt.AlignVCenter, bold = 1)

        currPos = 0

        if attrVals == "":  counts = [self.conditionalDict.get(val, 1) for val in values]
        else:               counts = [self.conditionalDict.get(attrVals + "-" + val, 1) for val in values]
        total = sum(counts)
        if total == 0:
            counts = [1]*len(values)
            total = sum(counts)

        for i in range(len(values)):
            val = values[i]
            perc = counts[i]/float(total)
            if side == 0:    OWCanvasText(self.canvas, str(val), x0+currPos+width*0.5*perc, y1 + self.attributeValueOffset, Qt.AlignCenter, bold = 0)
            elif side == 1:  OWCanvasText(self.canvas, str(val), x0-self.attributeValueOffset, y0+currPos+height*0.5*perc, Qt.AlignRight + Qt.AlignVCenter, bold = 0)
            elif side == 2:  OWCanvasText(self.canvas, str(val), x0+currPos+width*perc*0.5, y0 - self.attributeValueOffset, Qt.AlignCenter, bold = 0)
            else:            OWCanvasText(self.canvas, str(val), x1+self.attributeValueOffset, y0 + currPos + height*0.5*perc, Qt.AlignLeft + Qt.AlignVCenter, bold = 0)

            if side % 2 == 0: currPos += perc*width + self.cellspace*(totalAttrs-side)
            else :            currPos += perc*height+ self.cellspace*(totalAttrs-side)

    # draw a rectangle, set it to back and add it to rect list
    def addRect(self, x0, x1, y0, y1, condition = "", usedAttrs = [], usedVals = [], attrVals = "", **args):
        x0 = int(x0); x1 = int(x1); y0 = int(y0); y1 = int(y1)
        if x0 == x1: x1+=1
        if y0 == y1: y1+=1

        if x1-x0 + y1-y0 == 2: y1+=1        # if we want to show a rectangle of width and height 1 it doesn't show anything. in such cases we therefore have to increase size of one edge

        if args.has_key("selectionDict") and args["selectionDict"].has_key(tuple(usedVals)):
            d = 2
            OWCanvasRectangle(self.canvas, x0-d, y0-d, x1-x0+1+2*d, y1-y0+1+2*d, penColor = args["selectionDict"][tuple(usedVals)], penWidth = 2, z = -100)

        rect = OWCanvasRectangle(self.canvas, x0, y0, x1-x0, y1-y0, z = 30)

        # if we have selected a rule that contains this combination of attr values then show a kind of selection of this rectangle
        if self.activeRule and len(usedAttrs) == len(self.activeRule[0]) and sum([v in usedAttrs for v in self.activeRule[0]]) == len(self.activeRule[0]):
            for vals in self.activeRule[1]:
                if usedVals == [vals[self.activeRule[0].index(a)] for a in usedAttrs]:
                    values = list(self.attributeValuesDict.get(self.data.domain.classVar.name, [])) or getVariableValuesSorted(self.data, self.data.domain.classVar.name)
                    counts = [self.conditionalDict[attrVals + "-" + val] for val in values]
                    d = 2
                    r = OWCanvasRectangle(self.canvas, x0-d, y0-d, x1-x0+2*d+1, y1-y0+2*d+1, z = 50)
                    r.setPen(QPen(self.colorPalette[counts.index(max(counts))], 2, Qt.DashLine))

        if not self.conditionalDict[attrVals]: return rect

        # we have to remember which conditions were new in this update so that when we right click we can only remove the last added selections
        if self.selectionRectangle is not None and rect in self.canvas.collisions(self.selectionRectangle) and tuple(usedVals) not in self.selectionConditions:
            self.recentlyAdded = getattr(self, "recentlyAdded", []) + [tuple(usedVals)]
            self.selectionConditions = self.selectionConditions + [tuple(usedVals)]

        # show rectangle selected or not
        if tuple(usedVals) in self.selectionConditions:
            rect.setPen(QPen(Qt.black, 3, Qt.DotLine))

        if self.interiorColoring == CLASS_DISTRIBUTION and (not self.data.domain.classVar or not self.data.domain.classVar.varType == orange.VarTypes.Discrete):
            return rect

        aprioriDist = None; pearson = None; expected = None

        # draw pearsons residuals
        if self.interiorColoring == PEARSON or not self.data.domain.classVar or self.data.domain.classVar.varType != orange.VarTypes.Discrete:
            s = sum(self.aprioriDistributions[0])
            expected = s * reduce(lambda x, y: x*y, [self.aprioriDistributions[i][usedVals[i]]/float(s) for i in range(len(usedVals))])
            actual = self.conditionalDict[attrVals]
            pearson = float(actual - expected) / sqrt(expected)
            if abs(pearson) < 2:   ind = 0
            elif abs(pearson) < 4: ind = 1
            elif abs(pearson) < 8: ind = 2
            else:                  ind = 3

            if pearson > 0: color = self.blueColors[ind]
            else: color = self.redColors[ind]
            rect = OWCanvasRectangle(self.canvas, x0, y0, x1-x0, y1-y0, color, color, z = -20)

        # draw class distribution - actual and apriori
        # we do have a discrete class
        else:
            clsValues = list(self.attributeValuesDict.get(self.data.domain.classVar.name, [])) or getVariableValuesSorted(self.data, self.data.domain.classVar.name)
            aprioriDist = orange.Distribution(self.data.domain.classVar.name, self.data)
            total = 0
            for i in range(len(clsValues)):
                val = self.conditionalDict[attrVals + "-" + clsValues[i]]
                if self.horizontalDistribution:
                    if i == len(clsValues)-1: v = x1-x0 - total
                    else:                       v = int(((x1-x0)* val)/self.conditionalDict[attrVals])
                    OWCanvasRectangle(self.canvas, x0+total, y0, v, y1-y0, self.colorPalette[i], self.colorPalette[i], z = -20)
                else:
                    if i == len(clsValues)-1: v = y1-y0 - total
                    else:                       v = int(((y1-y0)* val)/self.conditionalDict[attrVals])
                    OWCanvasRectangle(self.canvas, x0, y0+total, x1-x0, v, self.colorPalette[i], self.colorPalette[i], z = -20)
                total += v

            # show apriori boxes and lines
            if (self.showAprioriDistributionLines or self.useBoxes) and abs(x1 - x0) > self.boxSize and abs(y1 - y0) > self.boxSize:
                apriori = [aprioriDist[val]/float(len(self.data)) for val in clsValues]
                if self.showAprioriDistributionBoxes or self.data.domain.classVar.name in usedAttrs:   # we want to show expected class distribution under independence hypothesis
                    boxCounts = apriori
                else:
                    contingencies = self.optimizationDlg.getContingencys(usedAttrs)
                    boxCounts = []
                    for clsVal in clsValues:
                        # compute: P(c_i) * prod (P(c_i|attr_k) / P(c_i))  for each class value
                        Pci = aprioriDist[clsVal]/float(sum(aprioriDist.values()))
                        tempVal = Pci
                        if Pci > 0:
                            #tempVal = 1.0 / Pci
                            for i in range(len(usedAttrs)):
                                tempVal *= contingencies[usedAttrs[i]][usedVals[i]][clsVal] / Pci
                        boxCounts.append(tempVal)
                        #boxCounts.append(aprioriDist[val]/float(sum(aprioriDist.values())) * reduce(operator.mul, [contingencies[usedAttrs[i]][usedVals[i]][clsVal]/float(sum(contingencies[usedAttrs[i]][usedVals[i]].values())) for i in range(len(usedAttrs))]))

                total1 = 0; total2 = 0
                if self.useBoxes:
                    if self.horizontalDistribution:  OWCanvasLine(self.canvas, x0, y0+self.boxSize, x1, y0+self.boxSize, z = 30)
                    else:                            OWCanvasLine(self.canvas, x0+self.boxSize, y0, x0+self.boxSize, y1, z = 30)

                for i in range(len(clsValues)):
                    val1 = apriori[i]
                    if self.showAprioriDistributionBoxes: val2 = apriori[i]
                    else:                                 val2 = boxCounts[i]/float(sum(boxCounts))
                    if self.horizontalDistribution:
                        if i == len(clsValues)-1:
                            v1 = x1-x0 - total1
                            v2 = x1-x0 - total2
                        else:
                            v1 = int((x1-x0)* val1)
                            v2 = int((x1-x0)* val2)
                        x,y,w,h, xL1, yL1, xL2, yL2 = x0+total2, y0, v2, self.boxSize, x0+total1+v1, y0, x0+total1+v1, y1
                    else:
                        if i== len(clsValues)-1:
                            v1 = y1-y0 - total1
                            v2 = y1-y0 - total2
                        else:
                            v1 = int((y1-y0)* val1)
                            v2 = int((y1-y0)* val2)
                        x,y,w,h, xL1, yL1, xL2, yL2 = x0, y0+total2, self.boxSize, v2, x0, y0+total1+v1, x1, y0+total1+v1

                    if self.useBoxes:
                        OWCanvasRectangle(self.canvas, x, y, w, h, self.colorPalette[i], self.colorPalette[i], z = 20)
                    if i < len(clsValues)-1 and self.showAprioriDistributionLines:
                        OWCanvasLine(self.canvas, xL1, yL1, xL2, yL2, z = 10)

                    total1 += v1
                    total2 += v2

            # show subset distribution
            if self.conditionalSubsetDict:
                # show a rect around the box if subset examples belong to this box
                if self.conditionalSubsetDict[attrVals]:
                    #counts = [self.conditionalSubsetDict[attrVals + "-" + val] for val in clsValues]
                    #if sum(counts) == 1:    color = self.colorPalette[counts.index(1)]
                    #else:                   color = Qt.black
                    #OWCanvasRectangle(self.canvas, x0-2, y0-2, x1-x0+5, y1-y0+5, color, Qt.white, penWidth = 2, z=-50, penStyle = Qt.DashLine)
                    counts = [self.conditionalSubsetDict[attrVals + "-" + val] for val in clsValues]
                    if sum(counts) == 1:
                        OWCanvasRectangle(self.canvas, x0-2, y0-2, x1-x0+5, y1-y0+5, self.colorPalette[counts.index(1)], Qt.white, penWidth = 2, z=-50, penStyle = Qt.DashLine)

                    if self.showSubsetDataBoxes:     # do we want to show exact distribution in the right edge of each cell
                        if self.horizontalDistribution:  OWCanvasLine(self.canvas, x0, y1-self.boxSize, x1, y1-self.boxSize, z = 30)
                        else:                            OWCanvasLine(self.canvas, x1-self.boxSize, y0, x1-self.boxSize, y1, z = 30)
                        total = 0
                        for i in range(len(aprioriDist)):
                            val = self.conditionalSubsetDict[attrVals + "-" + clsValues[i]]
                            if not self.conditionalSubsetDict[attrVals] or val == 0: continue
                            if self.horizontalDistribution:
                                if i == len(aprioriDist)-1: v = x1-x0 - total
                                else:                       v = int(((x1-x0)* val)/float(self.conditionalSubsetDict[attrVals]))
                                OWCanvasRectangle(self.canvas, x0+total, y1-self.boxSize, v, self.boxSize, self.colorPalette[i], self.colorPalette[i], z = 15)
                            else:
                                if i == len(aprioriDist)-1: v = y1-y0 - total
                                else:                       v = int(((y1-y0)* val)/float(self.conditionalSubsetDict[attrVals]))
                                OWCanvasRectangle(self.canvas, x1-self.boxSize, y0+total, self.boxSize, v, self.colorPalette[i], self.colorPalette[i], z = 15)
                            total += v

        self.addTooltip(x0, y0, x1-x0, y1-y0, condition, aprioriDist, attrVals, pearson, expected)


    # add tooltips
    def addTooltip(self, x, y, w, h, condition, apriori = None, attrVals = None, pearson = None, expected = None):
        tooltipText = "Examples in this area have:<br>" + condition

        if apriori:
            clsValues = list(self.attributeValuesDict.get(self.data.domain.classVar.name, [])) or getVariableValuesSorted(self.data, self.data.domain.classVar.name)
            actual = [self.conditionalDict[attrVals + "-" + clsValues[i]] for i in range(len(apriori))]
            if sum(actual) > 0:
                apriori = [apriori[key] for key in clsValues]
                aprioriText = ""; actualText = ""
                text = ""
                for i in range(len(clsValues)):
                    text += 4*"&nbsp;" + "<b>%s</b>: %d / %.1f%% (Expected %.1f / %.1f%%)<br>" % (clsValues[i], actual[i], 100.0*actual[i]/float(sum(actual)), (apriori[i]*sum(actual))/float(sum(apriori)), 100.0*apriori[i]/float(sum(apriori)))
                tooltipText += "Number of examples: " + str(int(sum(actual))) + "<br> Class distribution:<br>" + text[:-4]
        elif pearson and expected:
            tooltipText += "<hr>Expected number of examples: %.1f<br>Actual number of examples: %d<br>Standardized (Pearson) residual: %.1f" % (expected, self.conditionalDict[attrVals], pearson)
        tipRect = QRect(x, y, w, h)
        QToolTip.add(self.canvasView, tipRect, tooltipText)
        self.tooltips.append((tipRect, tooltipText))

    # draw the class legend below the square
    def DrawLegend(self, data, (x0, x1), (y0, y1)):
        if self.interiorColoring == CLASS_DISTRIBUTION and (not data.domain.classVar or data.domain.classVar.varType == orange.VarTypes.Continuous):
            return

        if self.interiorColoring == PEARSON:
            names = ["<-8", "-8:-4", "-4:-2", "-2:2", "2:4", "4:8", ">8", "Residuals:"]
            colors = self.redColors[::-1] + self.blueColors[1:]
        else:
            names = (list(self.attributeValuesDict.get(data.domain.classVar.name, [])) or getVariableValuesSorted(data, data.domain.classVar.name)) + [data.domain.classVar.name+":"]
            colors = [self.colorPalette[i] for i in range(len(data.domain.classVar.values))]

        self.names = [OWCanvasText(self.canvas, name) for name in names]
        totalWidth = sum([text.boundingRect().width() for text in self.names])

        # compute the x position of the center of the legend
        y = y1 + self.attributeNameOffset + 20
        distance = 30
        startX = (x0+x1)/2 - (totalWidth + (len(names))*distance)/2

        self.names[-1].move(startX+15, y+1); self.names[-1].show()
        xOffset = self.names[-1].boundingRect().width() + distance

        size = 16 # 8 + 8*(self.interiorColoring == PEARSON)

        for i in range(len(names)-1):
            if self.interiorColoring == PEARSON: edgeColor = Qt.black
            else: edgeColor = colors[i]

            OWCanvasRectangle(self.canvas, startX + xOffset, y-size/2, size, size, edgeColor, colors[i])
            self.names[i].move(startX + xOffset + 18, y)
            xOffset += distance + self.names[i].boundingRect().width()

    def saveToFileCanvas(self):
        sizeDlg = OWDlgs.OWChooseImageSizeDlg(self.canvas)
        sizeDlg.exec_loop()

    def setColors(self):
        dlg = self.createColorDialog()
        if dlg.exec_loop():
            self.colorSettings = dlg.getColorSchemas()
            self.selectedSchemaIndex = dlg.selectedSchemaIndex
            self.colorPalette = dlg.getDiscretePalette()
            self.updateGraph()

    def createColorDialog(self):
        c = OWDlgs.ColorPalette(self, "Color Palette")
        c.createDiscretePalette("Discrete palette", ColorBrewerColors)
        c.setColorSchemas(self.colorSettings, self.selectedSchemaIndex)
        return c

    # ########################################
    # cell/example selection
    def sendSelectedData(self):
        # send the selected examples
        self.send("Selected Examples", self.getSelectedExamples())

    # add a new rectangle. update the graph and see which mosaics does it intersect. add this mosaics to the recentlyAdded list
    def addSelection(self, rect):
        self.selectionRectangle = rect
        self.updateGraph(drillUpdateSelection = 0)
        self.sendSelectedData()

        if getattr(self, "recentlyAdded", []):
            self.selectionConditionsHistorically = self.selectionConditionsHistorically + [self.recentlyAdded]
            self.recentlyAdded = []

        self.optimizationDlg.mtUpdateState()            # we have already called this in self.updateGraph() call
        self.selectionRectangle = None

    # remove the mosaics that were added with the last selection rectangle
    def removeLastSelection(self):
        if self.selectionConditionsHistorically:
            vals = self.selectionConditionsHistorically.pop()
            for val in vals:
                if tuple(val) in self.selectionConditions:
                    self.selectionConditions.remove(tuple(val))

        self.updateGraph()
##        self.optimizationDlg.mtUpdateState()       # we have already called this in self.updateGraph() call
        self.sendSelectedData()

    def removeAllSelections(self):
        self.selectionConditions = []
        self.selectionConditionsHistorically = []
##        self.optimizationDlg.mtUpdateState()       # removeAllSelections is always called before updateGraph() - where mtUpdateState is called
        self.sendSelectedData()

    # return examples in currently selected boxes as example table or array of 0/1 values
    def getSelectedExamples(self, asExampleTable = 1, negate = 0, selectionConditions = None, data = None, attrs = None):
        if attrs == None:     attrs = self.getShownAttributeList()
        if data == None:      data = self.data
        if selectionConditions == None:    selectionConditions = self.selectionConditions

        if attrs == [] or not data:
            return None

        pp = orange.Preprocessor_take()
        sumIndices = numpy.zeros(len(data))
        for val in selectionConditions:
            for i, attr in enumerate(attrs):
                pp.values[data.domain[attr]] = val[i]
            indices = numpy.array(pp.selectionVector(data))
            sumIndices += indices
        selectedIndices = list(numpy.where(sumIndices > 0, 1 - negate, 0 + negate))

        if asExampleTable:
            return data.selectref(selectedIndices)
        else:
            return selectedIndices

    def saveSettings(self):
        OWWidget.saveSettings(self)
        self.optimizationDlg.saveSettings()
        


class SortAttributeValuesDlg(OWBaseWidget):
    def __init__(self, parentWidget = None, attrList = []):
        OWBaseWidget.__init__(self, None, None, "Sort Attribute Values", modal = TRUE)

        self.space = QVBox(self)
        self.layout = QVBoxLayout(self, 4)
        self.layout.addWidget(self.space)

        box1 = OWGUI.widgetBox(self.space, 1, orientation = "horizontal")

        self.attributeList = QListBox(box1)
        self.attributeList.setSelectionMode(QListBox.Extended)

        vbox = OWGUI.widgetBox(box1, "", orientation = "vertical")
        self.buttonUPAttr   = OWGUI.button(vbox, self, "", callback = self.moveAttrUP, tooltip="Move selected attribute values up")
        self.buttonDOWNAttr = OWGUI.button(vbox, self, "", callback = self.moveAttrDOWN, tooltip="Move selected attribute values down")
        self.buttonUPAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_up1.png")))
        self.buttonUPAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonUPAttr.setMaximumWidth(20)
        self.buttonDOWNAttr.setPixmap(QPixmap(os.path.join(self.widgetDir, r"icons\Dlg_down1.png")))
        self.buttonDOWNAttr.setSizePolicy(QSizePolicy(QSizePolicy.Fixed , QSizePolicy.Expanding))
        self.buttonDOWNAttr.setMaximumWidth(20)
        self.buttonUPAttr.setMaximumWidth(20)

        box2 = OWGUI.widgetBox(self.space, 1, orientation = "horizontal")
        self.okButton =     OWGUI.button(box2, self, "OK", callback = self.accept)
        self.cancelButton = OWGUI.button(box2, self, "Cancel", callback = self.reject)

        for attr in attrList:
            self.attributeList.insertItem(attr)

        self.resize(300, 300)

    # move selected attribute values
    def moveAttrUP(self):
        for i in range(1, self.attributeList.count()):
            if self.attributeList.isSelected(i):
                self.attributeList.insertItem(self.attributeList.text(i), i-1)
                self.attributeList.removeItem(i+1)
                self.attributeList.setSelected(i-1, TRUE)

    def moveAttrDOWN(self):
        for i in range(self.attributeList.count()-2,-1,-1):
            if self.attributeList.isSelected(i):
                self.attributeList.insertItem(self.attributeList.text(i), i+2)
                self.attributeList.removeItem(i)
                self.attributeList.setSelected(i+1, TRUE)


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow = OWMosaicDisplay()
    a.setMainWidget(ow)
    ow.show()
    for d in ["zoo.tab", "iris.tab", "zoo.tab"]:
        data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\\" + d)
        ow.setData(data)
        ow.handleNewSignals()
    a.exec_loop()

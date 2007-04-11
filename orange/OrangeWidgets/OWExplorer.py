from OWBaseWidget import *
import OWGUI, numpy
from orngCI import FeatureByCartesianProduct
from qtcanvas import QCanvas, QCanvasView
import orngMosaic
# valid visualizationMethodName values:
# Mosaic
# Scatterplot
# Linear projection

allExamplesText = "<All examples>"

class OWExplorerDialog(OWBaseWidget):
    settingsList = ["showDataSubset", "invertSelection", "lastSaveDirName"]
    def __init__(self, visualizationWidget, optimizationDialog, visualizationMethodName, signalManager = None):
        OWBaseWidget.__init__(self, None, signalManager, "Data Explorer Dialog", savePosition = True)

        self.visualizationWidget = visualizationWidget
        self.optimizationDialog = optimizationDialog
        self.visualizationMethodName = visualizationMethodName
        self.signalManager = signalManager
        self.wholeDataSet = None
        self.newClassValue = ""
        self.processingSubsetData = 0       # this is a flag that we set when we call visualizationWidget.setData function
        self.showDataSubset = 1
        self.invertSelection = 0
        self.lastSaveDirName = os.getcwd()

        # mosaic settings
        self.dontAddLeafs = 1
        self.mosaicSize = 300

        self.loadSettings()

        self.controlArea = QVBoxLayout(self)
        self.tabs = QTabWidget(self, 'tabWidget')
        self.controlArea.addWidget(self.tabs)

        self.TreeTab = QVGroupBox(self)
        self.tabs.insertTab(self.TreeTab, "Drilling")

        subsetBox = OWGUI.widgetBox(self.TreeTab, "Example Subset Analysis")
        self.splitter = QSplitter(Qt.Vertical, subsetBox, "main")
        self.subsetTree = QListView(self.splitter)
        self.subsetTree.setRootIsDecorated(1)
        self.subsetTree.setAllColumnsShowFocus(1)
        self.subsetTree.addColumn('Visualized Attributes')
        self.subsetTree.addColumn('# inst.')
        self.subsetTree.setColumnWidth(0, 300)
        self.subsetTree.setColumnWidthMode(0, QListView.Maximum)
        self.subsetTree.setColumnAlignment(0, QListView.AlignLeft)
        self.subsetTree.setColumnWidth(1, 50)
        self.subsetTree.setColumnWidthMode(1, QListView.Manual)
        self.subsetTree.setColumnAlignment(1, QListView.AlignRight)
        self.connect(self.subsetTree, SIGNAL("selectionChanged(QListViewItem *)"), self.subsetTreeSelectedItemChanged)
        self.connect(self.subsetTree, SIGNAL("rightButtonPressed(QListViewItem *, const QPoint &, int )"), self.subsetTreeRemoveItemPopup)

        self.selectionsList = QListBox(self.splitter)
        self.connect(self.selectionsList, SIGNAL("selectionChanged()"), self.selectionListSelectedItemChanged)

        self.subsetItems = {}
        self.subsetUpdateInProgress = 0
        self.treeRoot = None

        explorerBox = OWGUI.widgetBox(self.TreeTab, 1)
        OWGUI.button(explorerBox, self, "Explore Currently Selected Examples", callback = self.subsetEploreCurrentSelection, tooltip = "Visualize only selected examples and find interesting projections of them")
        OWGUI.checkBox(explorerBox, self, 'showDataSubset', 'Show unselected data as example subset', tooltip = "This option determines what to do with the examples that are not selected in the projection.\nIf checked then unselected examples will be visualized in the same way as examples that are received through the 'Example Subset' signal.")

        if self.visualizationMethodName == "Mosaic":
            autoBuildTreeBox = OWGUI.widgetBox(self.TreeTab, 1)
            OWGUI.button(autoBuildTreeBox, self, "Automatically Build Mosaic Tree", callback = self.mosaicAutoBuildTree, tooltip = "Evaluate different mosaic diagrams and automatically build a tree of mosaic diagrams with clear class separation")

            self.MosaicTreeTab = QVGroupBox(self)
            self.tabs.insertTab(self.MosaicTreeTab, "Mosaic Tree Settings")
            settingsBox = OWGUI.widgetBox(self.MosaicTreeTab, "Settings")
            OWGUI.checkBox(settingsBox, self, 'dontAddLeafs', 'Do not add leafs in the tree', tooltip = "Do you want to show leafs in the tree or not?")
            showTreeBox = OWGUI.widgetBox(self.MosaicTreeTab, "Mosaic Tree Visualization")
            OWGUI.lineEdit(showTreeBox, self, "mosaicSize", "Size of individual mosaics: ", orientation = "horizontal", tooltip = "What are the X and Y dimensions of individual mosaics in the tree?", valueType = int, validator = QIntValidator(self))
            OWGUI.button(showTreeBox, self, "Visualize Mosaic Tree", callback = self.visualizeMosaicTree, tooltip = "Visualize a tree where each node is a mosaic diagram")

            loadSaveBox = OWGUI.widgetBox(self.MosaicTreeTab, 1, orientation = "horizontal")
            OWGUI.button(loadSaveBox, self, "Load", callback = self.loadTree, tooltip = "Load a tree from a file")
            OWGUI.button(loadSaveBox, self, "Save", callback = self.saveTree, tooltip = "Save tree to a file")
        else:
            OWGUI.checkBox(explorerBox, self, 'invertSelection', 'Show inverted selection', tooltip = "Do you wish to show selected or unselected examples as example subset?")

            self.ClassChangeTab = QVGroupBox(self)
            self.tabs.insertTab(self.ClassChangeTab, "Class Change")
            self.newClassBox = OWGUI.widgetBox(self.ClassChangeTab, "Change Class Value For Selected Examples", orientation="horizontal")
            self.newClassCombo = OWGUI.comboBoxWithCaption(self.newClassBox, self, "newClassValue", "New class value: ", items = [], sendSelectedValue = 1, valueType = str, tooltip = "Change class value for examples that are selected in the projection")
            self.newClassCombo.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred))
            applyButt = OWGUI.button(self.newClassBox, self, "Apply", callback = self.changeClassForExamples, tooltip = "Apply the new class value to the selected examples")
            applyButt.setMaximumWidth(40)

        self.subsetPopupMenu = QPopupMenu(self)
        self.subsetPopupMenu.insertItem("Explore this selection", self.subsetEploreCurrentSelection)
        self.subsetPopupMenu.insertItem("Find interesting projection", self.optimizationDialog.evaluateProjections)
        self.subsetPopupMenu.insertSeparator()
        self.subsetPopupMenu.insertItem("Remove node", self.removeSelectedItem)
        self.subsetPopupMenu.insertItem("Clear tree", self.initSubsetTree)
        self.controlArea.activate()

        self.resize(320, 600)

    def setData(self, data):
        if self.processingSubsetData == 0:
            self.wholeDataSet = data
            self.initSubsetTree()
            if hasattr(self, "newClassCombo"):
                self.newClassCombo.clear()
                if data and data.domain.classVar and data.domain.classVar.varType == orange.VarTypes.Discrete:
                    for val in data.domain.classVar.values:
                        self.newClassCombo.insertItem(val)


    # clear subset tree and create a new root
    def initSubsetTree(self):
        self.subsetItems = {}
        self.subsetTree.clear()
        self.treeRoot = None
        self.subsetTree.setColumnWidth(0, self.subsetTree.width() - self.subsetTree.columnWidth(1)-4)

        if self.wholeDataSet:
            root = QListViewItem(self.subsetTree, allExamplesText, str(len(self.wholeDataSet)))
            root.setOpen(1)
            self.subsetTree.insertItem(root)
            self.treeRoot = root
            self.subsetItems[str(root)] = {"data": self.wholeDataSet, "exampleCount": len(self.wholeDataSet)}
            self.processingSubsetData = 1
            self.subsetTree.setSelected(root, 1)
            self.processingSubsetData = 0

    def getProjectionState(self, getSelectionIndices = 1):
        selectedIndices = None
        attrList = self.visualizationWidget.getShownAttributeList()
        exampleCount = self.visualizationWidget.data and len(self.visualizationWidget.data) or 0
        if self.visualizationMethodName == "Mosaic":
            retDict = {"attrs": list(attrList), "selectionConditions": list(self.visualizationWidget.selectionConditions), "selectionConditionsHistorically": list(self.visualizationWidget.selectionConditionsHistorically), "exampleCount": exampleCount}
            if getSelectionIndices:
                selectedIndices = self.visualizationWidget.getSelectedExamples(asExampleTable = 0)
                retDict["selectedIndices"] = selectedIndices
        else:
            retDict = {"attrs": list(attrList), "selectionConditions": self.visualizationWidget.graph.getSelections(), "anchorData": getattr(self.visualizationWidget.graph, "anchorData", None), "exampleCount": exampleCount}
            if getSelectionIndices:
                selectedIndices, unselIndices = self.visualizationWidget.graph.getSelectionsAsIndices(attrList)
                retDict["selectedIndices"] = selectedIndices
        return attrList, selectedIndices, retDict


    # restore visualized attributes and selections in the projections
    def setProjectionState(self, state):
        self.visualizationWidget.setShownAttributes(state.get("attrs", None))
        if self.visualizationMethodName == "Mosaic":
            self.visualizationWidget.selectionConditions = list(state.get("selectionConditions", []))
            self.visualizationWidget.selectionConditionsHistorically = list(state.get("selectionConditionsHistorically", []))
        else:
            if state.get("selectionConditions"): self.visualizationWidget.graph.setSelections(state["selectionConditions"])
            if state.get("anchorData"): self.visualizationWidget.graph.anchorData = state["anchorData"]
        self.visualizationWidget.updateGraph()


    # new element is added into the subsetTree
    def subsetEploreCurrentSelection(self):
        if not self.wholeDataSet:
            return

        attrList, selectedIndices, retDict = self.getProjectionState()

        if sum(selectedIndices) == 0:
            QMessageBox.information(self, "No data selection", "To explore a subset of examples you first have to select them in the projection.", QMessageBox.Ok)
            return

        selectedData = self.visualizationWidget.data.selectref(selectedIndices)
        unselectedData = self.visualizationWidget.data.selectref(selectedIndices, negate = 1)
        #self.visualizationWidget.setData(selectedData, onlyDrilling = 1)
        selectedItem = self.subsetTree.selectedItem()     # current selection

        attrListStr = self.attrsToString(attrList)
        newListItem = QListViewItem(selectedItem, attrListStr)

        # if newListItem was the first child bellow the root we have to add another child that will actually show only selected examples in newListItem
        if str(selectedItem.text(0)) == allExamplesText:
            self.subsetItems[str(newListItem)] = retDict
            newListItem.setText(1, str(len(self.visualizationWidget.data)))
            newListItem.setOpen(1)

            newnewListItem = QListViewItem(newListItem, attrListStr)
            self.subsetItems[str(newnewListItem)] = {"attrs": list(attrList), "exampleCount": len(selectedData)}
            newnewListItem.setText(1, str(len(selectedData)))
            newnewListItem.setOpen(1)
            self.subsetTree.setSelected(newnewListItem, 1)
        else:
            self.subsetItems[str(selectedItem)] = retDict
            self.subsetItems[str(newListItem)] = {"attrs": list(attrList), "exampleCount": len(selectedData)}
            newListItem.setText(1, str(len(selectedData)))
            newListItem.setOpen(1)
            self.subsetTree.setSelected(newListItem, 1)

    # a different attribute set was selected in mosaic. update the attributes in the selected node
    def updateState(self):
        if not self.wholeDataSet: return
        if self.processingSubsetData: return

        selectedItem = self.subsetTree.selectedItem()
        if not selectedItem or str(selectedItem.text(0)) == allExamplesText:   # we don't change the title of the root.
            attrList = self.visualizationWidget.getShownAttributeList()
            self.subsetItems[str(selectedItem)]["attrs"] = attrList
            return

        attrList, selectionIndices, retDict = self.getProjectionState(getSelectionIndices = 0)

        # if this is the last element in the tree, then update the element's values
        if not selectedItem.firstChild():
            selectedItem.setText(0, self.attrsToString(attrList))
            self.subsetItems[str(selectedItem)].update(retDict)

        # add a sibling if we changed any value
        else:
            if 0 in [self.subsetItems[str(selectedItem)][key] == retDict[key] for key in retDict.keys()]:
                parent = selectedItem.parent()
                newListItem = QListViewItem(parent, self.attrsToString(attrList))
                newListItem.setOpen(1)
                newListItem.setText(1, str(selectedItem.text(1)))   # new item has the same number of examples as the selected item
                self.subsetItems[str(newListItem)] = retDict
                self.subsetTree.setSelected(newListItem, 1)


    # we selected a different item in the tree
    def subsetTreeSelectedItemChanged(self, newSelection):
        if self.processingSubsetData:
            return
        self.processingSubsetData = 1

        if not newSelection or str(newSelection.text(0)) == allExamplesText:
            self.visualizationWidget.setData(self.wholeDataSet)
            self.visualizationWidget.setSubsetData(None)
            if hasattr(self.visualizationWidget, "handleNewSignals"):
                self.visualizationWidget.handleNewSignals()
        else:
            indices = self.getItemIndices(newSelection)
            selectedData = self.wholeDataSet
            unselectedData = orange.ExampleTable(self.wholeDataSet.domain)
            for ind in indices:
                unselectedData.extend(selectedData.selectref(ind, negate = 1))
                selectedData = selectedData.selectref(ind)

            # set data
            if self.invertSelection:
                temp = selectedData
                selectedData = unselectedData
                unselectedData = temp
            self.visualizationWidget.setData(selectedData)  #self.visualizationWidget.setData(selectedData, onlyDrilling = 1)
            if self.showDataSubset:
                self.visualizationWidget.setSubsetData(unselectedData)      #self.visualizationWidget.subsetData = unselectedData
            else:
                self.visualizationWidget.setSubsetData(None)
            # set projection state - selections, visualized attributes, ...
            self.setProjectionState(self.subsetItems[str(newSelection)])

        self.processingSubsetData = 0

    # a new selection was selected in the selection list. update the graph
    def selectionListSelectedItemChanged(self):
        pass


    # #####################
    # misc functions
    def getItemIndices(self, item):
        indices = []
        parent = item.parent()
        while parent:
            parentIndices = self.subsetItems[str(parent)].get("selectedIndices", None)
            if parentIndices:
                indices.insert(0, parentIndices)        # insert indices in reverse order
            parent = parent.parent()
        return indices

    # popup menu items
    def removeSelectedItem(self):
        item = self.subsetTree.selectedItem()
        if not item:
            return
        if str(item.text(0)) == allExamplesText:
            self.initSubsetTree()
        else:
            item.parent().takeItem(item)

    def subsetTreeRemoveItemPopup(self, item, point, i):
        self.subsetPopupMenu.popup(point, 0)

    def resizeEvent(self, ev):
        OWBaseWidget.resizeEvent(self, ev)
        self.subsetTree.setColumnWidth(0, self.subsetTree.width()-self.subsetTree.columnWidth(1)-4 - 20)

    def destroy(self, dw = 1, dsw = 1):
        self.saveSettings()
        OWBaseWidget.destroy(self, dw, dsw)

    def attrsToString(self, attrList):
        return reduce(lambda x,y: x+', '+y, attrList)

    # return actual item in the tree to that str(item) == strItem
    def strToItem(self, strItem, currItem = -1):
        if currItem == -1:
            currItem = self.treeRoot
        if currItem == None:
            return None
        if str(currItem) == strItem:
            return currItem
        child = currItem.firstChild()
        if child:
            item = self.strToItem(strItem, child)
            if item:
                return item
        return self.strToItem(strItem, currItem.nextSibling())


    # save tree to a file
    def saveTree(self):
        qname = QFileDialog.getSaveFileName( os.path.join(self.lastSaveDirName, "explorer tree.tree"), "Explorer tree (*.tree)", self, "", "Save tree")
        if qname.isEmpty():
            return
        name = str(qname)
        self.lastSaveDirName = os.path.split(name)[0]

        tree = {"None": [str(self.treeRoot)]}
        self.treeToDict(self.treeRoot, tree)
        import cPickle
        f = open(name, "w")
        cPickle.dump((tree, self.subsetItems), f)
        f.close()

    # load tree from a file
    def loadTree(self, name = None):
        self.subsetItems = {}
        self.subsetTree.clear()
        self.treeRoot = None

        if name == None:
            name = QFileDialog.getOpenFileName( self.lastSaveDirName, "Explorer tree (*.tree)", self, "", "Load tree")
            if name.isEmpty(): return
            name = str(name)

        self.lastSaveDirName = os.path.split(name)[0]
        import cPickle
        f = open(name, "r")
        (tree, subsetItems) = cPickle.load(f)
        self.createTreeFromDict(tree, subsetItems, self.subsetTree, tree["None"][0])

    # recursively create a tree from a loaded file
    def createTreeFromDict(self, tree, subsetItems, parentItem, currItemKey):
        attrList = subsetItems[currItemKey]["attrs"]
        exampleCount = subsetItems[currItemKey]["exampleCount"]
        strAttrs = self.attrsToString(attrList)
        item = QListViewItem(parentItem, strAttrs, str(exampleCount))
        item.setOpen(1)
        if not self.treeRoot:
            item.setText(0, allExamplesText)
            self.treeRoot = item

        #self.subsetTree.insertItem(item)
        self.subsetItems[str(item)] = subsetItems[currItemKey]

        if tree.has_key(currItemKey):
            for itm in tree[currItemKey]:
                self.createTreeFromDict(tree, subsetItems, item, itm[0])


    #################################################
    # build mosaic tree methods
    def mosaicAutoBuildTree(self):
        selectedItem = self.subsetTree.selectedItem()
        while selectedItem.firstChild():
            selectedItem.takeItem(selectedItem.firstChild())

        # create a mosaic so that we don't set data to the main mosaic (which would mean that we would have to prevent the user from clicking the current tree)
        mosaic = orngMosaic.orngMosaic()
#        mosaic.setData(self.visualizationWidget.data)
        for setting in self.optimizationDialog.settingsList:
            setattr(mosaic, setting, getattr(self.optimizationDialog, setting, None))
        if mosaic.qualityMeasure == orngMosaic.CN2_RULES:
            mosaic.qualityMeasure == orngMosaic.GINI_INDEX

        
    def visualizeMosaicTree(self):
        tree = {"None": [str(self.treeRoot)]}
        self.treeToDict(self.treeRoot, tree)
        dialog = MosaicTreeDialog(self, self.visualizationWidget, self.signalManager)
        dialog.visualizeTree(tree)
        dialog.show()

    def treeToDict(self, node, tree):
        if not node: return 0

        child = node.firstChild()
        if child:
            depth = self.treeToDict(child, tree)
        else:
            depth = 0

        #key = (str(node.parent()), str(node.text(0)), tuple(self.subsetItems[str(node)]["attrs"]))
        tree[str(node.parent())] = tree.get(str(node.parent()), []) + [
                                          (str(node),
                                          self.subsetItems[str(node)].get("selectedIndices", []),
                                          self.subsetItems[str(node)].get("selectionConditions", []),
                                          self.subsetItems[str(node)].get("selectionConditionsHistorically", []),
                                          self.subsetItems[str(node)].get("anchorData", []),
                                          depth,
                                          str(node.text(1)))]

        self.treeToDict(node.nextSibling(), tree)
        return depth + 1


    # set the new class value for selected examples
    def changeClassForExamples(self):
        pass


class MosaicTreeDialog(OWBaseWidget):
    def __init__(self, parentWidget, mosaicWidget, signalManager = None):
        OWBaseWidget.__init__(self, parentWidget, signalManager, "Mosaic Tree")
        self.mosaicWidget = mosaicWidget
        self.parentWidget = parentWidget

        self.canvasLayout = QVBoxLayout(self)
        self.canvasWidget = QWidget(self)

        self.canvasXSize = 10000
        self.canvasYSize = 10000
        self.canvas = QCanvas(self.canvasXSize, self.canvasYSize)
        self.canvasView = QCanvasView(self.canvas, self)
        self.canvasLayout.addWidget(self.canvasView)
        self.canvasLayout.activate()

        self.canvasView.show()
        #self.tree = parentWidget.subsetTree
        #self.canvas.resize(self.canvasView.size().width()-5, self.canvasView.size().height()-5)
        self.resize(1000, 1000)
        self.move(0,0)

    def visualizeTree(self, tree):
        mosaicCanvas = self.mosaicWidget.canvas
        mosaicCanvasView = self.mosaicWidget.canvasView
        self.mosaicWidget.canvas = self.canvas
        self.mosaicWidget.canvasView = self.canvasView

        #selectedItem = self.tree.selectedItem()
        originalData = self.mosaicWidget.data
        itemsToDraw = {0: [tree[tree["None"][0]] + [(tree["None"][0], 0, self.canvasXSize)]]}
        treeDepth = 0
        mosaicSize = self.parentWidget.mosaicSize
        mosOffset = 50
        print tree
        fullMosaicSize = mosaicSize + 2 * mosOffset     # we need some space also for text labels
        while itemsToDraw.has_key(treeDepth):
            groups = itemsToDraw[treeDepth]       # [('(__main__.qt.QListViewItem object at 0x066AD4E0)', 'milk, legs', ['milk', 'legs'], [0,1,...], [('0', '5')], 1), ...]
            treeDepth += 1

            xPos = 0
            yPos = 50 + treeDepth * fullMosaicSize
            for group in groups:
                #startXPos = xPos
                attrList = self.parentWidget.subsetItems[group[0][0]]["attrs"]

                key = self.parentWidget.strToItem(group[0])
                indices = self.parentWidget.getItemIndices(key)
                data = self.parentWidget.wholeDataSet
                unselectedData = orange.ExampleTable(self.parentWidget.wholeDataSet.domain)
                for ind in indices:
                    unselectedData.extend(data.selectref(ind, negate = 1))
                    data = data.selectref(ind)

                # call drawing of the mosaic
                self.mosaicWidget.updateGraph(data, unselectedData, attrList, erasePrevious = 0, positions = (xPos+mosOffset, yPos+mosOffset, mosaicSize), drawLegend = 0, drillUpdateSelection = 0)

                # what are the selections

                for item in group[:-1]:    # the last item is the middle x position
                    subsetItemsToDraw = []

                    # set the data used to generate the mosaic diagram
                    indices = self.parentWidget.getItemIndices(self.parentWidget.strToItem(item[0]))
                    data = self.parentWidget.wholeDataSet
                    unselectedData = orange.ExampleTable(self.parentWidget.wholeDataSet.domain)
                    for ind in indices:
                        unselectedData.extend(data.selectref(ind, negate = 1))
                        data = data.selectref(ind)

                    # call drawing of the mosaic
                    self.mosaicWidget.updateGraph(data, unselectedData, attrList, erasePrevious = 0, positions = (xPos+mosOffset, yPos+mosOffset, mosaicSize), drawLegend = 0, drillUpdateSelection = 0)

                    if tree.has_key(item[0]):
                        for child in tree[item[0]]:
                            subsetItemsToDraw.append(child)
                    if subsetItemsToDraw != []:
                        subsetItemsToDraw.append(xPos)
                        itemsToDraw[treeDepth] = itemsToDraw.get(treeDepth, []) + [subsetItemsToDraw]

                    # next mosaic will be to the right
                    xPos += fullMosaicSize

        self.mosaicWidget.canvas = mosaicCanvas
        self.mosaicWidget.canvasView = mosaicCanvasView



#test widget appearance
if __name__=="__main__":
    import sys
    import OWMosaicDisplay
    a = QApplication(sys.argv)
    #ow = OWExplorerDialog(None, None, "name")
    #ow = MosaicTreeDialog(None, None)
    ow = OWMosaicDisplay.OWMosaicDisplay()
    a.setMainWidget(ow)
    ow.show()
    data = orange.ExampleTable(r"e:\Development\Python23\Lib\site-packages\Orange Datasets\UCI\zoo.tab")
    ow.setData(data)
    #ow.explorerDlg.loadTree(r"E:\Development\Python23\Lib\site-packages\Orange\scripts\explorer tree.tree" )
    #ow.explorerDlg.visualizeMosaicTree()
    a.exec_loop()
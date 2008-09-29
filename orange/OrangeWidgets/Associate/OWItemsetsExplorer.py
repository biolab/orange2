"""
<name>Itemsests Explorer</name>
<description>Itemsets explorer.</description>
<icon>icons/ItemsetsExplorer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1100</priority>
"""
from OWWidget import *
import OWGUI, OWTools, orngMisc
import sys, re


class OWItemsetsExplorer(OWWidget):
    settingsList = ["treeDepth", "showWholeItemsets", "autoSend", "purgeAttributes", "purgeClasses"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Itemsets explorer")

        from OWItemsets import Itemsets
        self.inputs = [("Itemsets", Itemsets, self.setItemsets)]
        self.outputs = [("Itemsets", Itemsets), ("Examples", ExampleTable)]

        self.showWholeItemsets = 1
        self.treeDepth = 2
        self.autoSend = True
        self.dataChanged = False
        self.purgeAttributes = True
        self.purgeClasses = True

        self.nItemsets = self.nSelectedItemsets = self.nSelectedExamples = ""
        self.loadSettings()

        self.treeItemsets = QTreeWidget(self.mainArea)
        self.mainArea.layout().addWidget(self.treeItemsets)
        self.treeItemsets.setSelectionMode (QTreeWidget.ExtendedSelection)
        self.treeItemsets.setHeaderLabels(["Itemsets", "Examples"])
        self.treeItemsets.setAllColumnsShowFocus ( 1)
        self.treeItemsets.setAlternatingRowColors(1) 
        self.treeItemsets.setColumnCount(2) 
        self.connect(self.treeItemsets,SIGNAL("itemSelectionChanged()"),self. selectionChanged)

        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace = True)
        OWGUI.label(box, self, "Number of itemsets: %(nItemsets)s")
        OWGUI.label(box, self, "Selected itemsets: %(nSelectedItemsets)s")
        OWGUI.label(box, self, "Selected examples: %(nSelectedExamples)s")

        box = OWGUI.widgetBox(self.controlArea, "Tree", addSpace = True)
        OWGUI.spin(box, self, "treeDepth", label = "Tree depth", min = 0, max = 10, step = 1, callback = self.populateTree, callbackOnReturn = True)
        OWGUI.checkBox(box, self, "showWholeItemsets", "Display whole itemsets", callback = self.setWholeItemsets)
        OWGUI.button(box, self, "Expand All", callback = lambda: self.treeItemsets.expandAll())
        OWGUI.button(box, self, "Collapse", callback = lambda: self.treeItemsets.collapseAll())

        OWGUI.rubber(self.controlArea)

        boxSettings = OWGUI.widgetBox(self.controlArea, 'Send selection')
        OWGUI.checkBox(boxSettings, self, "purgeAttributes", "Purge attribute values/attributes", box=None, callback=self.purgeChanged)
        self.purgeClassesCB = OWGUI.checkBox(OWGUI.indentedBox(boxSettings), self, "purgeClasses", "Purge class attribute", callback=self.purgeChanged)
        if not self.purgeAttributes:
            self.purgeClassesCB.setEnabled(False)
            self.oldPurgeClasses = False

        cbSendAuto = OWGUI.checkBox(boxSettings, self, "autoSend", "Send immediately", box=None)
        btnUpdate = OWGUI.button(boxSettings, self, "Send", self.sendData)
        OWGUI.setStopper(self, btnUpdate, cbSendAuto, "dataChanged", self.sendData)

        self.itemsets= None

    def setWholeItemsets(self, node = None):
        print "da"
        if not node:
            for i in range(self.treeItemsets.topLevelItemCount()):
                self.setWholeItemsets(self.treeItemsets.topLevelItem(i))
        else:
            t = getattr(node, self.showWholeItemsets and "allItems" or "remainder", "")
            if t:
                node.setText(0, t)
            for i in range(node.childCount()):
                self.setWholeItemsets(node.child(i))

    def resizeEvent(self, *a):
        self.treeItemsets.setColumnWidth(0,self.treeItemsets.width()-60)
        OWWidget.resizeEvent(self, *a)

    def showEvent(self, *a):
        self.treeItemsets.setColumnWidth(0,self.treeItemsets.width()-60)
        OWWidget.showEvent(self, *a)

    def populateTree(self):
        self.treeItemsets.clear()
        if not self.itemsets:
            return
        
        itemset2Node = {}
        for itemset, examples, id in self.itemsets:
            if len(itemset) == 1:
                node = QTreeWidgetItem(self.treeItemsets, [itemset[0], str(len(examples))])
                node.itemsetId = id
                itemset2Node[itemset] = node
        
        treeDepth = self.treeDepth
        for depth in range(2, 1+treeDepth):
            newItemset2Node = {}
            for itemset, examples, id in self.itemsets:
                if len(itemset) == depth:
                    for itout in range(depth):
                        parentitemset = itemset[:itout]+itemset[itout+1:]
                        node = QTreeWidgetItem(itemset2Node[parentitemset], [itemset[itout], str(len(examples))])
                        node.itemsetId = id
                        newItemset2Node[itemset] = node
            itemset2Node = newItemset2Node
                        
        showWhole = self.showWholeItemsets
        for itemset, examples, id in sorted((i for i in self.itemsets if len(i[0]) > treeDepth), cmp = lambda x,y:cmp(len(x[0]), len(y[0]))):
            for itout in orngMisc.MofNCounter(treeDepth, len(itemset)):
                parentitemset = tuple([itemset[i] for i in itout])
                node.allItems = " ".join(itemset)
                node.remainder = " ".join(item for i, item in enumerate(itemset) if i not in itout)
                node = QTreeWidgetItem(itemset2Node[parentitemset], [showWhole and node.allItems or node.remainder, str(len(examples))])
                node.itemsetId = id

             
    def gatherItemsets(self, node, ids):
        if hasattr(node, "itemsetId"):
            ids.add(node.itemsetId)
        for i in range(node.childCount()):
            self.gatherItemsets(node.child(i), ids)

    def purgeChanged(self):
        if self.purgeAttributes:
            if not self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(True)
                self.purgeClasses = self.oldPurgeClasses
        else:
            if self.purgeClassesCB.isEnabled():
                self.purgeClassesCB.setEnabled(False)
                self.oldPurgeClasses = self.purgeClasses
                self.purgeClasses = False
    
    def selectionChanged(self):
        ids = set()
        for node in self.treeItemsets.selectedItems():
            self.gatherItemsets(node, ids)
        self.selectedItemsets = [self.itemsets[i] for i in ids]
        exampleids = set()
        for itemset in self.selectedItemsets:
            exampleids.update(itemset[1])
        self.exampleids = list(exampleids)
        self.setInfo()
        self.sendIf()

    def sendIf(self):
        if self.autoSend:
            self.sendData()
        else:
            self.dataChanged = True
            
    def sendData(self):
        self.dataChanged = False
        self.send("Itemsets", (self.dataset, self.selectedItemsets))
        print len(self.selectedItemsets)
        if self.selectedItemsets:
            examples = self.dataset.getitemsref(self.exampleids)
            if self.purgeAttributes or self.purgeClasses:
                examples = OWTools.domainPurger(examples, self.purgeClasses)
        else:
            examples = None

        self.send("Examples", examples)
       

    def setInfo(self):
        if self.itemsets:
            self.nItemsets = len(self.itemsets)
            if self.selectedItemsets:
                self.nSelectedItemsets = len(self.selectedItemsets)
                self.nSelectedExamples = len(self.exampleids)
            else:
                self.nSelectedItemsets = self.nSelectedExamples = ""
        else:
            self.nItemsets = self.nSelectedItemsets = self.nSelectedExamples = ""
            
    def setItemsets(self, data):
        if not data:
            self.dataset = self.origItemsets = itemsets = dataset = None
        else:
            self.dataset, self.origItemsets = data
            dataset, itemsets = self.dataset, self.origItemsets
        if itemsets:
            if isinstance(itemsets[0][0][0], tuple):
                vars = dataset.domain.variables
                self.itemsets = [(tuple(["%s=%s" % (vars[i].name, vars[i].values[j]) for i, j in itemset]),
                                  examples,
                                  i) for i, (itemset, examples) in enumerate(itemsets)]
            else:
                domain = dataset.domain
                self.itemsets = [(tuple([domain[i].name for i in itemset]),
                                  examples,
                                  i) for i, (itemset, examples) in enumerate(itemsets)]
        else:
            self.dataset = self.itemsets = None
        self.populateTree()
        self.selectionChanged()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWItemsetsExplorer()

    #dataset = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    dataset = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    rules=orange.AssociationRulesInducer(dataset, support = 0.5, maxItemSets=10000)
    ow.arules(rules)

    ow.show()
    a.exec_()
    ow.saveSettings()
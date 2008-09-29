"""
<name>Association Rules Explorer</name>
<description>Association rules tree viewer.</description>
<icon>icons/AssociationRulesTreeViewer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>300</priority>
"""
from OWWidget import *
import OWGUI, OWTools
import sys, re


class OWAssociationRulesTreeViewer(OWWidget):
    measures = [("Support",    "Supp", "support"),
                ("Confidence", "Conf", "confidence"),
                ("Lift",       "Lift", "lift"),
                ("Leverage",   "Lev",  "leverage"),
                ("Strength",   "Strg", "strength"),
                ("Coverage",   "Cov",  "coverage")]

    # only the last name can be taken for settings - the first two can be translated
    settingsList = ["treeDepth", "showWholeRules", "autoSend", "purgeAttributes", "purgeClasses"] \
                   + ["show%s" % m[2] for m in measures]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Association rules viewer")

        self.inputs = [("Association Rules", orange.AssociationRules, self.arules)]
        self.outputs = [("Association Rules", orange.AssociationRules), ("Covered Examples", ExampleTable), ("Matching Examples", ExampleTable), ("Mismatching Examples", ExampleTable)]

        self.showWholeRules = 1
        self.treeDepth = 2
        self.autoSend = True
        self.dataChanged = False
        self.purgeAttributes = True
        self.purgeClasses = True

        self.nRules = self.nSelectedRules = self.nSelectedExamples = self.nMatchingExamples = self.nMismatchingExamples = ""
        
        self.showsupport = self.showconfidence = 1
        self.showlift = self.showleverage = self.showstrength = self.showcoverage = 0
        self.loadSettings()

#        self.grid = QGridLayout()
#        rightUpRight = OWGUI.widgetBox(self.mainArea, box="Shown measures", orientation = self.grid)
#        self.cbMeasures = [OWGUI.checkBox(rightUpRight, self, "show"+attr, long, callback = self.showHideColumn, addToLayout = 0) for long, short, attr in self.measures]
#        for i, cb in enumerate(self.cbMeasures):
#            self.grid.addWidget(cb, i % 2, i / 2)

        box = OWGUI.widgetBox(self.mainArea, orientation = 0)
        OWGUI.widgetLabel(box, "Shown measures: ")
        self.cbMeasures = [OWGUI.checkBox(box, self, "show"+attr, long+"   ", callback = self.showHideColumn) for long, short, attr in self.measures]
        OWGUI.rubber(box)

        self.treeRules = QTreeWidget(self.mainArea)
        self.mainArea.layout().addWidget(self.treeRules)
        self.treeRules.setSelectionMode (QTreeWidget.ExtendedSelection)
        self.treeRules.setHeaderLabels(["Rules"] + [m[1] for m in self.measures])
        self.treeRules.setAllColumnsShowFocus ( 1)
        self.treeRules.setAlternatingRowColors(1) 
        self.showHideColumn()
        self.connect(self.treeRules,SIGNAL("itemSelectionChanged()"),self. selectionChanged)

        box = OWGUI.widgetBox(self.controlArea, "Info", addSpace = True)
        OWGUI.label(box, self, "Number of rules: %(nRules)s")
        OWGUI.label(box, self, "Selected rules: %(nSelectedRules)s")
        OWGUI.label(box, self, "Selected examples: %(nSelectedExamples)s")
        ibox = OWGUI.indentedBox(box)
        OWGUI.label(ibox, self, "... matching: %(nMatchingExamples)s")
        OWGUI.label(ibox, self, "... mismatching: %(nMismatchingExamples)s")

        box = OWGUI.widgetBox(self.controlArea, "Options", addSpace = True)
        OWGUI.spin(box, self, "treeDepth", label = "Tree depth", min = 0, max = 10, step = 1, callback = self.displayRules, callbackOnReturn = True)
        OWGUI.separator(box)
        OWGUI.checkBox(box, self, "showWholeRules", "Display whole rules", callback = self.setWholeRules)

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
        
        self.rules = None


    def setWholeRules(self, node = None):
        if not node:
            for i in range(self.treeRules.topLevelItemCount()):
                self.setWholeRules(self.treeRules.topLevelItem(i))
        else:
            t = getattr(node, self.showWholeRules and "wholeRule" or "remainder", "")
            if t:
                node.setText(0, t)
            for i in range(node.childCount()):
                self.setWholeRules(node.child(i))


    def showHideColumn(self):
        for i, cb in enumerate(self.cbMeasures):
            (cb.isChecked() and self.treeRules.showColumn or self.treeRules.hideColumn)(i+1)
        for i in range(1+len(self.measures)):
            self.treeRules.setColumnWidth(i, i and 50 or 300)


    def displayRules(self):
        self.treeRules.clear()
        self.buildLayer(None, self.rules, self.treeDepth)


    def buildLayer(self, parent, rulesLC, n, startOfRule = ""):
        if n==0:
            self.printRules(parent, rulesLC, startOfRule)
        elif n>0:
            children = set()
            for rule in rulesLC:
                children.update(rule[0])
            for childText in children:
                item = QTreeWidgetItem(parent or self.treeRules, [childText])
                rules2 = []
                newStartOfRule = startOfRule + childText + "  " 
                for rule in rulesLC:
                    if childText in rule[0]:
                        if len(rule[0])==1:
                            self.printRules(item, [(set(),) + rule[1:]], newStartOfRule)
                        else:
                            rules2.append((rule[0].difference([childText]),) + rule[1:])
                self.buildLayer(item, rules2, n-1, newStartOfRule)


    def printRules(self, parent, rulesLC, startOfRule):
        for rule in rulesLC:
            restOfRule = "  ".join(rule[0])
            remainder = restOfRule+"  ->   "+rule[1]
            wholeRule = startOfRule + remainder
            item = QTreeWidgetItem(parent,   [self.showWholeRules and wholeRule or remainder]
                                           + [str('  %.3f  ' % v) for v in rule[2]])
            item.remainder = remainder
            item.wholeRule = wholeRule
            item.ruleId = rule[3]

    def gatherRules(self, node, ids):
        if hasattr(node, "ruleId"):
            ids.add(node.ruleId)
        for i in range(node.childCount()):
            self.gatherRules(node.child(i), ids)

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
        for node in self.treeRules.selectedItems():
            self.gatherRules(node, ids)
        self.selectedRules = orange.AssociationRules([self.origRules[id] for id in ids])
        leftids = set()
        bothids = set()
        for rule in self.selectedRules:
            leftids.update(rule.matchLeft)
            bothids.update(rule.matchBoth)
        self.leftids = list(leftids)
        self.bothids = list(bothids)
        self.misids = list(leftids - bothids)
        self.setInfo()
        self.sendIf()

    def sendIf(self):
        if self.autoSend:
            self.sendData()
        else:
            self.dataChanged = True
            
    def sendData(self):
        self.dataChanged = False
        self.send("Association Rules", self.selectedRules)
        if self.selectedRules:
            examples = self.selectedRules[0].examples

            coveredExamples = examples.getitemsref(self.leftids)
            matchingExamples = examples.getitemsref(self.bothids)
            mismatchingExamples = examples.getitemsref(self.misids)
            
            if self.purgeAttributes or self.purgeClasses:
                coveredExamples = OWTools.domainPurger(coveredExamples, self.purgeClasses)
                matchingExamples = OWTools.domainPurger(matchingExamples, self.purgeClasses)
                mismatchingExamples = OWTools.domainPurger(mismatchingExamples, self.purgeClasses)
        else:
            coveredExamples = matchingExamples = mismatchingExamples = None

        self.send("Covered Examples", coveredExamples)
        self.send("Matching Examples", matchingExamples)
        self.send("Mismatching Examples", mismatchingExamples)
        

    def setInfo(self):
        if self.origRules:
            self.nRules = len(self.rules)
            if self.selectedRules:
                self.nSelectedRules = len(self.selectedRules)
                self.nSelectedExamples = len(self.leftids)
                self.nMatchingExamples = len(self.bothids)
                self.nMismatchingExamples = len(self.misids)
            else:
                self.nSelectedRules = self.nSelectedExamples = self.nMatchingExamples = self.nMismatchingExamples = ""
        else:
            self.nRules = self.nSelectedRules = self.nSelectedExamples = self.nMatchingExamples = self.nMismatchingExamples = ""
            

    def arules(self,arules):
        self.origRules = arules
        self.rules=[(set(str(val.variable.name) + (val.varType and "=" + str(val) or "") for val in rule.left if not val.isSpecial()),
                     "  ".join(str(val.variable.name) + (val.varType and "=" + str(val) or "") for val in rule.right if not val.isSpecial()),
                     (rule.support, rule.confidence, rule.lift, rule.leverage, rule.strength, rule.coverage),
                     id
                     )
                    for id, rule in enumerate(arules or [])]
        self.displayRules()
        self.selectionChanged()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesTreeViewer()

    #dataset = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    dataset = orange.ExampleTable(r"E:\Development\Orange Datasets\UCI\zoo.tab")
    rules=orange.AssociationRulesInducer(dataset, support = 0.5, maxItemSets=10000)
    ow.arules(rules)

    ow.show()
    a.exec_()
    ow.saveSettings()

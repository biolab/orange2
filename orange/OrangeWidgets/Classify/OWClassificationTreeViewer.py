"""
<name>Classification Tree Viewer</name>
<description>Classification tree viewer (hierarchical list view).</description>
<icon>icons/ClassificationTreeViewer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>2100</priority>
"""
from OWWidget import *
from orngTree import TreeLearner
import OWGUI

import orngTree

class ColumnCallback:
    def __init__(self, widget, attribute, f = None):
        self.widget = widget
        self.attribute = attribute
        self.f = f
        widget.callbackDeposit.append(self)

    def __call__(self, value):
        setattr(self.widget, self.attribute, self.f and self.f(value) or value)
        self.widget.setTreeView(1)

def checkColumn(widget, master, text, value):
    wa = QCheckBox(text, widget)
    widget.layout().addWidget(wa)
    wa.setChecked(getattr(master, value))
    master.connect(wa, SIGNAL("toggled(bool)"), ColumnCallback(master, value))
    return wa

class OWClassificationTreeViewer(OWWidget):
    settingsList = ["maj", "pmaj", "ptarget", "inst", "dist", "adist", "expslider", "sliderValue"]
    contextHandlers = {"": DomainContextHandler("", ["targetClass"], matchValues=1)}

    def __init__(self, parent=None, signalManager = None, name='Classification Tree Viewer'):
        OWWidget.__init__(self, parent, signalManager, name)

        self.dataLabels = (('Majority class', 'Class'), 
                  ('Probability of majority class', 'P(Class)'), 
                  ('Probability of target class', 'P(Target)'), 
                  ('Number of instances', '# Inst'), 
                  ('Relative distribution', 'Distribution (rel)'), 
                  ('Absolute distribution', 'Distribution (abs)'))

        self.callbackDeposit = []

        self.inputs = [("Classification Tree", orange.TreeClassifier, self.setClassificationTree)]
        self.outputs = [("Examples", ExampleTable)]

        # Settings
        for s in self.settingsList[:6]:
            setattr(self, s, 1)
        self.expslider = 5
        self.targetClass = 0
        self.loadSettings()

        self.tree = None
        self.sliderValue = 5
        self.precision = 3
        self.precFrmt = "%%2.%if" % self.precision

        # GUI
        # parameters

        self.dBox = OWGUI.widgetBox(self.controlArea, 'Displayed information')
        for i in range(len(self.dataLabels)):
            checkColumn(self.dBox, self, self.dataLabels[i][0], self.settingsList[i])

        OWGUI.separator(self.controlArea)
                
        self.slider = OWGUI.hSlider(self.controlArea, self, "sliderValue", box = 'Expand/shrink to level', minValue = 1, maxValue = 9, step = 1, callback = self.sliderChanged)

        OWGUI.separator(self.controlArea)
        self.targetCombo=OWGUI.comboBox(self.controlArea, self, "targetClass", items=[], box="Target class", callback=self.setTarget)

        self.infBox = OWGUI.widgetBox(self.controlArea, 'Tree size')
        self.infoa = OWGUI.widgetLabel(self.infBox, 'No tree.')
        self.infob = OWGUI.widgetLabel(self.infBox, ' ')

        OWGUI.rubber(self.controlArea)

        # list view
        self.splitter = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitter)

        self.v = QTreeWidget(self.splitter)
        self.splitter.addWidget(self.v)
        self.v.setAllColumnsShowFocus(1)
        self.v.setHeaderLabels(['Classification Tree'] + [label[1] for label in self.dataLabels])
        self.v.setColumnWidth(0, 250)

        # rule
        self.rule = QTextEdit(self.splitter)
        self.splitter.addWidget(self.rule)
        self.rule.setReadOnly(1)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)

        self.resize(800,400)

        self.resize(830, 400)

    def getTreeItemSibling(self, item):
            parent = item.parent()
            if not parent:
                parent = self.v.invisibleRootItem()
            ind = parent.indexOfChild(item)
            return parent.child(ind+1)

    # main part:

    def setTreeView(self, updateonly = 0):
        f = self.precFrmt

        def addNode(node, parent, desc, anew):
            return li

        def walkupdate(listviewitem):
            node = self.nodeClassDict[listviewitem]
            if not node: return
            ncl = node.nodeClassifier
            dist = node.distribution
            a = dist.abs
            if a < 1e-20:
                a = 1
            try:
                p_majclass = f % float(dist[int(ncl.defaultVal)]/a)
            except:
                p_majclass = "N/A"
            try:
                p_tarclass = f % float(dist[self.targetClass]/a)
            except:
                p_tarclass = "N/A"
            
            colf = (str(ncl.defaultValue), 
                    p_majclass, 
                    p_tarclass, 
                    "%d" % dist.cases, 
                    len(dist) and reduce(lambda x, y: x+':'+y, [self.precFrmt % (x/a) for x in dist]) or "N/A", 
                    len(dist) and reduce(lambda x, y: x+':'+y, ["%d" % int(x) for x in dist]) or "N/A"
                   )

            col = 1
            for j in range(6):
                if getattr(self, self.settingsList[j]):
                    listviewitem.setText(col, colf[j])
                    col += 1

            for i in range(listviewitem.childCount()):
                walkupdate(listviewitem.child(i))

        def walkcreate(node, parent):
            if not node: return
            if node.branchSelector:
                for i in range(len(node.branches)):
                    if node.branches[i]:
                        bd = node.branchDescriptions[i]
                        if not bd[0] in ["<", ">"]:
                            bd = node.branchSelector.classVar.name + " = " + bd
                        else:
                            bd = node.branchSelector.classVar.name + " " + bd
                        li = QTreeWidgetItem(parent, [bd])
                        li.setExpanded(1)
                        self.nodeClassDict[li] = node.branches[i]
                        walkcreate(node.branches[i], li)

        headerItemStrings = []
        for i in range(len(self.dataLabels)):
            if getattr(self, self.settingsList[i]):
                headerItemStrings.append(self.dataLabels[i][1])
        self.v.setHeaderLabels(["Classification Tree"] + headerItemStrings)
        self.v.setColumnCount(len(headerItemStrings)+1)
        self.v.setRootIsDecorated(1)
        self.v.header().setResizeMode(0, QHeaderView.Interactive)
        for i in range(len(headerItemStrings)):
            self.v.header().setResizeMode(1+i, QHeaderView.ResizeToContents)

        if not updateonly:
            self.v.clear()
            self.nodeClassDict = {}
            li = QTreeWidgetItem(self.v, ["<root>"])
            li.setExpanded(1)
            if self.tree:
                self.nodeClassDict[li] = self.tree.tree
                walkcreate(self.tree.tree, li)
            self.rule.setText("")
        if self.tree:
            walkupdate(self.v.invisibleRootItem().child(0))
        self.v.show()

    # slots: handle input signals

    def setClassificationTree(self, tree):
        self.closeContext()
        if tree and (not tree.classVar or tree.classVar.varType != orange.VarTypes.Discrete):
            self.error("This viewer only shows trees with discrete classes.\nThere is another viewer for regression trees")
            self.tree = None
        else:
            self.error()
            self.tree = tree

        self.setTreeView()
        self.sliderChanged()

        self.targetCombo.clear()
        if tree:
            self.infoa.setText('Number of nodes: %i' % orngTree.countNodes(tree))
            self.infob.setText('Number of leaves: %i' % orngTree.countLeaves(tree))
            self.targetCombo.addItems([name for name in tree.tree.examples.domain.classVar.values])
            self.targetClass = 0
            self.openContext("", tree.domain)
        else:
            self.infoa.setText('No tree on input.')
            self.infob.setText('')
            self.openContext("", None)

    def setTarget(self):
        def updatetarget(listviewitem):
            dist = self.nodeClassDict[listviewitem].distribution
            listviewitem.setText(targetindex, f % (dist[self.targetClass]/max(1, dist.abs)))

            child = listviewitem.firstChild()
            while child:
                updatetarget(child)
                child = self.getTreeItemSibling(child)

        if self.ptarget:
            targetindex = 1
            for st in range(5):
                if self.settingsList[st] == "ptarget":
                    break
                if getattr(self, self.settingsList[st]):
                    targetindex += 1

            f = self.precFrmt
            updatetarget(self.v.firstChild())

    def expandTree(self, lev):
        def expandTree0(listviewitem, lev):
            if not listviewitem:
                return
            if not lev:
                listviewitem.setExpanded(0)
            else:
                listviewitem.setExpanded(1)
                for i in range(listviewitem.childCount()):
                    child = listviewitem.child(i)
                    expandTree0(child, lev-1)

        expandTree0(self.v.invisibleRootItem().child(0), lev)

    # signal processing
    
    def viewSelectionChanged(self, item):
        """handles click on the tree"""
        self.handleSelectionChanged(item)
        if self.tree:
            data = self.nodeClassDict[item].examples
            self.send("Examples", data)

            tx = ""
            f = 1
            nodeclsfr = self.nodeClassDict[item].nodeClassifier
            while item and item.parent():
                if f:
                    tx = str(item.text(0))
                    f = 0
                else:
                    tx = str(item.text(0)) + " AND\n    "+tx

                item = item.parent()

            classLabel = str(nodeclsfr.defaultValue)
            className = str(nodeclsfr.classVar.name)
            if tx:
                self.rule.setText("IF %(tx)s\nTHEN %(className)s = %(classLabel)s" % vars())
            else:
                self.rule.setText("%(className)s = %(classLabel)s" % vars())
        else:
            self.send("Examples", None)
            self.rule.setText("")

    def handleSelectionChanged(self, item):
        pass

    def sliderChanged(self):
        self.expandTree(self.sliderValue)

##############################################################################
# Test the widget, run from DOS prompt
# > python OWDataTable.py)
# Make sure that a sample data set (adult_sample.tab) is in the directory

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWClassificationTreeViewer()
    #a.setMainWidget(ow)

    data = orange.ExampleTable(r'../../doc/datasets/adult_sample')

    tree = orange.TreeLearner(data, storeExamples = 1)
    ow.setClassificationTree(tree)
    a.exec_()
    ow.saveSettings()

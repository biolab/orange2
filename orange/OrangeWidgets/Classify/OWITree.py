"""
<name>Interactive Tree Builder</name>
<description>Interactive Tree Builder</description>
<icon>icons/ITree.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>50</priority>
"""
from OWWidget import *
from OWFile import *
from OWClassificationTreeViewer import *
import OWGUI, sys, orngTree

class FixedTreeLearner(orange.Learner):
    def __init__(self, classifier, name):
        self.classifier = classifier
        self.name = name

    def __call__(self, *d):
        return self.classifier

class OWITree(OWClassificationTreeViewer):
    settingsList = OWClassificationTreeViewer.settingsList

    def __init__(self,parent = None, signalManager = None):
        OWClassificationTreeViewer.__init__(self, parent, signalManager, 'I&nteractive Tree Builder')
        self.inputs = [("Examples", ExampleTable, self.setData), ("Tree Learner", orange.Learner, self.setLearner)]
        self.outputs = [("Examples", ExampleTable), ("Classifier", orange.TreeClassifier), ("Tree Learner", orange.Learner)]

        self.attridx = 0
        self.cutoffPoint = 0.0
        self.loadSettings()

        self.data = None
        self.treeLearner = None
        self.tree = None

        OWGUI.separator(self.space, height=40)
        box = OWGUI.widgetBox(self.space, "Split selection")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
#        OWGUI.widgetLabel(box, "Split By:")
        self.attrsCombo = OWGUI.comboBox(box, self, 'attridx', orientation="horizontal", callback=self.cbAttributeSelected)
        self.cutoffEdit = OWGUI.lineEdit(box, self, 'cutoffPoint', label = 'Cut off point: ', orientation='horizontal', validator=QDoubleValidator(self))
        OWGUI.button(box, self, "Split", callback=self.btnSplitClicked)

        OWGUI.separator(self.space)
        box = OWGUI.widgetBox(self.space, "Modify Tree")
        box.setSizePolicy(QSizePolicy(QSizePolicy.Minimum , QSizePolicy.Fixed ))
        self.btnPrune = OWGUI.button(box, self, "Cut", callback = self.btnPruneClicked)
        self.btnBuild = OWGUI.button(box, self, "Build", callback = self.btnBuildClicked)

        b = QVBox(self.controlArea)
        self.activateLoadedSettings()
        self.space.updateGeometry()

    def cbAttributeSelected(self):
        val = ""
        if self.data:
            attr = self.data.domain[self.attridx]
            if attr.varType == orange.VarTypes.Continuous:
                val = str(orange.Value(attr, self.basstat[attr].avg))
        self.cutoffEdit.setDisabled(not val)
        self.cutoffEdit.setText(val)

    def activateLoadedSettings(self):
        self.cbAttributeSelected()

    def updateTree(self):
        self.setTreeView()
        self.learner = FixedTreeLearner(self.tree, self.title)
#        self.send("Examples", self.tree)
        self.send("Classifier", self.tree)
        self.send("Tree Learner", self.learner)

    def newTreeNode(self, data):
        node = orange.TreeNode()
        node.examples = data
        node.contingency = orange.DomainContingency(data)
        node.distribution = node.contingency.classes
        nodeLearner = self.treeLearner and getattr(self.treeLearner, "nodeLearner", None) or orange.MajorityLearner()
        node.nodeClassifier = nodeLearner(data)
        return node

    def cutNode(self, node):
        if not node:
            return
        node.branchDescriptions = node.branchSelector = node.branchSizes = node.branches = None

    def findCurrentNode(self, exhaustively=0):
        sitem = self.v.selectedItem()
        if not sitem and (1 or exhaustively):
            sitem = self.v.currentItem() or (self.v.childCount() == 1 and self.v.firstChild())
            if sitem.childCount():
                return
        return sitem and self.nodeClassDict[sitem]

    def btnSplitClicked(self):
        node = self.findCurrentNode(1)
        if not node:
            return

        attr = self.data.domain[self.attridx]
        if attr.varType == orange.VarTypes.Continuous:
            cutstr = str(self.cutoffEdit.text())
            if not cutstr:
                return
            cutoff = float(cutstr)

            node.branchSelector = orange.ClassifierFromVarFD(position=self.attridx, domain=self.data.domain, classVar=attr)
            node.branchSelector.transformer = orange.ThresholdDiscretizer(threshold = cutoff)
            node.branchDescriptions = ["<%5.3f" % cutoff, ">=%5.3f" % cutoff]

            cutvar = orange.EnumVariable(node.examples.domain[self.attridx].name, values = node.branchDescriptions)
            cutvar.getValueFrom = node.branchSelector
            node.branchSizes = orange.Distribution(cutvar, node.examples)
            node.branchSelector.classVar = cutvar

        else:
            node.branchSelector = orange.ClassifierFromVarFD(position=self.attridx, domain=self.data.domain, classVar=attr)
            node.branchDescriptions=node.branchSelector.classVar.values
            node.branchSizes = orange.Distribution(attr, node.examples)

        splitter = self.treeLearner and getattr(self.treeLearner, "splitter", None) or orange.TreeExampleSplitter_IgnoreUnknowns()
        node.branches = [subset and self.newTreeNode(subset) or None   for subset in splitter(node, node.examples)[0]]
        self.updateTree()

    def btnPruneClicked(self):
        self.cutNode(node = self.findCurrentNode())
        self.updateTree()

    def btnBuildClicked(self):
        node = self.findCurrentNode()
        if not node:
            return

        newtree = (self.treeLearner or orngTree.TreeLearner(storeExamples = 1))(node.examples)
        if not hasattr(newtree, "tree"):
            QMessageBox.critical( None, "Invalid Learner", "The learner on the input built a classifier which is not a tree.", QMessageBox.Ok)

        for k, v in newtree.tree.__dict__.items():
            node.setattr(k, v)
        self.updateTree()

    def setData(self, data):
        if self.data and data and data.domain.checksum() == self.data.domain.checksum():
            return

        self.attrsCombo.clear()

        self.data = self.isDataWithClass(data, orange.VarTypes.Discrete) and data or None

        if self.data:
            for attr in data.domain.attributes:
                self.attrsCombo.insertItem(attr.name)
            self.basstat = orange.DomainBasicAttrStat(data)
#            self.attrsCombo.adjustSize()
            self.attridx = 0
            self.cbAttributeSelected()
            self.tree = orange.TreeClassifier(domain = data.domain)
            self.tree.descender = orange.TreeDescender_UnknownMergeAsBranchSizes()
            self.tree.tree = self.newTreeNode(self.data)
        else:
            self.tree = None
            self.send("Classifier", self.tree)
            self.send("Tree Learner", self.learner)

        self.send("Examples", None)
        self.updateTree()
        self.v.setSelected(self.v.firstChild(), TRUE)

    def setLearner(self, learner):
        self.treeLearner = learner


if __name__ == "__main__":
    a=QApplication(sys.argv)
    owi=OWITree()

#    d = orange.ExampleTable('d:\\ai\\orange\\test\\iris')
    d = orange.ExampleTable('d:\\ai\\orange\\test\\crush')
    owi.setData(d)

    a.setMainWidget(owi)
    owi.show()
    a.exec_loop()

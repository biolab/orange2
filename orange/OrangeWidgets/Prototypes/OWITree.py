"""
<name>Interactive Tree Builder</name>
<description>Interactive Tree Builder</description>
<icon>icons/Unknown.png</icon>
<priority>20</priority>
"""
from OWWidget import *
from OWFile import *
from OWIRankOptions import *
from OWClassificationTreeViewer import *
import OWGUI, sys, orngTree

class OWITree(OWClassificationTreeViewer):
    settingsList = OWClassificationTreeViewer.settingsList + ["discretizationMethod"]
    
    def __init__(self,parent = None, signalManager = None):
        OWClassificationTreeViewer.__init__(self, parent, signalManager, 'I&nteractive Tree Builder')
        self.inputs = [("Examples", ExampleTable, self.cdata, 1), ("Tree Learner", orange.Learner, self.learner, 1)]
        self.outputs = [("Classified Examples", ExampleTableWithClass), ("Classifier", orange.TreeClassifier)]

        self.discretizationMethod = 0
        self.attridx = 0
        self.cutoffPoint = 0.0
        self.loadSettings()

        self.data = None
        self.treeLearner = None
        self.tree = None

        box = OWGUI.widgetBox(self.space, "Interactive construction")
        sbox = OWGUI.widgetBox(box)
        self.attrsCombo = OWGUI.comboBox(sbox, self, 'attridx', label = "Split by", orientation="horizontal", callback=self.cbAttributeSelected)
        self.cutoffEdit = OWGUI.lineEdit(sbox, self, 'cutoffPoint', label = 'Cut off point: ', orientation='horizontal', validator=QDoubleValidator(self))
        OWGUI.button(sbox, self, "Split", callback=self.btnSplitClicked)
        self.btnPrune = OWGUI.button(box, self, "Cut", callback = self.btnPruneClicked)
        self.btnBuild = OWGUI.button(box, self, "Build", callback = self.btnBuildClicked)

        self.activateLoadedSettings()

    def cbAttributeSelected(self):
        if self.data:
            attr = self.data.domain[self.attridx]
            self.cutoffEdit.setDisabled(attr.varType != orange.VarTypes.Continuous)
        else:
            self.cutoffEdit.setDisabled(True)

    def activateLoadedSettings(self):
        self.cbAttributeSelected()

    def updateTree(self):
        self.setTreeView()
        self.send("ctree", self.tree)
        self.send("classifier", self.tree)

    def newTreeNode(self, data):
        node = orange.TreeNode()
        node.examples = data
        node.contingency = orange.DomainContingency(data)
        node.distribution = node.contingency.classes
        nodeLearner = self.treeLearner and self.treeLearner.nodeLearner or orange.MajorityLearner()
        node.nodeClassifier = nodeLearner(data)
        return node

    def cutNode(self, node):
        node.branchDescriptions = node.branchSelector = node.branchSizes = node.branches = None

    def btnSplitClicked(self):
        sitem = self.v.selectedItem()
        if not sitem:
            return
        node = self.nodeClassDict[sitem]
        
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

        splitter = self.treeLearner and self.treeLearner.splitter or orange.TreeExampleSplitter_IgnoreUnknowns()
        node.branches = [subset and self.newTreeNode(subset) or None   for subset in splitter(node, node.examples)[0]]
        self.updateTree()

    def btnPruneClicked(self):
        self.cutNode(node = self.nodeClassDict[self.v.selectedItem()])
        self.updateTree()

    def btnBuildClicked(self):
        sitem = self.v.selectedItem()
        if not sitem:
            return
        node = self.nodeClassDict[sitem]

        newtree = (self.treeLearner or orngTree.TreeLearner(storeExamples = 1))(node.examples)
        for k, v in newtree.tree.__dict__.items():
            node.setattr(k, v)
        self.updateTree()

    def cdata(self, data):
        if self.data and data.domain == self.data.domain:
            return

        self.attrsCombo.clear()
        self.data = data
        if self.data:
            for attr in data.domain.attributes:
                self.attrsCombo.insertItem(attr.name)
            self.attrsCombo.adjustSize()
            self.attridx = 0
            self.cbAttributeSelected()
            self.tree = orange.TreeClassifier()
            self.tree.tree = self.newTreeNode(self.data)
        else:
            self.tree = None
            
        self.updateTree()

    def learner(self, learner):
        self.treeLearner = learner


if __name__ == "__main__":
    a=QApplication(sys.argv)
    owi=OWITree()

    d = orange.ExampleTable('d:\\ai\\orange\\test\\iris')
#    d = orange.ExampleTable('d:\\ai\\orange\\test\\crush')
    owi.cdata(d)

    a.setMainWidget(owi)
    owi.show()
    a.exec_loop()

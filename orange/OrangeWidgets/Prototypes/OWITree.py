"""
<name>Interactive Tree Builder</name>
<description>Interactive Tree Builder</description>
<category>Classification</category>
<icon>icons/Unknown.png</icon>
<priority>20</priority>
"""
from OWWidget import *
from OWFile import *
from OWIRankOptions import *
from OWClassificationTreeViewer import *
import OWGUI
import sys

#global Atribut
class var:
    lock=None
    wnode=None
    cont=None
    attr=None
    cut=None
    learner=None

class TL:

    def divide(self, node, attr, cutoff):
        pos = [a.name for a in node.examples.domain].index(attr)

        node.branchSelector = orange.ClassifierFromVarFD(position=pos, domain=node.examples.domain, classVar=node.examples.domain[pos])
        dist = orange.DomainDistributions(node.examples)

        if cutoff<>None:
            node.branchSelector.transformer = orange.ThresholdDiscretizer(threshold = cutoff)
            values = ["<%5.3f" % cutoff, ">=%5.3f" % cutoff]
            cutvar = orange.EnumVariable(node.examples.domain[pos].name, values = values)
            cutvar.getValueFrom = node.branchSelector
            node.branchSelector.classVar = cutvar
            
            node.branchSizes = orange.Distribution(cutvar, node.examples)
            node.branchDescriptions = values
            
        else :
            node.branchSelector.classVar = node.examples.domain[pos]
            node.branchSizes = dist[pos]
            #cont=node.contingency[pos]
            node.branchDescriptions=node.branchSelector.classVar.values

        if not node.branchSelector:
            return node

        subsets, w = self.exampleSplitter(node, node.examples)
        node.branches = []
        for subset in subsets:
            if len(subset):
                node.branches.append(self.new(subset))
            else:
                node.branches.append(None)

    def new(self,data):
        node = orange.TreeNode()
        node.examples=data
        node.contingency = orange.DomainContingency(data)
        node.distribution=node.contingency.classes
        node.nodeClassifier = self.nodeLearner(data)
        return node

    def __call__(self, data):
        classifier = orange.TreeClassifier()
        classifier.tree = self.divide(data)
        classifier.domain = data.domain
        classifier.descender = orange.TreeDescender_UnknownMergeAsSelector()
        classifier.FD = orange.ClassifierFromVarFD()
        classifier.dist=orange.DomainDistributions()
        return classifier

#--------------novo

class OWITree(OWClassificationTreeViewer):
    def __init__(self,parent = None):
        OWClassificationTreeViewer.__init__(self, parent,
                          'I&nteractive Tree Builder')


        #set default settings
        self.Precision=3
        self.ReliefK=11
        self.ReliefN=20
        self.discretizationMethod = 0
        self.attrMeasure = 0

        self.inputs = [("Examples", ExampleTable, self.cdata, 1), ("Tree Learner", orange.TreeLearner, self.learner, 1)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Learner", orange.Learner), ("Classifier", orange.TreeClassifier)]

        self.settingsList.append(["ReliefK", "ReliefN", "discretizationMethod", "attrMeasure"])
        self.loadSettings()

        self.data = None

        box = OWGUI.widgetBox(self.space, 1)
        self.btnPrune = OWGUI.button(box, self, "Prune", callback = self.BtnPruneClicked)
        self.btnBuild = OWGUI.button(box, self, "Build", callback = self.BtnBuildClicked)

        self.bxMeasure = OWGUI.widgetBox(self.space, "Attributes")
        self.fillBxMeasure()

        self.var = var
        self.tree = None
        self.var.lock = 0
        self.oData = None
        self.tree = None
        self.iattrib = None

        #connect GUI controls of options in options dialog to settings

#        self.connect(self.v, SIGNAL("selectionChanged(QListViewItem *)"),self.showAttr)
#        self.connect(self.table, SIGNAL("doubleClicked(int,int,int,const QPoint&)"), self.tableDoubleClick)

    def fillBxMeasure(self):
        self.cmbMeasure = OWGUI.comboBox(self.bxMeasure, self, "attrMeasure", None, ["Information Gain", "Gain Ration", "Gini Index", "ReliefF"], "Measure for ranking the attributes", self.measureChanged, 0)
        if self.attrMeasure == 3:
            self.sldReliefN = OWGUI.qwtHSlider(self.bxMeasure, self, "ReliefN", None, "Reference points ", None, minValue = 20, maxValue = 1000, step = 10, precision = 1, callback = self.measureChanged)
            self.sldReliefK = OWGUI.qwtHSlider(self.bxMeasure, self, "ReliefK", None, "Neighbours ", None, minValue = 1, maxValue = 50, step = 1, precision = 1, callback = self.measureChanged)
        else:
            OWGUI.radioButton(self.bxMeasure, self, "discretizationMethod", "Quartile discretization", box = 0, callback = self.measureChanged)
            OWGUI.radioButton(self.bxMeasure, self, "discretizationMethod", "Fayyad-Irani discretization", box = 0, callback = self.measureChanged)


#            self.cmbDiscretization = OWGUI.comboBox(self.bxMeasure, self, "discretizationMethod", None, ["Quartiles", "Fayyad-Irani"], "Discretization method", self.measureChanged, 0)

    def tableDoubleClick(self, row, col, neki, point):
        self.CmbAttribClick(self.table.text(row, 0))
        self.v.setFocus()
        
    def measureChanged(self, value):
        pass

    def activateLoadedSettings(self):
        self.options.precisionSlider.setValue(self.Precision)
        self.options.kSlider.setValue(self.ReliefK)
        self.options.nSlider.setValue(self.ReliefN)
        self.options.discretization.setCurrentItem(self.options.discretizationStrings.index(self.DiscretizationMethod))
        self.options.displayReliefF.setChecked(self.DisplayReliefF)
        self.options.displayInfoGain.setChecked(self.DisplayInfoGain)
        self.options.displayGainRatio.setChecked(self.DisplayGainRatio)
        self.options.displayGini.setChecked(self.DisplayGini)
        self.options.displayAttribVals.setChecked(self.displayAttributeVals)
        self.recalculate()

    def displayReliefF(self,checked):
        self.DisplayReliefF=checked
        if checked:
            self.table.showColumn(4)
        else:
            self.table.hideColumn(4)
    
    def displayInfoGain(self,checked):
        self.DisplayInfoGain=checked
        if checked:
            self.table.showColumn(5)
        else:
            self.table.hideColumn(5)
    
    def displayGainRatio(self,checked):
        self.DisplayGainRatio=checked
        if checked:
            self.table.showColumn(6)
        else:
            self.table.hideColumn(6)
    
    def displayGini(self,checked):
        self.DisplayGini=checked
        if checked:
            self.table.showColumn(7)
        else:
            self.table.hideColumn(7)

    def displayAttribVals(self,checked):
        self.displayAttributeVals=checked
        if checked:
            self.table.showColumn(2)
        else:
            self.table.hideColumn(2)

            
    def setPrecision(self,num):
        self.Precision=num
        self.recalculate()            
            
    def setK(self,k):
        self.ReliefK=k
        self.recalculate()
    
    def setN(self,n):
        self.ReliefN=n
        self.recalculate()
        
    def setDM(self,dm):
        self.DiscretizationMethod=self.options.discretizationStrings[dm]
        self.recalculate()

    def sort(self,col):
        "Sort the column col"
#        print col
        if col == 2:
            return
        if col==self.sortby:
            samecolumn=TRUE
        else:
            samecolumn=FALSE
        if not samecolumn:
            if col<2: 
                #table.sortColumn sort in alphabetical order, so it can only by used for the attribute and C/D columns
                self.table.sortColumn(col,TRUE,TRUE)
            else:            
            #we must do our own sorting
            #should take advantage of QTable's swapRows() method
            #so how about a simple bubblesort?
                for i in range(0,self.table.numRows()-1):
                    x=float(str(self.table.text(i,col)))
                    for j in range(i+1,self.table.numRows()):
                        y=float(str(self.table.text(j,col)))
#                        print x,y, cmp(x,y)
                        if x>y:
#                            print "swap!"
                            self.swapRows(i,j)
                            x=y
        else:   #reversing selection is faster
            for i in range(0,self.table.numRows()/2):
                self.swapRows(i,self.table.numRows()-i-1)
        self.sortby=col
        self.repaint()

    def showAttr(self, item):
        self.var.cut=None
        self.var=var
        self.var.attr=""
        self.var.wnode=self.nodeClassDict[item]
        self.data = self.var.wnode.examples
        self.recalculate()

    def chooseAttribute(self,attrib):
        self.var.cut=None
        self.var.attr=str(attrib)

    def btnApplyClicked(self):
        var = self.tree.tree.examples.domain[self.var.attr]
        
        if (self.var.attr):
            if var.varType == orange.VarTypes.Continuous:
                try:
                    (string,ok) = QInputDialog.getText("Continous attribute", "Enter a value")
                    if not ok:
                        return
                    self.var.cut=float(str(string))
                except ValueError:
                    self.msg.warning(self, "Error", "Bad continous value!", 0, 0, 0)
                    return
            else:
                self.var.cut = None

            self.tree.divide(self.var.wnode,self.var.attr,self.var.cut)

            self.setTreeView()
            self.send("ctree", self.tree)
            self.send("classifier", self.tree)



    def BtnPruneClicked(self):
        self.var.wnode.branches = []
        self.setTreeView()
        self.send("ctree", self.tree)
        self.send("classifier", self.tree)

    def BtnBuildClicked(self):
        if self.var.wnode==self.tree.tree:
            if self.var.learner<>None:
                tren=self.var.learner(self.var.wnode.examples)
            else:
                tren=orange.TreeLearner(self.var.wnode.examples,storeExamples=1)
            self.tree.tree=tren.tree
        else:
            self.Find(self.tree.tree,self.var.wnode)
        self.setTreeView()
        self.send("ctree", self.tree)
        self.send("classifier", self.tree)

    def Find(self,item,xx):
        if (item):
            if (item.branches):
                x=range(len(item.branches))
                for a in x:
                    if xx==item.branches[a]:
                        if self.var.learner<>None:
                            tren=self.var.learner(self.var.wnode.examples)                
                        else:
                            tren=orange.TreeLearner(self.var.wnode.examples,storeExamples=1)
                        item.branches[a]=tren.tree
                    else:
                        if item.branches:
                            self.Find(item.branches[a],xx)


    def CmbAttribClick(self, text):
        self.chooseAttribute(text)
        if self.v.selectedItem()==None: 
            if self.tree.tree.examples.domain[str(text)].varType==orange.VarTypes.Continuous:
                pass
            else:
                if self.nodeClassDict[self.v.selectedItem()].examples.domain[str(text)].varType==orange.VarTypes.Continuous:  
                    pass
 #       self.data = self.nodeClassDict[self.v.selectedItem()].examples        
#        self.recalculate()
        self.btnApplyClicked()        

    def cdata(self, data):
        if self.tree == None:
            self.init(data)
        else :
            if self.tree.tree.branchSelector<>None :
                name=self.tree.tree.branchSelector.classVar.name
                dat=self.tree.tree.examples
                self.tree.tree.examples=data.data
                x=range(len(data.data.domain))
                brSelector=0
                for a in x:
                    if data.data.domain[a]==dat.domain[str(name)]:
                        brSelector=1
                        self.check(self.tree.tree,dat,x)
                if brSelector==0:
                    self.init(data)
            else :
                self.init(data)

        self.var.cut=None
        self.var=var
        self.var.attr=""
        self.var.wnode=self.tree.tree
        self.data = data
        self.recalculate()
        self.setTreeView()


    def init(self, data):
        self.tree = TL()
        self.tree.stop = orange.TreeStopCriteria_common()
        self.tree.split = orange.TreeSplitConstructor_Attribute()
        self.tree.split.measure = orange.MeasureAttribute_gainRatio()
        self.tree.exampleSplitter = orange.TreeExampleSplitter_IgnoreUnknowns()
        self.tree.nodeLearner = orange.MajorityLearner()
        self.tree.tree=self.tree.new(data)

    def check(self,cnode,dat,x):
        name=cnode.branchSelector.classVar.name
        cutoff=None
        if cnode.examples.domain[str(name)].varType==orange.VarTypes.Continuous:
            cutoff=cnode.branchSelector.transformer.threshold
        br=cnode.branches
        brDes=cnode.branchDescriptions
        yy=range(len(br))
        self.tree.divide(cnode,name,cutoff)
        xx=range(len(cnode.branches))
        for i in xx:
            brName=0
            for j in yy:
                if ((cnode.branches[i]<>None) and (br[j]<>None) and (cnode.branchDescriptions[i]==brDes[j])):
                    brName=1
                    if (br[j].branchSelector<>None):
                        for a in x:
                            if cnode.examples.domain[a]==dat.domain[br[j].branchSelector.classVar.name]:
                                pos = [a.name for a in cnode.branches[i].examples.domain].index(br[j].branchSelector.classVar.name)
                                cnode.branches[i].branchSelector=orange.ClassifierFromVarFD(position=pos, domain=cnode.branches[i].examples.domain, classVar=cnode.branches[i].examples.domain[pos])
                                if cnode.examples.domain[br[j].branchSelector.classVar.name].varType==orange.VarTypes.Continuous: #OK
                                    cnode.branches[i].branchSelector.transformer = orange.ThresholdDiscretizer(threshold = br[j].branchSelector.transformer.threshold)
                                cnode.branches[i].branches=br[j].branches
                                cnode.branches[i].branchDescriptions=br[j].branchDescriptions
                                self.check(cnode.branches[i],dat,x)



    def learner(self, learner):
#        if type(learner)==orange.TreeLearner:
        self.var.learner=learner
#        else:
#            self.var.learner=None



    def recalculate(self):
        if self.data:
            pass

    def swapRows(self,i,j):
        for k in range (0, self.table.numCols()):
#            print "swapping ", self.table.text(i,k), "with ", self.table.text(j,k)
            temp=self.table.text(i,k)
            self.table.setText(i,k,self.table.text(j,k))
            self.table.setText(j,k,temp)

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owi=OWITree()

    d = orange.ExampleTable('crush')
    owi.cdata(d)

    a.setMainWidget(owi)
    owi.show()
    a.exec_loop()

#a=QApplication(sys.argv)
#owi=OWITree()
#a.setMainWidget(owi)
#owi.show()
#a.exec_loop()



"""
<name>Interactive Tree Induction</name>
<description>Interactive Tree Induction</description>
<category>Classification</category>
<icon>icons/Outcome.png</icon>
<priority>20</priority>
"""
from OWWidget import *
from OData import *
from OWFile import *
from OWIRankOptions import *
from OWClassificationTreeViewer import *
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

    def divide(self,node,attr,cutoff):
        pos = [a.name for a in node.examples.domain].index(attr)

        node.branchSelector=orange.ClassifierFromVarFD(position=pos, domain=node.examples.domain, classVar=node.examples.domain[pos])
        dist=orange.DomainDistributions(node.examples)

        if cutoff<>None:
            node.branchSelector.transformer = orange.ThresholdDiscretizer(threshold = cutoff)
            values = ["<%5.3f" % cutoff, ">=%5.3f" % cutoff] #TO SEM POPRAVU je bilo narobe oznaceno
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
        node.nodeClassifier = self.nodeLearner(data) # recimo da ne rabim
        return node

    def __call__(self, data):
        classifier = orange.TreeClassifier()
        classifier.tree = self.divide(data)
        classifier.domain = data.domain
        classifier.descender = orange.TreeDescender_UnknownMergeAsSelector()
        classifier.FD = orange.ClassifierFromVarFD() #to je novo
        classifier.dist=orange.DomainDistributions() #to je novo
        return classifier

#--------------novo

class OWITree(OWClassificationTreeViewer):
    def __init__(self,parent = None):
        OWClassificationTreeViewer.__init__(self, parent,
                          'I&nteracitve Tree build Widget')


        #set default settings
        self.Precision=3
        self.ReliefK=11
        self.ReliefN=20
        self.DiscretizationMethod="equal-frequency intervals"
        self.DisplayReliefF=1
        self.DisplayInfoGain=1
        self.DisplayGainRatio=1
        self.DisplayGini=1
        self.displayAttributeVals=1

        self.data = None

        self.options=OWIRankOptions()
        self.activateLoadedSettings()
        
        self.aBox =  QVGroupBox(self.controlArea)
        self.aBox.setTitle ('Interaction')

        self.BtnPrune = QPushButton(self.aBox,'BtnPrune')
        self.BtnPrune.setText("&Prune")

        self.BtnBuild = QPushButton(self.aBox,'BtnBuild')
        self.BtnBuild.setText("&Auto Build")

        self.settingsButton = QPushButton(self.aBox,'settingsButton')
        self.settingsButton.setText("&Rank Settings")

        self.table=QTable(self.mainArea)
        self.table.setMaximumHeight(150)
        self.table.setSelectionMode(QTable.NoSelection) #hell, if i set this, swapRows() doesn't work - weird. and if i do not set this, the users can edit the fields
        self.layout.add(self.table)
        self.table.setNumCols(8)
        self.table.setNumRows(0)
        self.est_names = ["ReliefF", "InfoGain", "GainRatio", "Gini"]
        self.topheader=self.table.horizontalHeader()
        self.topheader.setLabel(0,"Attribute")
        self.table.adjustColumn(0)
        self.topheader.setLabel(1,"C/D")
        self.table.adjustColumn(1)
        self.topheader.setLabel(2,"Attribute values")
        self.table.adjustColumn(2)
        self.topheader.setLabel(3,"#")
        self.table.adjustColumn(3)
        self.connect(self.topheader,SIGNAL("pressed(int)"),self.sort)
        self.sortby=-1
        for i in range(4):
            self.topheader.setLabel(4+i,self.est_names[i])  
            self.table.adjustColumn(i)     

        self.msg = QMessageBox()

        self.var = var
        self.tree = None
        self.var.lock = 0
        self.oData = None
        self.tree = None
        self.iattrib = None

        #inputs & outputs

        self.addInput("cdata")
        self.addInput("learner")
        
        self.addOutput("classifier")
        self.addOutput("ctree")
        self.addOutput("cdata")
        self.addOutput("data")

        #connect settingsbutton to show options
        self.connect(self.settingsButton,SIGNAL("clicked()"),self.options.show),

        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.displayReliefF,SIGNAL("stateChanged(int)"),self.displayReliefF) 
        self.connect(self.options.displayInfoGain,SIGNAL("stateChanged(int)"),self.displayInfoGain) 
        self.connect(self.options.displayGainRatio,SIGNAL("stateChanged(int)"),self.displayGainRatio) 
        self.connect(self.options.displayGini,SIGNAL("stateChanged(int)"),self.displayGini) 
        self.connect(self.options.displayAttribVals,SIGNAL("stateChanged(int)"),self.displayAttribVals) 
        self.connect(self.options.kSlider,SIGNAL("valueChanged(int)"),self.setK)
        self.connect(self.options.nSlider,SIGNAL("valueChanged(int)"),self.setN)
        self.connect(self.options.discretization,SIGNAL("activated(int)"),self.setDM)
        self.connect(self.options.precisionSlider,SIGNAL("valueChanged(int)"),self.setPrecision)

        self.connect(self.v, SIGNAL("selectionChanged(QListViewItem *)"),self.showAttr)
        self.connect(self.table, SIGNAL("doubleClicked(int,int,int,const QPoint&)"), self.tableDoubleClick)
        self.connect(self.BtnPrune, SIGNAL("clicked()"),self.BtnPruneClicked)
        self.connect(self.BtnBuild, SIGNAL("clicked()"),self.BtnBuildClicked)

    def tableDoubleClick(self, row, col, neki, point):
        self.CmbAttribClick(self.table.text(row, 0))
        self.v.setFocus()
        

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
        self.data = data.data
        self.recalculate()
        self.setTreeView()


    def init(self, data):
        self.tree = TL()
        self.tree.stop = orange.TreeStopCriteria_common()
        self.tree.split = orange.TreeSplitConstructor_Attribute()
        self.tree.split.measure = orange.MeasureAttribute_gainRatio()
        self.tree.exampleSplitter = orange.TreeExampleSplitter_IgnoreUnknowns()
        self.tree.nodeLearner = orange.MajorityLearner()
        self.tree.tree=self.tree.new(data.data)

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
        self.var.learner=learner



    def recalculate(self):
        if self.data==None:
            
            return
        self.table.setNumRows(len(self.data.domain.attributes))
        
        ###############################################################
        # construct a data set with only continuous attributes that
        # are discretized
        
        # there will be three types of discretization that user
        # can choose
        
        disMet=self.DiscretizationMethod
        # equal-frequency intervals
        if disMet==self.options.discretizationStrings[0]:
            discretizer = orange.EquiNDiscretization(numberOfIntervals=5)
        #entropy-based discretization
        elif disMet==self.options.discretizationStrings[1]:
            discretizer = orange.EntropyDiscretization()
        # equal-width intervals
        elif disMet==self.options.discretizationStrings[2]:
            discretizer = orange.EquiDistDiscretization(numberOfIntervals=5)      
        
        at = []
        for i in self.data.domain.attributes:
            if i.varType == 2: # a continuous variable?
                d=discretizer(i, self.data)
                at.append(d)
        at.append(self.data.domain.classVar)
        
        # construct a new, discretized, data set
        disc_data = self.data.select(orange.Domain(at))
        
        ###############################################################
        # estimate the relevancy of parameters
        # print them out together with attribute names,
        # attribute type and cardinality
        
        # notice that Relief is the only measure that can estimate
        # relevancy for both discrete and continuous variables
        # for other measure, we have to use the discretized data set
        # to estimate the relevancy of continuous attributes
        
        # for Relief
        # number of reference examples (m)
        # number of closest neighbors that are observed (k)
        
        relieff = orange.MeasureAttribute_relief(k=self.ReliefK, m=self.ReliefN)
        entr = orange.MeasureAttribute_info()
        gainr = orange.MeasureAttribute_gainRatio()
        gini = orange.MeasureAttribute_gini()
        
        prec="%%.%df" % self.Precision
        prec1="%%.%df" % 2
        
        estimators = (relieff, entr, gainr, gini)
        handle_cont = (1,0,0,0) # does estimator handle cont values?
        
        fmt='%-20s '+'%20s '*len(self.est_names)
        fm=' '*13+'%7.4f'
#        print fmt % tuple(['Attribute']+self.est_names)
        cont = 0
        j=-1
        for i in self.data.domain.attributes:
            j+=1
            cell=QTableItem(self.table,QTableItem.Never,i.name)
            cell.setText(i.name)
            self.table.setItem(j,0,cell)
#            self.table.setText(j,0,i.name)
            cell=QTableItem(self.table,QTableItem.Never,('D','C')[i.varType==2])
            cell.setText(('D','C')[i.varType==2])
            self.table.setItem(j,1,cell)
#            self.table.setText(j,1,('D','C')[i.varType==2])
            if i.varType == 1:
                avals = ""
                for ii in i:
                    avals = avals + str(ii) + ","
                avals
            else:
                bas = orange.DomainBasicAttrStat(self.data)
                avals = ""
                for bb in bas:
                    if bb and bb.variable.name == i.name:
                        avals = str(prec1 % bb.min) + " - " + str(prec1 % bb.max)
            
            cell=QTableItem(self.table,QTableItem.Never,avals)
            cell.setText(avals)
            self.table.setItem(j,2,cell)
            #self.table.setText(j,2, avals)
            if i.varType==1:
#                print '%2d' % (len(i.values)),
                cell=QTableItem(self.table,QTableItem.Never,prec % (len(i.values)))
                cell.setText(prec % (len(i.values)))
                self.table.setItem(j,3,cell)
                #self.table.setText(j,3,prec % (len(i.values)))
                k=-1
                for e in estimators:
                    k+=1
                    cell=QTableItem(self.table,QTableItem.Never,prec % (e(i,self.data)))
                    cell.setText(prec % (e(i,self.data)))
                    self.table.setItem(j,4+k,cell)
                    #self.table.setText(j,4+k,prec % (e(i,self.data)))
#                    print fm % e(i,data)[0],
            else:
                v = disc_data.domain.attributes[cont]
#                print '%2d' % (len(v.values)),
                cell=QTableItem(self.table,QTableItem.Never,prec % (len(v.values)))
                cell.setText(prec % (len(v.values)))
                self.table.setItem(j,3,cell)
                #self.table.setText(j,3,prec % (len(v.values)))
                k=-1
                for ii in range(len(estimators)):
                    k+=1
                    e = estimators[ii]
                    if handle_cont[ii]:
                        cell=QTableItem(self.table,QTableItem.Never,prec % (e(i,self.data)))
                        cell.setText(prec % (e(i,self.data)))
                        self.table.setItem(j,4+k,cell)
#                        self.table.setText(j,4+k,prec % (e(i,self.data)))
#                        print fm % e(i,data)[0],
                    else:
                        cell=QTableItem(self.table,QTableItem.Never,prec % (e(v,disc_data)))
                        cell.setText(prec % (e(v,disc_data)))
                        self.table.setItem(j,4+k,cell)
#                        self.table.setText(j,4+k,prec % (e(v,disc_data)))
#                        print fm % e(v,disc_data)[0],
                cont = cont + 1
#            print
        self.table.adjustColumn(0)
        for k in range(3,2+len(estimators)):
            self.table.adjustColumn(k)

    def swapRows(self,i,j):
        for k in range (0, self.table.numCols()):
#            print "swapping ", self.table.text(i,k), "with ", self.table.text(j,k)
            temp=self.table.text(i,k)
            self.table.setText(i,k,self.table.text(j,k))
            self.table.setText(j,k,temp)

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owi=OWITree()
    a.setMainWidget(owi)
    owi.show()
    a.exec_loop()

#a=QApplication(sys.argv)
#owi=OWITree()
#a.setMainWidget(owi)
#owi.show()
#a.exec_loop()



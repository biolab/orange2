"""<name>CN2 Rules Viewer</name>
<description>Displays the rules of the CN2 classfier</description>
<icon>CN2RulesViewer.png</icon>
<priority>2120</priority>
"""

import orange
import orngCN2
import OWGUI
import qt
from OWWidget import *
import sys

class OWCN2RulesViewer(OWWidget):
    settingsList=["RuleLen","RuleQ","Coverage","Commit"]
    callbackDeposit=[]

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, "CN2RulesViewer")
        
        self.inputs=[("CN2UnorderedClassifier", orngCN2.CN2UnorderedClassifier, self.data)]
        self.outputs=[("ExampleTable", ExampleTable)]
        self.RuleLen=1
        self.RuleQ=1
        self.Coverage=1
        self.Commit=0
        self.loadSettings()

        self.layout=QVBoxLayout(self.mainArea)
        self.ruleView=QListView(self.mainArea,"Rule View")
        self.ruleView.setSelectionMode(QListView.Extended)
        self.layout.addWidget(self.ruleView)
        self.connect(self.ruleView,SIGNAL("selectionChanged()"),self.select)
        box=OWGUI.widgetBox(self.controlArea,1)
        OWGUI.checkBox(box,self,"RuleLen","Rule length",callback=self.showRules)
        OWGUI.checkBox(box,self,"RuleQ","Rule quality",callback=self.showRules)
        OWGUI.checkBox(box,self,"Coverage","Covearge",callback=self.showRules)
        
        box=OWGUI.widgetBox(self.controlArea,1)
        OWGUI.checkBox(box,self,"Commit", "Commit on change")
        OWGUI.button(box,self,"&Commit",callback=self.commit)
        
        OWGUI.button(self.controlArea,self,"&Save rules to file",callback=self.saveRules)

        self.examples=None

    def data(self, classifier):
        #print classifier
        if classifier:
            self.ruleView.clear()
            self.classifier=classifier
            self.rules=classifier.rules
            self.showRules()
        else:
            self.rules=None
            self.ruleView.clear()
        self.examples=None
        self.commit()
    
    def showRules(self):
        for i in range(self.ruleView.columns()):
            self.ruleView.removeColumn(0)
        #attrNames=["Rule","Rule length","Rule quality","Coverage"]
        attrNames=["Rule"]
        if self.RuleLen:
            attrNames.append("Rule length")
        if self.RuleQ:
            attrNames.append("Rule quality")
        if self.Coverage:
            attrNames.append("Coverage")
            
        for a in attrNames:
            self.ruleView.addColumn(a)
        self.items=[]
        for r in self.rules:
            i=0
            item=QListViewItem(self.ruleView)
            if "Rule" in attrNames:
                item.setText(i,orngCN2.ruleToString(r))
                i+=1
            if "Rule length" in attrNames:
                item.setText(i,str(r.complexity))
                i+=1
            if "Rule quality" in attrNames:
                item.setText(i,str(r.quality))
                i+=1
            if "Coverage" in attrNames:
                item.setText(i,str(len(r.examples)/float(len(self.classifier.examples))))
                i+=1
            self.items.append((item,r))
            
        
    def select(self,):
        examples=[]
        source=self.classifier.examples
        for a in self.items:
            if a[0].isSelected():
                examples.extend(source.filterref(a[1].filter))
                a[1].filter.negate=1
                source=source.filterref(a[1].filter)
                a[1].filter.negate=0
        if not examples:
            self.examples=None
            self.commit()

            return
        self.examples=examples
        if self.Commit:
            self.commit()

    def commit(self):
        if self.examples:
            self.send("ExampleTable",orange.ExampleTable(self.examples))
        else:
            self.send("ExampleTable",None)
    
    def saveRules(self):
        fileName=str(QFileDialog.getSaveFileName("Rules.txt",".txt"))
        try:
            f=open(fileName,"w")
        except :
            return 
        for r in self.rules:
            f.write(orngCN2.ruleToString(r)+"\n")



if __name__=="__main__":
    ap=QApplication(sys.argv)
    w=OWCN2RulesViewer()
    ap.setMainWidget(w)
    data=orange.ExampleTable("../../doc/datasets/titanic.tab")
    l=orngCN2.CN2UnorderedLearner()
    l.ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS()
    w.data(l(data))
    w.data(l(data))
    w.show()
    ap.exec_loop()

"""
<name>Association Rules Tree Viewer</name>
<description>Association rules tree viewer.</description>
<icon>icons/AssociationRulesTreeViewer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>300</priority>
"""

from OWWidget import *
import OWGUI

import sys, re
from qt import *
from OWTools import *


class OWAssociationRulesTreeViewer(OWWidget):
    measures = [("Support",    "Supp", "support"),
                ("Confidence", "Conf", "confidence"),
                ("Lift",       "Lift", "lift"),
                ("Leverage",   "Lev",  "leverage"),
                ("Strength",   "Strg", "strength"),
                ("Coverage",   "Cov",  "coverage")]

    # only the last name can be taken for settings - the first two can be translated
    settingsList = ["treeDepth", "showWholeRules"] + ["show%s" % m[2] for m in measures]

    print settingsList
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Association rules viewer")

        self.inputs = [("Association Rules", orange.AssociationRules, self.arules)]
        self.outputs = []
        
        # Settings
        self.showWholeRules = 1
        self.treeDepth = 2
 
        self.showsupport = self.showconfidence = 1
        self.showlift = self.showleverage = self.showstrength = self.showcoverage = 0
        self.loadSettings()

        self.layout=QVBoxLayout(self.mainArea)
        self.treeRules = QListView(self.mainArea)       #the rules and their properties are printed into this QListView
#        self.treeRules.setMultiSelection (1)              #allow multiple selection
        self.treeRules.setAllColumnsShowFocus ( 1) 
        self.treeRules.addColumn("Rules")        #column0

        mbox = OWGUI.widgetBox(self.controlArea, "Shown measures")
        self.cbMeasures = []
        for long, short, attr in self.measures:
            self.cbMeasures.append(OWGUI.checkBox(mbox, self, "show"+attr, long, callback = self.showHideColumn))
            self.treeRules.addColumn(short, 40)
            
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Options")
        OWGUI.widgetLabel(box, "Tree depth")
        OWGUI.hSlider(box, self, "treeDepth", minValue = 0, maxValue = 10, step = 1, callback = self.displayRules)
        OWGUI.separator(box)
        OWGUI.checkBox(box, self, "showWholeRules", "Display whole rules", callback = self.setWholeRules)

        OWGUI.rubber(self.controlArea)
        
        self.layout.addWidget(self.treeRules)
        
        self.rules = None


    def setWholeRules(self):
        d = self.showWholeRules and 1 or 2
        for line in self.wrlist:
            line[0].setText(0,line[d])
        

    def showHideColumn(self):
        for i, cb in enumerate(self.cbMeasures):
            self.treeRules.setColumnWidth(i+1, cb.isChecked() and 40)
                
    def displayRules(self):
        """ Display rules as a tree. """
        self.treeRules.clear()
        self.wrlist = []
        if self.rules:
            self.rulesLC=[]
            for rule in self.rules:                 # local copy of rules [[antecedens1,antecedens2,...], [consequens, support,...]] (without measures)
                values = filter(lambda val: not val.isSpecial(), rule.left)

                antecedens = []
                for x in values:
                    if x.varType:
                        antecedens.append(str(x.variable.name) + "=" + str(x))
                    else:
                        antecedens.append(str(x.variable.name))
                                
                values = filter(lambda val: not val.isSpecial(), rule.right)

                kons=""
                for x in values:
                    if x.varType:
                        kons=kons + str(x.variable.name) + "=" + str(x) + "  "
                    else:
                        kons=kons + str(x.variable.name) + "  "
                
                self.rulesLC.append([antecedens, [kons, rule.support, rule.confidence, rule.lift, rule.leverage, rule.strength, rule.coverage]])
            
            self.updateTree()
            self.removeSingleGrouping()        # if there is only 1 rule behind a +, the rule is
            self.setWholeRules()
            self.showHideColumn()
            self.item0.setOpen(1)                        # display the rules


    def updateTree(self):
        self.item0 = QListViewItem(self.treeRules,"")        #the first row is different
        self.buildLayer(self.item0, self.rulesLC, self.treeDepth)     # recursively builds as many layers as are in the.................

        
    def buildLayer(self, parent, rulesLC, n):
        if n==0:
           self.printRules(parent, rulesLC)
        elif n>0:
            children = []
            for rule in rulesLC:                                 # for every rule
                for a in rule[0]:                                # for every antecedens 
                    if a not in children:                        # if it is not in the list of children, add it
                        children.append(a)
            for childText in children:                           # for every entry in the list of children
                item=QListViewItem(parent,childText)             # add a branch with the text
                rules2=[]
                for rule in rulesLC:                             # find rules that go in this branch
                    if childText in rule[0]:
                        if len(rule[0])==1:
                            self.printRules(item, [[[],rule[1]]])
                        else:
                            rules2.append([rule[0][:],rule[1]]) # make a locla copy of rule
                            rules2[-1][0].remove( childText)    # remove the element childText from the antecedenses
                self.buildLayer(item, rules2, n-1)              # recursively build next levels


    def printRules(self, parent, rulesLC):
        """ Prints whole rule or the rest of the rule, depending on WholeRules.isChecked(), as a child of parent."""
        startOfRule=""                                          # the part of rule that is already in the tree
        gparent=parent
        while str(gparent.text(0))!="":
            startOfRule = str(gparent.text(0)) +"  "+startOfRule 
            gparent=gparent.parent()        
            
        for rule in rulesLC:
            restOfRule=""                         # concatenate the part that is already in the tree
            for s in rule[0]:                                   # with the rest of the antecedeses
                restOfRule=restOfRule+"  "+s
            
            item=QListViewItem(parent,"", str('%.3f' %rule[1][1]),str('%.3f' %rule[1][2]),str('%.3f' %rule[1][3]),str('%.3f' %rule[1][4]),str('%.3f' %rule[1][5]),str('%.3f' %rule[1][6]))             # add a branch with the text
            self.wrlist.append([item, startOfRule + restOfRule+"  ->   "+rule[1][0] , restOfRule+"  ->   "+rule[1][0]])


    def removeSingleGrouping(self):         
        """Removes a row if it has a "+" and only one child.  """
        for line in self.wrlist:                    # go through the leaves of the tree
            parent=line[0].parent()
            if (parent.childCount())==1:            # if the parent has only one child
                line[2]=str(parent.text(0))+"  "+line[2]    # add the text to the leaf
                gparent = parent.parent()                   # find the grand-parent
                gparent.takeItem(parent)                    # remove the parent
                gparent.insertItem(line[0])                 # insert a child        


    def arules(self,arules):
        self.rules = arules
        self.displayRules()

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesTreeViewer()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    rules=orange.AssociationRulesInducer(dataset, minSupport = 0.3, maxItemSets=15000)
    ow.arules(rules)
    
    ow.show()
    a.exec_loop()
    ow.saveSettings()

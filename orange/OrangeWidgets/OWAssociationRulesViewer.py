"""
<name>Association Rules Viewer</name>
<description>Association rules viewer</description>
<category>Associations</category>
<icon>icons/Unknown.png</icon>
<priority>300</priority>
"""

from OData import *
from OWWidget import *
#from orngAssoc import *        # to ne dela, ce si v napacnem direktoriju

import orngAssoc
import re  

import sys
from qt import *
from OWTools import *




class OWAssociationRulesViewer(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "Association rules viewer",
        """OWAssociationRulesViewer is an Orange Widget that
        displays the association rules as a table or as a 
        tree, depending on the choosen number of layers.
        """,
        FALSE,
        FALSE,
        "OrangeWidgetsIcon.png",
        "OrangeWidgetsLogo.png")
        
        self.rules=[]          # na zacetku nima vhodnih podatkov -v mainu jih dobi
        self.addInput("arules")  #declare the channel
        
        # Settings
        self.chbWholeRulesValue = 1       # default value of the check bow display Whole Rules
        self.NumOfLayers= 2               # default value of layers
 
        self.chbSupportValue=1
        self.chbConfidenceValue=1
        self.chbLiftValue= 0
        self.chbLeverageValue=0 
        self.chbStrengthValue=0 
        self.chbCoverageValue=0
        
        self.settingsList = ["NumOfLayers", "chbWholeRulesValue","chbSupportValue","chbConfidenceValue","chbLiftValue","chbLeverageValue","chbStrengthValue","chbCoverageValue"]  # list of settings
        self.loadSettings()         # function call

        # GUI: CONTROL AREA
        #QWidget(self.controlArea).setFixedSize(16, 16)

        
        self.chbWholwRules= QCheckBox( "Display whole rules", self.controlArea, "chbWholwRules" )
        self.chbWholwRules.setChecked(self.chbWholeRulesValue)
        self.connect(self.chbWholwRules, SIGNAL("clicked()"), self.setChbWholwRules)
      #  self.connect(self.chbWholwRules, SIGNAL("clicked()"), self.displayRules)
        
        self.lblNofLayers=QLabel("Number of layers.",self.controlArea)
        self.edtNumLayers = QSpinBox(0, 5, 1, self.controlArea, "edtNumLayers")
        self.edtNumLayers.setValue(self.NumOfLayers)
        self.connect(self.edtNumLayers, SIGNAL("valueChanged ( int )"), self.setNOfLayers)
        self.connect(self.edtNumLayers, SIGNAL("valueChanged ( int )"), self.displayRules)

        
                # the measures - a part of settings
        self.gbox= QVGroupBox ( "Measures", self.controlArea, "gbox" )
        
        self.chbSupport = QCheckBox( "Support", self.gbox, "chbSupport" ) # the checkboxes - which attribs to display
        self.chbSupport.setChecked(self.chbSupportValue)
        self.connect(self.chbSupport, SIGNAL("clicked()"), self.setChbSupport)
        
        self.chbConfidence= QCheckBox( "Confidence", self.gbox, "chbConfidence" )
        self.chbConfidence.setChecked(self.chbConfidenceValue)
        self.connect(self.chbConfidence, SIGNAL("clicked()"), self.setChbConfidence)
        
        self.chbLift= QCheckBox( "Lift", self.gbox, "chbLift" )
        self.chbLift.setChecked(self.chbLiftValue)
        self.connect(self.chbLift, SIGNAL("clicked()"), self.setChbLift)
        
        self.chbLeverage= QCheckBox( "Leverage", self.gbox, "chbLeverage" )
        self.chbLeverage.setChecked(self.chbLeverageValue)
        self.connect(self.chbLeverage, SIGNAL("clicked()"), self.setChbLeverage)
        
        self.chbStrength= QCheckBox( "Strength", self.gbox, "chbStrength" )
        self.chbStrength.setChecked(self.chbStrengthValue)
        self.connect(self.chbStrength, SIGNAL("clicked()"), self.setChbStrength)
        
        self.chbCoverage= QCheckBox( "Coverage", self.gbox, "chbCoverage" )
        self.chbCoverage.setChecked(self.chbCoverageValue)
        self.connect(self.chbCoverage, SIGNAL("clicked()"), self.setChbCoverage)
        
        
        
        
####### GUI : vizualization area 
        self.layout=QVBoxLayout(self.mainArea)
        self.treeRules = QListView(self.mainArea,'ListView')       #the rules and their properties are printed into this QListView
        self.treeRules.setMultiSelection (1)              #allow multiple selection
        self.treeRules.setAllColumnsShowFocus ( 1) 
        self.treeRules.addColumn(self.tr("Rules"))        #column0
        self.treeRules.addColumn(self.tr("Support"),60)      #column1
        self.treeRules.addColumn(self.tr("Confidence"),65)   #column2
        self.treeRules.addColumn(self.tr("Lift"),60)
        self.treeRules.addColumn(self.tr("Leverage"),60)
        self.treeRules.addColumn(self.tr("Strength"),60)
        self.treeRules.addColumn(self.tr("Coverage"),60)

        self.setChbSupport()
        self.setChbConfidence()
        self.setChbLift()
        self.setChbLeverage()
        self.setChbStrength()
        self.setChbCoverage()
       
        self.layout.addWidget(self.treeRules)
        
        

   #     self.btnTest=QPushButton("Test",self.controlArea)
   #     self.connect(self.btnTest, SIGNAL("clicked()"), self.printCN)
        
   # def printCN(self):  
   #     for col in range(7):
   #         print col, self.treeRules.columnText ( col ),"\t" ,self.treeRules.columnWidth (col)
   #     print


    def setChbSupport(self):
    #    print "setChbSupport"
        self.chbSupportValue=self.chbSupport.isChecked()
        if (self.chbSupportValue == 0): self.treeRules.setColumnWidth ( 1, 0 )
        else : self.treeRules.setColumnWidth ( 1, 60 )

    def setChbConfidence(self):
    #    print "setChbConfidence"
        self.chbConfidenceValue=self.chbConfidence.isChecked()
        if (self.chbConfidenceValue == 0): self.treeRules.setColumnWidth ( 2, 0 )
        else : self.treeRules.setColumnWidth ( 2, 65 )

    def setChbLift(self):
    #    print "setChbLift"
        self.chbLiftValue=self.chbLift.isChecked()
        if (self.chbLiftValue == 0): self.treeRules.setColumnWidth ( 3, 0 )
        else : self.treeRules.setColumnWidth ( 3, 60 )

    def setChbLeverage(self):
     #   print "setChbLeverage"
        self.chbLeverageValue=self.chbLeverage.isChecked()
        if (self.chbLeverageValue == 0): self.treeRules.setColumnWidth ( 4, 0 )
        else : self.treeRules.setColumnWidth ( 4, 60 )
    
    def setChbStrength(self):
    #    print "setChbStrength"
        self.chbStrengthValue=self.chbStrength.isChecked()
        if (self.chbStrengthValue == 0): self.treeRules.setColumnWidth ( 5, 0 )
        else : self.treeRules.setColumnWidth ( 5, 60 )

    def setChbCoverage(self):
     #   print "setChbCoverage"
        self.chbCoverageValue=self.chbCoverage.isChecked()
        if (self.chbCoverageValue == 0): self.treeRules.setColumnWidth ( 6, 0 )
        else : self.treeRules.setColumnWidth ( 6, 60 )

    def setChbWholwRules(self):
        self.chbWholeRulesValue=self.chbWholwRules.isChecked()
        if (self.chbWholeRulesValue==1): d=1
        else : d=2
        for line in self.wrlist:
            line[0].setText(0,line[d])
        
    def setNOfLayers(self):
        self.NumOfLayers=self.edtNumLayers.value()
        

    def displayRules(self):
        """ Display rules as a tree. """
        if self.rules!=0:        # this function can be called before the rules are built
            self.treeRules.clear()
            self.wrlist = []
            
            item0 = QListViewItem(self.treeRules,"")        #the first row is different
            
            rulesLC=[]
            for rule in self.rules:                 # local copy of rules [[antecedens1,antecedens2,...], [consequens, support,...]] (without measures)
                values = filter(lambda val: not val.isSpecial(), rule.left)

                #antecedens = ["%s=%s" % (x.variable.name, str(x)) for x in values]
                antecedens = []
                for x in values:
                    if x.varType:
                        antecedens.append(str(x.variable.name) + "=" + str(x))
                    else:
                        antecedens.append(str(x.variable.name))
                                
                values = filter(lambda val: not val.isSpecial(), rule.right)

                #pairsr = ["%s=%s" % (x.variable.name, str(x)) for x in values]
                #kons=""
                #for x in pairsr:
                #    kons=kons + x + "  "
                kons=""
                for x in values:
                    if x.varType:
                        kons=kons + str(x.variable.name) + "=" + str(x) + "  "
                    else:
                        kons=kons + str(x.variable.name) + "  "
                
                rulesLC.append([antecedens, [kons, rule.support, rule.confidence, rule.lift, rule.leverage, rule.strength, rule.coverage]])
            
            self.buildLayer(item0, rulesLC, self.edtNumLayers.value() )     # recursively builds as many layers as are in the.................
            self.removeSingleGrouping(item0)        # if there is only 1 rule behind a +, the rule is
            self.setChbWholwRules()
            item0.setOpen(1)                        # display the rules



    def buildLayer(self, parent, rulesLC, n):
        if n==0:
           self.printRules(parent, rulesLC)
        elif n>0:
            children = []
            for rule in rulesLC:                                 # for every role
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
        """ Prints whole rule or the rest of the rule, depending on chbWholwRules.isChecked(), as a child of parent."""
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

    def removeSingleGrouping(self,item0):         
        """Removes a row if it has a "+" and only one child.  """
        for line in self.wrlist:                    # go through the leafs of the trees
            parent=line[0].parent()
            if (parent.childCount())==1:            # if the parent has only one child
                line[2]=str(parent.text(0))+"  "+line[2]    # add the text to the leaf
                gparent = parent.parent()                   # find the grand-parent
                gparent.takeItem(parent)                    # remove the parent
                gparent.insertItem(line[0])                 # insert a child        

    def arules(self,arules):                # the channel - when the data arrives, this function is called
        self.rules=arules
        self.displayRules()

                
                
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesViewer()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('lenses.tab')
    rules=orngAssoc.build(dataset, 0.3, maxItemSets=15000)
    ow.arules(rules)
    
    ow.show()
    a.exec_loop()
    ow.saveSettings()

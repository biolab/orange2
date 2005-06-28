"""
<name>Association Rules Print</name>
<description>Textual display of association rules.</description>
<category>Associations</category>
<icon>icons/AssociationRulesPrint.png</icon>
<priority>400</priority>
"""

from OWWidget import *
#from orngAssoc import *        # to ne dela, ce si v napacnem direktoriju

import orngAssoc
import string 

import sys
from qt import *
from OWTools import *


class OWAssociationRulesPrint(OWWidget):
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Association rules viewer")
        
        self.inputs = [("AssociationRules", orange.AssociationRules, self.arules, 1)]
        self.outputs = []
        
        self.rules=[]          # na zacetku nima vhodnih podatkov -v mainu jih dobi

        # Settings

        self.chbSupportValue=1
        self.chbConfidenceValue=1
        self.chbLiftValue= 0
        self.chbLeverageValue=0 
        self.chbStrengthValue=0 
        self.chbCoverageValue=0
        
        self.settingsList = ["chbSupportValue","chbConfidenceValue","chbLiftValue","chbLeverageValue","chbStrengthValue","chbCoverageValue"]  # list of settings
        self.loadSettings()         # function call

        # GUI: CONTROL AREA
        #QWidget(self.controlArea).setFixedSize(16, 16)


                # the measures - a part of settings
        self.gbox= QVGroupBox ( "Measures", self.controlArea, "gbox" )
        
        self.chbSupport = QCheckBox( "Support", self.gbox, "chbSupport" ) # the checkboxes - which attribs to display
        self.chbSupport.setChecked(self.chbSupportValue)
        self.connect(self.chbSupport, SIGNAL("clicked()"), self.setChbSupport)
        self.connect(self.chbSupport, SIGNAL("clicked()"), self.displayRules)
        
        self.chbConfidence= QCheckBox( "Confidence", self.gbox, "chbConfidence" )
        self.chbConfidence.setChecked(self.chbConfidenceValue)
        self.connect(self.chbConfidence, SIGNAL("clicked()"), self.setChbConfidence)
        self.connect(self.chbConfidence, SIGNAL("clicked()"), self.displayRules)
        
        self.chbLift= QCheckBox( "Lift", self.gbox, "chbLift" )
        self.chbLift.setChecked(self.chbLiftValue)
        self.connect(self.chbLift, SIGNAL("clicked()"), self.setChbLift)
        self.connect(self.chbLift, SIGNAL("clicked()"), self.displayRules)
        
        self.chbLeverage= QCheckBox( "Leverage", self.gbox, "chbLeverage" )
        self.chbLeverage.setChecked(self.chbLeverageValue)
        self.connect(self.chbLeverage, SIGNAL("clicked()"), self.setChbLeverage)
        self.connect(self.chbLeverage, SIGNAL("clicked()"), self.displayRules)
        
        self.chbStrength= QCheckBox( "Strength", self.gbox, "chbStrength" )
        self.chbStrength.setChecked(self.chbStrengthValue)
        self.connect(self.chbStrength, SIGNAL("clicked()"), self.setChbStrength)
        self.connect(self.chbStrength, SIGNAL("clicked()"), self.displayRules)
        
        self.chbCoverage= QCheckBox( "Coverage", self.gbox, "chbCoverage" )
        self.chbCoverage.setChecked(self.chbCoverageValue)
        self.connect(self.chbCoverage, SIGNAL("clicked()"), self.setChbCoverage)
        self.connect(self.chbCoverage, SIGNAL("clicked()"), self.displayRules)
        
            #Save rules to file button
        self.btnSaveToFile = QPushButton("&Save rules to file...", self.controlArea)
        self.connect(self.btnSaveToFile,SIGNAL("clicked()"), self.saveRulesToFile)       


        
####### GUI : vizualization area 
        self.layout=QVBoxLayout(self.mainArea)
        self.edtRules = QMultiLineEdit(self.mainArea)                # we print the rules in this multi line edit
        self.edtRules.setReadOnly(TRUE)
        self.layout.addWidget(self.edtRules)
        

    def setChbSupport(self):
        self.chbSupportValue=self.chbSupport.isChecked()

    def setChbConfidence(self):
        self.chbConfidenceValue=self.chbConfidence.isChecked()

    def setChbLift(self):
        self.chbLiftValue=self.chbLift.isChecked()

    def setChbLeverage(self):
        self.chbLeverageValue=self.chbLeverage.isChecked()
        
    def setChbStrength(self):
        self.chbStrengthValue=self.chbStrength.isChecked()

    def setChbCoverage(self):
        self.chbCoverageValue=self.chbCoverage.isChecked()

    def arules(self,arules):                # nekaj dela s podatki - ta je edini channel (baje)
        self.rules=arules
        self.displayRules()

        
    def saveRulesToFile(self):
        dlg = QFileDialog()
        fileName = dlg.getSaveFileName( "myRules.txt", "Textfiles (*.txt)", self );
        if not fileName.isNull() :
            f=open(str(fileName), 'w')
            if self.rules!=0:        #this function can be called also when just the tabs are changed before the rules are built
                ms=[]                                    # measures to be displayed (based on chkboxes)
                if self.chbSupportValue: ms.append("support")        
                if self.chbConfidenceValue: ms.append("confidence")  #have to do this for other measures too
                if self.chbLiftValue: ms.append("lift")
                if self.chbLeverageValue: ms.append("leverage")
                if self.chbStrengthValue: ms.append("strength")
                if self.chbCoverageValue: ms.append("coverage")
                if ms!=[]:
                    f.write(str( string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%s' % (m[0:4]) for m in ms]))+"\trule")+'\n') # prints the first line
                for rule in self.rules:
                    s=(orngAssoc.printRule(rule))
                    L=string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%.3f' % getattr(rule,m) for m in ms],""))    # for each rule gets the measures
                    if L!="":
                        f.write( L+'\t'+s+'\n')             # prints the measures and the rule        
                    else:
                        f.write(s+'\n')
            f.close()

    def displayRules(self):
        """ Checkes which measures are checked and displays the measures and the rules in their "normal" form."""
        if self.rules!=0:        #this function can be called also when just the tabs are changed before the rules are built
            ms=[]                                    # measures to be displayed (based on chkboxes)
            if self.chbSupportValue: ms.append("support")        
            if self.chbConfidenceValue: ms.append("confidence")  #have to do this for other measures too
            if self.chbLiftValue: ms.append("lift")
            if self.chbLeverageValue: ms.append("leverage")
            if self.chbStrengthValue: ms.append("strength")
            if self.chbCoverageValue: ms.append("coverage")
            
            self.edtRules.clear()
            if ms!=[]:
                self.edtRules.append(str( string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%s' % (m[0:4]) for m in ms]))+"\trule")) # prints the first line
            for rule in self.rules:
                s= `rule`
                s=s.replace(", ",",")                           # the result of categorize might have the sequence ", " that is removed
                s=s.replace(" ","  ")                           # extend the spaces to achieve greater legibility
                L=string.lstrip(reduce(lambda a,b: a+"\t"+b, ['%.3f' % getattr(rule,m) for m in ms],""))    # for each rule gets the measures
                if L!="":
                    self.edtRules.append( L+'\t'+s)             # prints the measures and the rule        
                else:
                    self.edtRules.append(s)




if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesPrint()
    a.setMainWidget(ow)


    dataset = orange.ExampleTable('car.tab')
    rules=orange.AssociationRulesInducer(dataset, minSupport = 0.3, maxItemSets=15000)
    ow.arules(rules)
        
    ow.show()
    a.exec_loop()
    ow.saveSettings()

"""
<name>Association Rules</name>
<description>Association rules inducer</description>
<category>Associations</category>
<icon>icons/Unknown.png</icon>
<priority>100</priority>
"""

import orange
from OData import *
from OWWidget import *
import orngAssoc

import string

class OWAssociationRules(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,
            parent,
            "AssociationRules",
            """OWAssociationRules is orange widget\n for building Association rules.\n\nAuthors: Jure Germovsek, Petra Kralj, Matjaz Jursic\nMay 25, 2003
            """,
            FALSE,
            FALSE,
            "OrangeWidgetsIcon.png",
            "OrangeWidgetsLogo.png")

        # zaèetne vrednosti
        self.support = 0.5
        self.max_rules = 1000
        self.method = 0

        self.addInput("cdata")
        self.addOutput("arules")

        self.settingsList = ["support", "max_rules", "method"]    # list of settings
        self.loadSettings()                             # function call
                
        self.dataset = None

        # Build Method
        self.rulesGB1 = QGroupBox(1, QGroupBox.Horizontal , 'Build Method', self.controlArea)
        self.cbBuildMethod = QComboBox(FALSE, self.rulesGB1)
        self.cbBuildMethod.insertItem("Dense Data")
        self.cbBuildMethod.insertItem("Sparse Data")
        self.cbBuildMethod.setCurrentItem(self.method)
        self.connect(self.cbBuildMethod, SIGNAL("activated(int)"), self.methodSelected)
        
        self.rulesGB = QGroupBox(1, QGroupBox.Horizontal , 'Build Settings', self.controlArea)

        # Min Support
        self.lblSupport = QLabel("Min Support: %.2f" % self.support, self.rulesGB) 
        self.sliSupport = QSlider(0, 99, 1, self.support*100, QSlider.Horizontal, self.rulesGB)
        self.connect(self.sliSupport,SIGNAL("valueChanged(int)"), self.setSupport)       

        # Max Number of Rules
        self.lblNumRules = QLabel("Stop generating at %i rules:" %self.max_rules, self.rulesGB)
        self.edtNumRules = QSpinBox(100, 100000, 100, self.rulesGB)
        self.edtNumRules.setValue(self.max_rules)
        self.connect(self.edtNumRules, SIGNAL("valueChanged(int)"), self.setNumRules)

        # Comment
        self.lblComment=QLabel("\nThe building of rules will stop,\n when support will fall to %.2f\n"%self.support + " or the number of rules will exced %i.\n"%self.max_rules, self.controlArea)        


        # Generate button
        self.btnGenerate = QPushButton("&Build rules", self.controlArea)
        self.connect(self.btnGenerate,SIGNAL("clicked()"), self.generateAssociations)

       
        # Build Log
        self.buildLogFrame = QGroupBox(1, QGroupBox.Horizontal , 'Build Log', self.mainArea)
        self.buildLog = QMultiLineEdit(self.buildLogFrame)
        self.buildLog.setReadOnly(True)

        # Avtomatski layout
        self.vbox = QVBoxLayout(self.mainArea)
        self.hbox = QHBoxLayout(self.vbox)
        self.hbox.addWidget(self.buildLogFrame)

    def methodSelected(self, value):
        self.method = value

    def setSupport(self, value):            
        if str(value) == '':               
            value = 50
        v = int(str(value))                 
        if (v<0) or (v>100):
            v = 50
        self.support = float(v)/100         
        self.lblSupport.setText ("Min Support: %.2f" % self.support)
        self.lblComment.setText( "\nThe building of rules will stop,\n when support will fall to %.2f\n"%self.support + " or the number of rules will exced %i.\n"%self.max_rules )

    def setNumRules(self, value):
        self.max_rules = value;
        self.lblNumRules.setText("Stop generating at %i rules:" %self.max_rules)
        self.lblComment.setText( "\nThe building of rules will stop,\n when support will fall to %.2f\n"%self.support + " or the number of rules will exced %i.\n"%self.max_rules )

    def generateAssociations(self):
        self.buildLog.clear()
        if self.dataset != None:  # èe dataset ni prazen
            # èe je izbran argawal
            if self.method == 1:
                self.buildLog.insertLine('Build with Sparse Data method started.', 0)
            else:
                self.buildLog.insertLine('Build with Dense Data method started.', 0)
            
            rules = []

            num_of_steps = 20

            # korakaj dokler nimaš dovolj pravil, oziroma ne dosežeš minSupport
            for i in range(1, num_of_steps + 1):
                # zgradi pravila
                build_support = 1 - float(i) / num_of_steps * (1 - self.support)
                try:
                    # èe je izbran Sparse Data
                    if self.method == 1:
                        rules=orngAssoc.buildSparse(self.dataset.table,build_support)
                    else:
                        rules=orngAssoc.build(self.dataset.table,build_support)
                except:
                    self.buildLog.insertLine('Error occured during build.',0)
                    return
                                    
                rules_count = len(rules)
                self.buildLog.insertLine('Found ' + str(rules_count) + ' rules with support >= '+ str(build_support) + '.', 0)

                # Èe si že našel dovolj pravil
                if rules_count >= self.max_rules:
                    break

            # pošlji pravila
            if self.cbBuildMethod.currentItem() == 1:
                self.buildLog.insertLine('Build with Sparse Data method ended.', 0)
            else:
                self.buildLog.insertLine('Build with Dense Data method ended.', 0)
           
            self.send("arules", rules)
        else:
            self.buildLog.insertLine('No data. Load a file.', 0 )

    def cdata(self,dataset):                # channel po katerem dobi podatke
        self.dataset = dataset 

if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRules()
    a.setMainWidget(ow)

    dataset = orange.ExampleTable('lenses.tab')
    #dataSet=orange.ExampleTable('basket_data_o')
    od = OrangeData(dataset)
    ow.cdata(od)

    ow.show()
    a.exec_loop()
    ow.saveSettings()
    
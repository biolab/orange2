"""
<name>Nomogram</name>
<description>Visualizes Naive Bayesian classification using interactive nomogram.</description>
<category>Classification</category>
<icon>icons/NomogramVisualisation.png</icon>
<priority>9998</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with Naive Bayes classifier
#
# An element can be interactively
# entered using a mouse
#

from OWWidget import *
from OW_KN_NomogramOptions import *
from OW_KN_NomogramGraph import * #if using a graph

class OWNomogram(OWWidget):
    settingsList = []

    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "&Nomogram visualisation",
        """OWNomogram is an Orange Widget
for displaying a nomogram of a Bayes classifier.\n\nAuthors: Andrej Oresnik, Iztok Heric""",
        FALSE,
        TRUE)

        self.loadSettings()

        #inputs
        self.inputs=[("nbClassifier", orange.BayesClassifier, self.nbClassifier, 1)]        
        #self.addInput("nbClassifier")
        #self.addInput("target")

        #GUI
        self.viewOptions=QVGroupBox(self.space)
        self.viewOptions.setTitle("Options")
        self.showOutcomes=QCheckBox("Show outcome", self.viewOptions)
        self.connect(self.showOutcomes, SIGNAL('stateChanged ( int )'), self.setShowOutcomes)
        self.clearExamplesButton=QPushButton("Clear example", self.viewOptions)
        self.connect(self.clearExamplesButton, SIGNAL('pressed ()'), self.clearExample)

        #add a graph widget
        #the graph widget needs to be created separately, preferably by inheriting from OWGraph
        self.box=QVBoxLayout(self.mainArea)
        self.graph=OWNomogramGraph(self.mainArea)
        self.box.addWidget(self.graph)

        #connect graph saving button to graph
        self.connect(self.graphButton,SIGNAL("clicked()"),self.graph.saveToFile)

    def setShowOutcomes(self, state):
        self.graph.setShowOutcomes(state == 2)

    def clearExample(self):
        self.graph.clearExample()

    # Input channel: the Bayessan classifier (mandatory)    
    def nbClassifier(self, data):
        self.graph.setClassifier(data)

    # Input channel: the target outcome (optional)    
    def target(self, data):
        self.graph.setTarget(data)

# test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNomogram()
    a.setMainWidget(ow)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()

    # save settings
    ow.saveSettings()


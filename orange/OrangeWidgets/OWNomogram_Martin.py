"""
<name>Nomogram</name>
<description>Visualizes Naive Bayesian or logistic regression classifier using interactive nomogram.</description>
<category>Classification</category>
<icon>icons/NomogramVisualisation.png</icon>
<priority>9998</priority>
"""

#
# Nomogram is a Orange widget for
# for visualization of the knowledge
# obtained with Naive Bayes or logistic regression classifier
#

import Numeric
import orange
import OWGUI
from OWWidget import *
#from OW_KN_NomogramOptions import *
from OW_NomogramGraph_Martin import * 
#import OW_NomogramGraph_Martin
#reload(OW_NomogramGraph_Martin)


class Classifier_Properties:
    def __init__(self, classifier):
        self.classifier = classifier
        self.att_names = classifier.domain.attributes
        self.classVar = classifier.domain.classVar
        
    def getAtributes(self):
        return att_names

class OWNomogram_Martin(OWWidget):
    settingsList = ["alignType"]

    def __init__(self,parent=None):
        OWWidget.__init__(self,
        parent,
        "&Nomogram visualisation",
        """OWNomogram is an Orange Widget
for displaying a nomogram of a Naive Bayesian or logistic regression classifier.""",
        FALSE,
        TRUE)

        self.callbackDeposit = [] # deposit for OWGUI callback functions
        self.alignType = 0
        self.contType = 0
        self.yAxis = 0
        self.probability = 1
        self.table = 0
        self.verticalSpacing = 40
        self.fontSize = 9
        self.lineWidth = 1
        
        self.loadSettings()

        #inputs
        self.inputs=[("nbClassifier", orange.BayesClassifier, self.nbClassifier, 1), ("lrClassifier", orange.LogisticClassifier, self.lrClassifier, 1)]
        #self.addInput("nbClassifier")
        #self.addInput("lrClassifier")
        #self.addInput("target")

        # GUI definition
        self.tabs = QTabWidget(self.controlArea, 'tabWidget')
        
        # GENERAL TAB
        GeneralTab = QVGroupBox(self)

        self.alignRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'Align', ['Left', '0-point'], 'alignType',
                                tooltips=['Attributes in nomogram are left aligned', 'Attributes are not aligned, top scale represents true (normalized) regression coefficient value'],
                                callback=self.setAlignType)
        self.yAxisRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'yAxis', ['100', 'beta coeff', 'odds ratio'], 'yAxis',
                                tooltips=['values are normalized on 0-100 point scale','beta = regression coefficients','OR (odds ration) = exp(beta). This kind of nomogram shows actual attribute contribution and not log-linear one.'],
                                callback=self.setYAxis)
        self.ContRadio = OWGUI.radioButtonsInBox(GeneralTab, self, 'Continuous', ['1D', '2D'], 'contType',
                                tooltips=['Continuous attribute are presented on a single scale', 'Two dimensional space is used to present continuous attributes in nomogram.'],
                                callback=self.setContType)

        self.yAxisRadio.setDisabled(True)
        self.probabilityCheck = OWGUI.checkOnly(GeneralTab, self, 'Show probability', 'probability', tooltip='Show probability scale at the bottom of nomogram graph?')
        self.probabilityCheck.setDisabled(True)
        self.tableCheck = OWGUI.checkOnly(GeneralTab, self, 'Show table', 'table', tooltip='Show table of selected attribute values?')
        self.tableCheck.setDisabled(True)
        
        self.tabs.insertTab(GeneralTab, "General")
        
        # TREE TAB
        NomogramStyleTab = QVGroupBox(self)

        self.verticalSpacingLabel = OWGUI.labelWithSpin(NomogramStyleTab, self, 'Vertical spacing:', min=15, max=100, value='verticalSpacing', step = 1, tooltip='Define space (pixels) between adjacent attributes.')
        self.verticalSpacingLabel.setDisabled(True)
        self.fontSizeLabel = OWGUI.labelWithSpin(NomogramStyleTab, self, 'Font size:', min=4, max=14, value='fontSize', step = 1, tooltip='Font size of nomogram labels.')
        self.fontSizeLabel.setDisabled(True)
        self.lineWidthLabel = OWGUI.labelWithSpin(NomogramStyleTab, self, 'Line width:', min=1, max=10, value='lineWidth', step = 1, tooltip='Define width of lines shown in nomogram.')
        self.lineWidthLabel.setDisabled(True)
        
        self.tabs.insertTab(NomogramStyleTab, "Style")
        
        #add a graph widget
        #the graph widget needs to be created separately, preferably by inheriting from OWGraph
        self.box=QVBoxLayout(self.mainArea)
        self.graph=OWNomogramGraph(self.mainArea)
        self.box.addWidget(self.graph)


    # Input channel: the Bayesian classifier   
    def nbClassifier(self, cl):
        classVal = cl.domain.classVar
        att = cl.domain.attributes

        prior = cl.distribution[classVal[0]]/cl.distribution[classVal[1]]
        bnomogram = BasicNomogram(AttValue("Constant", Numeric.log(prior)))
        for at in range(len(att)):
            a = AttrLine(att[at].name, at+1)
            if att[at].varType == orange.VarTypes.Discrete:
                for cd in cl.conditionalDistributions[at].keys():
                    a.addAttValue(AttValue(str(cd), Numeric.log(cl.conditionalDistributions[at][cd][classVal[0]]/cl.conditionalDistributions[at][cd][classVal[1]]/prior)))
            else:
                d = cl.conditionalDistributions[at].keys()[len(cl.conditionalDistributions[at].keys())-1]-cl.conditionalDistributions[at].keys()[0]
                d = getDiff(d/50)
                if cl.conditionalDistributions[at].keys()[0]<0:
                    curr_num = arange(-cl.conditionalDistributions[at].keys()[0]+d)
                    curr_num = curr_num[len(curr_num)-1]
                    curr_num = -curr_num
                elif cl.conditionalDistributions[at].keys()[0] == 0:
                    curr_num = 0
                else:
                    print d, cl.conditionalDistributions[at].keys()
                    curr_num = arange(cl.conditionalDistributions[at].keys()[0]-d)
                    print curr_num
                    curr_num = curr_num[len(curr_num)-1]
                                    
                rndFac = math.floor(math.log10(d));
                if rndFac<-2:
                    rndFac = -rndFac
                else:
                    rndFac = 2
                for cd in cl.conditionalDistributions[at].keys():
                    if cd>=curr_num:
                        print curr_num, round, 
                        a.addAttValue(AttValue(str(round(curr_num,rndFac)), Numeric.log(cl.conditionalDistributions[at][cd][classVal[0]]/cl.conditionalDistributions[at][cd][classVal[1]]/prior)))
                        curr_num = curr_num + d
                    
                if att[at].varType == orange.VarTypes.Continuous:
                    a.continuous = True
            bnomogram.addAttribute(a)        
            
        bnomogram.printOUT()
        self.alignType = 1
        self.graph.setNomogramData(bnomogram)
        self.graph.setAlignType(self.alignType)

    # Input channel: the logistic regression classifier    
    def lrClassifier(self, cl):
        bnomogram = BasicNomogram(AttValue('Constant', -cl.beta[0], cl.beta_se[0]))
        at = cl.domain.attributes
        at_num = 1
        curr_att = ""
        for i in range(len(at)):
            if at[i].getValueFrom:
                name = at[i].getValueFrom.variable.name
                var = at[i].getValueFrom.variable
            else:
                return
                name = at[i].name
                var = at[i]
            if name != curr_att:
                print "name", name, "var", var
                curr_att = name
                a = AttrLine(at[i].getValueFrom.variable.name, at_num)
                at_num = at_num+1

                if type(var) == orange.EnumVariable:
                    for v in range(len(var.values)):
                        if v == 0:
                            a.addAttValue(AttValue(var.values[0],0.0))
                        else:
                            print "beta", cl.beta[i+v-1]
                            a.addAttValue(AttValue(var.values[v],-cl.beta[i+v]))
                bnomogram.addAttribute(a)        
#                   else:
        print "v set nomogram"
        self.alignRadio.setDisabled(True)
        bnomogram.printOUT()
        self.graph.setNomogramData(bnomogram)
        self.alignType = 0
        self.graph.setAlignType(0)

    # Input channel: the target outcome (optional)    
    def target(self, data):
        self.graph.setTarget(data)

    def setAlignType(self):
        self.graph.setAlignType(self.alignType)

    def setContType(self):
        self.graph.setContType(self.contType)

    def setYAxis(self):
        self.graph.setYAxis(self.yAxis)

# test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWNomogram_Martin()
    a.setMainWidget(ow)

    # here you can test setting some stuff
    ow.show()
    a.exec_loop()

    # save settings
    ow.saveSettings()


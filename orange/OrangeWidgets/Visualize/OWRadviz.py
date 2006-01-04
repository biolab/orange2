"""
<name>Radviz</name>
<description>Create a radviz projection.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/Radviz.png</icon>
<priority>3100</priority>
"""
# Radviz.py
#
# Show a radviz projection of the data
# 

from OWWidget import *
from OWLinProj import *
#from random import betavariate 
#from OWLinProjGraph import *
#from OWkNNOptimization import OWVizRank
#from OWClusterOptimization import *
#from OWFreeVizOptimization import *
#import time
#import OWToolbars, OWGUI, orngTest, orangeom
#import OWVisFuncts, OWDlgs
#import orngVizRank

###########################################################################################
##### WIDGET : Linear Projection
###########################################################################################
class OWRadviz(OWLinProj):
    settingsList = ["graph.pointWidth", "graph.jitterSize", "graph.globalValueScaling", "graph.showFilledSymbols", "graph.scaleFactor",
                    "graph.showLegend", "graph.optimizedDrawing", "graph.useDifferentSymbols", "autoSendSelection", "graph.useDifferentColors",
                    "graph.tooltipKind", "graph.tooltipValue", "toolbarSelection", "graph.showClusters", "VizRankClassifierName", "clusterClassifierName",
                    "showOptimizationSteps", "valueScalingType", "graph.showProbabilities", "showAllAttributes",
                    "learnerIndex", "colorSettings", "addProjectedPositions"]
            
    def __init__(self, parent=None, signalManager = None):
        OWLinProj.__init__(self, parent, signalManager, "Radviz")

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default), ("Example Subset", ExampleTable, self.subsetdata), ("Attribute Selection List", AttributeList, self.attributeSelection), ("Evaluation Results", orngTest.ExperimentResults, self.test_results), ("VizRank Learner", orange.Learner, self.vizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Attribute Selection List", AttributeList), ("Learner", orange.Learner)]


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

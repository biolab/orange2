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

from OWLinProj import *

class OWRadviz(OWLinProj):
    settingsList = ["graph.pointWidth", "graph.jitterSize", "graph.showFilledSymbols", "graph.scaleFactor",
                    "graph.showLegend", "graph.optimizedDrawing", "graph.useDifferentSymbols", "autoSendSelection", "graph.useDifferentColors",
                    "graph.tooltipKind", "graph.tooltipValue", "toolbarSelection", "graph.showClusters", "clusterClassifierName", "graph.useAntialiasing",
                    "valueScalingType", "graph.showProbabilities", "showAllAttributes",
                    "learnerIndex", "colorSettings", "selectedSchemaIndex", "addProjectedPositions", "VizRankLearnerName"]

    def __init__(self, parent=None, signalManager = None):
        OWLinProj.__init__(self, parent, signalManager, "Radviz")

        self.inputs = [("Data", ExampleTable, self.setData, Default), ("Data Subset", ExampleTable, self.setSubsetData), ("Features", AttributeList, self.setShownAttributes), ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults), ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Data", ExampleTable), ("Other Data", ExampleTable), ("Features", AttributeList)]



#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWRadviz()
    ow.show()
    #data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\wine.tab")
    data = orange.ExampleTable(r"e:\Development\Orange Datasets\UCI\iris.tab")
    ow.setData(data)
    ow.handleNewSignals()
    a.exec_()

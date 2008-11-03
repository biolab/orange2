"""
<name>Polyviz</name>
<description>Polyviz (multiattribute) visualization.</description>
<contact>Gregor Leban (gregor.leban@fri.uni-lj.si)</contact>
<icon>icons/Polyviz.png</icon>
<priority>3150</priority>
"""
# Polyviz.py
#
# Show data using Polyviz visualization method
#
from OWLinProj import *
from OWPolyvizGraph import *


###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWPolyviz(OWLinProj):
    settingsList = ["graph.pointWidth", "graph.jitterSize", "graph.scaleFactor", "graph.useAntialiasing",
                    "graph.showLegend", "graph.showFilledSymbols", "graph.optimizedDrawing", "graph.useDifferentSymbols", "autoSendSelection",
                    "graph.useDifferentColors", "graph.tooltipKind", "graph.tooltipValue", "toolbarSelection", "VizRankLearnerName",
                    "colorSettings", "selectedSchemaIndex", "addProjectedPositions", "showAllAttributes", "graph.lineLength"]

    def __init__(self,parent=None, signalManager = None):
        OWLinProj.__init__(self, parent, signalManager, "Polyviz", graphClass = OWPolyvizGraph)

        self.inputs = [("Examples", ExampleTable, self.setData, Default), ("Example Subset", ExampleTable, self.setSubsetData), ("Attribute Selection List", AttributeList, self.setShownAttributes), ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults), ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable), ("Unselected Examples", ExampleTable), ("Attribute Selection List", AttributeList)]

        # SETTINGS TAB
        self.extraTopBox.show()
        OWGUI.hSlider(self.extraTopBox, self, 'graph.lineLength', box=' Line length ', minValue=0, maxValue=10, step=1, callback = self.updateGraph)




#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWPolyviz()
    ow.show()
    a.exec_()

    #save settings
    ow.saveSettings()

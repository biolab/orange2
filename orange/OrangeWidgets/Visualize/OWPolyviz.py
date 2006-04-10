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
from OWPolyvizGraph import *
from OWLinProj import *

###########################################################################################
##### WIDGET : Polyviz visualization
###########################################################################################
class OWPolyviz(OWLinProj):
    settingsList = ["graph.pointWidth", "graph.jitterSize", "graph.globalValueScaling", "graph.scaleFactor",
                    "graph.showLegend", "graph.showFilledSymbols", "graph.optimizedDrawing", "graph.useDifferentSymbols", "autoSendSelection",
                    "graph.useDifferentColors", "graph.tooltipKind", "graph.tooltipValue", "toolbarSelection", "VizRankLearnerName",
                    "colorSettings", "addProjectedPositions", "showAllAttributes", "graph.lineLength"]
        
    def __init__(self,parent=None, signalManager = None):
        OWLinProj.__init__(self, parent, signalManager, "Polyviz", graphClass = OWPolyvizGraph)

        self.inputs = [("Classified Examples", ExampleTableWithClass, self.cdata, Default), ("Attribute Selection List", AttributeList, self.attributeSelection), ("VizRank Learner", orange.Learner, self.vizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTableWithClass), ("Unselected Examples", ExampleTableWithClass), ("Attribute Selection List", AttributeList)]

        # SETTINGS TAB
        self.extraTopBox.show()
        OWGUI.hSlider(self.extraTopBox, self, 'graph.lineLength', box=' Line Length ', minValue=0, maxValue=10, step=1, callback = self.updateGraph)

        self.freeVizDlgButton.hide()
        
                


#test widget appearance
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWPolyviz()
    a.setMainWidget(ow)
    ow.show()
    a.exec_loop()

    #save settings 
    ow.saveSettings()

"""
<name>RadViz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>2000</priority>
"""

from plot.owplot3d import *
from plot.owplotgui import OWPlotGUI
from OWLinProjQt import *

class OWRadviz3DPlot(OWPlot3D, orngScaleLinProjData):
    def __init__(self, widget, parent=None, name='None'):
        OWPlot3D.__init__(self, parent)
        orngScaleLinProjData.__init__(self)

        self.point_width = 5
        self.alpha_value = 255
        self.show_filled_symbols = True
        self.use_antialiasing = True
        self.sendSelectionOnUpdate = False
        self.setCanvasBackground = self.setCanvasColor

        self.gui = OWPlotGUI(self)

    def setData(self, data, subsetData=None, **args):
        orngScaleLinProjData.setData(self, data, subsetData, **args)

    def setCanvasColor(self, c):
        pass

    def updateData(self, labels=None, setAnchors=0, **args):
        pass

    def getSelectionsAsExampleTables(self, attrList, useAnchorData=1, addProjectedPositions=0):
        return (None, None)

    def replot(self):
        pass

class OWRadviz3D(OWLinProjQt):
    settingsList = ['showAllAttributes']

    def __init__(self, parent=None, signalManager=None):
        OWLinProjQt.__init__(self, parent, signalManager, "Radviz", graphClass=OWRadviz3DPlot)

        self.inputs = [("Examples", ExampleTable, self.setData, Default),
                       ("Example Subset", ExampleTable, self.setSubsetData),
                       ("Attribute Selection List", AttributeList, self.setShownAttributes),
                       ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults),
                       ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable),
                        ("Unselected Examples", ExampleTable),
                        ("Attribute Selection List", AttributeList)]

if __name__ == '__main__':
    app = QApplication(sys.argv)
    radviz = OWRadviz3D()
    radviz.show()
    data = orange.ExampleTable('../../doc/datasets/iris')
    radviz.setData(data)
    radviz.handleNewSignals()
    app.exec_()

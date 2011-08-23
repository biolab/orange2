'''
<name>Linear Projection 3D</name>
<icon>icons/LinearProjection.png</icon>
<priority>2002</priority>
'''

from plot.owplot3d import *

from OWLinProjQt import *
from OWLinProj3DPlot import OWLinProj3DPlot

class OWLinProj3D(OWLinProjQt):
    settingsList = ['showAllAttributes'] #TODO

    def __init__(self, parent=None, signalManager=None):
        OWLinProjQt.__init__(self, parent, signalManager, "Linear Projection 3D", graphClass=OWLinProj3DPlot)

        self.inputs = [("Examples", ExampleTable, self.setData, Default),
                       ("Example Subset", ExampleTable, self.setSubsetData),
                       ("Attribute Selection List", AttributeList, self.setShownAttributes),
                       ("Evaluation Results", orngTest.ExperimentResults, self.setTestResults),
                       ("VizRank Learner", orange.Learner, self.setVizRankLearner)]
        self.outputs = [("Selected Examples", ExampleTable),
                        ("Unselected Examples", ExampleTable),
                        ("Attribute Selection List", AttributeList)]
        self.resize(1000, 600)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viz = OWLinProj3D()
    viz.show()
    data = orange.ExampleTable('../../doc/datasets/iris')
    viz.setData(data)
    viz.handleNewSignals()
    app.exec_()

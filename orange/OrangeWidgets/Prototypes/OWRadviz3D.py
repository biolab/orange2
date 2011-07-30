"""
<name>RadViz 3D</name>
<icon>icons/Radviz.png</icon>
<priority>2000</priority>
"""

from plot.owplot3d import *
from plot.owprimitives3d import get_symbol_data
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

        self.sphere_data = get_symbol_data(Symbol.CIRCLE)
        self.show_axes = self.show_chassis = self.show_grid = False

    def setData(self, data, subsetData=None, **args):
        orngScaleLinProjData.setData(self, data, subsetData, **args)

    def updateGraph(self, attrList=None, setAnchors=0, insideColors=None, **args):
        pass

    def draw_callback(self):
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glColor4f(1,0,0,1)

        glScalef(5, 5, 5)
        glBegin(GL_TRIANGLES)
        for v0, v1, v2, n0, n1, n2 in self.sphere_data:
            glVertex3f(*v0)
            glVertex3f(*v1)
            glVertex3f(*v2)
        glEnd()

    def setCanvasColor(self, c):
        pass

    def updateData(self, labels=None, setAnchors=0, **args):
        self.commands.append(('custom', self.draw_callback))
        self.updateGL()

    def getSelectionsAsExampleTables(self, attrList, useAnchorData=1, addProjectedPositions=0):
        return (None, None)

    def removeAllSelections(self):
        pass

    def replot(self):
        pass

    # TODO: catch mouseEvents

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





from OWWidget import *
import OWGUI

class ExampleWidget(OWWidget):
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Example')
            self.inputs = [("Train Data", ExampleTable, self.trainset), ("Test Data", ExampleTable, self.testset, 1, 1), ("Learner", orange.Learner, self.learner, 0)]
    
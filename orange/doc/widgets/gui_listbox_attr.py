from OWWidget import *
import OWGUI

class Test(OWWidget):
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Listbox')

        self.attributes = []
        self.chosenAttribute = []
        self.values = []
        self.chosenValues = []

        OWGUI.listBox(self.controlArea, self, "chosenAttribute", "attributes",
                      box="Attributes", callback=self.setValues)
        OWGUI.separator(self.controlArea)
        OWGUI.listBox(self.controlArea, self, "chosenValues", "values",
                      box="Values", selectionMode=QListWidget.MultiSelection)

        self.controlArea.setFixedSize(150, 250)
        self.adjustSize()


        # The following assignments usually don't take place in __init__
        # but later on, when the widget receives some data
        import orange
        self.data = orange.ExampleTable(r"..\datasets\horse-colic.tab")
        self.attributes = [(attr.name, attr.varType) for attr in self.data.domain]
        self.chosenAttribute = [0]

    def setValues(self):
        attrIndex = self.chosenAttribute[0]
        attr = self.data.domain[attrIndex]
        if attr.varType == orange.VarTypes.Discrete:
            self.values = attr.values
        else:
            self.values = []
        self.chosenValues = []

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

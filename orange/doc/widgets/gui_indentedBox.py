from OWWidget import *
import OWGUI

class Test(OWWidget):
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Listbox')

        self.method = 0

        self.removeRedundant = self.removeContinuous = self.addNoise = self.classOnly = True
        self.removeClasses = False

        box = OWGUI.widgetBox(self.controlArea, "Redundant values")
        remRedCB = OWGUI.checkBox(box, self, "removeRedundant", "Remove redundant values")
        iBox = OWGUI.indentedBox(box)
        OWGUI.checkBox(iBox, self, "removeContinuous", "Reduce continuous attributes")
        OWGUI.checkBox(iBox, self, "removeClasses", "Reduce class attribute")
        remRedCB.disables.append(iBox)

        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Noise")        
        noiseCB = OWGUI.checkBox(box, self, "addNoise", "Add noise to data")
        classNoiseCB = OWGUI.checkBox(OWGUI.indentedBox(box), self, "classOnly", "Add noise to class only")
        noiseCB.disables.append(classNoiseCB)

        self.adjustSize()

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

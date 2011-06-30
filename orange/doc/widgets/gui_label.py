from OWWidget import *
import OWGUI

class Test(OWWidget):
    settingsList = ["val", "line"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Labels')

        self.val = 5
        self.line = "a parrot"

        OWGUI.spin(self.controlArea, self, "val", 1, 10, label="Value")
        OWGUI.lineEdit(self.controlArea, self, "line", label="Line: ", orientation="horizontal")

        OWGUI.label(self.controlArea, self, "Value is %(val)i and line edit contains %(line)s")        

        self.resize(100,100)

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

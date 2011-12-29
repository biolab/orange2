from OWWidget import *
import OWGUI

class Test(OWWidget):
    
    settingsList = ["val", "chk"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, 'CheckSpin')

        self.val = 20
        self.chk = 1
        OWGUI.checkWithSpin(self.controlArea, self, "Prunning, m=", 0, 100, "chk", "val", posttext = "%")

        self.resize(100,100)

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

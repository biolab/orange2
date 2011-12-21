from OWWidget import *
import OWGUI

class Test(OWWidget):
    
    settingsList = ["chkA", "chkB", "dx"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, 'Check')
        
        # GUI        
        self.spinval = 10
        self.chkA = 1
        self.chkB = 0
        self.dx = 15

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        gridbox = OWGUI.widgetBox(self.controlArea, "Grid Opions")
        gridbox.setEnabled(self.chkB)
        OWGUI.checkBox(box, self, "chkA", "Verbose")
        OWGUI.checkBox(box, self, "chkB", "Display Grid", disables=[gridbox])
        OWGUI.spin(gridbox, self, "dx", 10, 20)
        
        self.resize(100,50)

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

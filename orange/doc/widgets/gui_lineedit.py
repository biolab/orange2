from OWWidget import *
import OWGUI

class Test(OWWidget):
    
    settingsList = ["val1", "val2", "valF"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, 'Check')

        self.val1 = "Enter text ..."
        self.val2 = "Some more text ..."
        self.valF = 10.2
        
        OWGUI.lineEdit(self.controlArea, self, "val1", box="Text Entry")
        box = QVGroupBox("Options (with lineEdit)", self.controlArea)
        OWGUI.lineEdit(box, self, "val2", 
                       label="Name:", orientation="horizontal", labelWidth=40)
        OWGUI.lineEdit(box, self, "valF", label="Float:",
                       orientation="horizontal", labelWidth=40, valueType=float)

        self.resize(100,100)

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    appl.setMainWidget(ow)
    ow.show()
    appl.exec_loop()

from OWWidget import *
import OWGUI

class Test(OWWidget):
    
    settingsList = ["spinval", "alpha", "beta"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, 'Spin')
        
        # GUI
        self.spinval = 10
        OWGUI.spin(self.controlArea, self, "spinval", 0, 100, box="Value A")
        box = OWGUI.widgetBox(self.controlArea, "Options")
        self.alpha = 30
        self.beta = 4
        OWGUI.spin(box, self, "alpha", 0, 100, label="Alpha:", labelWidth=60,
                   orientation="horizontal", callback=self.setInfo)
        OWGUI.spin(box, self, "beta", -10, 10, label="Beta:", labelWidth=60,
                   orientation="horizontal", callback=self.setInfo)

        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.info = OWGUI.widgetLabel(box, "")
        self.setInfo()
        
        self.resize(100,50)

    def setInfo(self):
        self.info.setText("Alpha=%d, Beta=%d" % (self.alpha, self.beta))

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

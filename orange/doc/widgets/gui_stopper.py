from OWWidget import *
import OWGUI

class OWXLearner(OWWidget):
    settingsList = ["method", "maxi", "cheat", "autoApply"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='X Learner')

        self.method= 0
        self.maxi = 1
        self.cheat = 0
        self.autoApply = True

        self.settingsChanged = False
        
        OWGUI.radioButtonsInBox(self.controlArea, self, "method",
                       ["Vanishing", "Disappearing", "Invisibilisation"],
                       box="Minimization technique", 
                       callback = self.applyIf)
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Settings")
        OWGUI.checkBox(box, self, "maxi", "Post-maximize", callback = self.applyIf)
        OWGUI.checkBox(box, self, "cheat", "Quasi-cheating", callback = self.applyIf)
        OWGUI.separator(self.controlArea)

        box = OWGUI.widgetBox(self.controlArea, "Apply")
        applyButton = OWGUI.button(box, self, "Apply", callback = self.apply)
        autoApplyCB = OWGUI.checkBox(box, self, "autoApply", "Apply automatically")

        OWGUI.setStopper(self, applyButton, autoApplyCB, "settingsChanged", self.apply)        
        
        self.adjustSize()

    def applyIf(self):
        if self.autoApply:
            self.apply()
        else:
            self.settingsChanged = True
            
    def apply(self):
        # Constructs and sends the learner; here we shall just show a message box
        QMessageBox.information(self, "Applied", "New settings applied")
        self.settingsChanged = False
        

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWXLearner()
    ow.show()
    appl.exec_()

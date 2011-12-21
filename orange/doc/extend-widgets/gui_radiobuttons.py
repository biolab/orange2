from OWWidget import *
import OWGUI

class Test(OWWidget):
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Listbox')

        self.method = 0

        OWGUI.radioButtonsInBox(self.controlArea, self, "method",
                      box = "Probability estimation",
                      btnLabels = ["Relative", "Laplace", "m-estimate"],
                      tooltips = ["Relative frequency of the event",
                                  "Laplace-corrected estimate",
                                  "M-estimate of probability"])
        self.adjustSize()

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

from OWWidget import *
import OWGUI

class Test(OWWidget):
    settingsList = ["colors", "chosenColor", "numbers", "chosenNumbers"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Listbox')

        self.colors = ["Red", "Green", "Blue"]
        self.chosenColor = [2]
        self.numbers = ["One", "Two", "Three", "Four"]
        self.chosenNumbers = [0, 2, 3]

        OWGUI.listBox(self.controlArea, self, "chosenColor", "colors", box="Color", callback=self.checkAll)
        OWGUI.listBox(self.controlArea, self, "chosenNumbers", "numbers", box="Number", selectionMode=QListWidget.MultiSelection, callback=self.checkAll)

        OWGUI.separator(self.controlArea)
        
        box = OWGUI.widgetBox(self.controlArea, "Debug info")
        OWGUI.label(box, self, "Color: %(chosenColor)s")
        OWGUI.label(box, self, "Numbers: %(chosenNumbers)s", labelWidth=100)

        self.setFixedSize(110, 280)

    def checkAll(self):
        if len(self.chosenNumbers) == len(self.numbers) and self.chosenColor != [0]:
            self.chosenColor = [0]

##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

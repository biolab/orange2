from OWWidget import *
import OWGUI

class Test(OWWidget):
    settingsList = ["colors", "chosenColor", "numbers", "chosenNumbers"]
    def __init__(self, parent=None):
        OWWidget.__init__(self, parent, title='Combo box')

        self.chosenColor = 1
        self.chosenAttribute = ""

        box = OWGUI.widgetBox(self.controlArea, "Color &  Attribute")
        OWGUI.comboBox(box, self, "chosenColor", label="Color: ", items=["Red", "Green", "Blue"])
        self.attrCombo = OWGUI.comboBox(box, self, "chosenAttribute", label="Attribute: ", sendSelectedValue = 1, emptyString="(none)")

        self.adjustSize()

        # Something like this happens later, in a function which receives an example table
        import orange
        self.data = orange.ExampleTable(r"..\datasets\horse-colic.tab")

        self.attrCombo.clear()
        self.attrCombo.addItem("(none)")
        icons = OWGUI.getAttributeIcons()
        for attr in self.data.domain:
            self.attrCombo.addItem(icons[attr.varType], attr.name)

        self.chosenAttribute = self.data.domain[0].name


##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = Test()
    ow.show()
    appl.exec_()

"""
<name>Attribute Sampler</name>
<description>Lets the user select a list of attributes and the class attribute</description>
<icon>icons/AttributeSampler.png</icon>
<priority>1020</priority>
"""

from OWWidget import *
import OWGUI

class OWAttributeSampler(OWWidget):
    settingsList = []
    contextHandlers = {"": DomainContextHandler("", [
            ContextField("classAttribute", DomainContextHandler.Required),
            ContextField("attributeList", DomainContextHandler.List + DomainContextHandler.SelectedRequired,
                         selected="selectedAttributes")])}

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'AttributeSampler')

        self.inputs = [("Examples", ExampleTable, self.dataset)]
        self.outputs = [("Examples", ExampleTable)]

        self.icons = self.createAttributeIconDict()

        self.attributeList = []
        self.selectedAttributes = []
        self.classAttribute = None
        self.loadSettings()

        OWGUI.listBox(self.controlArea, self, "selectedAttributes", "attributeList", box="Selected attributes", selectionMode = QListWidget.ExtendedSelection)
        OWGUI.separator(self.controlArea)
        self.classAttrCombo = OWGUI.comboBox(self.controlArea, self, "classAttribute", box="Class attribute")
        OWGUI.separator(self.controlArea)
        OWGUI.button(self.controlArea, self, "Commit", callback = self.outputData)

        self.resize(150,400)


    def dataset(self, data):
        self.closeContext()

        self.classAttrCombo.clear()
        if data:
            self.attributeList = [(attr.name, attr.varType) for attr in data.domain]
            self.selectedAttributes = []
            for attrName, attrType in self.attributeList:
                self.classAttrCombo.addItem(self.icons[attrType], attrName)
            self.classAttribute = 0
        else:
            self.attributeList = []
            self.selectedAttributes = []
            self.classAttrCombo.addItem("")

        self.openContext("", data)

        self.data = data
        self.outputData()


    def outputData(self):
        if not self.data:
            self.send("Examples", None)
        else:
            newDomain = orange.Domain([self.data.domain[i] for i in self.selectedAttributes], self.data.domain[self.classAttribute])
            newData = orange.ExampleTable(newDomain, self.data)
            self.send("Examples", newData)


##############################################################################
# Test the widget, run from prompt

if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWAttributeSampler()
    ow.show()

    data = orange.ExampleTable('../datasets/iris.tab')
    ow.dataset(data)

    appl.exec_()

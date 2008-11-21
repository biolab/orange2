"""
<name>Information</name>
<description>The Info Widget allows users to add text information into canvas schemas.</description>
<icon>icons/Info.png</icon>
<priority>10</priority>
"""

from OWWidget import *

class OWInfo(OWWidget):
    settingsList=["text"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Info Widget", wantMainArea=0)
        self.text = ""
        self.loadSettings()

        self.controlArea.setLayout(QVBoxLayout())
        self.layout().setMargin(2)
        self.textBox  = QPlainTextEdit(self)
        self.connect(self.textBox, SIGNAL("textChanged()"), self.textChanged)
        self.layout().addWidget(self.textBox)
        self.resize(500,300)

        self.activateLoadedSettings()

    def textChanged(self) :
        self.text = str(self.textBox.toPlainText())

    def activateLoadedSettings(self):
        if self.text != "":
            self.textBox.setPlainText(self.text)

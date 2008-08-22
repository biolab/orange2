"""
<name>Information</name>
<description>The Info Widget allows users to add text information into canvas schemas.</description>
<icon>icons/Info.png</icon>
<priority>10</priority>
"""

#
# OWInfo.py
# The Info Widget
# Add extra textual information into canvas schemas
#

from OWWidget import *

##################################################################################
# we have to catch keypress event and send it to its parent to save current text
class MyMultiLineEdit(QMultiLineEdit):
    def __init__(self, parent):
        QMultiLineEdit.__init__(self, parent)
        self.parent = parent

    def keyPressEvent(self, ev):
        QMultiLineEdit.keyPressEvent(self, ev)
        self.parent.keyPressEvent(ev)


##################################################################################
# OWINFO - allows users to save important information in schemas
class OWInfo(OWWidget):
    settingsList=["text"]

    def __init__(self, parent=None, signalManager = None):
        OWBaseWidget.__init__(self, parent, signalManager, "Info Widget")
        self.title = self.captionTitle = "Info Widget"

        #the title
        self.setCaption(self.captionTitle)
        self.caption = QLabel("Information:", self)
        self.controlArea = QVBoxLayout(self)
        self.textBox  = MyMultiLineEdit(self)
        self.closeButton=QPushButton("&Close", self)
        self.connect(self.closeButton,SIGNAL("clicked()"),self.close)
        #self.controlArea.addWidget(self.caption)
        self.controlArea.addWidget(self.textBox)
        self.controlArea.addWidget(self.closeButton)
        self.resize(200,200)

        self.linkBuffer={}

        self.text = ""
        self.loadSettings()
        self.activateLoadedSettings()

    def keyPressEvent (self, ev) :
        self.text = str(self.textBox.text())

    def activateLoadedSettings(self):
        if self.text != "": self.textBox.setText(self.text)

if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWInfo()
    owf.show()
    a.exec_()
    owf.saveSettings()

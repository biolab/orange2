"""
<name>Text File</name>
<description>Loads a bunch of  file</description>
<contact></contact>
<icon>icons/TextFile.png</icon>
<priority>100</priority>
"""

import orange
import OWGUI
from qt import *
from OWWidget import *

class OWTextFile(OWWidget):	
    settingsList = ["recentFiles"]

    def __init__(self, parent=None, signalManager = None, name='Text File'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name) 
        self.outputs = [("Example Table", ExampleTable)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.takeAttributeNames = False
        self.data = None
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Data File")
        hbox = OWGUI.widgetBox(box, orientation = 0)
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex", callback = self.loadFile)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        button.setMaximumWidth(25)

        self.adjustSize()            

        if self.recentFiles:
            self.loadFile()

                
    def browseFile(self):
        if self.recentFiles:
            lastPath = os.path.split(self.recentFiles[0])[0]
        else:
            lastPath = "."

        fn = str(QFileDialog.getOpenFileName(lastPath, "Text files (*.*)", None, "Open Text Files"))
        if not fn:
            return
        
        fn = os.path.abspath(fn)
        if fn in self.recentFiles: # if already in list, remove it
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.fileIndex = 0
        self.loadFile()


    def loadFile(self):
        if self.fileIndex:
            fn = self.recentFiles[self.fileIndex]
            self.recentFiles.remove(fn)
            self.recentFiles.insert(0, fn)
            self.fileIndex = 0
        else:
            fn = self.recentFiles[0]

        self.filecombo.clear()
        for file in self.recentFiles:
            self.filecombo.insertItem(os.path.split(file)[1])
        self.filecombo.updateGeometry()

        self.error()
        data = None
        try:
            import orngText
            if fn[-4:] == ".xml":
                data = orngText.loadFromXML(fn)
            elif fn[-4:] == ".sgm":
                data = orngText.loadReuters(os.path.split(fn)[0])
            else:
                data = orngText.loadFromListWithCategories(fn)

            if not data:
                self.error("Unknown file format or no documents")
        except:
            self.error("Cannot read the file")
        
        self.send("Example Table", data)


if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWTextFile()
    a.setMainWidget(ow)

    ow.show()
    a.exec_loop()
    ow.saveSettings()

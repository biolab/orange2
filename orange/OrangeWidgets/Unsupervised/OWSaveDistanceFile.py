"""
<name>Save Distance File</name>
<description>Saves a distance matrix to a file</description>
<contact>Miha Stajdohar</contact>
<icon>icons/SaveDistanceFile.png</icon>
<priority>1150</priority>
"""

import orange
import OWGUI
from OWWidget import *
import os.path
import pickle

class OWSaveDistanceFile(OWWidget):
    settingsList = ["recentFiles"]

    def __init__(self, parent=None, signalManager = None, name='Distance File'):
        self.callbackDeposit = [] # deposit for OWGUI callback functions
        OWWidget.__init__(self, parent, signalManager, name, wantMainArea = 0, resizingEnabled = 0)
        self.inputs = [("Distance Matrix", orange.SymMatrix, self.setData)]

        self.recentFiles=[]
        self.fileIndex = 0
        self.takeAttributeNames = False
        self.data = None
        self.loadSettings()

        box = OWGUI.widgetBox(self.controlArea, "Distance File")
        hbox = OWGUI.widgetBox(box, orientation = "horizontal")
        self.filecombo = OWGUI.comboBox(hbox, self, "fileIndex")
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(hbox, self, '...', callback = self.browseFile)
        
        fbox = OWGUI.widgetBox(self.controlArea, "Save")
        self.save = OWGUI.button(fbox, self, "Save current data", callback = self.saveFile)
        self.save.setDisabled(1)
        
        self.setFilelist()
        self.filecombo.setCurrentIndex(0)
        
        button.setMaximumWidth(25)
        self.adjustSize()
            
    def setData(self, data):
        self.data = data
        self.save.setDisabled(data == None)

    def browseFile(self):
        if self.data == None:
            return 
        
        if self.recentFiles:
            lastPath = self.recentFiles[0]
        else:
            lastPath = "."
            
        fn = str(QFileDialog.getSaveFileName(self, "Save Distance Matrix File", lastPath, "SymMatrix files (*.sym)\nAll files (*.*)"))
        
        if not fn or not os.path.split(fn)[1]:
            return
        
        self.addFileToList(fn)
        self.saveFile()
        
    def saveFile(self):
        self.error()
        if self.data is not None:
            combotext = str(self.filecombo.currentText())
            if combotext == "(none)":
                QMessageBox.information( None, "Error saving data", "Unable to save data. Select first a file name by clicking the '...' button.", QMessageBox.Ok + QMessageBox.Default)
                return
            filename = self.recentFiles[self.filecombo.currentIndex()]
            try:
                fp = open(filename, 'wb')
                pickle.dump(self.data, fp, -1)
                fp.close()
            except Exception, (errValue):
                self.error(str(errValue))
                return
            self.error()    
            
    def addFileToList(self, fn):
        if fn in self.recentFiles:
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.setFilelist()

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()

        if self.recentFiles:
            self.filecombo.addItems([os.path.split(file)[1] for file in self.recentFiles])
        else:
            self.filecombo.addItem("(none)")
            
if __name__=="__main__":
    import orange
    a = QApplication(sys.argv)
    ow = OWDistanceFile()
    ow.show()
    a.exec_()
    ow.saveSettings()

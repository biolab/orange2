"""
<name>File</name>
<description>The File Widget is used for selecting and opening data files.</description>
<icon>icons/File.png</icon>
<priority>10</priority>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import OWGUI, string, os.path

class OWFile(OWWidget):
    settingsList=["recentFiles"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "File Widget")

        self.inputs = []
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass)]
    
        #set default settings
        self.recentFiles=["(none)"]
        #get settings from the ini file, if they exist
        self.loadSettings()
        
        #GUI
        self.box = QHGroupBox("Data File", self.controlArea)
        self.filecombo=QComboBox(self.box)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(self.box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

        # info
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel('No data loaded.', box)
        self.infob = QLabel('', box)
            
        self.resize(150,100)


    def activateLoadedSettings(self):
        # remove missing data set names
        self.recentFiles=filter(os.path.exists,self.recentFiles)
        self.setFileList()
        
        if len(self.recentFiles) > 0 and os.path.exists(self.recentFiles[0]):
            self.openFile(self.recentFiles[0])
            
        # connecting GUI to code
        self.connect(self.filecombo, SIGNAL('activated(int)'), self.selectFile)

    # user selected a file from the combo box
    def selectFile(self,n):
        if n < len(self.recentFiles) :
            name = self.recentFiles[n]
            self.recentFiles.remove(name)
            self.recentFiles.insert(0, name)
        if len(self.recentFiles) > 0:
            self.setFileList()
            self.openFile(self.recentFiles[0])

    # user pressed the "..." button to manually select a file to load
    def browseFile(self):
        "Display a FileDialog and select a file"
        if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
            startfile="."
        else:
            startfile=self.recentFiles[0]
        filename = str(QFileDialog.getOpenFileName(startfile,
        'Tab-delimited files (*.tab *.txt)\nC4.5 files (*.data)\nAssistant files (*.dat)\nRetis files (*.rda *.rdo)\nAll files(*.*)',
        None,'Open Orange Data File'))
    
        if filename == "": return
        if filename in self.recentFiles: self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)
        self.setFileList()
        self.openFile(self.recentFiles[0])

    # set the file combo box
    def setFileList(self):
        self.filecombo.clear()
        for file in self.recentFiles:
            if file == "(none)": self.filecombo.insertItem("(none)")
            else:                self.filecombo.insertItem(os.path.split(file)[1])
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.filecombo.updateGeometry()

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)            

    # Open a file, create data from it and send it over the data channel
    def openFile(self,fn):
        if fn != "(none)":
            fileExt=lower(os.path.splitext(fn)[1])
            if fileExt in (".txt",".tab",".xls"):
                data = orange.ExampleTable(fn)
            elif fileExt in (".c45",):
                data = orange.C45ExampleGenerator(fn)
            else:
                return
                
            # update data info
            def sp(l):
                n = len(l)
                if n <> 1: return n, 's'
                else: return n, ''

            self.infoa.setText('%d example%s, ' % sp(data) + '%d attribute%s, ' % sp(data.domain.attributes) + '%d meta attribute%s.' % sp(data.domain.getmetas()))
            cl = data.domain.classVar
            if cl:
                if cl.varType == orange.VarTypes.Continuous:
                    self.infob.setText('Regression; Numerical class.')
                elif cl.varType == orange.VarTypes.Discrete:
                    self.infob.setText('Classification; Discrete class with %d value%s.' % sp(cl.values))
                else:
                    self.infob.setText("Class neither descrete nor continuous.")
            else:
                self.infob.setText('Classless domain')

            # make new data and send it
            data.name = string.split(os.path.split(fn)[1], '.')[0]
            self.send("Examples", data)
            if data.domain.classVar:
                self.send("Classified Examples", data)
        else:
            self.send("Classified Examples", None)
            self.send("Examples", None)
        
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFile()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

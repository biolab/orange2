"""
<name>File</name>
<description>The File Widget is used for selecting and opening data files.</description>
<category>Data</category>
<icon>icons/File.png</icon>
<priority>10</priority>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
from OData import *
import OWGUI

class OWFile(OWWidget):
    settingsList=["recentFiles","selectedFileName"]

    def __init__(self,parent=None):
        OWWidget.__init__(self,parent,"&File Widget",
        "The File Widget is an Orange Widget\nused for selecting and opening data files.",
        FALSE)
        "Constructor"

        self.inputs = []
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass)]
    
        #set default settings
        self.recentFiles=[]
        self.selectedFileName = "None"
        #get settings from the ini file, if they exist
        self.loadSettings()
        
        #GUI
        box = QHGroupBox("Data File", self.controlArea)
        self.filecombo=QComboBox(box)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

        # info
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel('No data loaded.', box)
        self.infob = QLabel('', box)
            
        self.recentFiles=filter(os.path.exists,self.recentFiles)
        self.setFilelist()
        self.filecombo.setCurrentItem(0)
        # this makes no difference, because when the file widget is created there are no connection yet
        if self.recentFiles!=[]:
            self.openFile(self.recentFiles[0])
        
        # connecting GUI to code
        self.connect(self.filecombo,SIGNAL('activated(int)'),self.selectFile)
        self.resize(150,100)

    def browseFile(self):
        "Display a FileDialog and select a file"
        if self.recentFiles==[]:
            startfile="."
        else:
            startfile=self.recentFiles[0]
        filename=QFileDialog.getOpenFileName(startfile,
        'Tab-delimited files (*.tab *.txt)\nC4.5 files (*.data)\nAssistant files (*.dat)\nRetis files (*.rda *.rdo)\nAll files(*.*)',
        None,'Open Orange Data File')
        self.openFile(str(filename))

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)            

    def xsp(self, l):
        n = len(l)
        if n>1: return n, 's'
        else: return n, ''
        
    def openFile(self,fn):
        "Open a file, create data from it and send it over the data channel"
        if fn!="(none)":
            fileExt=lower(os.path.splitext(fn)[1])
            if fileExt in (".txt",".tab",".xls"):
                data = orange.ExampleTable(fn, dontCheckStored = 1)
            elif fileExt in (".c45",):
                data = orange.C45ExampleGenerator(fn)
            else:
                return
                
            # update recent file list
            self.addFileToList(fn)

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
                    self.infob.setText("This won't work. Class neither descrete nor continuous.")
            else:
                self.infob.setText('Classless domain')

            # set setting
            self.selectedFileName = fn
            
            # make new data and send it
            data.name = fn
            self.send("Examples", data)
            if data.domain.classVar:
                    self.send("Classified Examples", data)
        else:
            self.send("Classified Examples",None)
            self.send("Examples", None)

    def addFileToList(self,fn):
        # Add a file to the start of the file list. 
        # If it exists, move it to the start of the list
        if fn in self.recentFiles:
            self.recentFiles.remove(fn)
        self.recentFiles.insert(0, fn)
        self.setFilelist()       

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()
        if self.recentFiles!=[]:
            for file in self.recentFiles:
                (dir,filename)=os.path.split(file)
                #leave out the path
                self.filecombo.insertItem(filename)
        else:
            self.filecombo.insertItem("(none)")
        self.filecombo.adjustSize() #doesn't work properly :(
            
    def selectFile(self,n):
        "Slot that is called when a file is selected from the combo box"
        if self.recentFiles:
            self.openFile(self.recentFiles[n])
        else:
            self.openFile("(none)")
        
    def activateLoadedSettings(self):
        if self.selectedFileName != "":
            if os.path.exists(self.selectedFileName):
                self.openFile(self.selectedFileName)
            else:
                self.selectedFileName = ""
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFile()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

"""
<name>File</name>
<description>The File Widget is used for selecting and opening data files.</description>
<category>Input</category>
<icon>icons\File.png</icon>
"""

#
# OWFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
from OData import *

class OWFile(OWWidget):
    def __init__(self,parent=None):
        OWWidget.__init__(self,parent,"&File Widget",
        "The File Widget is an Orange Widget\nused for selecting and opening data files.",
        FALSE)
        "Constructor"        
        self.settingsList=["RecentFiles","selectedFileName"]
        #set default settings
        self.RecentFiles=[]
        self.selectedFileName = "None"
        #get settings from the ini file, if they exist
        self.loadSettings()
        
        #add the outputs
        self.addOutput("data")
               
        #GUI
        vb=QGridLayout(self.mainArea,3)
        rfbox=QVGroupBox("Recent Files",self.mainArea)
        self.filecombo=QComboBox(rfbox)
        fbox=QVGroupBox("New File",self.mainArea)
        browse=QPushButton("&Browse...",fbox)
        vb.addWidget(rfbox,0,0)
        vb.addWidget(fbox,0,1)
        x=QWidget(self.mainArea)
#        y=QWidget(self.mainArea)
        vb.addWidget(x,1,0)
        vb.addWidget(x,0,2)
        vb.setRowStretch(1,10)
        vb.setColStretch(2,10)
        self.adjustSize()
#        self.setFixedSize(self.size())
        
        #check if files actually exist
        self.RecentFiles=filter(os.path.exists,self.RecentFiles)
                
        #fill the file list
        self.setFilelist()
        
        #
        self.filecombo.setCurrentItem(0)
        #this makes no difference, because when the file widget is created there are no connection yet
        if self.RecentFiles!=[]:
            self.openFile(self.RecentFiles[0])
        
        #connecting GUI to code
        self.connect(browse,SIGNAL('clicked()'),self.browseFile)        
        self.connect(self.filecombo,SIGNAL('activated(int)'),self.selectFile)
    
    def browseFile(self):
        "Display a FileDialog and select a file"
        if self.RecentFiles==[]:
            startfile="."
        else:
            startfile=self.RecentFiles[0]
        filename=QFileDialog.getOpenFileName(startfile,
        'Tab-delimited files (*.tab *.txt)\nC4.5 files (*.data)\nAssistant files (*.dat)\nRetis files (*.rda *.rdo)\nAll files(*.*)',
        None,'Open Orange Data File')
        self.openFile(str(filename))

    def openFile(self,fn):
        "Open a file, create data from it and send it over the data channel"
        if fn!="(none)":
            fileExt=lower(os.path.splitext(fn)[1])
            if fileExt in (".txt",".tab"):
                tab=orange.TabDelimExampleGenerator(fn)
            elif fileExt in (".c45",):
                tab=orange.C45ExampleGenerator(fn)
            else:
                return
                
            #update recent file list
            self.addFileToList(fn)

            #set setting
            self.selectedFileName = fn
            
            #make new data and send it
            data=OrangeData(tab)
            data.title = fn
            self.send("data",data)
        else:
            self.send("data",None)

    def addFileToList(self,fn):
        """
        Add a file to the start of the file list. 
        If it exists, move it to the start of the list
        """
        if fn in self.RecentFiles:
            self.RecentFiles.remove(fn)
        self.RecentFiles.insert(0,fn)
        self.setFilelist()       

    def setFilelist(self):
        "Set the GUI filelist"
        self.filecombo.clear()
        if self.RecentFiles!=[]:
            for file in self.RecentFiles:
                (dir,filename)=os.path.split(file)
                #leave out the path
                self.filecombo.insertItem(filename)
        else:
            self.filecombo.insertItem("(none)")
        self.filecombo.adjustSize() #doesn't work properly :(
            
    def selectFile(self,n):
        "Slot that is called when a file is selected from the combo box"
        if self.RecentFiles:
            self.openFile(self.RecentFiles[n])
        else:
            self.openFile("(none)")
        
    def activateLoadedSettings(self):
        if self.selectedFileName != "":
            self.openFile(self.selectedFileName)
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFile()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

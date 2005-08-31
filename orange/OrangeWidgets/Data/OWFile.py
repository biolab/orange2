"""
<name>File</name>
<description>Reads data from a file.</description>
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

        self.inputs = [("Attribute Definitions", orange.Domain, self.setDomain)]
        self.outputs = [("Examples", ExampleTable), ("Classified Examples", ExampleTableWithClass), ("Attribute Definitions", orange.Domain)]
    
        #set default settings
        self.recentFiles=["(none)"]
        self.resetDomain = 0
        self.domain = None
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
            
        self.rbox = QHGroupBox("Reload", self.controlArea)
        self.reloadBtn = OWGUI.button(self.rbox, self, "Reload File", callback = self.reload)
        OWGUI.separator(self.rbox, 8, 0)
        self.resetDomainCb = OWGUI.checkBox(self.rbox, self, "resetDomain", "Reset domain at next reload")

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
        elif n:
            self.browseFile(1)
            
        if len(self.recentFiles) > 0:
            self.setFileList()
            self.openFile(self.recentFiles[0])

    # user pressed the "..." button to manually select a file to load
    def browseFile(self, inDemos=0):
        "Display a FileDialog and select a file"
        if inDemos:
            try:
                import win32api, win32con
                t = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, "SOFTWARE\\Python\\PythonCore\\%i.%i\\PythonPath\\Orange" % sys.version_info[:2], 0, win32con.KEY_READ)
                t = win32api.RegQueryValueEx(t, "")[0]
                startfile = t[:t.find("orange")] + "orange\\doc\\datasets"
            except:
                QMessageBox.information( None, "File", "Cannot find the directory with example data sets", QMessageBox.Ok + QMessageBox.Default)
                return
        else:
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
        if not self.recentFiles:
            self.filecombo.insertItem("(none)")
            self.reloadBtn.disabled = 1
        for file in self.recentFiles:
            if file == "(none)":
                self.filecombo.insertItem("(none)")
            else:
                self.filecombo.insertItem(os.path.split(file)[1])
                self.reloadBtn.disabled = 0
        self.filecombo.insertItem("Browse documentation data sets...")
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.filecombo.updateGeometry()

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)            

    def reload(self):
        if self.recentFiles:
            return self.openFile(self.recentFiles[0], 1)

    # Open a file, create data from it and send it over the data channel
    def openFile(self,fn, throughReload = 0):
        dontCheckStored = throughReload and self.resetDomain
        self.resetDomain = self.domain != None
        if fn != "(none)":
            fileExt=lower(os.path.splitext(fn)[1])
            if fileExt in (".txt",".tab",".xls"):
                data = orange.ExampleTable(fn, dontCheckStored = dontCheckStored, use = self.domain)
            elif fileExt in (".c45",):
                data = orange.C45ExampleGenerator(fn, dontCheckStored = dontCheckStored, use = self.domain)
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
                self.send("Attribute Definitions", data.domain)
        else:
            self.send("Classified Examples", None)
            self.send("Examples", None)
            self.send("Attribute Definitions", None)
        
    def setDomain(self, domain):
        domainChanged = self.domain != domain
        self.domain = domain
        
        if self.domain:
            self.resetDomainCb.setDisabled(1)
        else:
            self.resetDomainCb.setDisabled(0)
            
        if domainChanged and len(self.recentFiles) > 0 and os.path.exists(self.recentFiles[0]):
            self.resetDomain = 1
            self.openFile(self.recentFiles[0], 1)
            
    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWFile()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

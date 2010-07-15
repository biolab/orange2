"""
<name>Network File</name>
<description>Reads data from a graf file (Pajek networks (.net) files and GML network files).</description>
<icon>icons/NetworkFile.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3100</priority>
"""

#
# OWNetworkFile.py
# The Network File Widget
# A widget for opening Network related files
#
import OWGUI, string, os.path, user, sys
import orngNetwork

from OWWidget import *
from orange import Graph
from orange import GraphAsList

class OWNetworkFile(OWWidget):
    
    settingsList=["recentFiles", "recentDataFiles", "recentEdgesFiles"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Network File")

        self.inputs = []
        self.outputs = [("Network", orngNetwork.Network), ("Items", ExampleTable)]
    
        #set default settings
        self.recentFiles = ["(none)"]
        self.recentDataFiles = ["(none)"]
        self.recentEdgesFiles = ["(none)"]
        
        self.domain = None
        self.graph = None
        #get settings from the ini file, if they exist
        self.loadSettings()

        #GUI
        self.controlArea.layout().setMargin(4)
        self.box = OWGUI.widgetBox(self.controlArea, box = "Graph File", orientation = "horizontal")
        self.filecombo = OWGUI.comboBox(self.box, self, "filename")
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(self.box, self, '...', callback = self.browseNetFile, disabled=0, width=25)
        
        self.databox = OWGUI.widgetBox(self.controlArea, box = "Vertices Data File", orientation = "horizontal")
        self.datacombo = OWGUI.comboBox(self.databox, self, "dataname")
        self.datacombo.setMinimumWidth(250)
        button = OWGUI.button(self.databox, self, '...', callback = self.browseDataFile, disabled=0, width=25)
        
        self.edgesbox = OWGUI.widgetBox(self.controlArea, box = "Edges Data File", orientation = "horizontal")
        self.edgescombo = OWGUI.comboBox(self.edgesbox, self, "edgesname")
        self.edgescombo.setMinimumWidth(250)
        button = OWGUI.button(self.edgesbox, self, '...', callback = self.browseEdgesFile, disabled=0, width=25)
        
        # info
        box = OWGUI.widgetBox(self.controlArea, "Info")
        self.infoa = OWGUI.widgetLabel(box, 'No data loaded.')
        self.infob = OWGUI.widgetLabel(box, ' ')
        self.infoc = OWGUI.widgetLabel(box, ' ')
        self.infod = OWGUI.widgetLabel(box, ' ')

        self.resize(150,100)
        self.activateLoadedSettings()

        # connecting GUI to code
        self.connect(self.filecombo, SIGNAL('activated(int)'), self.selectNetFile)
        self.connect(self.datacombo, SIGNAL('activated(int)'), self.selectDataFile)
        self.connect(self.edgescombo, SIGNAL('activated(int)'), self.selectEdgesFile)
        
    # set the comboboxes
    def setFileLists(self):
        self.filecombo.clear()
        if not self.recentFiles:
            self.filecombo.addItem("(none)")
        for file in self.recentFiles:
            if file == "(none)":
                self.filecombo.addItem("(none)")
            else:
                self.filecombo.addItem(os.path.split(file)[1])
        self.filecombo.addItem("Browse documentation data sets...")
        
        self.datacombo.clear()
        if not self.recentDataFiles:
            self.datacombo.addItem("(none)")
        for file in self.recentDataFiles:
            if file == "(none)":
                self.datacombo.addItem("(none)")
            else:
                self.datacombo.addItem(os.path.split(file)[1])
                
        self.edgescombo.clear()
        if not self.recentEdgesFiles:
            self.edgescombo.addItem("(none)")
        for file in self.recentEdgesFiles:
            if file == "(none)":
                self.edgescombo.addItem("(none)")
            else:
                self.edgescombo.addItem(os.path.split(file)[1])
            
        self.filecombo.updateGeometry()
        self.datacombo.updateGeometry()
        self.edgescombo.updateGeometry()
     
    def activateLoadedSettings(self):
        # remove missing data set names
        self.recentFiles = filter(os.path.exists, self.recentFiles)
        self.recentDataFiles = filter(os.path.exists, self.recentDataFiles)
        self.recentEdgesFiles = filter(os.path.exists, self.recentEdgesFiles)
        
        self.recentFiles.append("(none)")
        self.recentDataFiles.append("(none)")
        self.recentEdgesFiles.append("(none)")
        self.setFileLists()
        
        if len(self.recentFiles) > 0 and os.path.exists(self.recentFiles[0]):
            self.selectNetFile(0)

        if len(self.recentDataFiles) > 1 and os.path.exists(self.recentDataFiles[1]):
            self.selectDataFile(1)
            
        if len(self.recentEdgesFiles) > 1 and os.path.exists(self.recentEdgesFiles[1]):
            self.selectEdgesFile(1)
        
    # user selected a graph file from the combo box
    def selectNetFile(self, n):
        if n < len(self.recentFiles) :
            name = self.recentFiles[n]
            self.recentFiles.remove(name)
            self.recentFiles.insert(0, name)
        elif n:
            self.browseNetFile(1)
            
        if len(self.recentFiles) > 0:
            self.setFileLists()  
            self.openFile(self.recentFiles[0])
    
    # user selected a data file from the combo box
    def selectEdgesFile(self, n):
        if n < len(self.recentEdgesFiles) :
            name = self.recentEdgesFiles[n]
            self.recentEdgesFiles.remove(name)
            self.recentEdgesFiles.insert(0, name)

        if len(self.recentEdgesFiles) > 0:
            self.setFileLists()
            self.addEdgesFile(self.recentEdgesFiles[0])
    
    def selectDataFile(self, n):
        if n < len(self.recentDataFiles) :
            name = self.recentDataFiles[n]
            self.recentDataFiles.remove(name)
            self.recentDataFiles.insert(0, name)

        if len(self.recentDataFiles) > 0:
            self.setFileLists()
            self.addDataFile(self.recentDataFiles[0])
    
    def openFile(self, fn):
        """Read network from file."""
        
        # read network file
        if fn != "(none)":
            fileExt = lower(os.path.splitext(fn)[1])
            if not fileExt in (".net", ".gml"):
                self.graph = None
                self.send("Network", None)
                self.send("Items", None)
                self.infoa.setText('No data loaded')
                self.infob.setText('File extension does not match')
                self.infoc.setText('')            
                self.infod.setText('')
                return
            
            data = self.readNetFile(fn)
            
            if data == None:
                self.graph = None
                self.send("Network", None)
                self.send("Items", None)
                self.infoa.setText('No data loaded')
                self.infob.setText('Error reading file')
                self.infoc.setText('')            
                self.infod.setText('')
                return

            self.infoa.setText("%d nodes" % data.nVertices)
            
            if data.directed:
                self.infob.setText("Directed graph")
            else:
                self.infob.setText("Undirected graph")
            
            # make new data and send it
            fName = os.path.split(fn)[1]
            if "." in fName:
                #data.name = string.join(string.split(fName, '.')[:-1], '.')
                pass
            else:
                #data.name = fName
                pass
                
            self.graph = data
        else:
            self.graph = None
            self.send("Network", None)
            self.send("Items", None)
            self.infoa.setText('No data loaded')
            self.infob.setText('')
            self.infoc.setText('')            
            self.infod.setText('')
        
        # Find items data file for selected network
        items_candidate = os.path.splitext(fn)[0] + ".tab"
        if os.path.exists(items_candidate):
            self.readDataFile(items_candidate)
            self.recentDataFiles.insert(0, items_candidate)
        elif os.path.exists(os.path.splitext(fn)[0] + "_items.tab"):
            items_candidate = os.path.splitext(fn)[0] + "_items.tab"
            self.readDataFile(items_candidate)
            self.recentDataFiles.insert(0, items_candidate)
        else:
            if '(none)' in self.recentDataFiles: 
                self.recentDataFiles.remove('(none)')
            self.recentDataFiles.insert(0, '(none)')
        
        # Find links data file for selected network
        links_candidate = os.path.splitext(fn)[0] + "_links.tab" 
        if os.path.exists(links_candidate):
            self.readEdgesFile(links_candidate)
            self.recentEdgesFiles.insert(0, links_candidate)
        else:
            if '(none)' in self.recentEdgesFiles: 
                self.recentEdgesFiles.remove('(none)')
            self.recentEdgesFiles.insert(0, '(none)')
        
        self.setFileLists()
        
        self.send("Network", self.graph)
        if self.graph != None and self.graph.items != None:
            self.send("Items", self.graph.items)
        else:
            self.send("Items", None)
        
    def addDataFile(self, fn):
        if fn == "(none)" or self.graph == None:
            self.infoc.setText("No vertices data file specified")
            self.send("Network", None)
            self.send("Items", None)
            return
         
        self.readDataFile(fn)
        
        self.send("Network", self.graph)
        self.send("Items", self.graph.items)
        
    def readDataFile(self, fn):
        table = ExampleTable(fn)
        
        if len(table) != self.graph.nVertices:
            self.infoc.setText("Vertices data length does not match number of vertices")
            
            if '(none)' in self.recentDataFiles: 
                self.recentDataFiles.remove('(none)')
                
            self.recentDataFiles.insert(0, '(none)')
            self.setFileLists()
            return
        
        self.graph.setattr("items", table)
        self.infoc.setText("Vertices data file added")
        
    def addEdgesFile(self, fn):
        if fn == "(none)" or self.graph == None:
            self.infod.setText("No edges data file specified")
            #self.graph.setattr("links", None)
            self.send("Network", self.graph)
            self.send("Items", None)
            return
        
        self.readEdgesFile(fn)
        
        self.send("Network", self.graph)
        self.send("Items", self.graph.items)
        
    def readEdgesFile(self, fn):
        table = ExampleTable(fn)
        if self.graph.directed:
            nEdges = len(self.graph.getEdges())
        else:
            nEdges = len(self.graph.getEdges()) / 2
            
        if len(table) != nEdges:
            self.infod.setText("Edges data length does not match number of edges")
            
            if '(none)' in self.recentEdgesFiles: 
                self.recentEdgesFiles.remove('(none)')
                
            self.recentEdgesFiles.insert(0, '(none)')
            self.setFileLists()
            return
        
        self.graph.setattr("links", table)
        self.infod.setText("Edges data file added")
        
    def browseNetFile(self, inDemos=0):
        """user pressed the '...' button to manually select a file to load"""
        
        "Display a FileDialog and select a file"
        if inDemos:
            import os
            try:
                import orngConfiguration
                startfile = orngConfiguration.datasetsPath
            except:
                startfile = ""
                
            if not startfile or not os.path.exists(startfile):
                try:
                    import win32api, win32con
                    t = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, "SOFTWARE\\Python\\PythonCore\\%i.%i\\PythonPath\\Orange" % sys.version_info[:2], 0, win32con.KEY_READ)
                    t = win32api.RegQueryValueEx(t, "")[0]
                    startfile = t[:t.find("orange")] + "orange\\doc\\networks"
                except:
                    startfile = ""

            if not startfile or not os.path.exists(startfile):
                d = OWGUI.__file__
                if d[-8:] == "OWGUI.py":
                    startfile = d[:-22] + "doc/networks"
                elif d[-9:] == "OWGUI.pyc":
                    startfile = d[:-23] + "doc/networks"

            if not startfile or not os.path.exists(startfile):
                d = os.getcwd()
                if d[-12:] == "OrangeCanvas":
                    startfile = d[:-12]+"doc/networks"
                else:
                    if d[-1] not in ["/", "\\"]:
                        d+= "/"
                    startfile = d+"doc/networks"

            if not os.path.exists(startfile):
                QMessageBox.information( None, "File", "Cannot find the directory with example networks", QMessageBox.Ok + QMessageBox.Default)
                return
        else:
            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
                if sys.platform == "darwin":
                    startfile = user.home
                else:
                    startfile = "."
            else:
                startfile = self.recentFiles[0]
                
        filename = str(QFileDialog.getOpenFileName(self, 'Open a Network File', startfile, "Pajek files (*.net)\nGML files (*.gml)\nAll files (*.*)"))
        
        if filename == "": return
        if filename in self.recentFiles: self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)
        self.setFileLists()
        self.selectNetFile(0)
        
    def browseDataFile(self, inDemos=0):
        if self.graph == None:
            return
        
        #Display a FileDialog and select a file
        if len(self.recentDataFiles) == 0 or self.recentDataFiles[0] == "(none)":
            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
                if sys.platform == "darwin":
                    startfile = user.home
                else:
                    startfile="."
            else:
                startfile = os.path.dirname(self.recentFiles[0])
                
        else:
            startfile = self.recentDataFiles[0]
                
        filename = str(QFileDialog.getOpenFileName(self, 'Open a Vertices Data File', startfile, 'Data files (*.tab)\nAll files(*.*)'))
    
        if filename == "": return
        if filename in self.recentDataFiles: self.recentDataFiles.remove(filename)
        self.recentDataFiles.insert(0, filename)
        self.setFileLists()
        self.addDataFile(self.recentDataFiles[0])
        
    def browseEdgesFile(self, inDemos=0):
        if self.graph == None:
            return
        
        #Display a FileDialog and select a file
        if len(self.recentEdgesFiles) == 0 or self.recentEdgesFiles[0] == "(none)":
            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
                if sys.platform == "darwin":
                    startfile = user.home
                else:
                    startfile="."
            else:
                startfile = os.path.dirname(self.recentFiles[0])
                
        else:
            startfile = self.recentEdgesFiles[0]
                
        filename = str(QFileDialog.getOpenFileName(self, 'Open a Edges Data File', startfile, 'Data files (*.tab)\nAll files(*.*)'))
    
        if filename == "": return
        if filename in self.recentEdgesFiles: self.recentEdgesFiles.remove(filename)
        self.recentEdgesFiles.insert(0, filename)
        self.setFileLists()
        self.addEdgesFile(self.recentEdgesFiles[0])

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)            

    def readNetFile(self, fn):
        try:
            net = orngNetwork.Network.read(fn)
            self.infoc.setText("Vertices data generated and added automatically")
        except:
            self.infoa.setText("Could not read file")
            self.infob.setText("")
            self.infoc.setText("")
            self.infod.setText("")
            return None
        
        return net

    def sendReport(self):
        self.reportSettings("Network file",
                            [("File name", self.filecombo.currentText()),
                             ("Vertices", self.graph.nVertices),
                             hasattr(self.graph, "directed") and ("Directed", OWGUI.YesNo[self.graph.directed])])
        self.reportSettings("Vertices meta data", [("File name", self.datacombo.currentText())])
        self.reportData(self.graph.items, None)
        self.reportSettings("Edges meta data", [("File name", self.edgescombo.currentText())])
        self.reportData(self.graph.links, None, None)
        
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFile()
    owf.activateLoadedSettings()
    owf.show()  
    a.exec_()
    owf.saveSettings()

"""
<name>Network File</name>
<description>Reads data from a graf file (Pajek networks (.net) files).</description>
<icon>icons/NetworkFile.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact>
<priority>2010</priority>
"""

#
# OWGrafFile.py
# The File Widget
# A widget for opening orange data files
#
import OWGUI, string, os.path, user, sys

from OWWidget import *
from orangeom import *

from orange import Graph
from orange import GraphAsList
from orange import ExampleTable

class OWNetworkFile(OWWidget):

    settingsList=["recentFiles"]

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Network File")

        self.inputs = []
        self.outputs = [("Graph with ExampleTable", Graph)]

        #set default settings
        self.recentFiles=["(none)"]
        self.domain = None
        #get settings from the ini file, if they exist
        self.loadSettings()

        #GUI
        self.box = QHGroupBox("Data File", self.controlArea)
        self.filecombo = QComboBox(self.box)
        self.filecombo.setMinimumWidth(250)
        button = OWGUI.button(self.box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

        # info
        box = QVGroupBox("Info", self.controlArea)
        self.infoa = QLabel('No data loaded.', box)
        self.infob = QLabel('', box)

        self.resize(150,100)

        # set the file combo box
    def setFileList(self):
        self.filecombo.clear()
        if not self.recentFiles:
            self.filecombo.addItem("(none)")
        for file in self.recentFiles:
            if file == "(none)":
                self.filecombo.addItem("(none)")
            else:
                self.filecombo.addItem(os.path.split(file)[1])
        #self.filecombo.insertItem("Browse documentation data sets...")

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


    def openFile(self,fn):
        self.openFileBase(fn)

    # user pressed the "..." button to manually select a file to load
    def browseFile(self, inDemos=0):
        "Display a FileDialog and select a file"
        if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
            if sys.platform == "darwin":
                startfile = user.home
            else:
                startfile="."
        else:
            startfile=self.recentFiles[0]

        filename = str(QFileDialog.getOpenFileName(startfile, 'Pajek files (*.net)\nAll files(*.*)', None,'Open a Graph File'))

        if filename == "": return
        if filename in self.recentFiles: self.recentFiles.remove(filename)
        self.recentFiles.insert(0, filename)
        self.setFileList()
        self.openFile(self.recentFiles[0])

    def setInfo(self, info):
        for (i, s) in enumerate(info):
            self.info[i].setText(s)

     # Open a file, create data from it and send it over the data channel
    def openFileBase(self, fn):
        if fn != "(none)":
            fileExt = lower(os.path.splitext(fn)[1])
            if fileExt in (".net"):
                pass
            else:
                return

            data = self.readNetFile(fn)

            self.infoa.setText("%d nodes" % data.nVertices)

            if data.directed:
                self.infob.setText("Directed graph")
            else:
                self.infob.setText("Undirected graph")

            # make new data and send it
            fName = os.path.split(fn)[1]
            if "." in fName:
                data.name = string.join(string.split(fName, '.')[:-1], '.')
            else:
                data.name = fName

            #print "nVertices graph: " + str(data.nVertices)
            self.graph = data
            self.send("Graph with ExampleTable", data)
#            drawer = OWGraphDrawer()
#            drawer.setGraph(data)
#            drawer.show()
        else:
            print "None"
            self.send("Graph with ExampleTable", None)

    #vrstica je ali '' ali pa nek niz, ki se zakljuci z \n (newline)
    def getwords(self, line):
        WHITESPACE=['\t','\n','\r','\f','\v',' ']
        words=[]
        word=''

        if line=='':          #ce je konec datoteke
            return ''  #words

        i=0
        done=False
        while not done:
            try:
                while line[i] in WHITESPACE:   #preskok presledkov
                    if line[i]=='\n':
                        done=True
                        break
                    else:
                        i=i+1
            except IndexError:
                return 'EOF'           #vrnemo tak niz, da bo v nadrejeni proceduri napaka

            if done==True:
                break

            if line[i]!='\"' and line[i]!='\'':              #obicajne besede
                try:
                    while line[i] not in WHITESPACE:
                        word=word+line[i]
                        i=i+1
                except IndexError:
                    return 'EOF'           #vrnemo tak niz, da bo v nadrejeni proceduri napaka
                words.append(word)
                word=''
            else:
                i=i+1
                while line[i]!='\"' and line[i]!='\'':      #imena v narekovajih
                    if line[i]=='\n':
                        return ''                #ime brez zakljucnega " ali '
                    else:
                        word=word+line[i]
                        i=i+1
                i=i+1
                words.append(word)
                word=''

        return words

    def readNetFile(self, fn):
#        try:
#            graphFile = file(fn, 'r')
#        except IOError:
#            return None

        graph, table = readNetwork(fn)
        #print "table: " + str(len(table))
        graph.setattr("items", table)
        #print "prebral"
        return graph


if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFile()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

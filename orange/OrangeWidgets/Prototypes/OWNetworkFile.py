"""
<name>Network File</name>
<description>Reads data from a graf file (Pajek networks (.net) files).</description>
<icon>icons/File.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>2010</priority>
"""

#
# OWGrafFile.py
# The File Widget
# A widget for opening orange data files
#

from OWWidget import *
import OWGUI, string, os.path
from orange import Graph
#from OWGraphDrawer import *

class OWNetworkFile(OWWidget):
    
    settingsList=["recentFiles"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "GraphFile")

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
            self.filecombo.insertItem("(none)")
        for file in self.recentFiles:
            if file == "(none)":
                self.filecombo.insertItem("(none)")
            else:
                self.filecombo.insertItem(os.path.split(file)[1])
        #self.filecombo.insertItem("Browse documentation data sets...")
        #self.filecombo.adjustSize() #doesn't work properly :(
        self.filecombo.updateGeometry()
     
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
        try:
            graphFile = file(fn, 'r')
        except IOError:
            return None

        lineNum = 0     
        nVertices = 0   
        graphName = ""
        #######################################################################
        #branje glave datoteke
        while True:
            line = graphFile.readline()
            lineNum = lineNum + 1

            words = self.getwords(line)

            if words == '':
                #raise InputSyntaxErrors
                raise
            elif words == []:
                continue
            elif lower(words[0]) == '*network':
                graphName = '"' + str(words[1]) + '"'
            elif lower(words[0]) == '*vertices':
                try:
                    nVertices = int(words[1])
                except ValueError:
                    #raise InputSyntaxError
                    raise
                if nVertices <= 0:
                    #raise InputSyntaxError
                    raise
                break
            
        #print "nVertices: " + str(nVertices)
        graph = orange.GraphAsList(nVertices, 0)
        graph.name = graphName
        
        dataTable = {}
        dataAttributes = []
        row = 0
        #######################################################################
        #sedaj beremo opise vozlisc
        while True:           #opisi vozlisc so brez praznih vrstic!
            line = graphFile.readline()
            lineNum = lineNum + 1
            words = self.getwords(line)
            col = 0
            
            if words=='' or words==[]:   #prazna vrstica ali EOF, prenehamo brati
                break
            if lower(words[0]) == '*arcs'  or lower(words[0]) == '*edges':
                break

            try:
                index = int(words[0])             #to je indeks vozlisca
            except ValueError:
                #raise InputSyntaxError
                raise
            
            if index <= 0 or index > nVertices:
                #raise InputSyntaxError
                raise
            
            dataTable[row] = {}
            dataTable[row][col] = index;
            col += 1
            # samo v 1. vrstici nastavimo Domeno za ExampleTable
            if row == 0:
                dataAttributes[len(dataAttributes):] = [orange.FloatVariable("index")]
            
            k = len(words)
            if k > 1:
                i = 1
                dataTable[row][col] = words[i];
                col += 1
                # samo v 1. vrstici nastavimo Domeno za ExampleTable
                if row == 0:
                    dataAttributes[len(dataAttributes):] = [orange.StringVariable("label")]
                i += 1
                # poskusimo prebrati coor
                j=0
                while j <= 2 and i < len(words):
                    try:
                        coor = float(words[i])
                    except ValueError:
                        i-=1
                        break
                    if coor < 0 or coor > 1:
                        #raise InputSyntaxError
                        raise
                    
                    dataTable[row][col] = coor*1000.0;
                    col += 1
                    
                    # samo v 1. vrstici nastavimo Domeno za ExampleTable
                    if row == 0:
                        dataAttributes[len(dataAttributes):] = [orange.FloatVariable("coor" + j)]
                    i += 1
                    j += 1
                # beremo atribute
                while i < len(words):
                    if words[i]=='ic':
                        i += 1
                        dataTable[row][col] = words[i];    
                        
                        # samo v 1. vrstici nastavimo Domeno za ExampleTable
                        if row == 0:
                            dataAttributes[len(dataAttributes):] = [orange.EnumVariable("color")]
                        
                        if words[i] not in dataAttributes[col].values:
                            dataAttributes[col].values.append(words[i])
                            
                        col += 1
                            
                    elif words[i]=='bc':
                        i += 1
                        dataTable[row][col] = words[i];
                        col += 1
                        
                        # samo v 1. vrstici nastavimo Domeno za ExampleTable
                        if row == 0:
                            dataAttributes[len(dataAttributes):] = [orange.StringVariable("borderColor")]
                    elif words[i]=='bw':
                        i += 1
                        try:
                            width = int(words[i])
                        except ValueError:
                            #raise InputSyntaxError
                            raise
                        if width < 0:
                            #raise InputSyntaxError
                            raise
                        
                        dataTable[row][col] = words[i];
                        col += 1
                        
                        # samo v 1. vrstici nastavimo Domeno za ExampleTable
                        if row == 0:
                            dataAttributes[len(dataAttributes):] = [orange.FloatVariable("borderWidth")]
                    i += 1
            row += 1
        data = orange.ExampleTable(orange.Domain(dataAttributes))
        
        # create example table with node attributes
        for index in range(len(dataTable)):
            items = []
            for key in dataTable[index].keys():
                items[len(items):] = [dataTable[index][key]]
                
            data.append(items)
            
        graph.setattr("items", data)
        #######################################################################
        #konstrukcija matrike sosednosti
        #sedaj beremo opise povezav
        #postopamo tako, kot deluje Pajek: ce je vrstica prazna, nehamo brati opise povezav


        #TU JE NAPAKA: ce imamo v datoteki prazen blok, oznacen z *arcs ali *edges
        mywords = words
        while True:
            if mywords == [] or mywords== '':  #prazna vrstica ali EOF
                break

            if lower(mywords[0]) == '*edges':
#                if self.Gtype == None or self.Gtype == UNDIRECTED:
#                    self.Gtype = UNDIRECTED
#                else:
#                    self.Gtype = MIXED
                while True:
                    line = graphFile.readline()
                    lineNum = lineNum + 1
                    words = self.getwords(line)
        
                    if words == [] or words=='' or lower(words[0]) == '*arcs' or lower(words[0]) == '*edges':
                        mywords = words
                        break
                    try:
                        i1 = int(words[0])
                        i2 = int(words[1])
                    except ValueError:
                        #raise InputSyntaxError
                        raise
                    if (i1 not in range(1, nVertices+1)) or (i2 not in range(1, nVertices+1)):
                        #raise InputSyntaxError
                        raise
        
                    #zanke in veckratne povezave preskocimo
                    if (i1==i2):
                        continue
#                    if i1-1 in self.edgDesc.keys():
#                        if i2-1 in self.edgDesc[i1-1].keys():
#                            continue
#                    if i2-1 in self.edgDesc.keys():
#                        if i1-1 in self.edgDesc[i2-1].keys():
#                            continue
        
                    graph[i1-1, i2-1] = 1
                    graph[i2-1, i1-1] = 1
        
#                    eparms=EdgeParams()
#                    eparms.type=UNDIRECTED
#        
#                    k=len(words)
#                    if k>2:
#                        i=2
#                        try:
#                            w=float(words[i])
#                        except ValueError:
#                            raise InputSyntaxError
#                        eparms.weight=w     #dana teza prekrije privzeto
#                        i+=1
#                        while i<len(words):
#                            if words[i]=='c':
#                                i+=1
#                                if words[i] not in PajekColors.keys():
#                                    raise InputSyntaxError
#                                else:
#                                    colorTuple=PajekColors[words[i]]
#                                    eparms.color=QColor(colorTuple[0], colorTuple[1], colorTuple[2])
#                                    eparms.colorFromFile=True
#                                    eparms.colorName=words[i]
#                            elif words[i]=='l':
#                                i=i+1
#                                eparms.label=words[i]
#        ##                    elif words[i]=='w'
#        ##                        i+=1
#        ##                        try:
#        ##                            width=int(words[i])
#        ##                        except ValueError:
#        ##                            raise InputSyntaxError
#        ##                        if width<0:
#        ##                            raise InputSyntaxError
#        ##                        eparms.width=width
#        ##                        eparms.widthFromFile=True
#                            i=i+1
#        
#                    if i1-1 not in self.edgDesc.keys():
#                        self.edgDesc[i1-1]={}
#                    self.edgDesc[i1-1][i2-1]=eparms
                    #posledica: sedaj lahko dobimo parametre za vsako povezavo, ki gre iz vertex1-->vertex2
                    #na naslednji nacin:     self.edgDesc[vertex1][vertex2].color   ipd.
                    #iskanje je hitro, ker delamo s slovarji
            elif lower(mywords[0]) == '*arcs':
#                if self.Gtype == None or self.Gtype == DIRECTED:
#                    self.Gtype = DIRECTED
#                else:
#                    self.Gtype = MIXED
                while True:
                    line = graphFile.readline()
                    lineNum = lineNum + 1
                    words = self.getwords(line)
        
                    if words==[] or words=='' or lower(words[0]) == '*edges' or lower(words[0]) == '*arcs':
                        mywords = words
                        break
                    try:
                        i1 = int(words[0])
                        i2 = int(words[1])
                    except ValueError:
                        #raise InputSyntaxError
                        raise
                    if (i1 not in range(1, nVertices + 1)) or (i2 not in range(1, nVertices + 1)):
                        #raise InputSyntaxError
                        raise
        
                    #zanke in veckratne povezave preskocimo
                    if (i1==i2):
                        continue
#                    if i1-1 in self.edgDesc.keys():
#                        if i2-1 in self.edgDesc[i1-1].keys():
#                            continue
#                    if i2-1 in self.edgDesc.keys():
#                        if i1-1 in self.edgDesc[i2-1].keys():
#                            continue
        
        
                    graph[i1-1, i2-1] = 1
        
#                    eparms=EdgeParams()
#                    eparms.type=DIRECTED
#        
#                    k=len(words)
#                    if k>2:
#                        i=2
#                        try:
#                            w=float(words[i])
#                        except ValueError:
#                            raise InputSyntaxError
#                        eparms.weight=w     #dana teza prekrije privzeto
#                        i+=1
#                        while i<len(words):
#                            if words[i]=='c':
#                                i+=1
#                                if words[i] not in PajekColors.keys():
#                                    raise InputSyntaxError
#                                else:
#                                    colorTuple=PajekColors[words[i]]
#                                    eparms.color=QColor(colorTuple[0], colorTuple[1], colorTuple[2])
#                                    eparms.colorFromFile=True
#                                    eparms.colorName=words[i]
#                            elif words[i]=='l':
#                                i=i+1
#                                eparms.label=words[i]
#        ##                    elif words[i]=='w'
#        ##                        i+=1
#        ##                        try:
#        ##                            width=int(words[i])
#        ##                        except ValueError:
#        ##                            raise InputSyntaxError
#        ##                        if width<0:
#        ##                            raise InputSyntaxError
#        ##                        eparms.width=width
#        ##                        eparms.widthFromFile=True
#                            i=i+1
#        
#                    if i1-1 not in self.edgDesc.keys():
#                        self.edgDesc[i1-1]={}
#                    self.edgDesc[i1-1][i2-1]=eparms
                    #posledica: sedaj lahko dobimo parametre za vsako povezavo, ki gre iz vertex1-->vertex2
                    #na naslednji nacin:     self.edgDesc[vertex1][vertex2].color   ipd.
                    #iskanje je hitro, ker delamo s slovarji

        return graph

    
if __name__ == "__main__":
    a=QApplication(sys.argv)
    owf=OWNetworkFile()
    owf.activateLoadedSettings()
    a.setMainWidget(owf)
    owf.show()
    a.exec_loop()
    owf.saveSettings()

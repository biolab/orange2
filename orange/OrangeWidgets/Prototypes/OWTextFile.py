"""
<name>Text File</name>
<description>Loads XML File</description>
<icon>icons/TextFile.png</icon>
<priority>3500</priority>
"""

from qt import *
from OWWidget import *
import OWGUI, OWToolbars, OWDlgs
from xml.sax import make_parser, handler
from orngTextCorpus import TextCorpusLoader, loadWordSet
import os
import modulTMT as lemmatizer
from OWTools import *

class XMLEcho(handler.ContentHandler):
    def __init__(self, lv):
        self.lv = lv
        self.chars = []
        self.lv.lastAdded = None
        self.lv.parent = self.lv
        self.tags = []
    def startElement(self, name, attrs):    
        if not name in self.tags:
            self.tags.append(name)
        parent = self.lv
        self.lv  = (self.lv.lastAdded == None) and QListViewItem(self.lv) or QListViewItem(self.lv, self.lv.lastAdded)
        parent.lastAdded = self.lv
        self.lv.parent = parent
        self.lv.lastAdded = None
        self.lv.setText(0, "<%s %s>" % (name, " ".join(["%s=\"%s\"" % (k, v) for k, v in attrs.items()])))

    def endElement(self, name):
        str =  "".join(self.chars).strip(" \n\t\r")
        if len(str):
            item = QListViewItem(self.lv)
            item.setText(0,"TEXT")
            item.myText = str
        self.chars = []
        self.lv = self.lv.parent
        new = QListViewItem(self.lv, self.lv.lastAdded)
        new.setText(0, "</%s>" % name)
        self.lv.lastAdded = new
   
    def characters(self, chrs):                              
        self.chars.append(chrs)     

class OWTextFile(OWWidget):
    settingsList = []                    
    contextHandlers = {}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Text File')
        
        self.inputs = []
        self.outputs = [("Documents", ExampleTable)]
            
        self.mainArea.setFixedWidth(0)
        ca = QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,5,3,5)      
        
        col1 = QVBox(ca)
        
        # file browser
        box = QHGroupBox("Data File", col1)
        self.fileNameLabel = QLabel(box)
        self.fileNameLabel.setMinimumWidth(350)
        button = OWGUI.button(box, self, '...', callback = self.browseFile, disabled=0)
        button.setMaximumWidth(25)

        # XML table
        QLabel(col1).setText("XML Document")
        self.listView = QListView(col1)
        self.listView.setAllColumnsShowFocus(1)
        self.listView.setRootIsDecorated(1) 
        self.listView.addColumn("Document", 500) 
        self.listView.setSorting(-1)        
        
        # text edit -- displat text node of XML
        QLabel(col1).setText("Node text")
        self.textEdit = QTextView(col1)

        gl.addMultiCellWidget(col1, 0, 4, 0, 0)
        
        self.connect( self.listView, SIGNAL( 'clicked( QListViewItem* )' ),  self.fillText)
    
        self.listTags = []
        self.listTagsSelected = []
        col2 = QVGroupBox("Tags", ca)
        self.listBoxTags = OWGUI.listBox(col2, self, "listTagsSelected", "listTags")
        gl.addMultiCellWidget(col2, 0, 4, 1, 1)
        
        preproc = QVGroupBox("Preprocessing info", ca)
        hboxLem = QHBox(preproc)
        hboxStop = QHBox(preproc)
        
        startfile = os.path.join(str(orangedir), 'OrangeWidgets', 'TextData','.')

        QLabel('Lemmatizer:', hboxLem)
        self.lemmatizer = '(none)'
        items = ['(none)']      
        items.extend([a for a in os.listdir(startfile) if a[-3:] == 'fsa'])              
        OWGUI.comboBox(hboxLem, self, 'lemmatizer', items = items, sendSelectedValue = 1)
            
        QLabel('Stop words:', hboxStop)
        self.stopwords = '(none)'
        items = ['(none)']
        items.extend([a for a in os.listdir(startfile) if a[-3:] == 'txt'])  
        OWGUI.comboBox(hboxStop, self, 'stopwords', items = items, sendSelectedValue = 1) 
        
        preproc.setFixedHeight(100)
        gl.addWidget(preproc, 0, 2)
        
        col3 = QVGroupBox("Separation tags", ca)
        self.documentTag = ""
        self.categoriesTag = ""
        
        hbox2 = QHGroupBox("Content tag", col3)
        vbox2 = QVBox(hbox2)
        OWGUI.button(vbox2, self, ">", self.onContentAdd)
        OWGUI.button(vbox2, self, "<", self.onContentRemove)
        self.contentTag = ""
        OWGUI.lineEdit(hbox2, self, "contentTag")        
        
        hbox4 = QHGroupBox("Category tag", col3)
        vbox4 = QVBox(hbox4)
        OWGUI.button(vbox4, self, ">", self.onCategoryAdd)
        OWGUI.button(vbox4, self, "<", self.onCategoryRemove)
        self.categoryTag = ""
        OWGUI.lineEdit(hbox4, self, "categoryTag") 
        
        hbox5 = QHGroupBox("Additional tags", col3)
        vbox5 = QVBox(hbox5)
        OWGUI.button(vbox5, self, ">", self.onInformativeAdd)
        OWGUI.button(vbox5, self, "<", self.onInformativeRemove)        
        self.informativeTags = []
        self.informativeTagsSelected = []
        OWGUI.listBox(hbox5, self, "informativeTagsSelected", "informativeTags")

        app = OWGUI.button(ca, self, "Apply", self.apply)
        self.catDoc = False
        chBox  = OWGUI.checkBox(col3, self, 'catDoc', label = 'Output category-word', box = '')

        
        gl.addMultiCellWidget(col3, 1, 3,  2, 2)
        gl.addWidget(app, 4, 2)
        
        self.resize(1200, 700)
        
    def openFile(self, fPath):
        self.listView.clear()
        #self.textEdit.clear()
        self.textEdit.setText("")
        self.listBoxTags.clear()
        f = open(fPath, "r")
        
        h = XMLEcho(self.listView)
        parser = make_parser()
        parser.reset()
        parser.setContentHandler(h)
        parser.parse(f)        
        f.close()
        
        self.listTags = h.tags[:]
        
    def browseFile(self, inDemos=0):                
        startfile = "."
        filename = str(QFileDialog.getOpenFileName(startfile,
        'XML files (*.xml)\nAll files(*.*)',None,'Open Orange XML File'))
    
        self.fileNameLabel.setText(filename)
        if filename == "": return
        self.openFile(filename)        
        
    def fillText(self, lvi):
        if hasattr(lvi, "myText"):
            self.textEdit.setText(lvi.myText)
        else:
            self.textEdit.setText("")
            
    def onContentAdd(self):
        if not len(self.listTagsSelected):
            return
        self.contentTag = self.listTags.pop(self.listTagsSelected[0])                
        self.listTagsSelected = []
        self.listTags = self.listTags[:]

    def onContentRemove(self):
        if self.contentTag:
            self.listTags.append(self.contentTag)
            self.contentTag = ""
            self.listTags = self.listTags            
            
    def onCategoryAdd(self):
        if not len(self.listTagsSelected):
            return
        self.categoryTag = self.listTags.pop(self.listTagsSelected[0])                
        self.listTagsSelected = []
        self.listTags = self.listTags[:]

    def onCategoryRemove(self):
        if self.contentTag:
            self.listTags.append(self.categoryTag)
            self.categoryTag = ""
            self.listTags = self.listTags[:]  
                
    def onInformativeAdd(self):
        if not len(self.listTagsSelected):
            return
        self.informativeTags.append(self.listTags.pop(self.listTagsSelected[0]))
        self.listTagsSelected = []
        self.listTags = self.listTags[:]
        self.informativeTags = self.informativeTags[:]

    def onInformativeRemove(self):
        if len(self.informativeTagsSelected):
            self.listTags.append(self.informativeTags.pop(self.informativeTagsSelected[0]))
            self.informativeTagsSelected = []
            self.listTags = self.listTags[:]
            self.informativeTags = self.informativeTags[:]
      
      
    def apply(self):
        tags = {
                        "document" : self.documentTag and self.documentTag or "document",
                        "content" : self.contentTag and self.contentTag or "content",
                        "categories" : self.categoriesTag and self.categoriesTag or "categories",
                        "category" : self.categoryTag and self.categoryTag or "category",
                    }
        if self.lemmatizer == '(none)':
            lem = lemmatizer.NOPLemmatization()
        else:
            lem = lemmatizer.FSALemmatization(os.path.join(str(orangedir), 'OrangeWidgets', 'TextData', self.lemmatizer))
        if not self.stopwords == '(none)':
            for word in loadWordSet(os.path.join(str(orangedir), 'OrangeWidgets', 'TextData', self.stopwords)):
                lem.stopwords.append(word)
        a = TextCorpusLoader(str(self.fileNameLabel.text()), tags, self.informativeTagsSelected, lem)
        if self.catDoc:
            self.send("Documents", CategoryDocument(a.data).dataCD)
        else:
            self.send("Documents", a.data)
if __name__=="__main__": 
    import os
    os.chdir('/home/mkolar/Docs/Diplomski/repository/orange/')
    appl = QApplication(sys.argv) 
    ow = OWTextFile() 
    appl.setMainWidget(ow) 
    ow.show() 
    appl.exec_loop()            

"""
<name>Text File</name>
<description>Loads XML File</description>
<icon></icon>
<priority>3500</priority>
"""

from qt import *
from OWWidget import *
import OWGUI, OWToolbars, OWDlgs
from xml.sax import make_parser, handler
from orngTextCorpus import orngTextCorpus, loadWordSet
import os
import modulTMT as lemmatizer

class XMLEcho(handler.ContentHandler):
    def __init__(self, lv):
##        self.categoryNumber = {}
##        self.categoryDocs = {}
        self.lv = lv
        self.chars = []
        self.lv.lastAdded = None
        self.lv.parent = self.lv
        self.tags = []
##        self.inCategory = 0
    def startElement(self, name, attrs):    
        if not name in self.tags:
            self.tags.append(name)
        parent = self.lv
        self.lv  = (self.lv.lastAdded == None) and QListViewItem(self.lv) or QListViewItem(self.lv, self.lv.lastAdded)
        parent.lastAdded = self.lv
        self.lv.parent = parent
        self.lv.lastAdded = None
        self.lv.setText(0, "<%s %s>" % (name, " ".join(["%s=\"%s\"" % (k, v) for k, v in attrs.items()])))
##        if name == "category":
##            self.inCategory += 1
##        elif name == "document":
##            self.documentName = " ".join(["%s=\"%s\"" % (k, v) for k, v in attrs.items()])

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
##        if name == "category":
##            self.inCategory -= 1
##            if self.categoryNumber.has_key(str):
##                self.categoryNumber[str] = self.category[str] + 1
##                self.categoryDocs[str] = self.categoryDocs[str].append(self.documentName)
##            else:
##                self.categoryNumber[str] = 1
##                self.categoryDocs[str] = [self.documentName]
##        
    def characters(self, chrs):                              
        self.chars.append(chrs)     

class OWTextFile(OWWidget):
    settingsList = []                    
    contextHandlers = {}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Text File')
        
        self.inputs = []
        self.outputs = [("Examples", ExampleTable)]
            
        self.mainArea.setFixedWidth(0)
        ca = QFrame(self.controlArea)
        ca.adjustSize()
        gl=QGridLayout(ca,4,3,5)      
        
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
        self.textEdit = QTextEdit(col1)

        gl.addMultiCellWidget(col1, 0, 3, 0, 0)
        
        self.connect( self.listView, SIGNAL( 'clicked( QListViewItem* )' ),  self.fillText)
    
        self.listTags = []
        self.listTagsSelected = []
        col2 = QVGroupBox("Tags", ca)
        self.listBoxTags = OWGUI.listBox(col2, self, "listTagsSelected", "listTags")
        gl.addMultiCellWidget(col2, 0, 3, 1, 1)
        
        preproc = QVGroupBox("Preprocessing info", ca)
        hboxLem = QHBox(preproc)
        hboxStop = QHBox(preproc)
        
        QLabel('Lemmatizer:', hboxLem)
        self.lemmatizer = '(none)'
        items = ['(none)']
        
        
##        items.extend([a for a in os.listdir('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData') if a[-3:] == 'fsa'])     
        
        #################
        # check if this is ok
        ##
        d = os.getcwd()
        if d[-12:] == "OrangeCanvas":
            startfile = d[:-12]+"/OrangeWidgets/TextData"
        else:
            startfile = d+"/OrangeWidgets/TextData"
            
        items.extend([a for a in os.listdir(startfile) if a[-3:] == 'fsa'])  
            
        #################
            
        OWGUI.comboBox(hboxLem, self, 'lemmatizer', items = items, sendSelectedValue = 1)
            
        QLabel('Stop words:', hboxStop)
        self.stopwords = '(none)'
        items = ['(none)']
##        items.extend([a for a in os.listdir('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData') if a[-3:] == 'txt'])
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
        
        hbox5 = QHGroupBox("Informative tags", col3)
        vbox5 = QVBox(hbox5)
        OWGUI.button(vbox5, self, ">", self.onInformativeAdd)
        OWGUI.button(vbox5, self, "<", self.onInformativeRemove)        
        self.informativeTags = []
        self.informativeTagsSelected = []
        OWGUI.listBox(hbox5, self, "informativeTagsSelected", "informativeTags")

        app = OWGUI.button(ca, self, "Apply", self.apply)
        
        gl.addMultiCellWidget(col3, 1, 2,  2, 2)
        gl.addWidget(app, 3, 2)
        
        self.resize(1200, 700)
        
    def openFile(self, fPath):
        self.listView.clear()
        self.textEdit.clear()
##        self.statDocPerCat.clear()
        self.listBoxTags.clear()
        f = open(fPath, "r")
        
        h = XMLEcho(self.listView)
        parser = make_parser()
        parser.reset()
        parser.setContentHandler(h)
        parser.parse(f)        
        f.close()
        
##        item = None
##        for c, nd in h.categoryNumber.items():
##            item = QListViewItem(self.statDocPerCat)
##            item.setText(0, c)
##            item.setText(1, str(nd))

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
            lem = lemmatizer.FSALemmatization('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/'+self.lemmatizer)
        if not self.stopwords == '(none)':
            for word in loadWordSet('/home/mkolar/Docs/Diplomski/repository/orange/OrangeWidgets/TextData/'+self.stopwords):
                lem.stopwords.append(word)
        a = orngTextCorpus(self.fileNameLabel.text(), tags, self.informativeTagsSelected, lem)
        self.send("Examples", a.data)
if __name__=="__main__": 
    appl = QApplication(sys.argv) 
    ow = OWTextFile() 
    appl.setMainWidget(ow) 
    ow.show() 
    appl.exec_loop()            

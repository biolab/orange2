"""
<name>XML File Data</name>
<description>Loads XML File</description>
<icon>icons/ca.png</icon>
<priority>3500</priority>
"""

from qt import *
from OWWidget import *
import OWGUI, OWToolbars, OWDlgs
from xml.sax import make_parser, handler
from orngXMLData import orngXMLData

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

class OWXMLFile(OWWidget):
    settingsList = []                    
    contextHandlers = {}
    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'XML File Data')
        
        self.inputs = []
        self.outputs = [("XML Data file", orngXMLData)]
        
##        self.catListBoxSelection = []
##        self.catListBoxElems = []        
##        
##        self.docListBoxSelection = []
##        self.docListBoxElems = []
        
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
##        gl.addWidget(box, 0, 0)

        # XML table
        QLabel(col1).setText("XML Document")
        self.listView = QListView(col1)
        self.listView.setAllColumnsShowFocus(1)
        self.listView.setRootIsDecorated(1) 
        self.listView.addColumn("Document", 500) 
        self.listView.setSorting(-1)        
##        self.listView.setMinimumSize(QSize(600, 400))
##        gl.addWidget(self.listView, 1, 0)
        
        # text edit -- displat text node of XML
        QLabel(col1).setText("Node text")
        self.textEdit = QTextEdit(col1)
##        self.textEdit.setMinimumSize(QSize(600, 200))
##        gl.addWidget(self.textEdit, 2, 0)
        gl.addMultiCellWidget(col1, 0, 3, 0, 0)
        
        self.connect( self.listView, SIGNAL( 'clicked( QListViewItem* )' ),  self.fillText);

##        # statistics - number of documents per category
####        frame = QFrame(self.controlArea, "Pero detlic")
####        gl.addMultiCellWidget(frame, 0, 2, 1, 1)        
####        frame.setMinimumWidth(200)
####        frame.adjustSize()
##        self.statDocPerCat = QListView(ca)
##        self.statDocPerCat.setAllColumnsShowFocus(1)
##        self.statDocPerCat.setRootIsDecorated(1) 
##        self.statDocPerCat.addColumn("Category", 150) 
##        self.statDocPerCat.addColumn("Num. documents", 100)
##        self.statDocPerCat.setColumnAlignment(1, Qt.AlignRight)
##        self.statDocPerCat.setSelectionMode(QListView.Single)        
##        gl.addMultiCellWidget(self.statDocPerCat, 0, 2, 1, 1)

##        
##        self.catListBox = OWGUI.listBox(ca, self, "catListBoxSelection", "catListBoxElems", selectionMode = QListBox.Single)
##        gl.addMultiCellWidget(self.catListBox, 0, 0, 2, 3)
##        self.catListBox.setMinimumWidth(300)
##        self.docListBox = OWGUI.listBox(ca, self, "docListBoxSelection", "docListBoxElems", selectionMode = QListBox.Single)
##        gl.addMultiCellWidget(self.docListBox, 1, 1, 2, 3)
##        
##        self.apply = OWGUI.button(ca, self, "Apply", self.onApplyClicked)
##        gl.addWidget(self.apply, 2, 2)
##
##        self.reset = OWGUI.button(ca, self, "Reset", self.onResetClicked)
##        gl.addWidget(self.reset, 2, 3)
        
##        self.categoryButton = OWGUI.button(self.mainArea, self, ">", self.onCategoryButtonClicked)
##        self.listCategory = OWGUI.listBox(self.mainArea, self, "listBoxSelection", "selectedCategories", selectionMode = QListBox.NoSelection)
    
        self.listTags = []
        self.listTagsSelected = []
        col2 = QVGroupBox("Tags", ca)
        self.listBoxTags = OWGUI.listBox(col2, self, "listTagsSelected", "listTags")
        gl.addMultiCellWidget(col2, 0, 3, 1, 1)
        
        col3 = QVGroupBox("Separation tags", ca)
        hbox1 = QHGroupBox("Document tag", col3)
        vbox1 = QVBox(hbox1)
        OWGUI.button(vbox1, self, ">", self.onDocumentAdd)
        OWGUI.button(vbox1, self, "<", self.onDocumentRemove)
        self.documentTag = ""
        OWGUI.lineEdit(hbox1, self, "documentTag")
        
        hbox2 = QHGroupBox("Content tag", col3)
        vbox2 = QVBox(hbox2)
        OWGUI.button(vbox2, self, ">", self.onContentAdd)
        OWGUI.button(vbox2, self, "<", self.onContentRemove)
        self.contentTag = ""
        OWGUI.lineEdit(hbox2, self, "contentTag")        

        hbox3 = QHGroupBox("Categories tag", col3)
        vbox3 = QVBox(hbox3)
        OWGUI.button(vbox3, self, ">", self.onCategoriesAdd)
        OWGUI.button(vbox3, self, "<", self.onCategoriesRemove)
        self.categoriesTag = ""
        OWGUI.lineEdit(hbox3, self, "categoriesTag")      
        
        hbox4 = QHGroupBox("Category tag", col3)
        vbox4 = QVBox(hbox4)
        OWGUI.button(vbox4, self, ">", self.onCategoryAdd)
        OWGUI.button(vbox4, self, "<", self.onCategoryRemove)
        self.categoryTag = ""
        OWGUI.lineEdit(hbox4, self, "categoryTag") 

        OWGUI.button(col3, self, "Apply", self.apply)
        

        gl.addMultiCellWidget(col3, 0, 3,  2, 2)
        
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
##        "Display a FileDialog and select a file"
##        if inDemos:
##            import os
##            try:
##                import win32api, win32con
##                t = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, "SOFTWARE\\Python\\PythonCore\\%i.%i\\PythonPath\\Orange" % sys.version_info[:2], 0, win32con.KEY_READ)
##                t = win32api.RegQueryValueEx(t, "")[0]
##                startfile = t[:t.find("orange")] + "orange\\doc\\datasets"
##            except:
##                d = os.getcwd()
##                if d[-12:] == "OrangeCanvas":
##                    startfile = d[:-12]+"doc/datasets"
##                else:
##                    startfile = d+"doc/datasets"
##
##            if not os.path.exists(startfile):                    
##                QMessageBox.information( None, "File", "Cannot find the directory with example data sets", QMessageBox.Ok + QMessageBox.Default)
##                return
##        else:
##            if len(self.recentFiles) == 0 or self.recentFiles[0] == "(none)":
##                startfile="."
##            else:
##                startfile=self.recentFiles[0]
                
        startfile = "."
        filename = str(QFileDialog.getOpenFileName(startfile,
        'XML files (*.xml)\nAll files(*.*)',None,'Open Orange XML File'))
    
        self.fileNameLabel.setText(filename)
        if filename == "": return
##        if filename in self.recentFiles: self.recentFiles.remove(filename)
##        self.recentFiles.insert(0, filename)
##        self.setFileList()
        self.openFile(filename)        
        
    def fillText(self, lvi):
        if hasattr(lvi, "myText"):
            self.textEdit.setText(lvi.myText)
        else:
            self.textEdit.setText("")
##    def onCategoryButtonClicked(self):
##        item = self.statDocPerCat.firstChild()
##        if item.isSelected():
##            self.selectedCategories.append(item.text())
##        while item.nextSibling():
##            item = item.nextSibling()
##            if item.isSelected():
##                self.selectedCategories.append(item.text())
                
##    def onApplyClicked(self):
##        pass
##    def onResetClicked(self):
##        pass
    def onDocumentAdd(self):
        if not len(self.listTagsSelected):
            return
        self.documentTag = self.listTags.pop(self.listTagsSelected[0])                
        self.listTagsSelected = []
        self.listTags = self.listTags[:]

    def onDocumentRemove(self):
        if self.documentTag:
            self.listTags.append(self.documentTag)
            self.documentTag = ""
            self.listTags = self.listTags
            
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
            
            
    def onCategoriesAdd(self):
        if not len(self.listTagsSelected):
            return
        self.categoriesTag = self.listTags.pop(self.listTagsSelected[0])                
        self.listTagsSelected = []
        self.listTags = self.listTags[:]

    def onCategoriesRemove(self):
        if self.contentTag:
            self.listTags.append(self.categoriesTag)
            self.categoriesTag = ""
            self.listTags = self.listTags[:]
            
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
    def apply(self):
        tags = {
                        "document" : self.documentTag and self.documentTag or "document",
                        "content" : self.contentTag and self.contentTag or "content",
                        "categories" : self.categoriesTag and self.categoriesTag or "categories",
                        "category" : self.categoryTag and self.categoryTag or "category",
                    }
        self.send("XML Data file", orngXMLData(self.fileNameLabel.text(), tags, None))
if __name__=="__main__": 
    appl = QApplication(sys.argv) 
    ow = OWXMLFile() 
    appl.setMainWidget(ow) 
    ow.show() 
    appl.exec_loop()            

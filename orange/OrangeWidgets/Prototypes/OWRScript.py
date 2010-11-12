"""<name>R Script</name>
<description>Run R scripts</description>
<icon>icons/RScript.png</icon>
"""


from OWWidget import *
from OWPythonScript import OWPythonScript, Script, ScriptItemDelegate, PythonConsole
from OWItemModels import PyListModel, ModelActionsWidget

import rpy2.robjects as robjects
import rpy2.rlike.container as rlc

if hasattr(robjects, "DataFrame"): # rpy2 version 2.1
    DataFrame = robjects.DataFrame
else: # rpy2 version 2.0
    DataFrame = robjects.RDataFrame
    
if hasattr(robjects, "Matrix"): # rpy2 version 2.1
    Matrix = robjects.Matrix
else: # rpy2 version 2.0
    Matrix = robjects.RMatrix
    
if hasattr(robjects, "globalenv"): # rpy2 version 2.1
    globalenv = robjects.globalenv
else: # rpy2 version 2.0
    globalenv = robjects.globalEnv
    
if hasattr(robjects, "NA_Real"):
    NA_Real = robjects.NA_Real
else:
    NA_Real = robjects.r("NA")
    
if hasattr(robjects, "NA_Integer"):
    NA_Real = robjects.NA_Integer
else:
    NA_Real = robjects.r("NA")
    
if hasattr(robjects, "NA_Character"):
    NA_Real = robjects.NA_Character
else:
    NA_Real = robjects.r("NA")
        

def ExampleTable_to_DataFrame(examples):
    attrs = [ attr for attr in examples.domain.variables if attr.varType in \
             [orange.VarTypes.Continuous, orange.VarTypes.Discrete, orange.VarTypes.String]]
    def float_or_NA(value):
        if value.isSpecial():
            return NA_Real
        else:
            return float(value)
        
    def int_or_NA(value):
        if value.isSpecial():
            return NA_Integer
        else:
            return int(value)
    
    def str_or_NA(value):
        if value.isSpecial():
            return NA_Character
        else:
            return str(value)
        
    data = []
    for attr in attrs:
        if attr.varType == orange.VarTypes.Continuous:
            data.append((attr.name, robjects.FloatVector([float_or_NA(ex[attr]) for ex in examples])))
        elif attr.varType == orange.VarTypes.Discrete:
            intvec = robjects.IntVector([int_or_NA(ex[attr]) for ex in examples])
#            factor.levels = robjects.StrVector(list(attr.values))
            data.append((attr.name, intvec))
        elif attr.varType == orange.VarTypes.String:
            data.append((attr.name, robjects.StrVector([str_or_NA(ex[attr]) for ex in examples])))
        
    r_obj = DataFrame(rlc.TaggedList([v for _, v in data], [t for t,_ in data]))
    return r_obj


def SymMatrix_to_Matrix(matrix):
    matrix = [[e for e in row for row in matrix]]
    r_obj = Matrix(matrix)
    return r_obj

    
class RPy2Console(PythonConsole):
    
    def interact(self, banner=None):
        if banner is None:
            banner ="R console:"
        return PythonConsole.interact(self, banner)
        
    def push(self, line):
        if self.history[0] != line:
            self.history.insert(0, line)
        self.historyInd = 0
        more = 0
        try:
            r_res = self._rpy(line)
            self.write(r_res.r_repr())
        except Exception, ex:
            self.write(repr(ex))
        
        self.write("\n")
        
    def _rpy(self, script):
        r_res = robjects.r(script)
        return r_res
    
    def setLocals(self, locals):
        r_obj = {}
        for key, val in locals.items():
            if isinstance(val, orange.ExampleTable):
                dataframe = ExampleTable_to_DataFrame(val)
                globalenv[key] =  dataframe
            elif isinstance(val, orange.SymMatrix):
                matrix = SymMatrix_to_Matrix(val)
                globalenv[key] = matrix
                
                
class RSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        self.keywordFormat = QTextCharFormat()
        self.keywordFormat.setForeground(QBrush(Qt.darkGray))
        self.keywordFormat.setFontWeight(QFont.Bold)
        self.stringFormat = QTextCharFormat()
        self.stringFormat.setForeground(QBrush(Qt.darkGreen))
        self.defFormat = QTextCharFormat()
        self.defFormat.setForeground(QBrush(Qt.black))
        self.defFormat.setFontWeight(QFont.Bold)
        self.commentFormat = QTextCharFormat()
        self.commentFormat.setForeground(QBrush(Qt.lightGray))
        self.decoratorFormat = QTextCharFormat()
        self.decoratorFormat.setForeground(QBrush(Qt.darkGray))
        self.constantFormat = QTextCharFormat()
        self.constantFormat.setForeground(QBrush(Qt.blue))
        
        self.keywords = ["TRUE", "FALSE", "if", "else", "NULL", r"\.\.\.", "<-", "for", "while", "repeat", "next",
                         "break", "switch", "function"]
        self.rules = [(QRegExp(r"\b%s\b" % keyword), self.keywordFormat) for keyword in self.keywords] + \
                     [
#                     [(QRegExp(r"\bdef\s+([A-Za-z_]+[A-Za-z0-9_]+)\s*\("), self.defFormat),
#                      (QRegExp(r"\bclass\s+([A-Za-z_]+[A-Za-z0-9_]+)\s*\("), self.defFormat),
                      (QRegExp(r"'.*'"), self.stringFormat),
                      (QRegExp(r'".*"'), self.stringFormat),
                      (QRegExp(r"#.*"), self.commentFormat),
                      (QRegExp(r"[0-9]+\.?[0-9]*"), self.constantFormat)
#                      (QRegExp(r"@[A-Za-z_]+[A-Za-z0-9_]+"), self.decoratorFormat)]
                     ]
                     
        QSyntaxHighlighter.__init__(self, parent)
        
    def highlightBlock(self, text):
        for pattern, format in self.rules:
            exp = QRegExp(pattern)
            index = exp.indexIn(text)
            while index >= 0:
                length = exp.matchedLength()
                if exp.numCaptures() > 0:
                    self.setFormat(exp.pos(1), len(str(exp.cap(1))), format)
                else:
                    self.setFormat(exp.pos(0), len(str(exp.cap(0))), format)
                index = exp.indexIn(text, index + length)
                
        ## Multi line strings
#        start = QRegExp(r"(''')|" + r'(""")')
#        end = QRegExp(r"(''')|" + r'(""")')
#        self.setCurrentBlockState(0)
#        startIndex, skip = 0, 0
#        if self.previousBlockState() != 1:
#            startIndex, skip = start.indexIn(text), 3
#        while startIndex >= 0:
#            endIndex = end.indexIn(text, startIndex + skip)
#            if endIndex == -1:
#                self.setCurrentBlockState(1)
#                commentLen = len(text) - startIndex
#            else:
#                commentLen = endIndex - startIndex + 3
#            self.setFormat(startIndex, commentLen, self.stringFormat)
#            startIndex, skip = start.indexIn(text, startIndex + commentLen + 3), 3

class RScriptEditor(QPlainTextEdit):
    pass

class RScript(Script):
    pass

class LibraryWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.setLayout(QVBoxLayout())
        
        self.layout().setSpacing(1)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.view = QListView(self)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.layout().addWidget(self.view)
        
        self.addScriptAction = QAction("+", self)
        self.addScriptAction.pyqtConfigure(toolTip="Add a new script to library")
        
        self.updateScriptAction = QAction("Update", self)
        self.updateScriptAction.pyqtConfigure(toolTip="Save changes in the editor to library", shortcut=QKeySequence.Save)
        
        self.removeScriptAction = QAction("-", self)
        self.removeScriptAction.pyqtConfigure(toolTip="Remove selected script from file")
        
        self.moreAction = QAction("More", self)
        self.moreAction.pyqtConfigure(toolTip="More actions")#, icon=self.style().standardIcon(QStyle.SP_ToolBarHorizontalExtensionButton))
        
        self.addScriptFromFile = QAction("Add script from file", self)
        self.addScriptFromFile.pyqtConfigure(toolTip="Add a new script to library from a file")
        
        self.saveScriptToFile = QAction("Save script to file", self)
        self.saveScriptToFile.pyqtConfigure(toolTip="Save script to a file", shortcut=QKeySequence.SaveAs)
        
        menu = QMenu(self)
        menu.addActions([self.addScriptFromFile, self.saveScriptToFile])
        self.moreAction.setMenu(menu)
        
        self.actionsWidget = ModelActionsWidget([self.addScriptAction, self.updateScriptAction, 
                                            self.removeScriptAction, self.moreAction], self)
#        self.actionsWidget.buttons[1].setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.actionsWidget.layout().setSpacing(1)
        
        self.layout().addWidget(self.actionsWidget)
        
        self.resize(800, 600)
        
        
    def setScriptModel(self, model):
        """ Set the script model to show 
        """
        self.model = model
        if self.view is not None:
            self.view.setModel(model)
        
        
    def setView(self, view):
        """ Set the view (QListView or subclass) to use
        """
        if self.view is not None:
            self.layout().removeItemAt(0) 
        self.view = view
        self.layout().insertItem(0, view)
        if self.model is not None:
            self.view.setModel(self.model)
            
            
    def setActionsWidget(self, widget):
        """ Set the widget below the view to widget (removing the previous widget)
        """
        self.layout().removeItemAt(1)
        self.actionsWidget = widget
        self.layout().insertWidget(1, widget)

    def setDocumentEditor(self, editor):
        self.editor = editor
        

class OWRScript(OWWidget):
    settingsList = ["scriptLibraryList", "selectedScriptIndex", "lastDir"]
    def __init__(self, parent=None, signalManager=None, name="R Script"):
        OWWidget.__init__(self, parent, signalManager, name)
        
        self.inputs = [("in_data", ExampleTable, self.setData), ("in_distance", orange.SymMatrix, self.setDistance)]
        self.outputs = [("out_data", ExampleTable), ("out_distance", orange.SymMatrix), ("out_learner", orange.Learner), ("out_classifier", orange.Classifier)]
        
        self.in_data, self.in_distance = None, None
        self.scriptLibraryList = [RScript("New script", "x <- c(1,2,5)\ny <- c(2,1 6)\nplot(x,y)\n")]
        self.selectedScriptIndex = 0
        
        self.lastDir = os.path.expanduser("~/")
        
        self.loadSettings()
        
        self.defaultFont = QFont("Monaco") if sys.platform == "darwin" else QFont("Courier")
        
        self.splitter = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitter)
        self.scriptEdit = RScriptEditor()
        self.splitter.addWidget(self.scriptEdit)
        self.console = RPy2Console({}, self)
        self.splitter.addWidget(self.console)
        self.splitter.setSizes([2,1])
        
        self.libraryModel = PyListModel([], self, Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
        self.libraryModel.wrap(self.scriptLibraryList)
        
        self.infoBox = OWGUI.widgetLabel(OWGUI.widgetBox(self.controlArea, "Info", addSpace=True), "")
        self.infoBox.setText("<p>Execute R script</p>Input variables<li><ul>in_data</ul><ul>in_distance</ul></li>Output variables:<li><ul>out_data</ul></p/>")
        
        box = OWGUI.widgetBox(self.controlArea, "Library", addSpace=True)
        
        self.libraryWidget = LibraryWidget()
        self.libraryView = self.libraryWidget.view
        box.layout().addWidget(self.libraryWidget)
        
        self.libraryWidget.view.setItemDelegate(ScriptItemDelegate(self))
        self.libraryWidget.setScriptModel(self.libraryModel)
        self.libraryWidget.setDocumentEditor(self.scriptEdit)
        
        box = OWGUI.widgetBox(self.controlArea, "Run")
        OWGUI.button(box, self, "Execute", callback=self.runScript, tooltip="Execute the script")
        OWGUI.rubber(self.controlArea)
        
        
        self.connect(self.libraryWidget.addScriptAction, SIGNAL("triggered()"), self.onAddNewScript)
        self.connect(self.libraryWidget.updateScriptAction, SIGNAL("triggered()"), self.onUpdateScript)
        self.connect(self.libraryWidget.removeScriptAction, SIGNAL("triggered()"), self.onRemoveScript)
        self.connect(self.libraryWidget.addScriptFromFile, SIGNAL("triggered()"), self.onAddScriptFromFile)
        self.connect(self.libraryWidget.saveScriptToFile, SIGNAL("triggered()"), self.onSaveScriptToFile)
        
        self.connect(self.libraryView.selectionModel(), SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.onScriptSelection)
        
        self._cachedDocuments = {}
        
        self.resize(800, 600)
        QTimer.singleShot(30, self.initGrDevice)
        
    def initGrDevice(self):
#        from rpy2.robjects.packages import importr
#        grdevices = importr('grDevices')
        import tempfile
        self.grDevFile = tempfile.NamedTemporaryFile("rwb", prefix="orange-rscript-grDev", suffix=".png",  delete=False)
        self.grDevFile.close()
        robjects.r('png("%s", width=512, height=512)' % self.grDevFile.name)
        print "Temp bitmap file", self.grDevFile.name
#        self.fileWatcher = QFileSystemWatcher(self)
#        self.connect(self.fileWatcher, SIGNAL("fileChanged(QString)"), self.updateGraphicsView)
        
    
    def updateGraphicsView(self, filename):
        self.grView = w = QLabel()
        w.setPixmap(QPixmap(filename))
        w.show()

        
    def selectScript(self, index):
        index = self.libraryModel.index(index)
        self.libraryView.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
        
        
    def selectedScript(self):
        rows = self.libraryView.selectionModel().selectedRows()
        rows = [index.row() for index in rows]
        if rows:
            return rows[0]
        else:
            return None
        
        
    def onAddNewScript(self):
        self.libraryModel.append(RScript("New Script", ""))
        self.selectScript(len(self.libraryModel) - 1)
        
        
    def onRemoveScript(self):
        row = self.selectedScript()
        if row is not None:
            del self.libraryModel[row]
        
        
    def onUpdateScript(self):
        row = self.selectedScript()
        if row is not None:
            self.libraryModel[row].script = str(self.scriptEdit.toPlainText())
            self.scriptEdit.document().setModified(False)
            self.libraryModel.emitDataChanged([row])
            
        
    def onAddScriptFromFile(self):
        filename = str(QFileDialog.getOpenFileName(self, "Open script", self.lastDir))
        if filename:
            script = open(filename, "rb").read()
            self.lastDir, name = os.path.split(filename)
            self.libraryModel.append(RScript(name, script, sourceFileName=filename))
            self.selectScript(len(self.libraryModel) - 1)
            
            
    def onSaveScriptToFile(self):
        row = self.selectedScript()
        if row is not None:
            script = self.libraryModel[row]
            filename = str(QFileDialog.getSaveFileName(self, "Save Script As", script.sourceFileName or self.lastDir))
            if filename:
                self.lastDir, name = os.path.split(filename)
                script.sourceFileName = filename
                script.flags = 0
                open(filename, "wb").write(script.script)
                
                
    def onScriptSelection(self, *args):
        row = self.selectedScript()
        if row is not None:
            self.scriptEdit.setDocument(self.documentForScript(row))
            
                
    def documentForScript(self, script=0):
        if type(script) != RScript:
            script = self.libraryModel[script]
        if script not in self._cachedDocuments:
            doc = QTextDocument(self)
            doc.setDocumentLayout(QPlainTextDocumentLayout(doc))
            doc.setPlainText(script.script)
            doc.setDefaultFont(QFont(self.defaultFont))
            doc.highlighter = RSyntaxHighlighter(doc)
            self.connect(doc, SIGNAL("modificationChanged(bool)"), self.onModificationChanged)
            doc.setModified(False)
            self._cachedDocuments[script] = doc
        return self._cachedDocuments[script]
    
        
    def onModificationChanged(self, changed):
        row = self.selectedScript()
        if row is not None:
            self.libraryModel[row].flags = RScript.Modified if changed else 0
            self.libraryModel.emitDataChanged([row]) 
                               
                            
    def setData(self, data):
        self.in_data = data
        self.console.setLocals(self.getLocals())


    def setDistance(self, matrix):
        self.in_distance = matrix
        self.console.setLocals(self.getLocals())

        
    def getLocals(self):
        return {"in_data": self.in_data,
                "in_distance": self.in_distance,
               }
        
    def runScript(self):
        self.console.push('png("%s", width=512, height=512)\n' % self.grDevFile.name)
        self.console.push(str(self.scriptEdit.toPlainText()))
        self.console.new_prompt(">>> ")
        robjects.r("dev.off()\n")
        self.updateGraphicsView(self.grDevFile.name)
        
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWRScript()
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    w.setData(data)
    w.show()
    app.exec_()
    w.saveSettings()
        
              
        
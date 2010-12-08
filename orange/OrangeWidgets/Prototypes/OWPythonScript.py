"""
<name>Python Script</name>
<description>Executes python script.</description>
<icon>icons/PythonScript.png</icon>
<contact>Miha Stajdohar (miha.stajdohar(@at@)gmail.com)</contact> 
<priority>3011</priority>
"""
from OWWidget import *

import sys, traceback
import OWGUI, orngNetwork

import code

class PythonSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        self.keywordFormat = QTextCharFormat()
        self.keywordFormat.setForeground(QBrush(Qt.blue))
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
        
        self.keywords = ["def", "if", "else", "elif", "for", "while", "with", "try", "except",
                         "finally", "not", "in", "lambda", "None", "import", "class", "return", "print",
                         "yield", "break", "continue", "raise", "or", "and", "True", "False", "pass",
                         "from", "as"]
        self.rules = [(QRegExp(r"\b%s\b" % keyword), self.keywordFormat) for keyword in self.keywords] + \
                     [(QRegExp(r"\bdef\s+([A-Za-z_]+[A-Za-z0-9_]+)\s*\("), self.defFormat),
                      (QRegExp(r"\bclass\s+([A-Za-z_]+[A-Za-z0-9_]+)\s*\("), self.defFormat),
                      (QRegExp(r"'.*'"), self.stringFormat),
                      (QRegExp(r'".*"'), self.stringFormat),
                      (QRegExp(r"#.*"), self.commentFormat),
                      (QRegExp(r"@[A-Za-z_]+[A-Za-z0-9_]+"), self.decoratorFormat)]
                     
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
        start = QRegExp(r"(''')|" + r'(""")')
        end = QRegExp(r"(''')|" + r'(""")')
        self.setCurrentBlockState(0)
        startIndex, skip = 0, 0
        if self.previousBlockState() != 1:
            startIndex, skip = start.indexIn(text), 3
        while startIndex >= 0:
            endIndex = end.indexIn(text, startIndex + skip)
            if endIndex == -1:
                self.setCurrentBlockState(1)
                commentLen = len(text) - startIndex
            else:
                commentLen = endIndex - startIndex + 3
            self.setFormat(startIndex, commentLen, self.stringFormat)
            startIndex, skip = start.indexIn(text, startIndex + commentLen + 3), 3
                
class PythonScriptEditor(QPlainTextEdit):
    INDENT = 4
    def lastLine(self):
        text = str(self.toPlainText())
        pos = self.textCursor().position()
        index = text.rfind("\n", 0, pos)
        text = text[index: pos].lstrip("\n")
        return text
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            text = self.lastLine()
            indent = len(text) - len(text.lstrip())
            if text.strip() == "pass" or text.strip().startswith("return"):
                indent = max(0, indent - self.INDENT)
            elif text.strip().endswith(":"):
                indent += self.INDENT
            QPlainTextEdit.keyPressEvent(self, event)
            self.insertPlainText(" " * indent)
        elif event.key() == Qt.Key_Tab:
            self.insertPlainText(" " * self.INDENT)
        elif event.key() == Qt.Key_Backspace:
            text = self.lastLine()
            if text and not text.strip():
                cursor = self.textCursor()
                for i in range(min(self.INDENT, len(text))):
                    cursor.deletePreviousChar()
            else:
                QPlainTextEdit.keyPressEvent(self, event)
                
        else:
            QPlainTextEdit.keyPressEvent(self, event)
        
class PythonConsole(QPlainTextEdit, code.InteractiveConsole):
    def __init__(self, locals = None, parent=None):
        QPlainTextEdit.__init__(self, parent)
        code.InteractiveConsole.__init__(self, locals)
        self.history, self.historyInd = [""], 0
        self.loop = self.interact()
        self.loop.next()
        
    def setLocals(self, locals):
        self.locals = locals
        
    def interact(self, banner=None):
        try:
            sys.ps1
        except AttributeError:
            sys.ps1 = ">>> "
        try:
            sys.ps2
        except AttributeError:
            sys.ps2 = "... "
        cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
        if banner is None:
            self.write("Python %s on %s\n%s\n(%s)\n" %
                       (sys.version, sys.platform, cprt,
                        self.__class__.__name__))
        else:
            self.write("%s\n" % str(banner))
        more = 0
        while 1:
            try:
                if more:
                    prompt = sys.ps2
                else:
                    prompt = sys.ps1
                self.new_prompt(prompt)
                yield
                try:
                    line = self.raw_input(prompt)
                except EOFError:
                    self.write("\n")
                    break
                else:
                    more = self.push(line)
            except KeyboardInterrupt:
                self.write("\nKeyboardInterrupt\n")
                self.resetbuffer()
                more = 0
                
    def raw_input(self, prompt):
        input = str(self.document().lastBlock().previous().text())
        return input[len(prompt):]
        
    def new_prompt(self, prompt):
        self.write(prompt)
        self.newPromptPos = self.textCursor().position()
        
    def write(self, data):
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)
        cursor.insertText(data)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
        
    def push(self, line):
        if self.history[0] != line:
            self.history.insert(0, line)
        self.historyInd = 0
        more = 0
        saved = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = self, self
            return code.InteractiveConsole.push(self, line)
        finally:
            sys.stdout, sys.stderr = saved
            
    def setLine(self, line):
        cursor = QTextCursor(self.document())
        cursor.movePosition(QTextCursor.End)
        cursor.setPosition(self.newPromptPos, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(line)
        self.setTextCursor(cursor)
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            self.write("\n")
            self.loop.next()
        elif event.key() == Qt.Key_Up:
            self.historyUp()
        elif event.key() == Qt.Key_Down:
            self.historyDown()
        elif event.key() == Qt.Key_Tab:
            self.complete()
        elif event.key() in [Qt.Key_Left, Qt.Key_Backspace]:
            if self.textCursor().position() > self.newPromptPos:
                QPlainTextEdit.keyPressEvent(self, event)
        else:
            QPlainTextEdit.keyPressEvent(self, event)
            
    def historyUp(self):
        self.setLine(self.history[self.historyInd])
        self.historyInd = min(self.historyInd + 1, len(self.history) - 1)
        
    def historyDown(self):
        self.setLine(self.history[self.historyInd])
        self.historyInd = max(self.historyInd - 1, 0)
        
    def complete(self):
        pass
    
from OWItemModels import PyListModel, ModelActionsWidget

class Script(object):
    Modified = 1
    MissingFromFilesystem = 2 
    def __init__(self, name, script, flags=0, sourceFileName=None):
        self.name = name
        self.script = script
        self.flags = flags
        self.sourceFileName = sourceFileName
        self.modifiedScript = None

class ScriptItemDelegate(QStyledItemDelegate):
    def __init__(self, parent):
        QStyledItemDelegate.__init__(self, parent)
        
    def displayText(self, variant, locale):
        script = variant.toPyObject()
        if script.flags & Script.Modified:
            return QString("*" + script.name)
        else:
            return QString(script.name)
    
    def paint(self, painter, option, index):
        script = index.data(Qt.DisplayRole).toPyObject()
        tmp_palette = None
        if script.flags & Script.Modified:
            option = QStyleOptionViewItemV4(option)
            option.palette.setColor(QPalette.Text, QColor(Qt.red))
            option.palette.setColor(QPalette.Highlight, QColor(Qt.darkRed))
        QStyledItemDelegate.paint(self, painter, option, index)

        
    def createEditor(self, parent, option, index):
        return QLineEdit(parent)
    
    def setEditorData(self, editor, index):
        script = index.data(Qt.DisplayRole).toPyObject()
        editor.setText(script.name)
        
    def setModelData(self, editor, model, index):
        model[index.row()].name = str(editor.text())
        
class OWPythonScript(OWWidget):
    
    settingsList = ["codeFile", "libraryListSource", "currentScriptIndex", "splitterState"]
                    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Python Script')
        
        self.inputs = [("in_data", ExampleTable, self.setExampleTable), ("in_distance", orange.SymMatrix, self.setDistanceMatrix), ("in_network", orngNetwork.Network, self.setNetwork), ("in_learner", orange.Learner, self.setLearner), ("in_classifier", orange.Classifier, self.setClassifier)]
        self.outputs = [("out_data", ExampleTable), ("out_distance", orange.SymMatrix), ("out_network", orngNetwork.Network), ("out_learner", orange.Learner), ("out_classifier", orange.Classifier, Dynamic)]
        
        self.in_data = None
        self.in_network = None
        self.in_distance = None
        self.in_learner = None
        self.in_classifier = None
        
        self.codeFile = ''
        self.libraryListSource = [Script("Hello world", "print 'Hello world'\n")]
        self.currentScriptIndex = 0
        self.splitterState = None
        self.loadSettings()
        
        for s in self.libraryListSource:
            s.flags = 0
        
        self._cachedDocuments = {}
        
        self.infoBox = OWGUI.widgetBox(self.controlArea, 'Info')
        label = OWGUI.label(self.infoBox, self, "<p>Execute python script.</p><p>Input variables:<ul><li> " + \
                    "<li>".join(t[0] for t in self.inputs) + "</ul></p><p>Output variables:<ul><li>" + \
                    "<li>".join(t[0] for t in self.outputs) + "</ul></p>")
        self.libraryList = PyListModel([], self, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
#        self.libraryList.append(Script("Hello world", "print 'Hello world'\n"))
        self.libraryList.wrap(self.libraryListSource)
        
        self.controlBox = OWGUI.widgetBox(self.controlArea, 'Library')
        self.controlBox.layout().setSpacing(1)
        self.libraryView = QListView()
        self.libraryView.pyqtConfigure(editTriggers=QListView.DoubleClicked | QListView.SelectedClicked)
        self.libraryView.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.libraryView.setItemDelegate(ScriptItemDelegate(self))
        self.libraryView.setModel(self.libraryList)
        self.connect(self.libraryView.selectionModel(), SIGNAL("selectionChanged(QItemSelection, QItemSelection)"), self.onSelectedScriptChanged)
        self.controlBox.layout().addWidget(self.libraryView)
        w = ModelActionsWidget()
        
        self.addNewScriptAction = action = QAction("+", self)
        action.pyqtConfigure(toolTip="Add a new script to the library")
        self.connect(action, SIGNAL("triggered()"), self.onAddScript)
        new_empty = QAction("Add a new empty script", action)
        new_from_file = QAction("Add a new script from a file", action)
        self.connect(new_empty, SIGNAL("triggered()"), self.onAddScript)
        self.connect(new_from_file, SIGNAL("triggered()"), self.onAddScriptFromFile)
        menu = QMenu(w)
        menu.addAction(new_empty)
        menu.addAction(new_from_file)
        
#        action.setMenu(menu)
        button = w.addAction(action)
        
        self.removeAction = action = QAction("-", self)
        action.pyqtConfigure(toolTip="Remove script from library")
        self.connect(action, SIGNAL("triggered()"), self.onRemoveScript)
        w.addAction(action)
        
        action = QAction("Update", self)
        action.pyqtConfigure(toolTip="Save changes in the editor to library")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        self.connect(action, SIGNAL("triggered()"), self.commitChangesToLibrary)
        b = w.addAction(action)
#        b.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        
        action = QAction("More", self)
        action.pyqtConfigure(toolTip="More actions") #, icon=self.style().standardIcon(QStyle.SP_ToolBarHorizontalExtensionButton))
        
        self.openScriptFromFileAction = new_from_file = QAction("Import a script from a file", self)
        self.saveScriptToFile = save_to_file = QAction("Save selected script to a file", self)
        save_to_file.setShortcut(QKeySequence(QKeySequence.SaveAs))
        
        self.connect(new_from_file, SIGNAL("triggered()"), self.onAddScriptFromFile)
        self.connect(save_to_file, SIGNAL("triggered()"), self.saveScript)
        
        menu = QMenu(w)
        menu.addAction(new_from_file)
        menu.addAction(save_to_file)
        action.setMenu(menu)
        b = w.addAction(action)
        b.setPopupMode(QToolButton.InstantPopup) 
        ## TODO: set the space for the indicator
        
        w.layout().setSpacing(1)
        
        self.controlBox.layout().addWidget(w)
                    
#        OWGUI.button(self.controlBox, self, "Open...", callback=self.openScript)
#        OWGUI.button(self.controlBox, self, "Save...", callback=self.saveScript)
        
        self.runBox = OWGUI.widgetBox(self.controlArea, 'Run')
        OWGUI.button(self.runBox, self, "Execute", callback=self.execute)
        
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)
        
        self.defaultFont = defaultFont = "Monaco" if sys.platform == "darwin" else "Courier"
        self.textBox = OWGUI.widgetBox(self, 'Python script')
        self.splitCanvas.addWidget(self.textBox)
        self.text = PythonScriptEditor(self)
        self.textBox.layout().addWidget(self.text)
        
        self.textBox.setAlignment(Qt.AlignVCenter)
        self.text.setTabStopWidth(4)
        
        self.connect(self.text, SIGNAL("modificationChanged(bool)"), self.onModificationChanged)
        
        self.saveAction = action = QAction("&Save", self.text)
        action.pyqtConfigure(toolTip="Save script to file")
        action.setShortcut(QKeySequence(QKeySequence.Save))
        action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self.connect(action, SIGNAL("triggered()"), self.saveScript)
        
        self.consoleBox = OWGUI.widgetBox(self, 'Console')
        self.splitCanvas.addWidget(self.consoleBox)
        self.console = PythonConsole(self.__dict__, self)
        self.consoleBox.layout().addWidget(self.console)
        self.console.document().setDefaultFont(QFont(defaultFont))
        self.consoleBox.setAlignment(Qt.AlignBottom)
        self.console.setTabStopWidth(4)
        
        self.openScript(self.codeFile)
        try:
            self.libraryView.selectionModel().select(self.libraryList.index(self.currentScriptIndex), QItemSelectionModel.ClearAndSelect)
        except Exception:
            pass
        self.splitCanvas.setSizes([2, 1])
        if self.splitterState is not None:
            self.splitCanvas.restoreState(QByteArray(self.splitterState))
        
        self.connect(self.splitCanvas, SIGNAL("splitterMoved(int, int)"), lambda pos, ind: setattr(self, "splitterState", str(self.splitCanvas.saveState())))
        self.controlArea.layout().addStretch(1)
        self.resize(800,600)
        
    def setExampleTable(self, et):
        self.in_data = et
        
    def setDistanceMatrix(self, dm):
        self.in_distance = dm
        
    def setNetwork(self, net):
        self.in_network = net
        
    def setLearner(self, learner):
        self.in_learner = learner
        
    def setClassifier(self, classifier):
        self.in_classifier = classifier
        
    def selectedScriptIndex(self):
        rows = self.libraryView.selectionModel().selectedRows()
        if rows:
            return  [i.row() for i in rows][0]
        else:
            return None
        
    def setSelectedScript(self, index):
        selection = self.libraryView.selectionModel()
        selection.select(self.libraryList.index(index), QItemSelectionModel.ClearAndSelect)
        
    def onAddScript(self, *args):
        self.libraryList.append(Script("New script", "", 0))
        self.setSelectedScript(len(self.libraryList) - 1)
        
    def onAddScriptFromFile(self, *args):
        file = QFileDialog.getOpenFileName(self, 'Open Python Script', self.codeFile, 'Python files (*.py)\nAll files(*.*)')
        if file:
            file = str(file)
            name = os.path.basename(file)
            self.libraryList.append(Script(name, open(file, "rb").read(), 0, file))
            self.setSelectedScript(len(self.libraryList) - 1)
    
    def onRemoveScript(self, *args):
        index = self.selectedScriptIndex()
        if index is not None:
            del self.libraryList[index]
    
    def onSaveScriptToFile(self, *args):
        index = self.selectedScriptIndex()
        if index is not None:
            self.saveScript()
            
    def onSelectedScriptChanged(self, selected, deselected):
        index = [i.row() for i in selected.indexes()]
        if index:
            current = index[0] 
            if current >= len(self.libraryList):
                self.addNewScriptAction.trigger()
                return
            self.text.setDocument(self.documentForScript(current))
            self.currentScriptIndex = current
            
    def documentForScript(self, script=0):
        if type(script) != Script:
            script = self.libraryList[script]
        if script not in self._cachedDocuments:
            doc = QTextDocument(self)
            doc.setDocumentLayout(QPlainTextDocumentLayout(doc))
            doc.setPlainText(script.script)
            doc.setDefaultFont(QFont(self.defaultFont))
            doc.highlighter = PythonSyntaxHighlighter(doc)
            self.connect(doc, SIGNAL("modificationChanged(bool)"), self.onModificationChanged)
            doc.setModified(False)
            self._cachedDocuments[script] = doc
        return self._cachedDocuments[script]
    
    def commitChangesToLibrary(self, *args):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].script = self.text.toPlainText()
            self.text.document().setModified(False)
            self.libraryList.emitDataChanged(index)
            
    def onModificationChanged(self, modified):
        index = self.selectedScriptIndex()
        if index is not None:
            self.libraryList[index].flags = Script.Modified if modified else 0
            self.libraryList.emitDataChanged(index)
        
    def updateSelecetdScriptState(self):
        index = self.selectedScriptIndex()
        if index is not None:
            script = self.libraryList[index]
            self.libraryList[index] = Script(script.name, self.text.toPlainText(), 0)
            
    def openScript(self, filename=None):
        if filename == None:
            self.codeFile = str(QFileDialog.getOpenFileName(self, 'Open Python Script', self.codeFile, 'Python files (*.py)\nAll files(*.*)'))    
        else:
            self.codeFile = filename
            
        if self.codeFile == "": return
            
        f = open(self.codeFile, 'r')
        self.text.setPlainText(f.read())
        f.close()
    
    def saveScript(self):
        index = self.selectedScriptIndex()
        if index is not None:
            script = self.libraryList[index]
            filename = script.sourceFileName or self.codeFile
        else:
            filename = self.codeFile
            
        self.codeFile = QFileDialog.getSaveFileName(self, 'Save Python Script', filename, 'Python files (*.py)\nAll files(*.*)')
        
        if self.codeFile:
            fn = ""
            head, tail = os.path.splitext(str(self.codeFile))
            if not tail:
                fn = head + ".py"
            else:
                fn = str(self.codeFile)
            
            f = open(fn, 'w')
            f.write(self.text.toPlainText())
            f.close()
            
    def execute(self):
        self._script = str(self.text.toPlainText())
        self.console.write("\nRunning script:\n")
        self.console.push("exec(_script)")
        self.console.new_prompt(sys.ps1)
        for out in self.outputs:
            signal = out[0]
            self.send(signal, getattr(self, signal, None))

if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWPythonScript()
    ow.show()
    appl.exec_()
    ow.saveSettings()

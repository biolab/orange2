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
        self.stringFormat.setForeground(QBrush(Qt.green))
        self.stringFormat.setFontWeight(QFont.Bold)
        self.defFormat = QTextCharFormat()
        self.defFormat.setForeground(QBrush(Qt.black))
        self.defFormat.setFontWeight(QFont.Bold)
        
        self.keywords = ["def", "if", "else", "elif", "for", "while", "with", "try", "except",
                         "finally", "not", "in", "lambda", "None", "import", "class", "return", "print",
                         "yield", "break", "continue", "raise", "or", "and", "True", "False", "pass"]
        self.rules = [(QRegExp(r"\b%s\b" % keyword), self.keywordFormat) for keyword in self.keywords] + \
                     [(QRegExp(r"\bclass|\bdef\s+([A-Za-z]+[A-Za-z0-9]+)\s*\("), self.defFormat),
                      (QRegExp(r"'.*'"), self.stringFormat),
                      (QRegExp(r'".*"'), self.stringFormat)]
                     
        QSyntaxHighlighter.__init__(self, parent)
        
    def highlightBlock(self, text):
        for pattern, format in self.rules:
            exp = QRegExp(pattern)
            index = exp.indexIn(text) #text.indexOf(exp)
            while index >= 0:
                length = exp.matchedLength()
                if exp.numCaptures() > 0:
                    self.setFormat(exp.pos(1), len(str(exp.cap(1))), format)
#                    print exp.pos(1), str(exp.cap(1))
                else:
                    self.setFormat(exp.pos(0), len(str(exp.cap(0))), format)
#                    print index, str(exp.cap(0))
                index = exp.indexIn(text, index + length)
                
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
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(data)
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
        except:
            raise
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
            
#    def mousePressEvent(self, event):
#        pos = event.pos()
#        cursor = self.cursorForPosition(pos)
#        if cursor.position() <= self.newPromptPos:
#            self.cursorPosAtMousePress = cursor.position()
#        QPlainTextEdit.mousePressEvent(self, event)
#        
#    def mouseReleaseEvent(self, event):
#        pos = event.pos()
#        cursor = self.cursorForPosition(pos)
#        pos = cursor.position()
#        QPlainTextEdit.mousePressEvent(self, event)
#        if cursor.position() <= self.newPromptPos:
#            cursor = QTextCursor(self.textCursor())
#            cursor.setPosition(self.cursorPosAtMousePress)
##            cursor.movePosition(QTextCursor.End)
#            self.setTextCursor(cursor)
            
    def historyUp(self):
        self.setLine(self.history[self.historyInd])
        self.historyInd = min(self.historyInd + 1, len(self.history) - 1)
        
    def historyDown(self):
        self.setLine(self.history[self.historyInd])
        self.historyInd = max(self.historyInd - 1, 0)
        
    def complete(self):
        pass

class OWPythonScript(OWWidget):
    
    settingsList = ["codeFile"] 
                    
    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'Python Script')
        
        self.inputs = [("inExampleTable", ExampleTable, self.setExampleTable), ("inDistanceMatrix", orange.SymMatrix, self.setDistanceMatrix), ("inNetwork", orngNetwork.Network, self.setNetwork), ("inLearner", orange.Learner, self.setLearner), ("inClassifier", orange.Classifier, self.setClassifier)]
        self.outputs = [("outExampleTable", ExampleTable), ("outDistanceMatrix", orange.SymMatrix), ("outNetwork", orngNetwork.Network), ("outLearner", orange.Learner), ("outClassifier", orange.Classifier)]
        
        self.inNetwork = None
        self.inExampleTable = None
        self.inDistanceMatrix = None
        self.codeFile = ''
        
        self.loadSettings()
        
        self.infoBox = OWGUI.widgetBox(self.controlArea, 'Info')
        OWGUI.label(self.infoBox, self, "Execute python script.\n\nInput variables:\n - " + \
                    "\n - ".join(t[0] for t in self.inputs) + "\n\nOutput variables:\n - " + \
                    "\n - ".join(t[0] for t in self.outputs))
        
        self.controlBox = OWGUI.widgetBox(self.controlArea, 'File')
        OWGUI.button(self.controlBox, self, "Open...", callback=self.openScript)
        OWGUI.button(self.controlBox, self, "Save...", callback=self.saveScript)
        
        self.runBox = OWGUI.widgetBox(self.controlArea, 'Run')
        OWGUI.button(self.runBox, self, "Execute", callback=self.execute)
        
        self.splitCanvas = QSplitter(Qt.Vertical, self.mainArea)
        self.mainArea.layout().addWidget(self.splitCanvas)
        
        self.textBox = OWGUI.widgetBox(self, 'Python script')
        self.splitCanvas.addWidget(self.textBox)
        self.text = PythonScriptEditor(self)
        self.textBox.layout().addWidget(self.text)
        self.text.setFont(QFont("Monospace"))
        self.highlighter = PythonSyntaxHighlighter(self.text.document())
        self.textBox.setAlignment(Qt.AlignVCenter)
        self.text.setTabStopWidth(4)
        
        self.consoleBox = OWGUI.widgetBox(self, 'Console')
        self.splitCanvas.addWidget(self.consoleBox)
        self.console = PythonConsole(self.__dict__, self)
        self.consoleBox.layout().addWidget(self.console)
        self.console.setFont(QFont("Monospace"))
        self.consoleBox.setAlignment(Qt.AlignBottom)
        self.console.setTabStopWidth(4)
        
        self.openScript(self.codeFile)
        
        self.controlArea.layout().addStretch(1)
        self.resize(800,600)
        
    def setExampleTable(self, et):
        self.inExampleTable = et
        
    def setDistanceMatrix(self, dm):
        self.inDistanceMatrix = dm
        
    def setNetwork(self, net):
        self.inNetwork = net
        
    def setLearner(self, learner):
        self.inLearner = learner
        
    def setClassifier(self, classifier):
        self.inClassifier = classifier
    
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
        self.codeFile = QFileDialog.getSaveFileName(self, 'Save Python Script', self.codeFile, 'Python files (*.py)\nAll files(*.*)')
        
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
        for signal, cls in self.outputs:
            self.send(signal, getattr(self, signal, None))

if __name__=="__main__":    
    appl = QApplication(sys.argv)
    ow = OWPythonScript()
    ow.show()
    appl.exec_()

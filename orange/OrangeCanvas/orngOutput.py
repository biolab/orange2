# Author: Gregor Leban (gregor.leban@fri.uni-lj.si)
# Description:
#     print system output and exceptions into a window. Enables copy/paste
#
from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import string
from time import localtime
import traceback
import os.path, os

class OutputWindow(QMdiSubWindow):
    def __init__(self, canvasDlg, *args):
        apply(QMdiSubWindow.__init__,(self,) + args)
        self.canvasDlg = canvasDlg
        #self.canvasDlg.workspace.addWindow(self)

        self.textOutput = QTextEdit(self)
        self.textOutput.setReadOnly(1)

        self.setWidget(self.textOutput)
        self.setWindowTitle("Output Window")
        self.setWindowIcon(QIcon(canvasDlg.outputPix))

        #self.defaultExceptionHandler = sys.excepthook
        #self.defaultSysOutHandler = sys.stdout
        self.focusOnCatchException = 1
        self.focusOnCatchOutput  = 0
        self.printOutput = 1
        self.printException = 1
        self.writeLogFile = 1

        self.logFile = open(os.path.join(canvasDlg.canvasSettingsDir, "outputLog.htm"), "w") # create the log file
        self.unfinishedText = ""
        self.verbosity = 0

        #sys.excepthook = self.exceptionHandler
        #sys.stdout = self
        #self.textOutput.setText("")
        #self.setFocusPolicy(QWidget.NoFocus)

        self.resize(700,500)
        self.showNormal()

    def stopCatching(self):
        self.catchException(0)
        self.catchOutput(0)

    def closeEvent(self,ce):
        #QMessageBox.information(self,'Orange Canvas','Output window is used to print output from canvas and widgets and therefore can not be closed.','Ok')
        self.catchException(0)
        self.catchOutput(0)
        wins = self.canvasDlg.workspace.getDocumentList()
        if wins != []:
            wins[0].setFocus()

    def focusInEvent(self, ev):
        self.canvasDlg.enableSave(1)

    def setVerbosity(self, verbosity):
        self.verbosity = verbosity

    def catchException(self, catch):
        if catch: sys.excepthook = self.exceptionHandler
        else:     sys.excepthook = self.defaultExceptionHandler

    def catchOutput(self, catch):
        if catch:    sys.stdout = self
        else:         sys.stdout = self.defaultSysOutHandler

    def setFocusOnException(self, focusOnCatchException):
        self.focusOnCatchException = focusOnCatchException

    def setFocusOnOutput(self, focusOnCatchOutput):
        self.focusOnCatchOutput = focusOnCatchOutput

    def printOutputInStatusBar(self, printOutput):
        self.printOutput = printOutput

    def printExceptionInStatusBar(self, printException):
        self.printException = printException

    def setWriteLogFile(self, write):
        self.writeLogFile = write

    def clear(self):
        self.textOutput.clear()

    # print text produced by warning and error widget calls
    def widgetEvents(self, text, eventVerbosity = 1):
        if self.verbosity >= eventVerbosity:
            if text != None:
                self.write(str(text))
            self.canvasDlg.setStatusBarEvent(QString(text))

    # simple printing of text called by print calls
    def write(self, text):
        # is this some extra info for debuging
        #if len(text) > 7 and text[0:7] == "<extra>":
        #    if not self.printExtraOutput: return
        #    text = text[7:]
        text = str(text)
        text = text.replace("<", "(").replace(">", ")")    # since this is rich text control, we have to replace special characters
        text = text.replace("(br)", "<br>")
        text = text.replace("(nobr)", "<nobr>").replace("(/nobr)", "</nobr>")
        text = text.replace("(b)", "<b>").replace("(/b)", "</b>")
        text = text.replace("(i)", "<i>").replace("(/i)", "</i>")
        text = text.replace("(hr)", "<hr>")
        text = text.replace("\n", "<br>\n")   # replace new line characters with <br> otherwise they don't get shown correctly in html output
        #text = "<nobr>" + text + "</nobr>"

        if self.focusOnCatchOutput:
            self.canvasDlg.menuItemShowOutputWindow()
            #self.canvasDlg.workspace.cascade()    # cascade shown windows

        if self.writeLogFile:
            #self.logFile.write(str(text) + "<br>\n")
            self.logFile.write(text)

        cursor = QTextCursor(self.textOutput.textCursor())                # clear the current text selection so that
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # the text will be appended to the end of the
        self.textOutput.setTextCursor(cursor)                             # existing text
        if text == " ": self.textOutput.insertHtml("&nbsp;")
        else:           self.textOutput.insertHtml(text)                                  # then append the text
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # and then scroll down to the end of the text
        self.textOutput.setTextCursor(cursor)

        if text[-1:] == "\n":
            if self.printOutput:
                self.canvasDlg.setStatusBarEvent(self.unfinishedText + text)
            self.unfinishedText = ""
        else:
            self.unfinishedText += text

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass

    def exceptionHandler(self, type, value, tracebackInfo):
        if self.focusOnCatchException:
            self.canvasDlg.menuItemShowOutputWindow()
            #self.canvasDlg.workspace.cascade()    # cascade shown windows

        text = ""
        if str(self.textOutput.toPlainText()) not in ["", "\n"]:
            text += "<hr>"
        t = localtime()
        text += "<nobr>Unhandled exception of type <b>%s </b> occured at %d:%02d:%02d:</nobr><br><nobr>Traceback:</nobr><br>" % ( str(type).replace("<", "(").replace(">", ")") , t[3],t[4],t[5])

        if self.printException:
            self.canvasDlg.setStatusBarEvent("Unhandled exception of type %s occured at %d:%02d:%02d. See output window for details." % ( str(type) , t[3],t[4],t[5]))

        # TO DO:repair this code to shown full traceback. when 2 same errors occur, only the first one gets full traceback, the second one gets only 1 item
        list = traceback.extract_tb(tracebackInfo, 10)
        space = "&nbsp; &nbsp; "
        totalSpace = space
        for i in range(len(list)):
            (file, line, funct, code) = list[i]
            if code == None: continue
            (dir, filename) = os.path.split(file)
            text += "<nobr>" + totalSpace + "File: <u>" + filename + "</u>  in line %4d</nobr><br>" %(line)
            text += "<nobr>" + totalSpace + "<nobr>Function name: %s</nobr><br>" % (funct)
            if i == len(list)-1:
                text += "<nobr>" + totalSpace + "Code: <b>" + code + "</b></nobr><br>"
            else:
                text += "<nobr>" + totalSpace + "Code: " + code + "</nobr><br>"
                totalSpace += space

        value = str(value).replace("<", "(").replace(">", ")")    # since this is rich text control, we have to replace special characters
        text += "<nobr>" + totalSpace + "Exception type: <b>" + str(type).replace("<", "(").replace(">", ")") + "</b></nobr><br>"
        text += "<nobr>" + totalSpace + "Exception value: <b>" + value+ "</b></nobr><hr>"
        text = text.replace("<br>","<br>\n")

        cursor = QTextCursor(self.textOutput.textCursor())                # clear the current text selection so that
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # the text will be appended to the end of the
        self.textOutput.setTextCursor(cursor)                             # existing text
        self.textOutput.insertHtml(text)                                  # then append the text
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # and then scroll down to the end of the text
        self.textOutput.setTextCursor(cursor)

        if self.writeLogFile:
            self.logFile.write(str(text) + "<br>\n")

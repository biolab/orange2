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

try:
    __IPYTHON__  #We are running orange from ipython - it already has redirected sys.stdout
    __DISABLE_OUTPUT__ = True
except NameError:
    __DISABLE_OUTPUT__ = False
    
class OutputWindow(QDialog):
    def __init__(self, canvasDlg, *args):
        QDialog.__init__(self, None, Qt.Window)
        self.canvasDlg = canvasDlg

        self.textOutput = QPlainTextEdit(self)
        self.textOutput.setReadOnly(1)
#        self.textOutput.zoomIn(1)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.textOutput)
        self.layout().setMargin(2)
        self.setWindowTitle("Output Window")
        self.setWindowIcon(QIcon(canvasDlg.outputPix))

        self.defaultExceptionHandler = sys.excepthook
        self.defaultSysOutHandler = sys.stdout

        self.logFile = open(os.path.join(canvasDlg.canvasSettingsDir, "outputLog.html"), "w") # create the log file
        self.unfinishedText = ""

        w = h = 500
        if canvasDlg.settings.has_key("outputWindowPos"):
            desktop = qApp.desktop()
            deskH = desktop.availableGeometry(self).height()
            deskW = desktop.availableGeometry(self).width()
            w, h, x, y = canvasDlg.settings["outputWindowPos"]
            if x >= 0 and y >= 0 and deskH >= y+h and deskW >= x+w: 
                self.move(QPoint(x, y))
            else: 
                w = h = 500
        self.resize(w, h)
            
        self.hide()

    def stopCatching(self):
        self.catchException(0)
        self.catchOutput(0)

    def showEvent(self, ce):
        ce.accept()
        QDialog.showEvent(self, ce)
        settings = self.canvasDlg.settings
        if settings.has_key("outputWindowPos"):
            w, h, x, y = settings["outputWindowPos"]
            self.move(QPoint(x, y))
            self.resize(w, h)
        
    def hideEvent(self, ce):
        self.canvasDlg.settings["outputWindowPos"] = (self.width(), self.height(), self.pos().x(), self.pos().y())
        ce.accept()
        QDialog.hideEvent(self, ce)
                
    def closeEvent(self,ce):
        self.canvasDlg.settings["outputWindowPos"] = (self.width(), self.height(), self.pos().x(), self.pos().y())
        if getattr(self.canvasDlg, "canvasIsClosing", 0):
            self.catchException(0)
            self.catchOutput(0)
            ce.accept()
            QDialog.closeEvent(self, ce)
        else:
            self.hide()

    def catchException(self, catch):
        if __DISABLE_OUTPUT__:
            return
        if catch: sys.excepthook = self.exceptionHandler
        else:     sys.excepthook = self.defaultExceptionHandler

    def catchOutput(self, catch):
        if __DISABLE_OUTPUT__:
            return 
        if catch:    sys.stdout = self
        else:         sys.stdout = self.defaultSysOutHandler

    def clear(self):
        self.textOutput.clear()

    # print text produced by warning and error widget calls
    def widgetEvents(self, text, eventVerbosity = 1):
        if self.canvasDlg.settings["outputVerbosity"] >= eventVerbosity:
            if text != None:
                self.write(str(text))
            self.canvasDlg.setStatusBarEvent(QString(text))

    # simple printing of text called by print calls
    def write(self, text):
        Text = self.getSafeString(text)
        Text = Text.replace("\n", "<br>\n")   # replace new line characters with <br> otherwise they don't get shown correctly in html output
#        text = "<nobr>" + text + "</nobr>"

        if self.canvasDlg.settings["focusOnCatchOutput"]:
            self.canvasDlg.menuItemShowOutputWindow()

        if self.canvasDlg.settings["writeLogFile"]:
            #self.logFile.write(str(text) + "<br>\n")
            self.logFile.write(Text)

        cursor = QTextCursor(self.textOutput.textCursor())                # clear the current text selection so that
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # the text will be appended to the end of the
        self.textOutput.setTextCursor(cursor)                             # existing text
#        if text == " ": self.textOutput.insertHtml("&nbsp;")
#        else:           self.textOutput.insertHtml(Text)                                  # then append the text

        self.textOutput.insertPlainText(text)
        
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # and then scroll down to the end of the text

        self.textOutput.setTextCursor(cursor)

#        self.textOutput.insertPlainText(text)

        if Text[-1:] == "\n":
            if self.canvasDlg.settings["printOutputInStatusBar"]:
                self.canvasDlg.setStatusBarEvent(self.unfinishedText + text)
            self.unfinishedText = ""
        else:
            self.unfinishedText += text

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        pass
    
    def getSafeString(self, s):
        return str(s).replace("<", "&lt;").replace(">", "&gt;")

    def exceptionHandler(self, type, value, tracebackInfo):
        if self.canvasDlg.settings["focusOnCatchException"]:
            self.canvasDlg.menuItemShowOutputWindow()

        t = localtime()
        text = "<nobr>Unhandled exception of type %s occured at %d:%02d:%02d:</nobr><br><nobr>Traceback:</nobr><br>\n" % ( self.getSafeString(type.__name__), t[3],t[4],t[5])

        if self.canvasDlg.settings["printExceptionInStatusBar"]:
            self.canvasDlg.setStatusBarEvent("Unhandled exception of type %s occured at %d:%02d:%02d. See output window for details." % ( str(type) , t[3],t[4],t[5]))

        # TO DO:repair this code to shown full traceback. when 2 same errors occur, only the first one gets full traceback, the second one gets only 1 item
        list = traceback.extract_tb(tracebackInfo, 10)
        space = "&nbsp; "
        totalSpace = space
        for i in range(len(list)):
            (file, line, funct, code) = list[i]
            if code == None: continue
            (dir, filename) = os.path.split(file)
            text += "<nobr>" + totalSpace + "File: <b>" + filename + "</b>, line %4d" %(line) + " in <b>%s</b></nobr><br>\n" % (self.getSafeString(funct))
            text += "<nobr>" + totalSpace + "Code: " + code + "</nobr><br>\n"
            totalSpace += space

        lines = traceback.format_exception_only(type, value)
        for line in lines[:-1]:
            text += "<nobr>" + totalSpace + self.getSafeString(line) + "</nobr><br>\n"
        text += "<nobr><b>" + totalSpace + self.getSafeString(lines[-1]) + "</b></nobr><br>\n"

        cursor = QTextCursor(self.textOutput.textCursor())                # clear the current text selection so that
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # the text will be appended to the end of the
#        self.textOutput.setTextCursor(cursor)                             # existing text
#        self.textOutput.insertHtml(text)                                  # then append the text
        self.textOutput.appendHtml(text)
        cursor.movePosition(QTextCursor.End, QTextCursor.MoveAnchor)      # and then scroll down to the end of the text
        self.textOutput.setTextCursor(cursor)

#        self.textOutput.appendPlainText(traceback.format_exc(10))

        if self.canvasDlg.settings["writeLogFile"]:
            self.logFile.write(str(text) + "<br>\n")

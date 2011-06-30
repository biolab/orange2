from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtWebKit import *
import os

class HelpWindow(QDialog):
    def __init__(self, canvasDlg):
        QDialog.__init__(self)
        self.setWindowTitle("Orange Canvas Help")
        self.canvasDlg = canvasDlg
        
        self.setWindowIcon(QIcon(os.path.join(self.canvasDlg.widgetDir, "icons/Unknown.png")))


        self.setLayout(QVBoxLayout())
        self.layout().setMargin(2)        
#===============================================================================
#        hbox = QWidget(self)
#        self.layout().addWidget(hbox)
#        hbox.setLayout(QHBoxLayout())
#        cb = QCheckBox(self)
#        cb.setChecked(canvasDlg.settings["synchronizeHelp"])
#        self.connect(cb, SIGNAL("stateChanged(int)"), self.synchronizeHelpClicked)
#        hbox.layout().addWidget(cb)
#        hbox.layout().addWidget(QLabel("Show context sensitive help", self))
#        hbox.layout().addStretch(100)
#===============================================================================
        
        self.helpBrowser = QWebView(self)
        self.layout().addWidget(self.helpBrowser)
        
    def showHelpFor(self, widgetInfo, bringToFront=False):
        helpFileName = os.path.join(widgetInfo.docDir(), "%s.htm" % (widgetInfo.fileName[2:])).replace("\\", "/")
        if not os.path.exists(helpFileName):
            QMessageBox.warning( None, "Not available", "Sorry, there is no documentation available for this widget.", QMessageBox.Ok)
            return
        self.open("file:///"+helpFileName)
            
    def open(self, url, bringToFront=False, modal=False):
        self.helpBrowser.load(QUrl(url))
        if modal:
            self.exec_()
        else:
            self.show()
            if bringToFront:
                self.raise_()

    def synchronizeHelpClicked(self, st):
        canvasDlg.settings["synchronizeHelp"] = st == Qt.Checked
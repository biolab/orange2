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
        
    def showHelpFor(self, widget, bringToFront=False):
        if os.path.relpath(widget.thisWidgetDir, widget.addOnsDir).startswith(widget._category):
            catalogDir = os.path.join(widget.addOnsDir,widget._category,"doc","catalog")
        else:
            catalogDir = os.path.join(widget.orangeDir, "doc", "catalog", widget._category)
        helpFileName = "%s/%s.htm" % (catalogDir, widget.__class__.__name__[2:])
        if not os.path.exists(helpFileName):
            QMessageBox.warning( None, "Not available", "Sorry, there is no documentation available for this widget.", QMessageBox.Ok)
            return
        self.helpBrowser.load(QUrl("file:///"+helpFileName))
        self.show()
        if bringToFront:
            self.raise_()

    def synchronizeHelpClicked(self, st):
        canvasDlg.settings["synchronizeHelp"] = st == Qt.Checked
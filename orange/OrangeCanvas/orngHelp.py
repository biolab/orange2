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
        self.helpBrowser.load(QUrl("file:///%s/doc/widgets/catalog/%s/%s.htm" % (widget.orangeDir, widget._category, widget.__class__.__name__[2:])))
        self.show()
        if bringToFront:
            self.raise_()

    def synchronizeHelpClicked(self, st):
        canvasDlg.settings["synchronizeHelp"] = st == Qt.Checked
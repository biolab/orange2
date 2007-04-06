#
# OWWidget.py
# Orange Widget
# A General Orange Widget, from which all the Orange Widgets are derived
#

from OWBaseWidget import *

class OWWidget(OWBaseWidget):
    def __init__( self, parent = None, signalManager = None, title = "Qt Orange Widget", wantGraph = FALSE, wantStatusBar = FALSE, savePosition = True, noReport = False):
        """
        Initialization
        Parameters:
            title - The title of the\ widget, including a "&" (for shortcut in about box)
            wantGraph - displays a save graph button or not
        """

        OWBaseWidget.__init__(self, parent, signalManager, title, savePosition = savePosition)

        self.mainArea=QWidget(self)
        self.controlArea=QVBox(self)
        self.buttonBackground=QVBox(self)
        self.space = self.controlArea
        #self.controlArea.setMaximumWidth(250)
        #self.space=QVBox(self)
        #self.grid=QGridLayout(self,2,2,5)
        self.grid=QGridLayout(self,3,2,5)
        self.grid.addWidget(self.controlArea,0,0)
        #self.grid.addWidget(self.space,1,0)
        self.grid.addWidget(self.buttonBackground,1,0)
        self.grid.setRowStretch(0,20)
        self.grid.setColStretch(0,10)
        self.grid.setColStretch(1,50)
        self.grid.addMultiCellWidget(self.mainArea,0,2,1,1)
        self.space.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        #self.controlArea.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding))

        #self.setSizeGripEnabled(1)

        if wantGraph:    self.graphButton=QPushButton("&Save Graph",self.buttonBackground)

        self.reportData = None
        if hasattr(self, "sendReport") and not noReport:
            self.reportButton = QPushButton("&Report", self.buttonBackground)
            self.connect(self.reportButton, SIGNAL("clicked()"), self.sendReport)

        self.widgetStatusArea = QHBox(self)
        #self.widgetStatusArea.setFrameStyle (QFrame.Panel + QFrame.Sunken)
        self.grid.addMultiCellWidget(self.widgetStatusArea, 3, 3, 0, 1)
        #self.statusBar = QStatusBar(self.widgetStatusArea)
        #self.statusBar.setSizeGripEnabled(0)
        self.statusBarIconArea = QHBox(self.widgetStatusArea)
        self.statusBarTextArea = QLabel("", self.widgetStatusArea)
        self.statusBarIconArea.setFrameStyle (QFrame.Panel + QFrame.Sunken)
        self.statusBarTextArea.setFrameStyle (QFrame.Panel + QFrame.Sunken)
        self.statusBarTextArea.setSizePolicy(QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred))
        #self.statusBar.addWidget(self.statusBarIconArea, 0)
        #self.statusBar.addWidget(self.statusBarTextArea, 1)
        #self.statusBarIconArea.setMinimumSize(16*3,16)
        #self.statusBarIconArea.setMaximumSize(16*3,16)
        self.statusBarIconArea.setMinimumSize(16*2,18)
        self.statusBarIconArea.setMaximumSize(16*2,18)

        # create pixmaps used in statusbar to show info, warning and error messages
        #self._infoWidget, self._infoPixmap = self.createPixmapWidget(self.statusBarIconArea, self.widgetDir + "icons/triangle-blue.png")
        self._warningWidget, self._warningPixmap = self.createPixmapWidget(self.statusBarIconArea, self.widgetDir + "icons/triangle-orange.png")
        self._errorWidget, self._errorPixmap = self.createPixmapWidget(self.statusBarIconArea, self.widgetDir + "icons/triangle-red.png")
##        spacer = QWidget(self.statusBarIconArea)
##        spacer.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, 0))

        if wantStatusBar == 0:
            self.widgetStatusArea.hide()

        self.resize(640,480)

    def createPixmapWidget(self, parent, iconName):
        w = QWidget(parent)
        w.setMinimumSize(16,16)
        w.setMaximumSize(16,16)
        if os.path.exists(iconName):
            pix = QPixmap(iconName)
        else:
            pix = None

        return w, pix

    def setState(self, stateType, id, text):
        stateChanged = OWBaseWidget.setState(self, stateType, id, text)
        if not stateChanged:
            return

        #for state, widget, icon, use in [("Info", self._infoWidget, self._infoPixmap, self._owInfo), ("Warning", self._warningWidget, self._warningPixmap, self._owWarning), ("Error", self._errorWidget, self._errorPixmap, self._owError)]:
        for state, widget, icon, use in [("Warning", self._warningWidget, self._warningPixmap, self._owWarning), ("Error", self._errorWidget, self._errorPixmap, self._owError)]:
            if use and self.widgetState[state] != {}:
                widget.setBackgroundPixmap(icon)
                QToolTip.add(widget, "\n".join(self.widgetState[state].values()))
            else:
                widget.setBackgroundPixmap(QPixmap())
                QToolTip.remove(widget)

##        if self.widgetStateHandler:
##            self.widgetStateHandler()

        #if (stateType == "Info" and self._owInfo) or (stateType == "Warning" and self._owWarning) or (stateType == "Error" and self._owError):
        if (stateType == "Warning" and self._owWarning) or (stateType == "Error" and self._owError):
            if text:
                self.setStatusBarText(stateType + ": " + text)
            else:
                self.setStatusBarText("")
        self.updateStatusBarState()
        #qApp.processEvents()

    def updateStatusBarState(self):
        if self._owShowStatus and (self.widgetState["Warning"] != {} or self.widgetState["Error"] != {}):
            self.widgetStatusArea.show()
        else:
            self.widgetStatusArea.hide()

    def setStatusBarText(self, text):
        self.statusBarTextArea.setText("  " + text)
        #qApp.processEvents()

    def startReport(self, name, needDirectory = False):
        if self.reportData:
            print "Cannot open a new report when an old report is still active"
            return False
        self.reportData = "<H1>%s</H1>\n" % name
        if needDirectory:
            import OWReport
            return OWReport.createDirectory()
        else:
            return True

    def reportSection(self, title):
        self.reportData += "<H2>%s</H2>\n" % title

    def reportSubsection(self, title):
        self.reportData += "<H3>%s</H3>\n" % title

    def reportList(self, items):
        self.startReportList()
        for item in items:
            self.addToReportList(item)
        self.finishReportList()

    def reportImage(self, filename):
        self.reportData += '<IMG src="%s"/>' % filename

    def startReportList(self):
        self.reportData += "<UL>\n"

    def addToReportList(self, item):
        self.reportData += "    <LI>%s</LI>\n" % item

    def finishReportList(self):
        self.reportData += "</UL>\n"

    def reportSettings(self, settingsList, closeList = True):
        self.startReportList()
        for item in settingsList:
            if item:
                self.addToReportList("<B>%s:</B> %s" % item)
        if closeList:
            self.finishReportList()

    def reportRaw(self, text):
        self.reportData += text

    def finishReport(self):
        import OWReport
        OWReport.feed(self.reportData or "")
        self.reportData = None

if __name__ == "__main__":
    a=QApplication(sys.argv)
    oww=OWWidget()
    a.setMainWidget(oww)
    oww.show()
    a.exec_loop()
    oww.saveSettings()

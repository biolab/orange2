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


        if wantStatusBar:
            self.widgetStatusArea = QHBox(self)
            self.grid.addMultiCellWidget(self.widgetStatusArea, 3, 3, 0, 1)
            self.statusBar = QStatusBar(self.widgetStatusArea, )
            self.statusBar.setSizeGripEnabled(0)
            self.widgetStatusArea.setFrameStyle (QFrame.Panel + QFrame.Sunken)
            #self.statusBar.hide()
        
        self.resize(640,480)

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

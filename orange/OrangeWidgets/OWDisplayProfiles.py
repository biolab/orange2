"""
<name>Display Profiles</name>
<description>None.</description>
<category>Genomics</category>
<icon>icons\DisplayProfiles.png</icon>
<priority>10</priority>
"""

from OData import *
from OWTools import *
from OWWidget import *
from OWGraph import *
from OWGUI import *
from OWDisplayProfilesOptions import *

import statc

class curveWithDataQwtPlotCurve(QwtPlotCurve):
    def __init__(self, parent = None, text = None, info = None):
        QwtPlotCurve.__init__(self, parent, text)


class profilesGraph(OWGraph):
    def __init__(self, parent = None, name = None, title = ""):
        OWGraph.__init__(self, parent, name)
        self.setYRlabels(None)
        self.enableGridXB(0)
        self.enableGridYL(0)
        self.setAxisMaxMajor(QwtPlot.xBottom, 10)
        self.setAxisMaxMinor(QwtPlot.xBottom, 0)
        self.setAxisMaxMajor(QwtPlot.yLeft, 10)
        self.setAxisMaxMinor(QwtPlot.yLeft, 5)
        self.setShowMainTitle(1)
        self.setMainTitle(title)
        self.setAxisAutoScale(QwtPlot.xBottom)
        self.setAxisAutoScale(QwtPlot.xTop)
        self.setAxisAutoScale(QwtPlot.yLeft)
        self.setAxisAutoScale(QwtPlot.yRight)

        self.showAverageProfile = 1
        self.showSingleProfiles = 0
        self.groups = None ##[('grp1', ['0', '2', '4']), ('grp2', ['4', '6', '8', '10', '12', '14']), ('grp3', ['16', '18'])]

        self.removeCurves()
##        self.connect(self, SIGNAL("plotMouseMoved(const QMouseEvent &)"), self.onMouseMoved)

    def removeCurves(self):
        OWGraph.removeCurves(self)
        self.classColor = None
        self.profileCurveKeys = []
        self.averageProfileCurveKeys = []
        self.showClasses = []

    def setData(self, data, classColor, ShowAverageProfile, ShowSingleProfiles):
        self.removeCurves()
        self.classColor = classColor
        self.showAverageProfile = ShowAverageProfile
        self.showSingleProfiles = ShowSingleProfiles
        print self.showAverageProfile, self.showSingleProfiles

        self.groups = [('grp', data.domain.attributes)]
        ## go group by group
        avgCurveData = []
        ccn = 0
        for c in data.domain.classVar.values:
            classSymb = QwtSymbol(QwtSymbol.Ellipse, QBrush(self.classColor[ccn]), QPen(self.classColor[ccn]), QSize(7,7)) ##self.black
            self.showClasses.append(0)

            self.profileCurveKeys.append([])
            self.averageProfileCurveKeys.append([])
            grpcnx = 0
            for (grpname, grpattrs) in self.groups:
                oneClassData = data.select({data.domain.classVar.name:c})
                oneGrpData = oneClassData.select(orange.Domain(grpattrs, oneClassData.domain))

                ## single profiles
                nativeData = oneGrpData.native(2)
                for e in nativeData:
                    y = []
                    x = []
                    xcn = grpcnx
                    en = e.native(1)
                    for v in en:
                        if not v.isSpecial():
                            y.append( v.native() )
                            x.append( xcn )
                        xcn += 1
                    ckey = self.insertCurve('')
                    self.setCurvePen(ckey, QPen(self.classColor[ccn], 1))
                    self.setCurveData(ckey, x, y)
                    self.setCurveSymbol(ckey, classSymb)
                    self.profileCurveKeys[-1].append(ckey)

                ## average profile
                y = []
                x = []
                xcn = grpcnx
                bas = orange.DomainBasicAttrStat(oneGrpData)
                for a in bas:
                    if a:
                        y.append( a.avg )
                        x.append( xcn )
                    xcn += 1

                avgCurveData.append((x, y, ccn) ) ## postpone rendering until the very last, so average curves are on top of all others
                grpcnx += len(grpattrs)
            ccn += 1

        for (x, y, tmpCcn) in avgCurveData:
            ckey = self.insertCurve('')
            self.setCurvePen(ckey, QPen(self.classColor[tmpCcn], 3))
            self.setCurveData(ckey, x, y)
            self.averageProfileCurveKeys[tmpCcn].append(ckey)

        ## generate labels for attributes
        labels = []
        for (grpname, grpattrs) in self.groups:
            for a in grpattrs:
                labels.append( a.name)

        self.setXlabels(labels)
        self.updateCurveDisplay()

    def updateCurveDisplay(self):
        for cNum in range(len(self.showClasses)):
            showCNum = (self.showClasses[cNum] <> 0)

            ## single profiles
            b = showCNum and self.showSingleProfiles
            for ckey in self.profileCurveKeys[cNum]:
                curve =  self.curve(ckey)
                if curve <> None: curve.setEnabled(b)

            ## average profiles
            b = showCNum and self.showAverageProfile ## 1 = show average profiles for now
            for ckey in self.averageProfileCurveKeys[cNum]:
                curve =  self.curve(ckey)
                if curve <> None: curve.setEnabled(b)

        self.updateLayout()
        self.update()

    def setShowClasses(self, list):
        self.showClasses = list
        self.updateCurveDisplay()

    def setShowAverageProfile(self, v):
        self.showAverageProfile = v
        self.updateCurveDisplay()

    def setShowSingleProfiles(self, v):
        self.showSingleProfiles = v
        self.updateCurveDisplay()

    def setPointWidth(self, v):
        for cNum in range(len(self.showClasses)):
            for ckey in self.profileCurveKeys[cNum]:
                symb = self.curveSymbol(ckey)
                symb.setSize(v, v)
                self.setCurveSymbol(ckey, symb)
        self.update()

    def setCurveWidth(self, v):
        for cNum in range(len(self.showClasses)):
            for ckey in self.profileCurveKeys[cNum]:
                self.setCurvePen(ckey, QPen(self.classColor[cNum], v))
        self.update()

    def setAverageCurveWidth(self, v):
        for cNum in range(len(self.showClasses)):
            for ckey in self.averageProfileCurveKeys[cNum]:
                self.setCurvePen(ckey, QPen(self.classColor[cNum], v))
        self.update()

    def sizeHint(self):
        return QSize(170, 170)

    def onMouseMoved(self, e):
        (key, foo1, x, y, foo2) = self.closestCurve(e.pos().x(), e.pos().y())
        print e.pos().x(), e.pos().y(), key, foo1, x, y, foo2
        print self.invTransform(QwtPlot.xBottom, e.pos().x()), self.invTransform(QwtPlot.yLeft, e.pos().y())


class OWDisplayProfiles(OWWidget):
    settingsList = ["PointWidth", "CurveWidth", "AverageCurveWidth", "ShowAverageProfile", "ShowSingleProfiles"]
    def __init__(self,parent=None):
        "Constructor"
        OWWidget.__init__(self,
        parent,
        "&Display Profiles",
        """None.
        """,
        TRUE,
        TRUE)

        #set default settings
        self.ShowAverageProfile = 1
        self.ShowSingleProfiles = 0
        self.PointWidth = 4
        self.CurveWidth = 1
        self.AverageCurveWidth = 6

        #load settings
        self.loadSettings()

        # GUI
        self.box = QVBoxLayout(self.mainArea)
        self.graph = profilesGraph(self.mainArea, "")
        self.connect(self.graphButton, SIGNAL("clicked()"), self.graph.saveToFile)
        self.box.addWidget(self.graph)
        
        # inputs
        # data and graph temp variables
        self.addInput("cdata")
        self.addInput("data")

        # temp variables
        self.MAdata = None
        self.MAnoclass = 1 
        self.classColor = None
        self.numberOfClasses  = 0

        self.options = OWDisplayProfilesOptions()
        self.setOptions()

        #connect settingsbutton to show options
        self.connect(self.settingsButton, SIGNAL("clicked()"), self.options.show)

        #connect GUI controls of options in options dialog to settings
        self.connect(self.options.pointWidthSlider, SIGNAL("valueChanged(int)"), self.setPointWidth)
        self.connect(self.options.lineWidthSlider, SIGNAL("valueChanged(int)"), self.setCurveWidth)
        self.connect(self.options.averageLineWidthSlider, SIGNAL("valueChanged(int)"), self.setAverageCurveWidth)

        # GUI connections
        ## class selection (classQLB)
        self.classQVGB = QVGroupBox(self.space)
        self.classQVGB.setTitle("Classes")
        self.classQLB = QListBox(self.classQVGB)
        self.classQLB.setSelectionMode(QListBox.Multi)
        self.unselectAllClassedQLB = QPushButton("(Un)select all", self.classQVGB)
        self.connect(self.unselectAllClassedQLB, SIGNAL("clicked()"), self.SUAclassQLB)
        self.connect(self.classQLB, SIGNAL("selectionChanged()"), self.classSelectionChange)

        ## show single/average profile
        self.showAverageQLB = QPushButton("Show Average", self.classQVGB)
        self.showAverageQLB.setToggleButton(1)
        self.showAverageQLB.setOn(self.ShowAverageProfile)
        self.showSingleQLB = QPushButton("Show Single", self.classQVGB)
        self.showSingleQLB.setToggleButton(1)
        self.showSingleQLB.setOn(self.ShowSingleProfiles)
        self.connect(self.showAverageQLB, SIGNAL("toggled(bool)"), self.setShowAverageProfile)
        self.connect(self.showSingleQLB, SIGNAL("toggled(bool)"), self.setShowSingleProfiles)

        self.graph.canvas().setMouseTracking(1)

        self.zoomStack = []
        self.connect(self.graph,
                     SIGNAL('plotMousePressed(const QMouseEvent&)'),
                     self.onMousePressed)
        self.connect(self.graph,
                     SIGNAL('plotMouseReleased(const QMouseEvent&)'),
                     self.onMouseReleased)

    def onMousePressed(self, e):
        if Qt.LeftButton == e.button():
            # Python semantics: self.pos = e.pos() does not work; force a copy
            self.xpos = e.pos().x()
            self.ypos = e.pos().y()
            self.graph.enableOutline(1)
            self.graph.setOutlinePen(QPen(Qt.black))
            self.graph.setOutlineStyle(Qwt.Rect)
            self.zooming = 1
            if self.zoomStack == []:
                self.zoomState = (
                    self.graph.axisScale(QwtPlot.xBottom).lBound(),
                    self.graph.axisScale(QwtPlot.xBottom).hBound(),
                    self.graph.axisScale(QwtPlot.yLeft).lBound(),
                    self.graph.axisScale(QwtPlot.yLeft).hBound(),
                    )
        elif Qt.RightButton == e.button():
            self.zooming = 0
        # fake a mouse move to show the cursor position

    # onMousePressed()

    def onMouseReleased(self, e):
        if Qt.LeftButton == e.button():
            xmin = min(self.xpos, e.pos().x())
            xmax = max(self.xpos, e.pos().x())
            ymin = min(self.ypos, e.pos().y())
            ymax = max(self.ypos, e.pos().y())
            self.graph.setOutlineStyle(Qwt.Cross)
            xmin = self.graph.invTransform(QwtPlot.xBottom, xmin)
            xmax = self.graph.invTransform(QwtPlot.xBottom, xmax)
            ymin = self.graph.invTransform(QwtPlot.yLeft, ymin)
            ymax = self.graph.invTransform(QwtPlot.yLeft, ymax)
            if xmin == xmax or ymin == ymax:
                return
            self.zoomStack.append(self.zoomState)
            self.zoomState = (xmin, xmax, ymin, ymax)
            self.graph.enableOutline(0)
        elif Qt.RightButton == e.button():
            if len(self.zoomStack):
                xmin, xmax, ymin, ymax = self.zoomStack.pop()
            else:
                self.graph.setAxisAutoScale(QwtPlot.xBottom)
                self.graph.setAxisAutoScale(QwtPlot.yLeft)
                self.graph.replot()
                return

        self.graph.setAxisScale(QwtPlot.xBottom, xmin, xmax)
        self.graph.setAxisScale(QwtPlot.yLeft, ymin, ymax)
        self.graph.replot()

    def saveToFile(self):
        qfileName = QFileDialog.getSaveFileName("graph.png","Portable Network Graphics (.PNG)\nWindows Bitmap (.BMP)\nGraphics Interchange Format (.GIF)", None, "Save to..")
        fileName = str(qfileName)
        if fileName == "": return
        (fil,ext) = os.path.splitext(fileName)
        ext = ext.replace(".","")
        ext = ext.upper()
        cl = 0
        for g in self.graphs:
            if g.isVisible():
                clfname = fil + "_" + str(cl) + "." + ext
                g.saveToFileDirect(clfname, ext)
            cl += 1

    def setShowAverageProfile(self, v):
        self.ShowAverageProfile = v
        self.graph.setShowAverageProfile(v)

    def setShowSingleProfiles(self, v):
        self.ShowSingleProfiles = v
        self.graph.setShowSingleProfiles(v)

    def setPointWidth(self, v):
        self.PointWidth = v
        self.graph.setPointWidth(v)

    def setCurveWidth(self, v):
        self.CurveWidth = v
        self.graph.setCurveWidth(v)

    def setAverageCurveWidth(self, v):
        self.AverageCurveWidth = v
        self.graph.setAverageCurveWidth(v)

    def setOptions(self):
        self.options.pointWidthSlider.setValue(self.PointWidth)
        self.options.pointWidthLCD.display(self.PointWidth)
        self.setPointWidth(self.PointWidth)
        #
        self.options.lineWidthSlider.setValue(self.CurveWidth)
        self.options.lineWidthLCD.display(self.CurveWidth)
        self.setCurveWidth(self.CurveWidth)
        #
        self.options.averageLineWidthSlider.setValue(self.AverageCurveWidth)
        self.options.averageLineWidthLCD.display(self.AverageCurveWidth)
        self.setAverageCurveWidth(self.AverageCurveWidth)

    ##
    def selectUnselectAll(self, qlb):
        selected = 0
        for i in range(qlb.count()):
            if qlb.isSelected(i):
                selected = 1
                break
        qlb.selectAll(not(selected))

    def SUAclassQLB(self):
        self.selectUnselectAll(self.classQLB)
    ##

    ## class selection (classQLB)
    def classSelectionChange(self):
        list = []
        for i in range(self.classQLB.count()):
            if self.classQLB.isSelected(i):
                list.append( 1 )
            else:
                list.append( 0 )
        self.graph.setShowClasses(list)
    ##

    def calcGraph(self):
        self.graph.setData(self.MAdata, self.classColor, self.ShowAverageProfile, self.ShowSingleProfiles)
        self.graph.setPointWidth(self.PointWidth)
        self.graph.setCurveWidth(self.CurveWidth)
        self.graph.setAverageCurveWidth(self.AverageCurveWidth)

        self.graph.setAxisAutoScale(QwtPlot.xBottom)
        self.graph.setAxisAutoScale(QwtPlot.yLeft)
        self.graph.replot()

    def newdata(self):
        self.classQLB.clear()
        if self.MAdata <> None:
            ## classQLB
            self.numberOfClasses = len(self.MAdata.domain.classVar.values)
            self.classColor = []
            if self.numberOfClasses > 1:
                allCforHSV = self.numberOfClasses - 1
            else:
                allCforHSV = self.numberOfClasses
            for i in range(self.numberOfClasses):
                newColor = QColor()
                newColor.setHsv(i*255/allCforHSV, 255, 255)
                self.classColor.append( newColor )

            self.calcGraph()
            ## update graphics
            ## classQLB
            self.classQLB.show()
            classValues = self.MAdata.domain.classVar.values.native()
            for cn in range(len(classValues)):
                self.classQLB.insertItem(ColorPixmap(self.classColor[cn]), classValues[cn])
            self.classQLB.selectAll(1)  ##or: if numberOfClasses > 0: self.classQLB.setSelected(0, 1)

            if self.MAnoclass:
                self.classQLB.hide()
        else:
            self.classColor = None

    def data(self, MAdata):
        ## if there is no class attribute, create a dummy one
        if MAdata.data.domain.classVar == None:
            noClass = orange.EnumVariable('noclass', values=['none'])
            noClass.getValueFrom = lambda ex, w: 0
            newDomain = orange.Domain(MAdata.data.domain.variables + [noClass])
            self.MAdata = MAdata.data.select(newDomain)
            self.MAnoclass = 1 ## remember that there is no class to display
        else:
            self.MAdata = MAdata.data
            self.MAnoclass = 0 ## there are classes
        self.newdata()

    def cdata(self, MAcdata):
        self.MAdata = MAcdata.data
        self.MAnoclass = 0
        self.newdata()

if __name__ == "__main__":
    a = QApplication(sys.argv)
    owdm = OWDisplayProfiles()
    a.setMainWidget(owdm)
    owdm.show()
    a.exec_loop()
    owdm.saveSettings()

"""<name>Timeline</name>
<description>Timeline Visualization</description>
<icon>icons/Timeline.png</icon>
<priority>30</priority>
<contact>Janez Demsar (janez.demsar@fri.uni-lj.si)</contact>"""

from OWWidget import *
from OWGUI import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from OWDlgs import OWChooseImageSizeDlg
import sip
from math import ceil, log
from random import Random

class GraphicsViewWithScrollSync(QGraphicsView):
    def __init__(self, *args, **argkw):
        QGraphicsView.__init__(self, *args)
        self.syncVertical = self.syncHorizontal = None
        if argkw.get("forceScrollBars", True):
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
                
    def scrollContentsBy(self, dx, dy):
        QGraphicsView.scrollContentsBy(self, dx, dy)
        if dx and self.syncHorizontal:
            self.syncHorizontal.horizontalScrollBar().setValue(self.horizontalScrollBar().value())
        elif dy and self.syncVertical:
            self.syncVertical.verticalScrollBar().setValue(self.verticalScrollBar().value())
        
        
TM_YEAR, TM_MON, TM_DAY, TM_HOUR, TM_MIN, TM_SEC, TM_WDAY, TM_YDAY = range(8)

def sym0(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y-5, x+5, y+5))
    c.addItem(QGraphicsLineItem(x+5, y-5, x-5, y+5))
def sym1(c, x, y):
    c.addItem(QGraphicsEllipseItem(x-5, y-5, 10, 10))
def sym2(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y+3.16, x+5, y+3.16))
    c.addItem(QGraphicsLineItem(x-5, y+3.16, x, y-5.16))
    c.addItem(QGraphicsLineItem(x, y-5.16, x+5, y+3.16))
def sym3(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y-3.16, x+5, y-3.16))
    c.addItem(QGraphicsLineItem(x-5, y-3.16, x, y+5.16))
    c.addItem(QGraphicsLineItem(x, y+5.45, x+5, y-3.16))
def sym4(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y-5, x+5, y-5))
    c.addItem(QGraphicsLineItem(x+5, y-5, x+5, y+5))
    c.addItem(QGraphicsLineItem(x+5, y+5, x-5, y+5))
    c.addItem(QGraphicsLineItem(x-5, y+5, x-5, y-5))
def sym5(c, x, y):
    c.addItem(QGraphicsLineItem(x, y-5, x, y+5))
    c.addItem(QGraphicsLineItem(x-4, y-5, x+4, y-5))
    c.addItem(QGraphicsLineItem(x-4, y+5, x+4, y+5))
def sym6(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y-4, x+5, y+4))
    c.addItem(QGraphicsLineItem(x-5, y+4, x+5, y-4))
    c.addItem(QGraphicsLineItem(x, y+4, x, y-4))
def sym7(c, x, y):
    t = QGraphicsEllipseItem(x-6, y-3, 12, 12)
    t.setSpanAngle(2880)
    c.addItem(t)
def sym8(c, x, y):
    t = QGraphicsEllipseItem(x-6, y-8, 12, 12)
    t.setStartAngle(2880)
    t.setSpanAngle(2880)
    c.addItem(t)
def sym9(c, x, y):
    c.addItem(QGraphicsLineItem(x-5, y-5, x, y+5))
    c.addItem(QGraphicsLineItem(x, y-5, x, y+5))
    c.addItem(QGraphicsLineItem(x, y-5, x+5, y+5))

_font16 = QFont("", 13)
def overflowSymbol(c, x, y, sym):
    t = QGraphicsSimpleTextItem(sym)
    t.setFont(_font16)
    b = t.boundingRect()
    t.setPos(x-b.width()/2, y-b.height()/2)
    c.addItem(t)

_rr = range(ord("A"), ord("Z")+1) + range(ord("0"), ord("9")+1) + range(ord("a"), ord("z")+1)
symFunc = [sym0, sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9] + [lambda c, x, y, s=chr(s): overflowSymbol(c, x, y, s) for s in _rr]


class LegendWindow(QDialog):
    def __init__(self, legendcanvas):
        QDialog.__init__(self)
        self.setWindowTitle("Timeline Legend")
        self.lay = QVBoxLayout()
        self.setLayout(self.lay)
        self.legendview = QGraphicsView(legendcanvas, self)
        self.lay.addWidget(self.legendview)
        self.legendview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.position = None
     
    def closeEvent(self, ev):
        self.position = self.pos()
        
    def showEvent(self, ev):
        if self.position is not None:
            self.move(self.position)

    def resetBounds(self):
        brh = int(self.legendview.scene().itemsBoundingRect().height())
        self.legendview.resize(250, brh)
        self.updateGeometry()
        self.resize(280, min(450, brh+40))
        
class OWTimeline(OWWidget):
    settingsList = ["lineDistance", "wrapTime", "normalizeEvent"]
    contextHandlers = {"": DomainContextHandler("", ["groupAtt", "eventAtt", "timeAtt", "labelAtt", "timeScale", "timeInTicks", "customTimeRange", "jitteringAmount"])}

    def __init__(self, parent=None, signalManager=None, name="Timeline"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = []
        self.lineDistance = 40
        self.timeScale = 2
        self.eventAtt = 0
        self.timeAtt = 0
        self.groupAtt = 0
        self.labelAtt = 0
        self.wrapTime = True
        self.normalizeEvent = True
        self.timeInTicks = True
        self.jitteringAmount = 0
        self.customTimeRange = ""
#        self.showDensity = False
        self.loadSettings()

        self.icons = self.createAttributeIconDict()
        self.data = None
        
        self.timeScales = [("Year", 3600*24*365.25, "%Y"), ("Month", 3600*24*30, "%B %Y"), ("Week", 3600*24*7, "%B %Y"),
                            ("Day", 3600*24, "%B %d, %Y"), ("Hour", 3600, "%D %d, %Y @ %H:%M"),#("Minute", 60, "%D %d, %Y @ %H:%M"), ("Second", 1, "%D %d, %Y @ %H:%M:%S")
                          ]
        self.wrapScales = [[(str(i+1), 3600*24*sum([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30][:i])) for i in range(12)],
                           [(str(i+1), i*24*3600) for i in range(31)],
                           [(time.asctime((0, 0, 0, 0, 0, 0, i, 0, 0)), i*24*3600) for i in range(7)],
                           [(str(i), i*3600) for i in range(24)],
                           [(str(i), i*60) for i in range(60)]
                          ]
        
        self.jitteringAmounts = [0, 5, 10, 15, 20]
                
        b1 = OWGUI.widgetBox(self.controlArea, "Plotted Data", addSpace=True)
        self.attrEventCombo = OWGUI.comboBox(b1, self, "eventAtt", label = "Symbol shape / Tick size", callback = self.updateDisplay, sendSelectedValue = True, valueType = str, emptyString="(Ticks)")
        self.cbNormalize = OWGUI.checkBox(b1, self, "normalizeEvent", label = "Normalize values", callback = self.updateDisplay)
#        self.cbDensity = OWGUI.checkBox(b1, self, "showDensity", label = "Show density (or average)", callback = self.updateDisplay)
        OWGUI.separator(b1)
        self.attrLabelCombo = OWGUI.comboBox(b1, self, "labelAtt", label = "Label", callback = self.updateDisplay, sendSelectedValue = True, valueType = str, emptyString="(None)")

        b1 = OWGUI.widgetBox(self.controlArea, "Grouping", addSpace=True)
        self.attrGroupCombo = OWGUI.comboBox(b1, self, "groupAtt", label = "Group by", callback = self.updateDisplay, sendSelectedValue = 1, valueType = str, emptyString="(None)", addSpace=True)
        self.spDistance = OWGUI.spin(b1, self, "lineDistance", label = "Distance between groups", min=30, max=200, step=10, callback = self.updateDisplay, callbackOnReturn=True)

        b1 = OWGUI.widgetBox(self.controlArea, "Time Scale")
        self.attrTimeCombo = OWGUI.comboBox(b1, self, "timeAtt", label = "Time", callback = self.updateDisplay, sendSelectedValue = 1, valueType = str)
        OWGUI.checkBox(b1, self, "timeInTicks", label="Time is in ticks (from Jan 1, 1970)", callback = self.updateDisplay)
        bg = OWGUI.radioButtonsInBox(b1, self, "timeScale", [], label="Approximate amount of data per screen", callback = self.updateDisplay, addSpace=True)
        self.rbTickedTimes = [OWGUI.appendRadioButton(bg, self, "timeScale", lab[0], callback = self.updateDisplay) for lab in self.timeScales]
        OWGUI.appendRadioButton(bg, self, "timeScale", "Entire time range", callback = self.updateDisplay)
        OWGUI.appendRadioButton(bg, self, "timeScale", "Custom", callback = self.updateDisplay)
        OWGUI.lineEdit(OWGUI.indentedBox(bg), self, "customTimeRange", callback=self.updateDisplay, enterPlaceholder=True, validator=QDoubleValidator(self.controlArea))
        OWGUI.separator(b1)
        self.cbAggregate = OWGUI.checkBox(b1, self, "wrapTime", "Aggregate data within time scale", callback = self.updateDisplay)
        OWGUI.separator(b1)
        OWGUI.comboBox(b1, self, "jitteringAmount", label="Jittering", orientation=0, items = [("%i px" % i)if i else "None" for i in self.jitteringAmounts], callback=self.updateDisplay)
        
        OWGUI.rubber(self.controlArea)
        
        sip.delete(self.mainArea.layout())
        self.layout = QGridLayout(self.mainArea)
        self.layout.setHorizontalSpacing(2)
        self.layout.setVerticalSpacing(2)

        self.legendbox = OWGUI.widgetBox(self.mainArea, "Legend", addToLayout = False)
        self.legendbox.setFixedHeight(80)
        self.legendbox.setVisible(False)
        self.mainArea.layout().addWidget(self.legendbox, 0, 0)

        self.legendcanvas = QGraphicsScene(self)
        self.legendview = QGraphicsView(self.legendcanvas, self.mainArea)
        self.legendview.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.legendview.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        self.legendview.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        self.legendbox.layout().addWidget(self.legendview)
        self.legendWindow = LegendWindow(self.legendcanvas)
        
        self.legendLabel = OWGUI.widgetLabel(self.legendbox, "")
        self.legendButton = OWGUI.button(self.legendbox, self, "Show Legend", callback=self.showLegend)

        self.canvas = QGraphicsScene(self)
        self.canvasview = GraphicsViewWithScrollSync(self.canvas, self.mainArea)
        self.mainArea.layout().addWidget(self.canvasview, 1, 1)

        self.groupscanvas = QGraphicsScene(self)
        self.groupsview = GraphicsViewWithScrollSync(self.groupscanvas, self.mainArea)
        self.mainArea.layout().addWidget(self.groupsview, 1, 0)
        self.groupsview.setLayoutDirection(Qt.RightToLeft)
        
        self.timecanvas = QGraphicsScene(self)
        self.timeview = GraphicsViewWithScrollSync(self.timecanvas, self.mainArea)
        self.timeview.setFixedHeight(80)
        self.mainArea.layout().addWidget(self.timeview, 0, 1)
        
        self.timeview.syncHorizontal = self.canvasview
        self.canvasview.syncHorizontal = self.timeview
        self.groupsview.syncVertical = self.canvasview
        self.canvasview.syncVertical = self.groupsview

        self.nSymbols = len(symFunc)
                                        
        self.resize(qApp.desktop().screenGeometry(self).width()-30, 600)
    
    def setData(self, data):
        self.closeContext()
        self.attrGroupCombo.clear()
        self.attrEventCombo.clear()
        self.attrTimeCombo.clear()
        self.attrLabelCombo.clear()
        self.data = data
        if self.data:
            discAttrs = [att for att in data.domain if att.varType==orange.VarTypes.Discrete]
            contAttrs = [att for att in data.domain if att.varType==orange.VarTypes.Continuous]
            if not discAttrs or not contAttrs:
                self.error("The data needs to have at least one discrete and one continuous attribute")
                self.data = None
            else:
                self.error("")
                self.basStat = orange.DomainBasicAttrStat(data)
                self.attrGroupCombo.addItem("(None)")
                if discAttrs:
                    for att in discAttrs:
                        self.attrGroupCombo.addItem(self.icons[att.varType], att.name)
                    self.groupAtt = discAttrs[0].name
                else:
                    self.groupAtt = ""

                satt = None
                for att, bas in zip(data.domain, self.basStat):
                    self.attrTimeCombo.addItem(self.icons[att.varType], att.name)
                    if not satt and bas and 946652400 <= bas.min <= bas.max <= 1577804400:
                        satt = att
                self.timeAtt = satt.name if satt else contAttrs[0].name
        
                self.attrEventCombo.addItem("(Ticks)")
                self.attrLabelCombo.addItem("(None)")
                overvalued = False
                for att in data.domain:
                    self.attrLabelCombo.addItem(self.icons[att.varType], att.name)
                    if att.varType==orange.VarTypes.Discrete and len(att.values) > len(symFunc):
                        overvalued = True
                    elif att.varType in [orange.VarTypes.Discrete, orange.VarTypes.Continuous]:
                        self.attrEventCombo.addItem(self.icons[att.varType], att.name)
                if overvalued:
                    self.warning(1, "Discrete features with more than %i different values are omitted\nfrom the list of possible symbol shapes." % len(symFunc))
                else:
                    self.warning(1, "")
                self.eventAtt = self.labelAtt = ""
        
                self.openContext("", self.data)
        self.updateDisplay()

    def showLegend(self):
        self.legendWindow.show()
        
    def closeEvent(self, ev):
        self.legendWindow.close()
    
    def makeConsistent(self):
        for tt in self.rbTickedTimes:
            tt.setDisabled(not self.timeInTicks)
        if not self.timeInTicks:
            self.wrapTime = False 
            if self.timeScale < len(self.timeScales):
                self.timeScale = len(self.timeScales)
        self.cbAggregate.setDisabled(self.timeScale>=len(self.timeScales))
        if self.timeScale >= len(self.timeScales):
            self.wrapTime = False 
            if self.timeScale > len(self.timeScales) and not self.customTimeRange and self.data:
                timePos = self.data.domain.index(self.timeAtt)
                bas = self.basStat[timePos]
                self.customTimeRange = str((bas.max-bas.min) or 1)

    def updateDisplay(self):
        self.canvas.clear()
        self.timecanvas.clear()
        self.groupscanvas.clear() 
        self.legendcanvas.clear() 
        if not self.data:
            return

        self.makeConsistent()
        
        font16 = QFont("", 13)
        font12 = QFont("", 10)
        font10 = QFont("", 8)
        
        timePos = self.data.domain.index(self.timeAtt)
        if self.wrapTime:
            timespan = self.timeScales[self.timeScale][1]
            xFactor = 1000. / timespan
        else: # ticks-based non-custom scale
            bas = self.basStat[timePos]
            minTime, maxTime = bas.min, bas.max
            timespan = maxTime - minTime
            if self.timeScale > len(self.timeScales):
                xFactor = 1000. / (float(self.customTimeRange) or 1)
            elif self.timeScale == len(self.timeScales):
                xFactor = 1000. / timespan
            else:
                xFactor = 1000. / self.timeScales[self.timeScale][1]  
            
        if timespan*xFactor > 120000:
            t = QGraphicsSimpleTextItem("Graph is too wide. Decrease the scale by choosing a larger time interval.")
            t.setFont(font12)
            t.setPos(10, 10)
            self.canvas.addItem(t)
            return
        
        jitter = self.jitteringAmounts[self.jitteringAmount]
        randint = Random(0).randint
                
        self.setCursor(Qt.WaitCursor)
        plotTicks = not self.eventAtt
        if plotTicks:
            self.cbNormalize.setDisabled(True)
            self.legendbox.setVisible(False)
            self.legendWindow.close()
        else:
            eventAttr = self.data.domain[self.eventAtt]
            eventPos = self.data.domain.index(eventAttr)
            discreteEvent = eventAttr.varType == orange.VarTypes.Discrete
            self.cbNormalize.setDisabled(discreteEvent)
            if discreteEvent:
                self.legendbox.setVisible(True)
                nEvents = len(eventAttr.values)
                self.legendview.setVisible(nEvents <= 2)
                self.legendButton.setVisible(nEvents > 2)
                self.legendLabel.setVisible(nEvents > 2)
                self.legendLabel.setText("There are %i different symbols" % nEvents)
                if nEvents <= 2:
                    self.legendWindow.close()
                for i, val in enumerate(eventAttr.values):
                    t = QGraphicsSimpleTextItem(str(val).decode("utf-8"))
                    t.setFont(font12)
                    if i < self.nSymbols:
                        symFunc[i](self.legendcanvas, 0, 20*i+t.boundingRect().height()/2)
                    else:
                        overflowSymbol(self.legendcanvas, 0, 20*i, i)
                    t.setPos(15, 20*i)
                    self.legendcanvas.addItem(t)
                br = self.legendcanvas.itemsBoundingRect()
                self.legendcanvas.setSceneRect(-10,-5, br.width(), br.height())
                self.legendWindow.resetBounds()
            else:
                self.legendbox.setVisible(False)
                self.legendWindow.close()
                bas = self.basStat[eventPos]
                minCls, maxCls = bas.min, bas.max
                if self.normalizeEvent:
                    yFactor = (self.lineDistance-5) / ((maxCls-minCls) or 1)
                else:
                    yFactor = (self.lineDistance-5) / (max(abs(maxCls), abs(minCls)) or 1)

        grouped = bool(self.groupAtt)
        self.groupsview.setVisible(grouped)
        if grouped:
            groupPos = self.data.domain.index(self.groupAtt)
            groupAttr = self.data.domain[groupPos]
            filt = orange.Filter_sameValue(position=self.data.domain.index(groupAttr))
            groupValues = groupAttr.values
        else:
            groupValues = [""]

        labeled = bool(self.labelAtt)
        if labeled:
            labelPos = self.data.domain.index(self.labelAtt)
            labelOff = -20 if plotTicks else (-22 if discreteEvent else 4) 

        filt_unk = orange.Filter_isDefined(domain = self.data.domain)
        filt_unk.check = [(i==timePos) or (not plotTicks and i==eventPos) or (grouped and i==groupPos) for i in range(len(filt_unk.check))]
        data = filt_unk(self.data)
        if len(data) != len(self.data):
            self.warning(2, "Instances with unknown values of plotted data were removed.")
        else:
            self.warning(2, "")
        maxw = 30
        for index, group in enumerate(groupValues):
#            if self.showDensity:
#                density = [[] for i in eventAttr.values] if (discreteEvent and not plotTicks) else [] 
            y = self.lineDistance * (1 + index)
            self.canvas.addItem(QGraphicsLineItem(-15, y, xFactor*timespan + 15, y))
            if grouped:
                filt.value = index
                thisAxis = data.filterref(filt)
                t = QGraphicsSimpleTextItem(group.decode("utf-8"))
                t.setFont(font12)
                brect = t.boundingRect()
                t.setPos(-10 - brect.width(), y - brect.height()/2)
                self.groupscanvas.addItem(t)
                maxw = max(maxw, brect.width())
            else:
                thisAxis = data

            for ex in thisAxis:
                if self.wrapTime:
                    timetup = time.localtime(float(ex[timePos]))
                    if self.timeScale == 0:
                        x = xFactor * (86400*(timetup[TM_YDAY] + (timetup[TM_MON]>2 and timetup[TM_YEAR]%4 != 0)) + 3600*timetup[3] + 60*timetup[4] + timetup[5])
                    elif self.timeScale == 1:
                        x = xFactor * (86400*timetup[TM_DAY] + 3600*timetup[TM_HOUR] + 60*timetup[TM_MIN] + timetup[TM_SEC])
                    elif self.timeScale == 2:
                        x = xFactor * (86400*timetup[TM_WDAY] + 3600*timetup[TM_HOUR] + 60*timetup[TM_MIN] + timetup[TM_SEC])
                    elif self.timeScale == 3:
                        x = xFactor * (3600*timetup[TM_HOUR] + 60*timetup[TM_MIN] + timetup[TM_SEC])
                    elif self.timeScale == 4:
                        x = xFactor * (60*timetup[TM_MIN] + timetup[TM_SEC])
                else:
                    x = xFactor * (float(ex[timePos]) - minTime)

                rx = x
                if jitter:
                    x += randint(-jitter, jitter)
                    
                if labeled:
                    t = QGraphicsSimpleTextItem(str(ex[labelPos]).decode("utf-8"))
                    t.setFont(font10)
                    t.setPos(x - t.boundingRect().width()/2, y + labelOff)
                    self.canvas.addItem(t)
                    
                if plotTicks:
                    self.canvas.addItem(QGraphicsLineItem(x, y-3, x, y+3))
#                    if self.showDensity:
#                        density.append(rx)
                elif discreteEvent:
                    clsi = int(ex[eventPos])
                    if clsi < self.nSymbols:
                        symFunc[clsi](self.canvas, x, y)
                    else:
                        self.overflowSymbol(x, y, clsi)
#                    if self.showDensity:
#                        density[clsi].append(rx)
                else:
                    l = yFactor*(float(ex[eventPos])-minCls) if self.normalizeEvent else yFactor*float(ex[eventPos]) 
                    self.canvas.addItem(QGraphicsLineItem(x, y, x, y-l))
#                    if self.showDensity:
#                        density.append((rx, l))
                        
#            if self.showDensity:
#                if not discreteEvent or plotTicks:
#                    import statc
#                    loessCurve = statc.loess(sorted(density), 100, 0.2, 1)
#                    q = QPolygonF()
#                    for i, (x, l, foo) in enumerate(loessCurve):
#                        q.insert(i, QPointF(x, y-l))
#                    q.insert(i+1, QPointF(x, y))
#                    q.insert(0, QPointF(loessCurve[0][0], y))
#                    self.canvas.addItem(QGraphicsPolygonItem(q))

        if self.timeScale >= len(self.timeScales):
            if self.timeScale == len(self.timeScales):
                cr = timespan
            else:
                cr = float(self.customTimeRange)
            if cr:
                decs = max(0, 2 + ceil(-log(cr)))
                step = 10**decs
                mt = round(minTime, decs)
                while mt <= maxTime:
                    t = QGraphicsSimpleTextItem("%.*f" % (decs, mt))
                    t.setFont(font12)
                    t.setPos(xFactor*(mt-minTime), 30)
                    self.timecanvas.addItem(t)
                    mt += cr/10                            
        elif self.wrapTime:
            for label, pos in self.wrapScales[self.timeScale]:
                t = QGraphicsSimpleTextItem(label)
                t.setFont(font12)
                t.setPos(xFactor*pos, 30)
                self.timecanvas.addItem(t)
        else:
            try:
                realScale = self.timeScale if self.timeScale < 2 else self.timeScale-1
                timeFmt = self.timeScales[self.timeScale][2]
                mintup = time.localtime(minTime)
                maxtup = time.localtime(maxTime)
                starttime = time.mktime(mintup[:realScale+2] + ((0,)*9)[realScale+2:])
                stoptime = time.mktime(maxtup[:realScale+2] + (0,0, 31, 24, 60, 60, 0, 0, 0)[realScale+2:])
                prevReal = None
                while starttime < stoptime:
                    x = xFactor*(starttime-minTime)
                    
                    bk = time.localtime(starttime)
                    if bk[realScale] != prevReal:
                        t = QGraphicsSimpleTextItem(time.strftime(timeFmt, bk))
                        t.setFont(font12)
                        t.font().setBold(True)
                        t.setPos(x, 0)
                        self.timecanvas.addItem(t)
                        self.timecanvas.addItem(QGraphicsLineItem(x-8, 0, x-8, 50))
                        
                    t = QGraphicsSimpleTextItem(str(bk[realScale+1]))
                    t.setPos(x, 25)
                    self.timecanvas.addItem(t)
                    prevReal = bk[realScale]
                    starttime = time.mktime(bk[:realScale+1] + (bk[realScale+1]+1, ) + bk[realScale+2:])
            except:
                self.timecanvas.clear()
                t = QGraphicsSimpleTextItem("Error occurred while converting ticks to dates. Please verify that the time indeed represents valid ticks in seconds from January 1 1970.")
                t.setFont(font12)
                t.setPos(10, 10)
                self.timecanvas.addItem(t)

        swidth, sheight = 40+xFactor*timespan, self.lineDistance*(len(groupValues)+.5) 
        self.groupscanvas.setSceneRect(-maxw-20, 0, maxw+20, sheight)
        self.groupsview.setFixedWidth(min(250, maxw)+45)
        if not plotTicks and discreteEvent:
            self.legendbox.setFixedWidth(min(250, maxw)+45)
        self.canvas.setSceneRect(-20, 0, swidth, sheight)
        self.timecanvas.setSceneRect(-20, -5, swidth, 60)
        self.canvasview.resetMatrix()
        self.groupsview.resetMatrix()
        self.timeview.resetMatrix()
        self.setCursor(Qt.ArrowCursor)

    def sendReport(self):
        if not self.data:
            return

        self.reportData(self.data)
        self.reportSettings("Visualization",
                    [("Symbol shape/size", self.eventAtt),
                     ("Normalized", OWGUI.YesNo[self.normalizeEvent]),
                     ("Label", self.labelAtt),
                     ("Grouping", self.groupAtt),
                     ("Time", self.timeAtt),
                     ("Time scale", self.timeScales[self.timeScale][0] if self.timeScale<len(self.timeScales) else ("Custom, "+self.customTimeRange if self.timeScale>len(self.timeScales) else "Entire range")),
                     ("Time jittering", "%i pixels" % self.jitteringAmounts[self.jitteringAmount] if self.jitteringAmount else "None"),
                     ("Aggregate data within time scale", OWGUI.YesNo[self.wrapTime]),
                     ])

        self.reportSection("Timeline")
        groups, times, canvas = self.groupscanvas, self.timecanvas, self.canvas
        if canvas.width()+groups.width() > 10000:
            self.reportRaw("<p>Figure is too wide for the report.</p>")
            return
            
        if self.eventAtt and self.data.domain[self.eventAtt].varType == orange.VarTypes.Discrete:
            self.reportImage(lambda *x: OWChooseImageSizeDlg(self.legendcanvas).saveImage(*x))

        painter = QPainter()
        buffer = QPixmap(groups.width()+canvas.width(), times.height()+canvas.height())
        painter.begin(buffer)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        painter.fillRect(buffer.rect(), QBrush(QColor(255, 255, 255)))
        groups.render(painter, QRectF(0, times.height(), groups.width(), groups.height()))
        times.render(painter, QRectF(groups.width(), 0, times.width(), times.height()))
        canvas.render(painter, QRectF(groups.width(), times.height(), canvas.width(), canvas.height()))
        painter.end()
        self.reportImage(lambda filename: buffer.save(filename, os.path.splitext(filename)[1][1:]))

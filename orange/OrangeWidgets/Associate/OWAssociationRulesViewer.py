"""
<name>Association Rules Viewer</name>
<description>Association rules filter and viewer.</description>
<icon>icons/AssociationRulesViewer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact> 
<priority>200</priority>
"""

import orange, sys
from qt import *
from qtcanvas import *
from qttable import *
from OWWidget import *
import OWGUI

class AssociationRulesViewerCanvas(QCanvas):
    def __init__(self, master, widget):
        QCanvas.__init__(self, widget)
        self.master = master
        self.rect = None
        self.unselect()
        self.draw()

    def unselect(self):
        if self.rect:
            self.rect.hide()
            self.rect = None
        
        
    def draw(self):
        master = self.master
        nc, nr, cw, ch, ig = master.numcols, master.numrows, master.cellwidth, master.cellheight, master.ingrid
        scmin, scmax, srmin, srmax = master.sel_colmin, master.sel_colmax, master.sel_rowmin, master.sel_rowmax

        self.resize(nc * cw +1, nr * ch +1)

        for a in self.allItems():
            a.hide()
                
        maxcount = max([max([len(cell) for cell in row]) for row in master.ingrid])
        maxcount = float(max(10, maxcount))

        pens = [QPen(QColor(200,200,200), 1), QPen(QColor(200,200,255), 1)]
        brushes = [QBrush(QColor(255, 255, 255)), QBrush(QColor(250, 250, 255))]
        self.cells = []
        for x in range(nc):
            selx = x >= scmin and x <= scmax
            for y in range(nr):
                sel = selx and y >= srmin and y <= srmax
                cell = QCanvasRectangle(x*cw, y*ch, cw+1, ch+1, self)
                cell.setPen(pens[sel])
                if not ig[y][x]:
                    cell.setBrush(brushes[sel])
                else:
                    if sel:
                        color = 220 - 220 * len(ig[y][x]) / maxcount
                        cell.setBrush(QBrush(QColor(color, color, 255)))
                    else:
                        color = 255 - 235 * len(ig[y][x]) / maxcount
                        cell.setBrush(QBrush(QColor(color-20, color-20, color)))
                cell.show()

        if self.rect:
            self.rect.hide()
        if scmin > -1:
            self.rect = QCanvasRectangle(scmin*cw, srmin*ch, (scmax-scmin+1)*cw, (srmax-srmin+1)*ch, self)
            self.rect.setPen(QPen(QColor(128, 128, 255), 2))
            self.rect.show()
        else:
            self.rect = None

        self.update()
        self.master.shownSupport.setText('%3i%% - %3i%%' % (int(master.supp_min*100), int(master.supp_max*100)))
        self.master.shownConfidence.setText('%3i%% - %3i%%' % (int(master.conf_min*100), int(master.conf_max*100)))
        self.master.shownRules.setText('%3i' % sum([sum([len(cell) for cell in row]) for row in master.ingrid]))


class AssociationRulesViewerView(QCanvasView):
    def __init__(self, master, canvas, widget):
        QCanvasView.__init__(self, canvas, widget)
        self.master = master
        self.canvas = canvas
        self.setFixedSize(365, 365)
        self.selecting = False
        self.update()

    def contentsMousePressEvent(self, ev):
        self.sel_startX = ev.pos().x()
        self.sel_startY = ev.pos().y()
        master = self.master
        self.master.sel_colmin = self.master.sel_colmax = self.sel_startX / self.master.cellwidth
        self.master.sel_rowmin = self.master.sel_rowmax = self.sel_startY / self.master.cellheight
        self.canvas.draw()
        self.master.updateRuleList()

    def contentsMouseMoveEvent(self, ev):
        self.sel_endX = ev.pos().x()
        self.sel_endY = ev.pos().y()
        t = self.sel_startX /self.master.cellwidth, self.sel_endX /self.master.cellwidth
        self.master.sel_colmin, self.master.sel_colmax = min(t), max(t)
        t = self.sel_startY /self.master.cellheight, self.sel_endY /self.master.cellheight
        self.master.sel_rowmin, self.master.sel_rowmax = min(t), max(t)

        self.master.sel_colmin = max(self.master.sel_colmin, 0)
        self.master.sel_rowmin = max(self.master.sel_rowmin, 0)
        self.master.sel_colmax = min(self.master.sel_colmax, self.master.numcols-1)
        self.master.sel_rowmax = min(self.master.sel_rowmax, self.master.numrows-1)

        self.canvas.draw()
        self.master.updateRuleList()

    def contentsMouseReleaseEvent(self, ev):
        self.master.sendIfAuto()


class OWAssociationRulesViewer(OWWidget):
    measures = [("Support",    "Supp", "support"),
                ("Confidence", "Conf", "confidence"),
                ("Lift",       "Lift", "lift"),
                ("Leverage",   "Lev",  "leverage"),
                ("Strength",   "Strg", "strength"),
                ("Coverage",   "Cov",  "coverage")]

    settingsList = ["autoSend", "sortedBy"] + [vn[2] for vn in measures]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "AssociationRulesViewer")

        self.inputs = [("Association Rules", orange.AssociationRules, self.arules)]
        self.outputs = [("Association Rules", orange.AssociationRules)]

        self.supp_min, self.supp_max = self.conf_min, self.conf_max = 0., 1.
        self.numcols = self.numrows = 20
        self.cellwidth = self.cellheight = 18

        for m in self.measures:
            setattr(self, m[2], False)
        self.support = self.confidence = True
        self.sortedBy = 0
        self.autoSend = True
        
        self.loadSettings()

        self.rules = None
        self.selectedRules = []
        self.noZoomButton()


        self.mainLayout = QHBoxLayout(self.mainArea)
        self.mainLayout.setAutoAdd(True)
        mainLeft = OWGUI.widgetBox(self.mainArea, "Filter")
        sep = OWGUI.separator(self.mainArea, 16, 0)
        mainRight = OWGUI.widgetBox(self.mainArea, "Rules")
        mainRight.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))


        info = QWidget(mainLeft)
        infoGrid = QGridLayout(info, 3, 4, 0, 5)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Shown"), 1, 0)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Selected"), 2, 0)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Support (H)"), 0, 1)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Confidence (V)"), 0, 2)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "# Rules"), 0, 3)
        
        self.shownSupport = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.shownSupport, 1, 1)
        self.shownConfidence = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.shownConfidence, 1, 2)
        self.shownRules = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.shownRules, 1, 3)

        self.selSupport = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.selSupport, 2, 1)
        self.selConfidence = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.selConfidence, 2, 2)
        self.selRules = OWGUI.widgetLabel(info, " ")
        infoGrid.addWidget(self.selRules, 2, 3)
        
        OWGUI.separator(mainLeft, 0, 4)
        self.ruleCanvas = AssociationRulesViewerCanvas(self, mainLeft)
        self.canvasView = AssociationRulesViewerView(self, self.ruleCanvas, mainLeft)

        boxb = OWGUI.widgetBox(mainLeft, box=None, orientation="horizontal")
        OWGUI.button(boxb, self, 'Zoom', callback = self.zoomButton)
        OWGUI.button(boxb, self, 'Show All', callback = self.showAllButton)
        OWGUI.button(boxb, self, 'No Zoom', callback = self.noZoomButton)
        OWGUI.separator(boxb, 16, 8)
        OWGUI.button(boxb, self, 'Unselect', callback = self.unselect)
        

        rightUpRight = QWidget(mainRight)
        self.grid=QGridLayout(rightUpRight,2,3,5,5)
        for i, m in enumerate(self.measures):
            cb = OWGUI.checkBox(rightUpRight, self, m[2], m[0], callback = self.showHideColumns)
            self.grid.addWidget(cb.parentWidget(), i % 2, i / 2)

#        rightUpRight = OWGUI.widgetBox(mainRight, orientation=0)
#        for i, m in enumerate(self.measures):
#            cb = OWGUI.checkBox(rightUpRight, self, m[2], m[0], callback = self.showHideColumns)

        OWGUI.separator(mainRight, 0, 4)
        
        trules = self.trules = QTable(0, 0, mainRight)
        trules.setLeftMargin(0)
        trules.verticalHeader().hide()
        trules.setSelectionMode(QTable.NoSelection)
        trules.setNumCols(len(self.measures)+1)

        header = trules.horizontalHeader()
        for i, m in enumerate(self.measures):
            trules.setColumnStretchable(i, 0)
            header.setLabel(i, m[1])
        trules.setColumnStretchable(len(self.measures), 1)
        header.setLabel(len(self.measures), "Rule")
        self.connect(header, SIGNAL("clicked(int)"), self.sort)

        bottom = QWidget(mainRight)
        bottomGrid = QGridLayout(bottom, 2, 2, 5, 5)

        self.saveButton = OWGUI.button(bottom, self, "Save Rules", callback = self.saveRules)
        commitButton = OWGUI.button(bottom, self, "Send Rules", callback = self.sendRules)        
        autoSend = OWGUI.checkBox(bottom, self, "autoSend", "Send rules automatically", disables=[(-1, commitButton)])
        autoSend.makeConsistent()

        bottomGrid.addWidget(self.saveButton, 1, 0)
        bottomGrid.addWidget(autoSend.parentWidget(), 0, 1)
        bottomGrid.addWidget(commitButton, 1, 1)


        self.controlArea.setFixedSize(0, 0)
        self.resize(1000, 380)

    
    def checkScale(self):
        if self.supp_min == self.supp_max:
            self.supp_max += 0.01
        if self.conf_max == self.conf_min:
            self.conf_max += 0.01
        self.suppInCell = (self.supp_max - self.supp_min) / self.numcols
        self.confInCell = (self.conf_max - self.conf_min) / self.numrows


    def unselect(self):
        self.sel_colmin = self.sel_colmax = self.sel_rowmin = self.sel_rowmax = -1

        self.selectedRules = []
        for row in self.ingrid:
            for cell in row:
                for rule in cell:
                    self.selectedRules.append(rule)

        self.displayRules()                
        if hasattr(self, "selConfidence"):
            self.updateConfSupp()

        if hasattr(self, "ruleCanvas"):
            self.ruleCanvas.unselect()
            self.ruleCanvas.draw()

        self.sendIfAuto()            


    def updateConfSupp(self):
        if self.sel_colmin >= 0:
            smin, cmin = self.coordToSuppConf(self.sel_colmin, self.sel_rowmin)
            smax, cmax = self.coordToSuppConf(self.sel_colmax+1, self.sel_rowmax+1)
        else:
            smin, cmin = self.supp_min, self.conf_min
            smax, cmax = self.supp_max, self.conf_max
            
        self.selConfidence.setText("%3i%% - %3i%%" % (round(100*cmin), round(100*cmax)))
        self.selSupport.setText("%3i%% - %3i%%" % (round(100*smin), round(100*smax)))
        self.selRules.setText("%3i" % len(self.selectedRules))

    # This function doesn't send anything to output! (Shouldn't because it's called by the mouse move event)            
    def updateRuleList(self):
        self.selectedRules = []
        for row in self.ingrid[self.sel_rowmin : self.sel_rowmax+1]:
            for cell in row[self.sel_colmin : self.sel_colmax+1]:
                for rule in cell:
                    self.selectedRules.append(rule)

        self.displayRules()
        self.updateConfSupp()
        self.saveButton.setEnabled(len(self.selectedRules) > 0)

    def displayRules(self):
        if hasattr(self, "trules"):
            trules = self.trules
            trules.setNumRows(len(self.selectedRules))

            rulecol = len(self.measures)
            for row, rule in enumerate(self.selectedRules):
                for col, m in enumerate(self.measures):
                    trules.setText(row, col, "  %.3f  " % getattr(rule, m[2]))
                trules.setText(row, rulecol, `rule`.replace(" ", "  "))

            for i in range(len(self.measures)+1):
                self.trules.adjustColumn(i)
                
            self.showHideColumns()
            self.sort()


    def showHideColumns(self):
        for i, m in enumerate(self.measures):
            show = getattr(self, m[2])
            if bool(self.trules.columnWidth(i)) != bool(show):
                if show:
                    self.trules.showColumn(i)
                    self.trules.adjustColumn(i)
                else:
                    self.trules.hideColumn(i)

    def sort(self, i = None):
        if i is None:
            i = self.sortedBy
        else:
            self.sortedBy = i
        self.trules.sortColumn(i, i == len(self.measures), True)
        self.trules.horizontalHeader().setSortIndicator(i)
        
            
    def saveRules(self):
        fileName = QFileDialog.getSaveFileName( "myRules.txt", "Textfiles (*.txt)", self );
        if not fileName.isNull() :
            f = open(str(fileName), 'w')
            if self.selectedRules:
                toWrite = [m for m in self.measures if getattr(self, m[2])]
                if toWrite:
                    f.write("\t".join([m[1] for m in toWrite]) + "\n")
                for rule in self.selectedRules:
                    f.write("\t".join(["%.3f" % getattr(rule, m[2]) for m in toWrite] + [`rule`.replace(" ", "  ")]) + "\n")


    def setIngrid(self):
        smin, sic, cmin, cic = self.supp_min, self.suppInCell, self.conf_min, self.confInCell
        self.ingrid = [[[] for x in range(self.numcols)] for y in range(self.numrows)]
        if self.rules:
            for r in self.rules:
                self.ingrid[min(self.numrows-1, int((r.confidence - cmin) / cic))][min(self.numcols-1, int((r.support - smin) / sic))].append(r)


    def coordToSuppConf(self, col, row):
        return self.supp_min + col * self.suppInCell, self.conf_min + row * self.confInCell
    
    def zoomButton(self):
        if self.sel_rowmin >= 0:
            # have to compute both at ones!
            self.supp_min, self.conf_min, self.supp_max, self.conf_max = self.coordToSuppConf(self.sel_colmin, self.sel_rowmin) + self.coordToSuppConf(self.sel_colmax+1, self.sel_rowmax+1)
            self.checkScale()

            smin, sic, cmin, cic = self.supp_min, self.suppInCell, self.conf_min, self.confInCell
            newingrid = [[[] for x in range(self.numcols)] for y in range(self.numrows)]
            for row in self.ingrid[self.sel_rowmin : self.sel_rowmax+1]:
                for cell in row[self.sel_colmin : self.sel_colmax+1]:
                    for rule in cell:
                        inrow = (rule.confidence - cmin) / cic
                        if inrow >= 0 and inrow < self.numrows + 1e-3:
                            incol = (rule.support - smin) / sic
                            if incol >= 0 and incol < self.numcols + 1e-3:
                                newingrid[min(int(inrow), self.numrows-1)][min(int(incol), self.numcols-1)].append(rule)
            self.ingrid = newingrid

            self.unselect()
            self.ruleCanvas.draw()
            self.sendIfAuto()


    def rezoom(self, smi, sma, cmi, cma):
        self.supp_min, self.supp_max, self.conf_min, self.conf_max = smi, sma, cmi, cma
        self.checkScale() # to set the inCell
        self.setIngrid()
        self.unselect()
        if hasattr(self, "ruleCanvas"):
            self.ruleCanvas.draw()
        self.sendIfAuto()
        
    def showAllButton(self):
        self.rezoom(self.supp_allmin, self.supp_allmax, self.conf_allmin, self.conf_allmax)

    def noZoomButton(self):
        self.rezoom(0., 1., 0., 1.)
        
    def sendIfAuto(self):
        if self.autoSend:
            self.sendRules()
        
    def sendRules(self):
        self.send("Association Rules", orange.AssociationRules(self.selectedRules))
    
    def arules(self,rules):
        self.rules = rules
        if self.rules:
            self.supp_min = self.conf_min = 1
            self.supp_max = self.conf_max = 0
            for rule in self.rules:
                self.conf_min = min(self.conf_min, rule.confidence)
                self.conf_max = max(self.conf_max, rule.confidence)
                self.supp_min = min(self.supp_min, rule.support)
                self.supp_max = max(self.supp_max, rule.support)
            self.checkScale()
        else:
            self.supp_min, self.supp_max = self.conf_min, self.conf_max = 0., 1.

        self.supp_allmin, self.supp_allmax, self.conf_allmin, self.conf_allmax = self.supp_min, self.supp_max, self.conf_min, self.conf_max
        self.rezoom(self.supp_allmin, self.supp_allmax, self.conf_allmin, self.conf_allmax)


       
if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesViewer()
    a.setMainWidget(ow)


    dataset = orange.ExampleTable('../../doc/datasets/car.tab')
    rules=orange.AssociationRulesInducer(dataset, minSupport = 0.3, maxItemSets=15000)
    ow.arules(rules)

    ow.show()
    a.exec_loop()
    ow.saveSettings()

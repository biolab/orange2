"""
<name>Association Rules Filter</name>
<description>Association rules filter and viewer.</description>
<icon>icons/AssociationRulesViewer.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>200</priority>
"""
import orange, sys
from OWWidget import *
import OWGUI

class AssociationRulesViewerScene(QGraphicsScene):
    def __init__(self, master, widget):
        QGraphicsScene.__init__(self, widget)
        self.master = master
        self.rect = None
        self.mousePressed = False
        self.unselect()
        self.draw()

    def unselect(self):
        if self.rect:
            self.rect.hide()


    def draw(self):
        master = self.master
        nc, nr, cw, ch, ig = master.numcols, master.numrows, master.cellwidth, master.cellheight, master.ingrid
        scmin, scmax, srmin, srmax = master.sel_colmin, master.sel_colmax, master.sel_rowmin, master.sel_rowmax

        self.setSceneRect(0, 0, nc * cw +1, nr * ch +1)

        for a in self.items():
            self.removeItem(a)

        maxcount = max([max([len(cell) for cell in row]) for row in master.ingrid])
        maxcount = float(max(10, maxcount))

        pens = [QPen(QColor(200,200,200), 1), QPen(QColor(200,200,255), 1)]
        brushes = [QBrush(QColor(255, 255, 255)), QBrush(QColor(250, 250, 255))]
        self.cells = []
        for x in range(nc):
            selx = x >= scmin and x <= scmax
            for y in range(nr):
                sel = selx and y >= srmin and y <= srmax
                cell = QGraphicsRectItem(x*cw, y*ch, cw+1, ch+1, None, self)
                cell.setZValue(0)
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

        self.rect = QGraphicsRectItem(scmin*cw, srmin*ch, (scmax-scmin+1)*cw, (srmax-srmin+1)*ch, None, self)
        self.rect.setPen(QPen(QColor(128, 128, 255), 2))
        self.rect.setZValue(1)
        self.rect.hide()

        self.update()
        self.master.shownSupport.setText('%3i%% - %3i%%' % (int(master.supp_min*100), int(master.supp_max*100)))
        self.master.shownConfidence.setText('%3i%% - %3i%%' % (int(master.conf_min*100), int(master.conf_max*100)))
        self.master.shownRules.setText('%3i' % sum([sum([len(cell) for cell in row]) for row in master.ingrid]))

    def updateSelectionRect(self):
        master = self.master
        self.rect.setRect(master.sel_colmin*master.cellwidth,
                          master.sel_rowmin*master.cellheight,
                          (master.sel_colmax-master.sel_colmin+1)*master.cellwidth,
                          (master.sel_rowmax-master.sel_rowmin+1)*master.cellheight)
        self.update()

    def mousePressEvent(self, ev):
        self.sel_startX = int(ev.scenePos().x())
        self.sel_startY = int(ev.scenePos().y())
        master = self.master
        master.sel_colmin = master.sel_colmax = self.sel_startX / master.cellwidth
        master.sel_rowmin = master.sel_rowmax = self.sel_startY / master.cellheight
        self.rect.show()
        self.updateSelectionRect()
        self.mousePressed = True

    def mouseMoveEvent(self, ev):
        if self.mousePressed:
            self.sel_endX = int(ev.scenePos().x())
            self.sel_endY = int(ev.scenePos().y())
            t = self.sel_startX /self.master.cellwidth, self.sel_endX /self.master.cellwidth
            self.master.sel_colmin, self.master.sel_colmax = min(t), max(t)
            t = self.sel_startY /self.master.cellheight, self.sel_endY /self.master.cellheight
            self.master.sel_rowmin, self.master.sel_rowmax = min(t), max(t)

            self.master.sel_colmin = max(self.master.sel_colmin, 0)
            self.master.sel_rowmin = max(self.master.sel_rowmin, 0)
            self.master.sel_colmax = min(self.master.sel_colmax, self.master.numcols-1)
            self.master.sel_rowmax = min(self.master.sel_rowmax, self.master.numrows-1)

            self.updateSelectionRect()

    def mouseReleaseEvent(self, ev):
        self.master.updateRuleList()
        self.master.sendIfAuto()
        self.mousePressed = False


class AssociationRulesViewerView(QGraphicsView):
    def __init__(self, master, scene, widget):
        QGraphicsView.__init__(self, scene, widget)
        self.master = master
        self.scene = scene
        self.setFixedSize(365, 365)
        self.selecting = False
        self.update()

    def contentsMousePressEvent(self, ev):
        self.sel_startX = ev.pos().x()
        self.sel_startY = ev.pos().y()
        master = self.master
        master.sel_colmin = master.sel_colmax = self.sel_startX / master.cellwidth
        master.sel_rowmin = master.sel_rowmax = self.sel_startY / master.cellheight
        self.scene().draw()
        master.updateRuleList()

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

        self.scene.draw()
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
        OWWidget.__init__(self, parent, signalManager, "AssociationRulesViewer", wantMainArea=0)

        self.inputs = [("Association Rules", orange.AssociationRules, self.arules)]
        self.outputs = [("Association Rules", orange.AssociationRules)]

        self.supp_min, self.supp_max = self.conf_min, self.conf_max = 0., 1.
        self.numcols = self.numrows = 20
        self.showBars = True

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
        self.mainArea = OWGUI.widgetBox(self.topWidgetPart, orientation = "horizontal", sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding), margin = 0)

        mainLeft = OWGUI.widgetBox(self.mainArea, "Filter")
        OWGUI.separator(self.mainArea, 16, 0)
        mainRight = OWGUI.widgetBox(self.mainArea, "Rules")
        mainRight.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding))
        mainLeft.setSizePolicy(QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum))

        infoGrid = QGridLayout()
        info = OWGUI.widgetBox(mainLeft, orientation = infoGrid)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Shown", addToLayout = 0), 1, 0)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Selected", addToLayout = 0), 2, 0)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Support (H)", addToLayout = 0), 0, 1)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "Confidence (V)", addToLayout = 0), 0, 2)
        infoGrid.addWidget(OWGUI.widgetLabel(info, "# Rules", addToLayout = 0), 0, 3)

        self.shownSupport = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.shownSupport, 1, 1)
        self.shownConfidence = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.shownConfidence, 1, 2)
        self.shownRules = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.shownRules, 1, 3)

        self.selSupport = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.selSupport, 2, 1)
        self.selConfidence = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.selConfidence, 2, 2)
        self.selRules = OWGUI.widgetLabel(info, " ", addToLayout = 0)
        infoGrid.addWidget(self.selRules, 2, 3)

        OWGUI.separator(mainLeft, 0, 4)
        self.ruleScene = AssociationRulesViewerScene(self, mainLeft)
        self.sceneView = AssociationRulesViewerView(self, self.ruleScene, mainLeft)
        mainLeft.layout().addWidget(self.sceneView)

        boxb = OWGUI.widgetBox(mainLeft, box=None, orientation="horizontal")
        OWGUI.button(boxb, self, 'Zoom', callback = self.zoomButton)
        OWGUI.button(boxb, self, 'Show All', callback = self.showAllButton)
        OWGUI.button(boxb, self, 'No Zoom', callback = self.noZoomButton)
        OWGUI.separator(boxb, 16, 8)
        OWGUI.button(boxb, self, 'Unselect', callback = self.unselect)

        self.grid = QGridLayout()
        rightUpRight = OWGUI.widgetBox(mainRight, orientation = self.grid)
        for i, m in enumerate(self.measures):
            cb = OWGUI.checkBox(rightUpRight, self, m[2], m[0], callback = self.showHideColumns, addToLayout = 0)
            self.grid.addWidget(cb, i % 2, i / 2)

        OWGUI.separator(mainRight, 0, 4)
        
        trules = self.trules = QTableWidget(0, 0, mainRight)
        mainRight.layout().addWidget(trules)
        trules.verticalHeader().hide()
        trules.setSelectionMode(QTableWidget.NoSelection)
        trules.setColumnCount(len(self.measures)+1)

        header = trules.horizontalHeader()
        trules.setHorizontalHeaderLabels([m[1] for m in self.measures]+["Rule"])
        trules.setItemDelegate(OWGUI.TableBarItem(self, trules))
        trules.setSortingEnabled(True)
        trules.normalizers = []

        bottomGrid = QGridLayout()
        bottom = OWGUI.widgetBox(mainRight, orientation = bottomGrid)

        self.saveButton = OWGUI.button(bottom, self, "Save Rules", callback = self.saveRules, addToLayout=0)
        commitButton = OWGUI.button(bottom, self, "Send Rules", callback = self.sendRules, addToLayout=0)
        autoSend = OWGUI.checkBox(bottom, self, "autoSend", "Send rules automatically", disables=[(-1, commitButton)], addToLayout=0)
        autoSend.makeConsistent()

        bottomGrid.addWidget(self.saveButton, 1, 0)
        bottomGrid.addWidget(autoSend, 0, 1)
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
        self.selectedRules = sum(sum(self.ingrid, []), [])

        self.displayRules()
        if hasattr(self, "selConfidence"):
            self.updateConfSupp()

        if hasattr(self, "ruleScene"):
            self.ruleScene.unselect()
            self.ruleScene.draw()

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


    def updateRuleList(self):
        self.selectedRules = sum(sum((row[self.sel_colmin : self.sel_colmax+1] for row in self.ingrid[self.sel_rowmin : self.sel_rowmax+1]), []), []) 
        self.displayRules()
        self.updateConfSupp()
        self.saveButton.setEnabled(len(self.selectedRules) > 0)


    def displayRules(self):
        if hasattr(self, "trules"):
            trules = self.trules
            trules.setRowCount(len(self.selectedRules))

            rulecol = len(self.measures)
            trules.normalizers = []
            try:
                self.progressBarInit()
                progressStep = 100./(rulecol+1)
                for col, m in enumerate(self.measures):
                    mname = m[2]
                    values = [getattr(rule, mname) for rule in self.selectedRules]
                    if m[1] in ["Supp", "Conf", "Cov"]:
                        trules.normalizers.append((1, 1))
                    elif values:
                        mi, ma = min(values), max(values)
                        div = ma - mi
                        trules.normalizers.append((ma, div or 1))
                    else:
                        trules.normalizers.append((0, 1))
                    for row, meas in enumerate(values):
                        trules.setItem(row, col, QTableWidgetItem("  %.3f  " % meas))
                    self.progressBarAdvance(progressStep)
    
                for row, rule in enumerate(self.selectedRules):
                    trules.setItem(row, rulecol, QTableWidgetItem(str(rule).replace(" ", "  ")))
            finally:
                self.progressBarFinished()

            self.trules.resizeColumnsToContents()
            self.trules.resizeRowsToContents()
            
            self.showHideColumns()


    def showHideColumns(self):
        for i, m in enumerate(self.measures):
            (getattr(self, m[2]) and self.trules.showColumn or self.trules.hideColumn)(i)


    def saveRules(self):
        fileName = QFileDialog.getSaveFileName(self, "Save Rules", "myRules.txt", "Textfiles (*.txt)" );
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
            self.ruleScene.draw()
            self.sendIfAuto()


    def rezoom(self, smi, sma, cmi, cma):
        self.supp_min, self.supp_max, self.conf_min, self.conf_max = smi, sma, cmi, cma
        self.checkScale() # to set the inCell
        self.setIngrid()
        self.unselect()
        if hasattr(self, "ruleScene"):
            self.ruleScene.draw()
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
            supps = [rule.support for rule in self.rules]
            self.supp_min = min(supps)
            self.supp_max = max(supps)
            del supps

            confs = [rule.confidence for rule in self.rules]
            self.conf_min = min(confs)
            self.conf_max = max(confs)
            del confs

            self.checkScale()
        else:
            self.supp_min, self.supp_max = self.conf_min, self.conf_max = 0., 1.

        self.supp_allmin, self.supp_allmax, self.conf_allmin, self.conf_allmax = self.supp_min, self.supp_max, self.conf_min, self.conf_max
        self.rezoom(self.supp_allmin, self.supp_allmax, self.conf_allmin, self.conf_allmax)



if __name__=="__main__":
    a=QApplication(sys.argv)
    ow=OWAssociationRulesViewer()

    dataset = orange.ExampleTable('../../doc/datasets/car.tab')
    rules=orange.AssociationRulesInducer(dataset, minSupport = 0.3, maxItemSets=15000)
    ow.arules(rules)

    ow.show()
    a.exec_()
    ow.saveSettings()

"""
<name>Predictions</name>
<description>Displays predictions of models for a particular data set.</description>
<icon>icons/Predictions.png</icon>
<contact>Blaz Zupan (blaz.zupan(@at@)fri.uni-lj.si)</contact>
<priority>300</priority>
"""

import orngOrangeFoldersQt4
from OWWidget import *
import OWGUI
import statc

##############################################################################

class colorItem(QTableWidgetItem):
    def __init__(self, table, editType, text):
        QTableWidgetItem.__init__(self, table, editType, str(text))

    def paint(self, painter, colorgroup, rect, selected):
        g = QPalette(colorgroup)
        g.setColor(QPalette.Base, Qt.lightGray)
        QTableWidgetItem.paint(self, painter, g, rect, selected)

##############################################################################

class OWPredictions(OWWidget):
    settingsList = ["showProb", "showClass",
                    "ShowAttributeMethod", "sendDataType", "sendOnChange",
                    "sendPredictions", "sendSelection", "classvalues",
                    "rbest", "rpercentile"]

    def __init__(self, parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Predictions")

        self.callbackDeposit = []
        self.inputs = [("Examples", ExampleTable, self.setData),("Classifiers", orange.Classifier, self.setClassifier, Multiple)]
        self.outputs = [("Predictions", ExampleTable), ("Selected Examples", ExampleTable)]
        self.classifiers = {}

        # saveble settings
        self.showProb = 1; self.showClass = 1
        self.ShowAttributeMethod = 0
        self.sendDataType = 0; self.sendOnChange = 1
        self.sendPredictions = 0
        self.sendSelection = 1
        self.rbest = 0
        self.loadSettings()
        self.outvar = None # current output variable (set by the first predictor send in)

        self.freezeAttChange = 0 # block table update after changes in attribute list box?
        self.data = None

        # GUI - Options

        # Options - classification
        self.copt = OWGUI.widgetBox(self.controlArea, "Options (classification)")
        self.copt.setDisabled(1)
        OWGUI.checkBox(self.copt, self, 'showProb', "Show predicted probabilities", callback=self.updateTableOutcomes)

        self.lbClasses = OWGUI.listBox(self.copt, self, selectionMode = QListWidget.MultiSelection, callback = self.updateTableOutcomes)

        OWGUI.checkBox(self.copt, self, 'showClass', "Show predicted class", callback=[self.updateTableOutcomes, self.checksendpredictions])

        # Options - regression
        # self.ropt = QVButtonGroup("Options (regression)", self.controlArea)
        # OWGUI.checkBox(self.ropt, self, 'showClass', "Show predicted class",
        #                callback=[self.updateTableOutcomes, self.checksendpredictions])
        # self.ropt.hide()

        OWGUI.separator(self.controlArea)

        self.att = OWGUI.widgetBox(self.controlArea, "Data attributes")
        OWGUI.radioButtonsInBox(self.att, self, 'ShowAttributeMethod', ['Show all', 'Hide all'], callback=self.updateAttributes)
        self.att.setDisabled(1)

        OWGUI.separator(self.controlArea)
        self.outbox = OWGUI.widgetBox(self.controlArea, "Output")

        self.dsel = OWGUI.checkBox(self.outbox, self, "sendSelection", "Send data selection")
        # data selection for classification
        self.csel = OWGUI.radioButtonsInBox(self.outbox, self, 'sendDataType',
                                ['Examples with class conflict', 'Examples with class agreement'],
                                box='Data selection',
                                tooltips=['Data instances with different true and predictied class.',
                                          'Data instances with matching true and predictied class.'],
                                callback=self.checksendselection)
        # data selection for regression
        self.rsel = OWGUI.widgetBox(self.outbox, "Data selection")
        OWGUI.radioButtonsInBox(self.rsel, self, "rbest", ["Highest variance", "Lowest variance"],
                                callback = self.checksendselection)
        hb = OWGUI.widgetBox(self.rsel, orientation = "horizontal")
        OWGUI.widgetLabel(hb, 'Percentiles: ')
        OWGUI.comboBox(hb, self, "rpercentile",
                       items = [0.01, 0.02, 0.05, 0.1, 0.2],
                       sendSelectedValue = 1, valueType = float, callback = self.checksendselection)

        self.dsel.disables = [self.csel, self.rsel]

        OWGUI.checkBox(self.outbox, self, 'sendPredictions', "Send predictions",
                       callback=self.updateTableOutcomes)
        OWGUI.separator(self.controlArea)

        self.commitBtn = OWGUI.button(self.outbox, self, "Send data", callback=self.senddata)
        OWGUI.checkBox(self.outbox, self, 'sendOnChange', 'Send automatically')

        self.outbox.setDisabled(1)

        # GUI - Table
        self.table = OWGUI.table(self.mainArea, selectionMode = QTableWidget.NoSelection)
        self.header = self.table.horizontalHeader()
        self.vheader = self.table.verticalHeader()
        # manage sorting (not correct, does not handle real values)
        self.connect(self.header, SIGNAL("pressed(int)"), self.sort)
        self.sortby = -1


    ##############################################################################
    # Contents painting

    def updateTableOutcomes(self):
        """updates the columns associated with the classifiers"""
        if self.freezeAttChange: # program-based changes should not alter the table immediately
            return
        if not self.data or not self.classifiers:
            return

        classification = None
        if self.outvar:
            classification = self.outvar.varType == orange.VarTypes.Discrete
            if classification:
                selclass = [self.lbClasses.item(i).isSelected() for i in range(len(self.data.domain.classVar.values))]
                showclass = selclass.count(1)
            else:
                showclass = 1

        # sindx is the column where these start
        sindx = 1 + len(self.data.domain.attributes) + 1 * (self.data.domain.classVar<>None)
        col = sindx
        if self.showClass or self.showProb:
            for (cid, c) in enumerate(self.classifiers.values()):
                if classification:
                    for (i, d) in enumerate(self.data):
                        (cl, p) = c(d, orange.GetBoth)

                        self.classifications[i].append(cl)
                        if self.showProb and showclass:
                            s = " : ".join(["%5.3f" % p for (vi,p) in enumerate(p) if selclass[vi]])
                            if self.showClass: s += " -> "
                        else:
                            s = ""
                        if self.showClass:
                            s += str(cl)
                        self.table.setText(self.rindx[i], col, s)
                else:
                    # regression
                    for (i, d) in enumerate(self.data):
                        cl = c(d)
                        self.classifications[i].append(cl)
                        self.table.setText(self.rindx[i], col, str(cl))
                col += 1
        else:
            for i in range(len(self.data)):
                for c in range(len(self.classifiers)):
                    self.table.setText(self.rindx[i], col+c, '')
            col += len(self.classifiers)

        for i in range(sindx, col):
            self.table.adjustColumn(i)
            if self.showClass or self.showProb:
                self.table.showColumn(i)
            else:
                self.table.hideColumn(i)

    def updateTrueClass(self):
        col = 1+len(self.data.domain.attributes)
        if self.data.domain.classVar:
            self.table.showColumn(col)
            self.table.adjustColumn(col)
        else:
            self.table.hideColumn(col)

    def updateAttributes(self):
        if self.ShowAttributeMethod == 0:
            for i in range(len(self.data.domain.attributes)):
                self.table.showColumn(i+1)
                self.table.adjustColumn(i+1)
        if self.ShowAttributeMethod == 1:
            for i in range(len(self.data.domain.attributes)):
                self.table.hideColumn(i+1)

    def setTable(self):
        """defines the attribute/predictions table and paints its contents"""
        if self.data==None:
            return

        self.table.setColumnCount(1 + len(self.data.domain.attributes) + (self.data.domain.classVar <> None) + len(self.classifiers))
        self.table.setRowCount(len(self.data))

        # HEADER: set the header (attribute names)
        self.header.setLabel(0, '#')
        for col in range(len(self.data.domain.attributes)):
            self.header.setLabel(col+1, self.data.domain.attributes[col].name)
        col = len(self.data.domain.attributes)+1
        self.header.setLabel(col, self.data.domain.classVar.name)
        col += 1
        for (i,c) in enumerate(self.classifiers.values()):
            self.header.setLabel(col+i, c.name)

        # ATTRIBUTE VALUES: set the contents of the table (values of attributes), data first
        for i in range(len(self.data)):
            self.table.setText(i, 0, str(i+1))
            for j in range(len(self.data.domain.attributes)):
                self.table.setText(i, j+1, str(self.data[i][j]))
        col = 1+len(self.data.domain.attributes)

        self.classifications = [[]] * len(self.data)
        if self.data.domain.classVar:
            for i in range(len(self.data)):
                c = self.data[i].getclass()
                item = colorItem(self.table, QTableItem.WhenCurrent, str(c))
                self.table.setItem(i, col, item)
                self.classifications[i] = [c]
            col += 1

        for i in range(col):
            self.table.adjustColumn(i)

        # include classifications
        self.updateTableOutcomes()
        self.updateAttributes()
        self.updateTrueClass()
        self.table.hideColumn(0) # hide column with indices, we will use vertical header to show this info

    def sort(self, col):
        "sorts the table by column col"
        self.sortby = - self.sortby
        self.table.sortColumn(col, self.sortby>=0, TRUE)

        # the table may be sorted, figure out data indices
        for i in range(len(self.data)):
            self.rindx[int(str(self.table.item(i,0).text()))-1] = i
        for (i, indx) in enumerate(self.rindx):
            self.vheader.setLabel(i, self.table.item(i,0).text())

    def checkenable(self):
        # following should be more complicated and depends on what data are we showing
        cond = len(self.classifiers)
        self.outbox.setEnabled(cond)
        self.att.setEnabled(cond)
        self.copt.setEnabled(cond)
        e = (self.data and (self.data.domain.classVar <> None) + len(self.classifiers)) >= 2
        # need at least two classes to compare predictions
        self.dsel.setEnabled(e)
        if e and self.sendSelection:
            self.csel.setEnabled(1)
            self.rsel.setEnabled(1)

    ##############################################################################
    # Input signals

    def consistenttarget(self, target):
        """returns TRUE if target is consistent with current predictiors and data"""
        if self.classifiers:
            return target == self.classifiers.values()[0]
        return True

    def setData(self, data):
        if not data:
            self.data = data
            self.table.hide()
            self.send("Selected Examples", None)
            self.send("Predictions", None)
            self.att.setDisabled(1)
            self.outbox.setDisabled(1)
        else:
            vartypes = {1:"discrete", 2:"continuous"}
            if len(self.classifiers) and data.domain.classVar and data.domain.classVar <> self.outvar:
                self.warning(id, "Data set %s ignored, inconsistent outcome variables\n%s/%s <> %s/%s (type or variable mismatch)" % (data.name, data.domain.classVar.name, vartypes.get(data.domain.classVar.varType, "?"), self.outvar.name, vartypes.get(self.outvar.varType, "?")))
                return
            self.data = data
            self.rindx = range(len(self.data))
            self.setTable()
            self.table.show()
            self.checksenddata()
        self.checkenable()

    def setClassifier(self, c, id):
        """handles incoming classifier (prediction, as could be a regressor as well)"""
        if not c:
            if self.classifiers.has_key(id):
                del self.classifiers[id]
                if len(self.classifiers) == 0: self.outvar = None
            else:
                self.warning(id, "")
        else:
            if len(self.classifiers) and c.classVar <> self.outvar:
                vartypes = {1:"discrete", 2:"continuous"}
                self.warning(id, "Predictor %s ignored, inconsistent outcome variables\n%s/%s <> %s/%s (type or variable mismatch)" % (c.name, c.classVar.name, vartypes.get(c.classVar.varType, "?"), self.outvar.name, vartypes.get(self.outvar.varType, "?")))
                return
            else:
                self.outvar = c.classVar
            self.classifiers[id] = c

        if len(self.classifiers) == 1 and c:
            # defines the outcome variable and the type of the problem we are dealing with (regression/classification)
            self.outvar == c.classVar
            if self.outvar.varType == orange.VarTypes.Continuous:
                # regression
                self.copt.hide(); self.csel.hide(); self.rsel.show()
            else:
                # classification
                self.rsel.hide(); self.copt.show(); self.csel.show()
                lb = self.lbClasses
                lb.clear()
                for v in self.outvar.values:
                    lb.addItem(str(v))
                self.freezeAttChange = 1
                for i in range(len(self.outvar.values)):
                    lb.item(i).setSelected(1)
                lb.show()
                self.freezeAttChange = 0

        if self.data:
            self.setTable()
            self.table.show()
            self.checksenddata()
        self.checkenable()

    ##############################################################################
    # Ouput signals

    def checksenddata(self):
        # if self.sendOnChange and self.outbox.isEnabled():
        if len(self.classifiers) and self.sendOnChange: self.senddata()

    def checksendselection(self):
        if len(self.classifiers) and self.sendOnChange: self.selection()

    def checksendpredictions(self):
        if len(self.classifiers) and self.sendOnChange: self.predictions()

    def senddata(self):
        self.predictions()
        self.selection()

    # assumes that the data and display conditions
    # (enough classes are displayed) have been checked

    def predictions(self):
        if self.freezeAttChange: return
        if not self.data or not self.classifiers:
            self.send("Predictions", None)

        if self.sendPredictions:
            # predictions, data set with class predictions
            classification = self.outvar.varType == orange.VarTypes.Discrete

            metas = []
            if classification:
                selclass = [self.lbClasses.item(i).isSelected() for i in range(len(self.data.domain.classVar.values))]
                showclass = selclass.count(1)
                if showclass:
                    for c in self.classifiers.values():
                        m = [orange.FloatVariable(name="%s(%s)" % (c.name, str(v)),
                                                  getValueFrom = lambda ex, rw, cindx=i: orange.Value(c(ex, c.GetProbabilities)[cindx])) \
                             for (i, v) in enumerate(self.data.domain.classVar.values) if selclass[i]]
                        metas.extend(m)
                if self.showClass:
                    mc = [orange.EnumVariable(name="%s" % c.name, values = self.data.domain.classVar.values,
                                             getValueFrom = lambda ex, rw: orange.Value(c(ex)))
                          for c in self.classifiers.values()]
                    metas.extend(mc)
            else:
                # regression
                mc = [orange.FloatVariable(name="%s" % c.name,
                                           getValueFrom = lambda ex, rw: orange.Value(c(ex)))
                      for c in self.classifiers.values()]
                metas.extend(mc)

            domain = orange.Domain(self.data.domain.attributes + [self.data.domain.classVar])
            for m in metas:
                domain.addmeta(orange.newmetaid(), m)
            predictions = orange.ExampleTable(domain, self.data)
            predictions.name = self.data.name
            self.send("Predictions", predictions)

    def selection(self):
        def cmpclasses(clist):
            """returns True if all elements in clist are the same"""
            clist = filter(lambda x: not x.isSpecial(), clist)
            ref = clist[0]
            for c in clist[1:]:
                if c<>ref: return 0
            return 1

        if not self.sendSelection:
            return

        if not self.data or not self.classifiers:
            self.send("Selected Examples", None)

        classification = self.outvar.varType == orange.VarTypes.Discrete
        if classification:
            s = [cmpclasses(cls) for cls in self.classifications]
            if self.sendDataType == 1:
                s = [not x for x in s]
            data_selection = self.data.select(s)
        else:
            if self.data.domain.classVar:
                variance = [(i, statc.var(cls)) for (i, cls) in enumerate(self.classifications)
                            if not cls[0].isSpecial()]
            else:
                variance = [(i, statc.var(cls)) for (i, cls) in enumerate(self.classifications)]
            variance.sort(lambda x, y: cmp(x[1], y[1]))
            if not self.rbest:
                variance.reverse()
            n = int(len(self.data) * self.rpercentile)
            if not n:
                return
            sel = [r[0] for r in variance[:n]]
            data_selection = self.data.getitems(sel)
        data_selection.name = self.data.name
        self.send("Selected Examples", data_selection)

##############################################################################
# Test the widget, run from DOS prompt

if __name__=="__main__":
    a = QApplication(sys.argv)
    ow = OWPredictions()
    ow.show()

    if 0: # data set only
        data = orange.ExampleTable('sailing')
        ow.setData(data)
    if 0:
        data = orange.ExampleTable('outcome')
        test = orange.ExampleTable('cheat', uses=data.domain)
        data = orange.ExampleTable('iris')

        bayes = orange.BayesLearner(data, name="NBC")

        import orngTree
        tree = orngTree.TreeLearner(data, name="Tree")
        ow.setClassifier(bayes, 1)
        ow.setClassifier(tree, 2)
        ow.setData(test)
    if 1: # two classifiers
        data = orange.ExampleTable('sailing.txt')
        bayes = orange.BayesLearner(data)
        bayes.name = "NBC"
        ow.setClassifier(bayes, 1)
        maj = orange.MajorityLearner(data)
        maj.name = "Majority"
        import orngTree
        tree = orngTree.TreeLearner(data, name="Tree")
        knn = orange.kNNLearner(data, k = 10)
        knn.name = "knn"
        ow.setClassifier(maj, 2)
        ow.setClassifier(knn, 3)
        ow.setData(data)
    if 0: # regression
        data = orange.ExampleTable('auto-mpg')
        data.name = 'auto-mpg'
        knn = orange.kNNLearner(data, name="knn")
        knn.name = "knn"
        maj = orange.MajorityLearner(data)
        maj.name = "Majority"
        ow.setClassifier(knn, 10)
        ow.setClassifier(maj, 2)
        ow.setData(data)

    a.exec_()
    ow.saveSettings()

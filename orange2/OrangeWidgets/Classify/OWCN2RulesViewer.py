"""
<name>CN2 Rules Viewer</name>
<description>Viewer of classification rules.</description>
<icon>icons/CN2RulesViewer.png</icon>
<contact>Ales Erjavec (ales.erjavec(@at@)fri.uni-lj.si)</contact>
<priority>2120</priority>
"""

import orngEnviron

import orange, orngCN2
from OWWidget import *
from OWPredictions import PyTableModel
import OWGUI, OWColorPalette
import sys, re, math

def _toPyObject(variant):
    val = variant.toPyObject()
    if isinstance(val, type(NotImplemented)): # PyQt 4.4 converts python int, floats ... to C types
        qtype = variant.type()
        if qtype == QVariant.Double:
            val, ok = variant.toDouble()
        elif qtype == QVariant.Int:
            val, ok = variant.toInt()
        elif qtype == QVariant.LongLong:
            val, ok = variant.toLongLong()
        elif qtype == QVariant.String:
            val = variant.toString()
    return val
        
    
class DistributionItemDelegate(QStyledItemDelegate):
    def __init__(self, parent):
        QStyledItemDelegate.__init__(self, parent)

    def displayText(self, value, locale):
        dist = value.toPyObject()
        if isinstance(dist, orange.Distribution):
            return QString("<" + ",".join(["%.1f" % c for c in dist]) + ">")
        else:
            return QStyledItemDelegate.displayText(value, locale)
        
    def sizeHint(self, option, index):
        metrics = QFontMetrics(option.font)
        height = metrics.lineSpacing() * 2 + 8 # 4 pixel margin
        width = metrics.width(self.displayText(index.data(Qt.DisplayRole), QLocale())) + 8
        return QSize(width, height)
    
    def paint(self, painter, option, index):
        dist = index.data(Qt.DisplayRole).toPyObject()
        rect = option.rect.adjusted(4, 4, -4, -4)
        rect_w = rect.width() - len([c for c in dist if c]) # This is for the separators in the distribution bar
        rect_h = rect.height()
        colors = OWColorPalette.ColorPaletteHSV(len(dist))
        abs = dist.abs
        dist_sum = 0
        
        painter.save()
        painter.setFont(option.font)
        
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter)
        
        showText = getattr(self, "showDistText", True)
        metrics = QFontMetrics(option.font)
        drect_h = metrics.height()
        lineSpacing = metrics.lineSpacing()
        leading = metrics.leading()
        distText = self.displayText(index.data(Qt.DisplayRole), QLocale())
        
        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
#        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color))
            
        if showText:
            textPos = rect.topLeft()
            textRect = QRect(textPos, QSize(rect.width(), rect.height() / 2 - leading))
            painter.drawText(textRect, Qt.AlignHCenter | Qt.AlignBottom, distText)
            
        painter.setPen(QPen(Qt.black))
        painter.translate(QPoint(rect.topLeft().x(), rect.center().y() - (drect_h/2 if not showText else  0)))
        for i, count in enumerate(dist):
            if count:
                color = colors[i]
                painter.setBrush(color)
                painter.setRenderHint(QPainter.Antialiasing)
                width = round(rect_w * float(count) / abs)
                painter.drawRoundedRect(QRect(1, 3, width, 5), 1, 2)
                painter.translate(width, 0)
        painter.restore()
        
class MultiLineStringItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        metrics = QFontMetrics(option.font)
        text = index.data(Qt.DisplayRole).toString()
        size = metrics.size(0, text)
        return QSize(size.width() + 8, size.height() + 8) # 4 pixel margin
    
    def paint(self, painter, option, index):
        text = self.displayText(index.data(Qt.DisplayRole), QLocale())
        painter.save()
        
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewItem, option, painter)
        
        rect = option.rect.adjusted(4, 4, -4, -4)
            
        if option.state & QStyle.State_Selected:
            color = option.palette.highlightedText().color()
        else:
            color = option.palette.text().color()
#        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color))
        
            
        painter.drawText(rect, option.displayAlignment, text)
        painter.restore()
        
        
class PyObjectItemDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        obj = _toPyObject(value) #value.toPyObject()
        return QString(str(obj))
    
    
class PyFloatItemDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        obj = _toPyObject(value)
        if isinstance(obj, float):
            return QString("%.2f" % obj)
        else:
            return QString(str(obj))
        
def rule_to_string(rule, show_distribution = True):
    """
    Write a string presentation of rule in human readable format.
    
    :param rule: rule to pretty-print.
    :type rule: :class:`Orange.classification.rules.Rule`
    
    :param show_distribution: determines whether presentation should also
        contain the distribution of covered instances
    :type show_distribution: bool
    
    """
    import Orange
    def selectSign(oper):
        if oper == Orange.core.ValueFilter_continuous.Less:
            return "<"
        elif oper == Orange.core.ValueFilter_continuous.LessEqual:
            return "<="
        elif oper == Orange.core.ValueFilter_continuous.Greater:
            return ">"
        elif oper == Orange.core.ValueFilter_continuous.GreaterEqual:
            return ">="
        else: return "="

    if not rule:
        return "None"
    conds = rule.filter.conditions
    domain = rule.filter.domain
    
    def pprint_values(values):
        if len(values) > 1:
            return "[" + ",".join(values) + "]"
        else:
            return str(values[0])
        
    ret = "IF "
    if len(conds)==0:
        ret = ret + "TRUE"

    for i,c in enumerate(conds):
        if i > 0:
            ret += " AND "
        if type(c) == Orange.core.ValueFilter_discrete:
            ret += domain[c.position].name + "=" + pprint_values( \
                   [domain[c.position].values[int(v)] for v in c.values])
        elif type(c) == Orange.core.ValueFilter_continuous:
            ret += domain[c.position].name + selectSign(c.oper) + str(c.ref)
    if rule.classifier and type(rule.classifier) == Orange.classification.ConstantClassifier\
            and rule.classifier.default_val:
        ret = ret + " THEN "+domain.class_var.name+"="+\
        str(rule.classifier.default_value)
        if show_distribution:
            ret += str(rule.class_distribution)
    elif rule.classifier and type(rule.classifier) == Orange.classification.ConstantClassifier\
            and type(domain.class_var) == Orange.core.EnumVariable:
        ret = ret + " THEN "+domain.class_var.name+"="+\
        str(rule.class_distribution.modus())
        if show_distribution:
            ret += str(rule.class_distribution)
    return ret        

        
class OWCN2RulesViewer(OWWidget):
    settingsList = ["show_Rule_length", "show_Rule_quality", "show_Coverage",
                    "show_Predicted_class", "show_Distribution", "show_Rule"]
    
    def __init__(self, parent=None, signalManager=None, name="CN2 Rules Viewer"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs = [("Rule Classifier", orange.RuleClassifier, self.setRuleClassifier)]
        self.outputs = [("Data", ExampleTable), ("Features", AttributeList)]
        
        self.show_Rule_length = True
        self.show_Rule_quality = True
        self.show_Coverage = True
        self.show_Predicted_class = True
        self.show_Distribution = True
        self.show_Rule = True
        
        self.autoCommit = False
        self.selectedAttrsOnly = True
        
        
        self.loadSettings()
        
        #####
        # GUI
        #####
        
        box = OWGUI.widgetBox(self.controlArea, "Show Info", addSpace=True)
        box.layout().setSpacing(3)
        self.headers = ["Rule length",
                        "Rule quality",
                        "Coverage",
                        "Predicted class",
                        "Distribution",
                        "Rule"]
        
        for i, header in enumerate(self.headers):
            OWGUI.checkBox(box, self, "show_%s" % header.replace(" ", "_"), header,
                           tooltip="Show %s column" % header.lower(),
                           callback=self.updateVisibleColumns)
            
        box = OWGUI.widgetBox(self.controlArea, "Output")
        box.layout().setSpacing(3)
        cb = OWGUI.checkBox(box, self, "autoCommit", "Commit on any change",
                            callback=self.commitIf)
        
        OWGUI.checkBox(box, self, "selectedAttrsOnly", "Selected attributes only",
                       tooltip="Send selected attributes only",
                       callback=self.commitIf)
        
        b = OWGUI.button(box, self, "Commit", callback=self.commit, default=True)
        OWGUI.setStopper(self, b, cb, "changedFlag", callback=self.commit)
        
        OWGUI.rubber(self.controlArea)
        
        self.tableView = QTableView()
        self.tableView.setItemDelegate(PyObjectItemDelegate(self))
        self.tableView.setItemDelegateForColumn(1, PyFloatItemDelegate(self))
        self.tableView.setItemDelegateForColumn(2, PyFloatItemDelegate(self))
        self.tableView.setItemDelegateForColumn(4, DistributionItemDelegate(self))
        self.tableView.setItemDelegateForColumn(5, MultiLineStringItemDelegate(self))
        self.tableView.setSortingEnabled(True)
        self.tableView.setSelectionBehavior(QTableView.SelectRows)
        self.tableView.setAlternatingRowColors(True)
        
        self.rulesTableModel = PyTableModel([], self.headers)
        self.proxyModel = QSortFilterProxyModel(self)
        self.proxyModel.setSourceModel(self.rulesTableModel)
        
        self.tableView.setModel(self.proxyModel)
        self.connect(self.tableView.selectionModel(),
                     SIGNAL("selectionChanged(QItemSelection, QItemSelection)"),
                     lambda is1, is2: self.commitIf())
        self.connect(self.tableView.horizontalHeader(), SIGNAL("sectionClicked(int)"), lambda section: self.tableView.resizeRowsToContents())
        self.mainArea.layout().addWidget(self.tableView)

        self.updateVisibleColumns()
        
        self.changedFlag = False
        self.classifier = None
        self.rules = []
        self.resize(800, 600)

    def sendReport(self):
        nrules = self.rulesTableModel.rowCount()
        print nrules
        if not nrules:
            self.reportRaw("<p>No rules.</p>")
            return
        
        shown = [i for i, header in enumerate(self.headers) if getattr(self, "show_%s" % header.replace(" ", "_"))]
        rep = '<table>\n<tr style="height: 2px"><th colspan="11"  style="border-bottom: thin solid black; height: 2px;">\n'
        rep += "<tr>"+"".join("<th>%s</th>" % self.headers[i] for i in shown)+"</tr>\n"
        for row in range(nrules):
            rep += "<tr>"
            for col in shown:
                data = _toPyObject(self.rulesTableModel.data(self.rulesTableModel.createIndex(row, col)))
                if col==4:
                    rep += "<td>%s</td>" % ":".join(map(str, data))
                elif col in (0, 3):
                    rep += '<td align="center">%s</td>' % data
                elif col in (1, 2):
                    rep += '<td align="right">%.3f&nbsp;</td>' % data
                else:
                    rep += '<td>%s</td>' % data
        rep += '<tr style="height: 2px"><th colspan="11"  style="border-bottom: thin solid black; height: 2px;">\n</table>\n'
        self.reportRaw(rep)
        
    def setRuleClassifier(self, classifier=None):
        self.classifier = classifier
        if classifier is not None:
            self.rules = classifier.rules
        else:
            self.rules = []
        
    def handleNewSignals(self):
        self.updateRulesModel()
        self.commit()
    
    def updateRulesModel(self):
        table = []
        if self.classifier is not None:
            for i, r in enumerate(self.classifier.rules):
                table.append((int(r.complexity),
                              r.quality,
                              r.classDistribution.abs,
                              str(r.classifier.defaultValue),
                              r.classDistribution,
                              self.ruleText(r)))

        self.rulesTableModel = PyTableModel(table, self.headers)
        self.proxyModel.setSourceModel(self.rulesTableModel)
        self.tableView.resizeColumnsToContents()
        self.tableView.resizeRowsToContents()
        self.updateVisibleColumns() # if the widget got data for the first time
            
    
    def ruleText(self, rule):
        text = rule_to_string(rule, show_distribution=False)
        p = re.compile(r"[0-9]\.[0-9]+")
        text = p.sub(lambda match: "%.2f" % float(match.group()[0]), text)
        text = text.replace("AND", "AND\n   ")
        text = text.replace("THEN", "\nTHEN")
        return text
    
    def updateVisibleColumns(self):
        anyVisible = False
        for i, header in enumerate(self.headers):
            visible = getattr(self, "show_%s" % header.replace(" ", "_"))
            self.tableView.horizontalHeader().setSectionHidden(i, not visible)
            anyVisible = anyVisible or visible
        
        # report button is not available if not running canvas
        if hasattr(self, "reportButton"):
            self.reportButton.setEnabled(anyVisible)

    
    def commitIf(self):
        if self.autoCommit:
            self.commit()
        else:
            self.changedFlag = True
            
    def selectedAttrsFromRules(self, rules):
        selected = []
        for rule in rules:
            for c in rule.filter.conditions:
                selected.append(rule.filter.domain[c.position])
        return set(selected)
    
    def selectedExamplesFromRules(self, rules, examples):
        selected = []
        for rule in rules:
            selected.extend(examples.filterref(rule.filter))
            rule.filter.negate=1
            examples = examples.filterref(rule.filter)
            rule.filter.negate=0
        return selected
             
    def commit(self):
        rows = self.tableView.selectionModel().selectedRows()
        rows = [self.proxyModel.mapToSource(index) for index in rows]
        rows = [index.row() for index in rows]
        selectedRules = [self.classifier.rules[row] for row in rows]
        
        if selectedRules:
            examples = self.classifier.examples
            selectedExamples = self.selectedExamplesFromRules(selectedRules, self.classifier.examples)
            selectedAttrs = self.selectedAttrsFromRules(selectedRules)
            selectedAttrs = [attr for attr in examples.domain.attributes if attr in selectedAttrs] # restore the order
            if self.selectedAttrsOnly:
                domain = orange.Domain(selectedAttrs, examples.domain.classVar)
                domain.addmetas(examples.domain.getmetas())
                selectedExamples = orange.ExampleTable(domain, selectedExamples)
            else:
                selectedExamples = orange.ExampleTable(selectedExamples)
                
            self.send("Data", selectedExamples)
            self.send("Features", orange.VarList(list(selectedAttrs)))
        
        else:
            self.send("Data", None)
            self.send("Features", None)
        
        self.changedFlag = False
        
    
if __name__=="__main__":
    ap=QApplication(sys.argv)
    w=OWCN2RulesViewer()
    #data=orange.ExampleTable("../../doc/datasets/car.tab")
    data = orange.ExampleTable("../../doc/datasets/car.tab")
    l=orngCN2.CN2UnorderedLearner()
    l.ruleFinder.ruleStoppingValidator=orange.RuleValidator_LRS()
    w.setRuleClassifier(l(data))
    w.setRuleClassifier(l(data))
    w.handleNewSignals()
    w.show()
    ap.exec_()


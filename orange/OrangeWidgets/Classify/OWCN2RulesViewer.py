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
#            return QString("")
        else:
            return QStyledItemDelegate.displayText(value, locale)
        
    
    def sizeHint(self, option, index):
        metrics = QFontMetrics(option.font)
        height = metrics.lineSpacing() * 2 + 8
        width = metrics.width(self.displayText(index.data(Qt.DisplayRole), QLocale()))
        return QSize(width, height)
    
    
    def paint(self, painter, option, index):
        dist = index.data(Qt.DisplayRole).toPyObject()
        rect = option.rect
        rect_w = rect.width() - len([c for c in dist if c]) - 2
        rect_h = rect.height() - 2
        colors = OWColorPalette.ColorPaletteHSV(len(dist))
        abs = dist.abs
        dist_sum = 0
        
        painter.save()
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        
        showText = getattr(self, "showDistText", True)
        metrics = QFontMetrics(option.font)
        drect_h = metrics.height()
        lineSpacing = metrics.lineSpacing()
        distText = self.displayText(index.data(Qt.DisplayRole), QLocale())
        if showText:
            textPos = QPoint(rect.topLeft().x(), rect.center().y() - lineSpacing)
            painter.drawText(QRect(textPos, QSize(rect.width(), lineSpacing)), Qt.AlignCenter, distText)
        
        painter.translate(QPoint(rect.topLeft().x(), rect.center().y() - (drect_h/2 if not showText else  - 2)))
        for i, count in enumerate(dist):
            if count:
                color = colors[i]
                painter.setBrush(color)
                width = round(rect_w * float(count) / abs)
                painter.drawRect(QRect(1, 1, width, drect_h))
                painter.translate(width, 0)
        
        painter.restore()
        
        
class MultiLineStringItemDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        metrics = QFontMetrics(option.font)
        text = index.data(Qt.DisplayRole).toString()
        return metrics.size(0, text)
    
    
    def paint(self, painter, option, index):
        text = self.displayText(index.data(Qt.DisplayRole), QLocale())
        painter.save()
        qApp.style().drawPrimitive(QStyle.PE_PanelItemViewRow, option, painter)
        painter.drawText(option.rect, Qt.AlignLeft | Qt.AlignVCenter, text)
        painter.restore()
        
        
class PyObjectItemDelegate(QStyledItemDelegate):
    def displayText(self, value, locale):
        obj = _toPyObject(value) #value.toPyObject()
        return QString(str(obj))
            

class OWCN2RulesViewer(OWWidget):
    settingsList = ["show_Rule_length", "show_Rule_quality", "show_Coverage",
                    "show_Predicted_class", "show_Distribution", "show_Rule"]
    
    def __init__(self, parent=None, signalManager=None, name="CN2 Rules Viewer"):
        OWWidget.__init__(self, parent, signalManager, name)
        self.inputs = [("Rule Classifier", orange.RuleClassifier, self.setRuleClassifier)]
        self.outputs = [("Examples", ExampleTable), ("Attribute List", AttributeList)]
        
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
        cb = OWGUI.checkBox(box, self, "autoCommit", "Commit on any change",
                            callback=self.commitIf)
        
        OWGUI.checkBox(box, self, "selectedAttrsOnly", "Selected attributes only",
                       tooltip="Send selected attributes only",
                       callback=self.commitIf)
        
        b = OWGUI.button(box, self, "Commit", callback=self.commit)
        OWGUI.setStopper(self, b, cb, "changedFlag", callback=self.commit)
        
        OWGUI.rubber(self.controlArea)
        
        self.tableView = QTableView()
        self.tableView.setItemDelegate(PyObjectItemDelegate(self))
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
        
        self.changedFlag = False
        self.classifier = None
        self.rules = []
        self.resize(800, 600)
        
        
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
        if self.classifier is not None:
            table = []
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
            
    
    def ruleText(self, rule):
        text = orngCN2.ruleToString(rule, showDistribution=False)
        p = re.compile(r"[0-9]\.[0-9]+")
        text = p.sub(lambda match: "%.2f" % float(match.group()[0]), text)
        text = text.replace("AND", "AND\n   ")
        text = text.replace("THEN", "\nTHEN")
        return text
        
    
    def updateVisibleColumns(self):
        for i, header in enumerate(self.headers):
            self.tableView.horizontalHeader().setSectionHidden(i, not getattr(self, "show_%s" % header.replace(" ", "_")))
    
    
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
                
            self.send("Examples", selectedExamples)
            self.send("Attribute List", orange.VarList(list(selectedAttrs)))
        
        else:
            self.send("Examples", None)
            self.send("Attribute List", None)
        
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


"""
<name>Feature Constructor</name>
<description>Construct a new continuous attribute computed from existing attributes.</description>
<icon>icons/FeatureConstructor.png</icon>
<priority>11</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI, math, re

re_identifier = re.compile(r'([a-zA-Z_]\w*)|("[^"]+")')

class IdentifierReplacer:
    def __init__(self, reinserted, attributes):
        self.reinserted = reinserted
        self.attributes = attributes

    def __call__(self, id):
        id = id.group()
        if id in self.reinserted:
            return "(%s)" % self.reinserted[id]
        if (id[0] == id[-1] == '"') and (id[1:-1] in self.attributes):
            return "_ex[%s]" % id
        if id in self.attributes:
            return "_ex['%s']" % id
        return id


class AttrComputer:
    def __init__(self, expression):
        self.expression = expression

    def __call__(self, ex, weight):
        try:
            return float(eval(self.expression, math.__dict__, {"_ex": ex}))
        except:
            return "?"


class OWFeatureConstructor(OWWidget):
    contextHandlers = {"": PerfectDomainContextHandler()}

    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "FeatureConstructor")

        self.inputs = [("Examples", orange.ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable)]

        self.expression = self.attrname = ""
        self.selectedDef = []
        self.defLabels = []
        self.data = None
        self.definitions = []

        self.selectedAttr = 0
        self.selectedFunc = 0
        self.autosend = True

        db = OWGUI.widgetBox(self.controlArea, "Attribute definitions", addSpace = True)

        OWGUI.separator(db, 4)

        hb = OWGUI.widgetBox(db, None, "horizontal")

        vb = OWGUI.widgetBox(hb, None, "vertical")
        self.leAttrName = OWGUI.lineEdit(vb, self, "attrname", "New attribute")
        OWGUI.rubber(vb)

        OWGUI.separator(hb, 8, 8)

        vb = OWGUI.widgetBox(hb, None, "vertical")
        self.leExpression = OWGUI.lineEdit(vb, self, "expression", "Expression")
        hhb = OWGUI.widgetBox(vb, None, "horizontal")
        self.cbAttrs = OWGUI.comboBox(hhb, self, "selectedAttr", items = ["(all attributes)"], callback = self.attrListSelected)
        self.cbFuncs = OWGUI.comboBox(hhb, self, "selectedFunc", items = ["(all functions)"] + [m for m in math.__dict__.keys() if m[:2]!="__"], callback = self.funcListSelected)

        OWGUI.separator(hb, 8, 8)
        OWGUI.button(hb, self, "Clear", callback = self.clearLineEdits)

        OWGUI.separator(db)

        hb = OWGUI.widgetBox(db, None, "horizontal")
        OWGUI.button(hb, self, "Add", callback = self.addAttr)
        OWGUI.button(hb, self, "Update", callback = self.updateAttr)
        OWGUI.button(hb, self, "Remove", callback = self.removeAttr)
        OWGUI.button(hb, self, "Remove All", callback = self.removeAllAttr)

        OWGUI.separator(db)
        self.lbDefinitions = OWGUI.listBox(db, self, "selectedDef", "defLabels", callback=self.selectAttr)
        self.lbDefinitions.setFixedHeight(160)

        hb = OWGUI.widgetBox(self.controlArea, "Apply", "horizontal")
        OWGUI.button(hb, self, "Apply", callback = self.apply)
        cb = OWGUI.checkBox(hb, self, "autosend", "Apply automatically", callback=self.enableAuto)
        cb.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.adjustSize()


    def settingsFromWidgetCallback(self, handler, context):
        context.definitions = self.definitions


    def settingsToWidgetCallback(self, handler, context):
        self.definitions = getattr(context, "definitions", [])
        self.defLabels = ["%s := %s" % t for t in self.definitions]
        self.selectedDef = []


    def setData(self, data):
        self.closeContext()
        self.data = data
        self.cbAttrs.clear()
        self.cbAttrs.addItem("(all attributes)")
        if self.data:
            self.cbAttrs.addItems([attr.name for attr in self.data.domain])

        self.removeAllAttr()
        self.openContext("", data)
        self.apply()


    def clearLineEdits(self):
        self.expression = self.attrname = ""


    def addAttr(self):
        attrname = self.attrname.strip()
        if not attrname:
            self.leAttrName.setFocus()
            return

        expression = self.expression.strip()
        if not expression:
            self.leExpression.setFocus()
            return
        self.defLabels = self.defLabels + ["%s := %s" % (attrname, expression)]
        self.definitions.append((attrname, expression))
        self.expression = self.attrname = ""
        self.applyIf()


    def removeAttr(self):
        if self.selectedDef:
            selected = self.selectedDef[0]
            if 0 <= selected < self.lbDefinitions.count():
                self.defLabels = self.defLabels[:selected] + self.defLabels[selected+1:]
                del self.definitions[selected]
                self.applyIf()


    def removeAllAttr(self):
        self.defLabels = []
        self.definitions = []
        self.selectedDef = []
        self.clearLineEdits()
        self.applyIf()


    def updateAttr(self):
        if self.selectedDef:
            selected = self.selectedDef[0]
            if 0 <= selected < self.lbDefinitions.count():
                self.defLabels = self.defLabels[:selected] + ["%s := %s" % (self.attrname, self.expression)] + self.defLabels[selected+1:]
                self.definitions[selected] = (self.attrname, self.expression)
                self.applyIf()


    def selectAttr(self):
        selected = self.selectedDef[0]
        if 0 <= selected < self.lbDefinitions.count():
            self.attrname, self.expression = self.definitions[selected]
        else:
            self.attrname = self.expression = ""


    def insertIntoExpression(self, what):
        if self.leExpression.hasMarkedText():
            self.leExpression.delChar()

        cp = self.leExpression.cursorPosition()
        self.expression = self.expression[:cp] + what + self.expression[cp:]
        self.leExpression.setFocus()


    def attrListSelected(self):
        if self.selectedAttr:
            attr = str(self.cbAttrs.text(self.selectedAttr))
            mo = re_identifier.match(attr)
            if not mo or mo.span()[1] != len(attr):
                attr = '"%s"' % attr

            self.insertIntoExpression(attr)
            self.selectedAttr = 0


    def funcListSelected(self):
        if self.selectedFunc:
            func = str(self.cbFuncs.text(self.selectedFunc))
            if func in ["atan2", "fmod", "ldexp", "log", "pow"]:
                self.insertIntoExpression(func + "(,)")
                self.leExpression.cursorLeft(False, 2)
            elif func == "pi":
                self.insertIntoExpression(func)
            else:
                self.insertIntoExpression(func + "()")
                self.leExpression.cursorLeft(False)

            self.selectedFunc = 0


    def applyIf(self):
        self.dataChanged = True
        if self.autosend:
            self.apply()


    def enableAuto(self):
        if self.dataChanged:
            self.apply()


    def apply(self):
        self.dataChanged = False
        if not self.data:
            self.send("Examples", None)
            return

        oldDomain = self.data.domain

        names = [d[0] for d in self.definitions]
        for name in names:
            if names.count(name)>1 or name in oldDomain > 1:
                self.error(1, "Multiple attributes with the same name (%s)" % name)
                self.send("Examples", None)
                return

        unknown = [[name, exp, set([id[0] or id[1] for id in re_identifier.findall(exp) if id[0] in names or id[1][1:-1] in names])] for name, exp in self.definitions]
        reinserted = {}
        replacer = IdentifierReplacer(reinserted, [n.name for n in oldDomain])
        while unknown:
            solved = set()
            for i, (name, exp, unk_attrs) in enumerate(unknown):
                if not unk_attrs:
                    reinserted[name] = re_identifier.sub(replacer, exp)
                    del unknown[i]
                    solved.add(name)
            if not solved:
                self.error(1, "Circular attribute definitions (%s)" % ", ".join([x[0] for x in unknown]))
                self.send("Examples", None)
                return
            for name, exp, unk_attrs in unknown:
                unk_attrs -= solved

        self.error(1)

        newDomain = orange.Domain(oldDomain.attributes + [orange.FloatVariable(str(attrname), getValueFrom = AttrComputer(reinserted[attrname])) for attrname in names], oldDomain.classVar)
        newDomain.addmetas(oldDomain.getmetas())
        self.send("Examples", orange.ExampleTable(newDomain, self.data))

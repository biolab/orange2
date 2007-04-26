"""
<name>Feature Constructor</name>
<description>Construct a new continuous attribute computed from existing attributes with an.</description>
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
        
        hb = OWGUI.widgetBox(self.controlArea, None, "horizontal")
        
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
        
        OWGUI.separator(self.controlArea, 12, 12)

        hb = OWGUI.widgetBox(self.controlArea, None, "horizontal")
        OWGUI.button(hb, self, "Add", callback = self.addAttr)
        OWGUI.button(hb, self, "Update", callback = self.updateAttr)
        OWGUI.button(hb, self, "Remove", callback = self.removeAttr)
        OWGUI.button(hb, self, "Remove All", callback = self.removeAllAttr)
        
        OWGUI.separator(self.controlArea)
        self.lbDefinitions = OWGUI.listBox(self.controlArea, self, "selectedDef", "defLabels", callback=self.selectAttr)
        self.lbDefinitions.setFixedHeight(160)

        OWGUI.separator(self.controlArea)
        OWGUI.button(self.controlArea, self, "Apply", callback = self.apply)

        self.definitions = [("x", '"petal length"+2'), ("z", "x+2"), ("u", "x+z"), ("zz", '"petal length"**3+"sepal length"')]
        self.loadDefinitions()
        self.adjustSize()


    def loadDefinitions(self):
        self.defLabels = ["%s := %s" % t for t in self.definitions]
        self.selectedDef = []

    def setData(self, data):
        if not self.data or self.data.domain != data.domain:
            self.clearLineEdits()
        self.data = data
        self.cbAttrs.clear()
        self.cbAttrs.insertItem("(all attributes)")
        if self.data:
            for attr in self.data.domain:
                self.cbAttrs.insertItem(attr.name)
        
        
    def clearLineEdits(self):
        self.expression = self.attrname = ""
        
    def addAttr(self):
        if not self.attrname:
            self.leAttrName.setFocus()
            return
        if not self.expression:
            self.leExpression.setFocus()
            return
        self.defLabels = self.defLabels + ["%s := %s" % (self.attrname, self.expression)]
        self.definitions.append((self.attrname, self.expression))
        self.expression = self.attrname = ""
        
    def removeAttr(self):
        if self.selectedDef:
            selected = self.selectedDef[0]
            if 0 <= selected < self.lbDefinitions.count():
                self.defLabels = self.defLabels[:selected] + self.defLabels[selected+1:]
                del self.definitions[selected]

    def removeAllAttr(self):
        self.defLabels = []
        self.definitions = []
        self.clearLineEdits()
        
    def updateAttr(self):
        selected = self.selectedDef[0]
        if 0 <= selected < self.lbDefinitions.count():
            self.defLabels = self.defLabels[:selected] + ["%s := %s" % (self.attrname, self.expression)] + self.defLabels[selected+1:]
            self.definitions[selected] = (self.attrname, self.expression)
    
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
        
    
    def apply(self):
        if not self.data:
            self.send("Examples", None)
            return

        oldDomain = self.data.domain

        names = [d[0] for d in self.definitions]
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

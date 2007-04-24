"""
<name>Feature Constructor</name>
<description>Construct a new continuous attribute computed from existing attributes with an.</description>
<icon>icons/FeatureConstructor.png</icon>
<priority>11</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI, math, re

re_identifier = re.compile(r'((?<=\W)[a-zA-Z_]\w*(?=(\Z|\W)))|("[^"]+")')

def identifier_replacer(id):
    id = id.group()
    if id in math.__dict__:
        return id
    if id[0] == id[-1] == '"':
        return "_ex[%s]" % id
    else:
        return "_ex['%s']" % id

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
        self.data = None
        OWGUI.lineEdit(self.controlArea, self, "attrname", "Attribute name")
        OWGUI.separator(self.controlArea)
        OWGUI.lineEdit(self.controlArea, self, "expression", "Expression")
        OWGUI.separator(self.controlArea)
        OWGUI.button(self.controlArea, self, "Apply", callback = self.apply)
        self.adjustSize()

    def setData(self, data):
        if not self.data or self.data.domain != data.domain:
            self.expression = self.attrname = ""
        self.data = data

    def apply(self):
        if not self.data:
            self.send("Examples", None)
            return

        oldDomain = self.data.domain
        
        if self.attrname:
            attrname = self.attrname
        else:
            t = 1
            while "T%04i" % t in oldDomain:
                t += 1
            attrname = "T%04i" % t

        exp = re_identifier.sub(identifier_replacer, " "+self.expression)
        newattr = orange.FloatVariable(str(attrname), getValueFrom = AttrComputer(exp))

        newDomain = orange.Domain(oldDomain.attributes + [newattr], oldDomain.classVar)
        newDomain.addmetas(oldDomain.getmetas())
        
        self.send("Examples", orange.ExampleTable(newDomain, self.data))


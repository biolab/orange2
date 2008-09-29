"""
<name>Purge Domain</name>
<description>Removes redundant values and attributes, sorts values.</description>
<icon>icons/PurgeDomain.png</icon>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
<priority>1105</priority>
"""
from OWWidget import *
import OWGUI

class OWPurgeDomain(OWWidget):

    def __init__(self, parent=None, signalManager=None):
        OWWidget.__init__(self, parent, signalManager, 'PurgeDomain')
        self.settingsList=["removeValues", "removeAttributes", "removeClassAttribute", "removeClasses", "autoSend", "sortValues", "sortClasses"]

        self.inputs = [("Examples", ExampleTable, self.setData)]
        self.outputs = [("Examples", ExampleTable)]

        self.data = None

        self.preRemoveValues = self.removeValues = 1
        self.removeAttributes = 1
        self.removeClassAttribute = 1
        self.preRemoveClasses = self.removeClasses = 1
        self.autoSend = 1
        self.dataChanged = False

        self.sortValues = self.sortClasses = True

        self.loadSettings()

        self.removedAttrs = self.reducedAttrs = self.resortedAttrs = self.classAttr = "-"

        boxAt = OWGUI.widgetBox(self.controlArea, "Attributes")
        OWGUI.checkBox(boxAt, self, 'sortValues', 'Sort attribute values', callback = self.optionsChanged)
        rua = OWGUI.checkBox(boxAt, self, "removeAttributes", "Remove attributes with less than two values", callback = self.removeAttributesChanged)
        boxH = OWGUI.widgetBox(boxAt, orientation="horizontal")
        OWGUI.separator(boxH, width=30, height=0)
        ruv = OWGUI.checkBox(boxH, self, "removeValues", "Remove unused attribute values", callback = self.optionsChanged)
        rua.disables = [ruv]

        OWGUI.separator(self.controlArea)

        boxAt = OWGUI.widgetBox(self.controlArea, "Classes")
        OWGUI.checkBox(boxAt, self, 'sortClasses', 'Sort classes', callback = self.optionsChanged)
        rua = OWGUI.checkBox(boxAt, self, "removeClassAttribute", "Remove class attribute if there are less than two classes", callback = self.removeClassesChanged)
        boxH = OWGUI.widgetBox(boxAt, orientation="horizontal")
        OWGUI.separator(boxH, width=30, height=0)
        ruv = OWGUI.checkBox(boxH, self, "removeClasses", "Remove unused class values", callback = self.optionsChanged)
        rua.disables = [ruv]

        OWGUI.separator(self.controlArea)
        box2 = OWGUI.widgetBox(self.controlArea)
        btSend = OWGUI.button(box2, self, "Send data", callback = self.process)
        cbAutoSend = OWGUI.checkBox(box2, self, "autoSend", "Send automatically")

        OWGUI.setStopper(self, btSend, cbAutoSend, "dataChanged", self.process)

        OWGUI.separator(self.controlArea, height=24)

        box3 = OWGUI.widgetBox(self.controlArea, 'Statistics')
        OWGUI.label(box3, self, "Removed attributes: %(removedAttrs)s")
        OWGUI.label(box3, self, "Reduced attributes: %(reducedAttrs)s")
        OWGUI.label(box3, self, "Resorted attributes: %(resortedAttrs)s")
        OWGUI.label(box3, self, "Class attribute: %(classAttr)s")

        #self.adjustSize()

    def setData(self, dataset):
        if dataset:
            self.data = dataset
            self.process()
        else:
            self.reducedAttrs = self.removedAttrs = self.resortedAttrs = self.classAttr = ""
            self.send("Examples", None)
            self.data = None
        self.dataChanged = False

    def removeAttributesChanged(self):
        if not self.removeAttributes:
            self.preRemoveValues = self.removeValues
            self.removeValues = False
        else:
            self.removeValues = self.preRemoveValues
        self.optionsChanged()

    def removeClassesChanged(self):
        if not self.removeClassAttribute:
            self.preRemoveClasses = self.removeClasses
            self.removeClasses = False
        else:
            self.removeClasses = self.preRemoveClasses
        self.optionsChanged()

    def optionsChanged(self):
        if self.autoSend:
            self.process()
        else:
            self.dataChanged = True

    def sortAttrValues(self, attr, interattr=None):
        if not interattr:
            interattr = attr

        newvalues = list(interattr.values)
        newvalues.sort()
        if newvalues == list(interattr.values):
            return interattr

        newattr = orange.EnumVariable(interattr.name, values=newvalues)
        newattr.getValueFrom = orange.ClassifierByLookupTable(newattr, attr)
        lookupTable = newattr.getValueFrom.lookupTable
        distributions = newattr.getValueFrom.distributions
        for val in interattr.values:
            idx = attr.values.index(val)
            lookupTable[idx] = val
            distributions[idx][newvalues.index(val)] += 1
        return newattr

    def process(self):
        if self.data == None:
            return

        self.reducedAttrs = 0
        self.removedAttrs = 0
        self.resortedAttrs = 0
        self.classAttribute = 0

        if self.removeAttributes or self.sortValues:
            newattrs = []
            for attr in self.data.domain.attributes:
                if attr.varType == orange.VarTypes.Continuous:
                    if orange.RemoveRedundantOneValue.hasAtLeastTwoValues(self.data, attr):
                        newattrs.append(attr)
                    else:
                        self.removedAttrs += 1
                    continue

                if attr.varType != orange.VarTypes.Discrete:
                    newattrs.append(attr)
                    continue

                if self.removeValues:
                    newattr = orange.RemoveUnusedValues(attr, self.data)
                    if not newattr:
                        self.removedAttrs += 1
                        continue

                    if newattr != attr:
                        self.reducedAttrs += 1
                else:
                    newattr = attr

                if self.removeValues and len(newattr.values) < 2:
                    self.removedAttrs += 1
                    continue

                if self.sortValues:
                    newnewattr = self.sortAttrValues(attr, newattr)
                    if newnewattr != newattr:
                        self.resortedAttrs += 1
                        newattr = newnewattr

                newattrs.append(newattr)
        else:
            newattrs = self.data.domain.attributes


        klass = self.data.domain.classVar
        classChanged = False
        if not klass:
            newclass = klass
            self.classAttr = "No class"
        elif klass.varType != orange.VarTypes.Discrete:
            newclass = klass
            self.classAttr = "Class is not discrete"
        elif not (self.removeClassAttribute or self.sortClasses):
            newclass = klass
            self.classAttr = "Class is not checked"
        else:
            self.classAttr = ""

            if self.removeClasses:
                newclass = orange.RemoveUnusedValues(klass, self.data)
            else:
                newclass = klass

            if not newclass or self.removeClassAttribute and len(newclass.values) < 2:
                newclass = None
                self.classAttr = "Class is removed"
            elif len(newclass.values) != len(klass.values):
                    self.classAttr = "Class is reduced"

            if newclass and self.sortClasses:
                newnewclass = self.sortAttrValues(klass, newclass)
                if newnewclass != newclass:
                    if self.classAttr:
                        self.classAttr = "Class is reduced and sorted"
                    else:
                        self.classAttr = "Class is sorted"
                    newclass = newnewclass

            if not self.classAttr:
                self.classAttr = "Class is unchanged"

        if self.reducedAttrs or self.removedAttrs or self.resortedAttrs or newclass != klass:
            newDomain = orange.Domain(newattrs, newclass)
            newData = orange.ExampleTable(newDomain, self.data)
        else:
            newData = self.data

        self.send("Examples", newData)

        self.dataChanged = False


if __name__=="__main__":
    appl = QApplication(sys.argv)
    ow = OWPurgeDomain()
    #data = orange.ExampleTable('..\\..\\doc\\datasets\\car.tab')
    #data.domain.attributes[3].values.append("X")
    #ow.setData(data)
    ow.show()
    appl.exec_()
    ow.saveSettings()

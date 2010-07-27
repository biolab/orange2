"""
<name>Concatenate</name>
<description>Concatenates Example Tables.</description>
<icon>icons/Concatenate.png</icon>
<priority>1111</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI

class OWConcatenate(OWWidget):
    settingsList = ["mergeAttributes"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Concatenate", wantMainArea=0)
        self.inputs = [("Primary Table", orange.ExampleTable, self.setData), ("Additional Tables", orange.ExampleTable, self.setMoreData, Multiple)]
        self.outputs = [("Examples", ExampleTable)]

        self.mergeAttributes = 0

        self.primary = None
        self.additional = {}
        
        bg = self.bgMerge = OWGUI.radioButtonsInBox(self.controlArea, self, "mergeAttributes", [], "Domains merging", callback = self.apply)
        OWGUI.widgetLabel(bg, "When there is no primary table, the domain should be:")
        OWGUI.appendRadioButton(bg, self, "mergeAttributes", "Union of attributes appearing in all tables")
        OWGUI.appendRadioButton(bg, self, "mergeAttributes", "Intersection of attributes in all tables")
        OWGUI.widgetLabel(bg, "The resulting table will have class only if there is no conflict between input classes.")
        
        OWGUI.rubber(self.controlArea)

        self.adjustSize()


    def setData(self, data):
        self.primary = data
        self.bgMerge.setEnabled(not data)
        self.apply()
        

    def setMoreData(self, data, id):
        if not data:
            if id in self.additional:
                del self.additional[id]
        else:
            self.additional[id] = data
        self.apply()
        
    
    def apply(self):
        if self.primary:
            if not self.additional:
                newTable = self.primary

            else:
                newTable = orange.ExampleTable(self.primary)
                for additional in self.additional.values():
                    newTable.extend(additional)

        else:
            if not self.additional:
                newTable = None
                
            else:
                classVar = False
                for additional in self.additional.values():
                    if additional.domain.classVar:
                        if classVar == False: # can also be None
                            classVar = additional.domain.classVar
                        elif classVar != additional.domain.classVar:
                            classVar = None
                            
                if self.mergeAttributes: # intersection
                    attributes = metas = None
                    for additional in self.additional.values():
                        if attributes == None:
                            if classVar:
                                attributes = additional.domain.attributes
                            else:
                                attributes = additional.domain
                            metas = dict((attr, id) for id, attr in additional.domain.getmetas().items())
                        else:
                            attributes = [attr for attr in attributes if attr in additional.domain and not attr == classVar]
                            metas = dict((attr, id) for id, attr in additional.domain.getmetas().items() if attr in metas)
                    if attributes == None:
                        attributes = []
                        metas = {}
                else: # union
                    attributes = []
                    metas = {}
                    for additional in self.additional.values():
                        for attr in additional.domain:
                            if attr not in attributes and attr != classVar:
                                attributes.append(attr)
                        for id, attr in additional.domain.getmetas().items():
                            if not attr in metas:
                                metas[attr] = id
                if not attributes and not classVar:
                    self.error(1, "The output domain is empty.")
                    newTable = None
                else:
                    self.error(1)
                    newDomain = orange.Domain(attributes, classVar)
                    newDomain.addmetas(dict((x[1], x[0]) for x in metas.items())) 
                    newTable = orange.ExampleTable(newDomain)
                    for additional in self.additional.values():
                        newTable.extend(additional)

        self.dataReport = self.prepareDataReport(newTable)
        self.send("Examples", newTable)

    def sendReport(self):
        self.reportData(self.primary, "Primary table", 
                        "None; outputting the %s of attributes from all tables" % ["union", "intersection"][self.mergeAttributes]) 
        for additional in self.additional.values():
            self.reportData(additional, "Additional table")
        if not self.additional:
            self.reportData(None, "Additional table")
        self.reportData(self.dataReport, "Merged data")


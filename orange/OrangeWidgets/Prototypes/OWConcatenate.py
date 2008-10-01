"""
<name>Concatenate</name>
<description>Concatenates Example Tables.</description>
<icon>icons/Concatenate.png</icon>
<priority>12</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI
from itertools import izip

class OWConcatenate(OWWidget):
    settingsList = ["mergeAttributes", "dataSourceSelected", "addIdAs", "dataSourceName"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "FeatureConstructor")
        self.inputs = [("Primary Table", orange.ExampleTable, self.setData), ("Additional Tables", orange.ExampleTable, self.setMoreData, Multiple)]
        self.outputs = [("Examples", ExampleTable)]

        self.mergeAttributes = 0
        self.dataSourceSelected = 1
        self.addIdAs = 0
        self.dataSourceName = "clusterId"
        
        self.primary = None
        self.additional = {}
        
        self.loadSettings()
        
        bg = self.bgMerge = OWGUI.radioButtonsInBox(self.controlArea, self, "mergeAttributes", [], "Domains merging", callback = self.apply)
        OWGUI.widgetLabel(bg, "When there is no primary table, the domain should be")
        OWGUI.appendRadioButton(bg, self, "mergeAttributes", "Union of attributes appearing in all tables")
        OWGUI.appendRadioButton(bg, self, "mergeAttributes", "Intersection of attributes in all tables")
        OWGUI.widgetLabel(bg, "The resulting table will have class only if there is no conflict betwen input classes.")
        
        box = OWGUI.widgetBox(self.controlArea, "Data source IDs")
        cb = OWGUI.checkBox(box, self, "dataSourceSelected", "Append data source IDs")
        self.classificationBox = ib = OWGUI.widgetBox(box)
        le = OWGUI.lineEdit(ib, self, "dataSourceName", "Name" + "  ", orientation='horizontal', valueType = str)
        OWGUI.separator(ib, height = 4)
        aa = OWGUI.comboBox(ib, self, "addIdAs", label = "Place" + "  ", orientation = 'horizontal', items = ["Class attribute", "Attribute", "Meta attribute"])
        cb.disables.append(ib)
        cb.makeConsistent()
        OWGUI.separator(box)
        OWGUI.button(box, self, "Apply Changes", callback = self.apply)

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
        dataSourceIDs = []
        currentID = 1
        
        if self.primary:
            if not self.additional:
                newTable = self.primary
                dataSourceIDs.extend([currentID] * len(self.primary))
                currentID += 1
            else:
                newTable = orange.ExampleTable(self.primary)
                dataSourceIDs.extend([currentID] * len(self.primary))
                currentID += 1
                
                for additional in self.additional.values():
                    newTable.extend(additional)
                    dataSourceIDs.extend([currentID] * len(additional))
                    currentID += 1

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
                    attributes = None
                    for additional in self.additional.values():
                        if attributes == None:
                            if classVar:
                                attributes = additional.domain.attributes
                            else:
                                attributes = additional.domain
                        else:
                            attributes = [attr for attr in attributes if attr in additional.domain and not attr == classVar]
                    if attributes == None:
                        attributes = []
                else: # union
                    attributes = []
                    for additional in self.additional.values():
                        for attr in additional.domain:
                            if attr not in attributes and attr != classVar:
                                attributes.append(attr)
                    
                if not attributes and not classVar:
                    self.error(1, "The output domain is empty.")
                    newTable = None
                else:
                    self.error(1)
                    newTable = orange.ExampleTable(orange.Domain(attributes, classVar))
                    for additional in self.additional.values():
                        newTable.extend(additional)
                        dataSourceIDs.extend([currentID] * len(additional))
                        currentID += 1
        
        if newTable != None:
            tableCount = 0
            if self.primary:
                tableCount += 1
            if self.additional:
                tableCount += len(self.additional)
                
            dataSourceVar = orange.EnumVariable(self.dataSourceName, values = [str(x) for x in range(1, 1 + tableCount)])
            
            origDomain = newTable.domain
            if self.addIdAs == 0:
                domain = orange.Domain(origDomain.attributes, dataSourceVar)
                if origDomain.classVar:
                    domain.addmeta(orange.newmetaid(), origDomain.classVar)
                aid = -1
            elif self.addIdAs == 1:
                domain=orange.Domain(origDomain.attributes+[dataSourceVar], origDomain.classVar)
                aid = len(origDomain.attributes)
            else:
                domain=orange.Domain(origDomain.attributes, origDomain.classVar)
                aid=orange.newmetaid()
                domain.addmeta(aid, dataSourceVar)
                
            domain.addmetas(origDomain.getmetas())
            
            table1 = orange.ExampleTable(domain)
            table1.extend(newTable)
            
            for ex, id in izip(table1, dataSourceIDs):
                ex[aid] = dataSourceVar(str(id))
            
            newTable = table1
        #for ex, midx in izip(table1, self.mc.mapping):
        #    ex[aid] = clustVar(str(midx))

        self.send("Examples", newTable)

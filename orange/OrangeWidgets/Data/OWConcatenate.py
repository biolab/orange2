"""
<name>Concatenate</name>
<description>Concatenates Example Tables.</description>
<icon>icons/Concatenate.png</icon>
<priority>1111</priority>
<contact>Janez Demsar (janez.demsar(@at@)fri.uni-lj.si)</contact>
"""

from OWWidget import *
import OWGUI
from itertools import izip

class OWConcatenate(OWWidget):
    settingsList = ["mergeAttributes", "dataSourceSelected", "addIdAs", "dataSourceName"]
    
    def __init__(self,parent=None, signalManager = None):
        OWWidget.__init__(self, parent, signalManager, "Concatenate", wantMainArea=0)
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
        OWGUI.widgetLabel(bg, "The resulting table will have class only if there is no conflict between input classes.")

        OWGUI.separator(self.controlArea)
        box = OWGUI.widgetBox(self.controlArea, "Data source IDs", addSpace=True)
        cb = OWGUI.checkBox(box, self, "dataSourceSelected", "Append data source IDs")
        self.classificationBox = ib = OWGUI.indentedBox(box, sep=OWGUI.checkButtonOffsetHint(cb))
        le = OWGUI.lineEdit(ib, self, "dataSourceName", "Name" + "  ", orientation='horizontal', valueType = str)
        OWGUI.separator(ib, height = 4)
        aa = OWGUI.comboBox(ib, self, "addIdAs", label = "Place" + "  ", orientation = 'horizontal', items = ["Class attribute", "Attribute", "Meta attribute"])
        cb.disables.append(ib)
        cb.makeConsistent()
        
        OWGUI.button(self.controlArea, self, "Apply Changes", callback = self.apply, default=True)
        
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
                        dataSourceIDs.extend([currentID] * len(additional))
                        currentID += 1
        
        if newTable != None:
            tableCount = 0
            if self.primary:
                tableCount += 1
            if self.additional:
                tableCount += len(self.additional)
            
            origDomain = newTable.domain
            if self.dataSourceSelected:
                dataSourceVar = orange.EnumVariable(self.dataSourceName, values = [str(x) for x in range(1, 1 + tableCount)])
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
            else:
                domain = orange.Domain(origDomain.attributes, origDomain.classVar)

            domain.addmetas(origDomain.getmetas())
            
            table1 = orange.ExampleTable(domain)
            table1.extend(newTable)
            
            if self.dataSourceSelected:
                for ex, id in izip(table1, dataSourceIDs):
                    ex[aid] = dataSourceVar(str(id))

            newTable = table1

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
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = OWConcatenate()
    data = orange.ExampleTable("../../doc/datasets/iris.tab")
    w.setData(data)
    w.setMoreData(data, 0)
    w.show()
    app.exec_()


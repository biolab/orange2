#
# OData.py
# the orange data with helper functions, gui independent 
#

import orange

CONTINUOUS = 2
DISCRETE = 1

class OrangeData:

    def __init__(self,tab):
        self.tab=tab
        self.setVariables()
        
    #helper functions, copied and search/replaced from deData
    def setVariables(self):
        self.table = orange.ExampleTable(self.tab)
        self.originalNumInstances = len(self.table)
        self.data = orange.Filter_hasClassValue(self.table)
        self.dc = orange.DomainContingency(self.table)
        self.outcome = self.data.domain.classVar
        self.targetValIndx = 0
        self.targetVal = self.getOutcomeValues()[0]

    def getOutcomeValues(self): return self.data.domain.classVar.values.native()

    def getVarValues(self,var):
        var = self.data.domain[var]
        if var.varType == CONTINUOUS: return None
        else: return var.values.native()
        
    def getOutcomeName(self):
        return self.outcome.name
        
    def setOutcomeByName(self, name):
        attributes = self.tab.domain.attributes.native()
        if self.tab.domain[name] in attributes:
            attributes.remove(self.tab.domain[name])
            if self.tab.domain.classVar:
                attributes.append(self.tab.domain.classVar)
            attributes.append(self.tab.domain[name])
            self.tab = self.tab.select(orange.Domain(attributes))
        self.setVariables()

    def getVarNames(self, varType=None, classname=0):
        var_dict = []
        for i in range(len(self.data.domain)-1):
            var = self.data.domain[i]
            if (varType == None) or (var.varType == varType):
                var_dict.append(var.name)
        
        if classname: var_dict.append(self.getOutcomeName())
        return var_dict      
        
    def getCategoricalNames(self):
        return self.getVarNames(DISCRETE)
        
    def getPotentialOutcomes(self):
        list = self.getVarNames(DISCRETE)
        list.append(self.getOutcomeName())
        return list
        
    def getContinuousNames(self):
        return self.getVarNames(CONTINUOUS)
        
    def getTargetValIndx(self):
        return self.targetValIndx

    def setTargetValByName(self, name):
        outcomes = self.getOutcomeValues()
        self.targetValIndx = outcomes.index(name)
        self.targetVal = name
         
    def getTargetValName(self):
        return self.getOutcomeValues()[self.targetValIndx]
    def getTargetValIndx(self): return self.targetValIndx
        
    def getDC(self): return self.dc
    def getInstances(self): return self.data
    def getNumInstances(self): return len(self.data)
    def getOriginalNumInstances(self): return self.originalNumInstances
    def getNumOutcomes(self): return len(self.outcome.values)

if __name__== "__main__":
    print "This module is not supposed to be run by itself."

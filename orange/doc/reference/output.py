# Description: Shows how to define new output format
# Category:    kernel
# Classes:     Contingency, ContingencyAttrClass
# Uses:        monk1
# Referenced:  contingency.htm

import orange

def printTabDelimContingency(c):
    if c.innerVariable.varType != orange.VarTypes.Discrete or \
       c.outerVariable.varType != orange.VarTypes.Discrete:
        raise "printTabDelimContingency can only handle discrete contingencies"
    
    res = ""
    for v in c.innerVariable.values:
        res += "\t%s" % v
    res += "\n"
    for i in range(len(c.outerVariable.values)):
        res += c.outerVariable.values[i]
        for v in c[i]:
            res += "\t%5.3f" % v
        res += "\n"
    return res

orange.setoutput(orange.Contingency, "tab", printTabDelimContingency)

data = orange.ExampleTable("monk1")
cont = orange.ContingencyAttrClass("e", data)

print "\n*** Dump in format 'tab' ***\n"
print cont.dump("tab")

orange.setoutput(orange.Contingency, "repr", printTabDelimContingency)
print "\n*** Print after 'repr' is set ***\n"
print cont

print "\n*** Reverse-quoting after 'repr' is set ***\n"
print `cont`

print "\n*** Print after 'str' is set ***\n"
orange.setoutput(orange.Contingency, "str", printTabDelimContingency)
print cont

orange.removeoutput(orange.Contingency, "repr")
orange.removeoutput(orange.Contingency, "str")
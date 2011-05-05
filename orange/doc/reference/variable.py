# Description: Shows how to construct new attributes from the existing
# Category:    feature construction, constructive induction
# Classes:     Variable
# Uses:        monk1
# Referenced:  Variable.htm

import orange

data = orange.ExampleTable("monk1")

e2 = orange.EnumVariable("e2", values = ["not 1", "1"])

def checkE(example, returnWhat):
    if example["e"]=="1":
        return orange.Value(e2, "1")
    else:
        return orange.Value(e2, "not 1")

e2.getValueFrom = checkE
newDomain = orange.Domain([data.domain["a"], data.domain["b"], e2, data.domain.classVar])
newData = orange.ExampleTable(newDomain, data)

dist = orange.Distribution(e2, data)
print dist

cont = orange.ContingencyAttrClass(e2, data)
print "Class distribution when e=1:", cont["1"]
print "Class distribution when e<>1:", cont["not 1"]


# Description: Shows how to use ClassifierFromVar and transformers
# Category:    sampling
# Classes:     ClassifierFromVar, TransformValue
# Uses:        monk1
# Referenced:  RandomIndices.htm

import orange

data = orange.ExampleTable("monk1")
e = data.domain["e"]

def eTransformer(value):
    if int(value) == 0:
        return 0
    else:
        return 1


e1 = orange.EnumVariable("e1", values = ["1", "not 1"])
e1.getValueFrom = orange.ClassifierFromVar()
e1.getValueFrom.whichVar = e
e1.getValueFrom.transformer = eTransformer

data2 = data.select(["a", "b", "e", e1, "y"])
for i in data2[:5]:
    print i
print


e1.getValueFrom = orange.ClassifierFromVarFD()
e1.getValueFrom.domain = data.domain
e1.getValueFrom.position = data.domain.attributes.index(e)
e1.getValueFrom.transformer = eTransformer

data2 = data.select(["a", "b", "e", e1, "y"])
for i in data2[:5]:
    print i
print


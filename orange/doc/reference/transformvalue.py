# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import orange

data = orange.ExampleTable("monk1")

e = data.domain["e"]

e1 = orange.FloatVariable("e=1")
e1.getValueFrom = orange.ClassifierFromVar(whichVar = e)
e1.getValueFrom.transformer = orange.Discrete2Continuous()
e1.getValueFrom.transformer.value = int(orange.Value(e, "1"))

e10 = orange.FloatVariable("e=1")
e10.getValueFrom = orange.ClassifierFromVar(whichVar = e)
e10.getValueFrom.transformer = orange.Discrete2Continuous()
e10.getValueFrom.transformer.value = int(orange.Value(e, "1"))
e10.getValueFrom.transformer.zeroBased = False

e01 = orange.FloatVariable("e=1")
e01.getValueFrom = orange.ClassifierFromVar(whichVar = e)
transformer = e01.getValueFrom.transformer = orange.Discrete2Continuous()
transformer.value = int(orange.Value(e, "1"))
transformer.zeroBased = False
transformer.invert = True


newDomain = orange.Domain([e, e1, e10, e01], data.domain.classVar)
newData = orange.ExampleTable(newDomain, data)
for ex in newData[:10]:
    print ex
print "\n\n"

attributes = [e]
for v in e.values:
    newattr = orange.FloatVariable("e=%s" % v)
    newattr.getValueFrom = orange.ClassifierFromVar(whichVar = e)
    newattr.getValueFrom.transformer = orange.Discrete2Continuous()
    newattr.getValueFrom.transformer.value = int(orange.Value(e, v))
    attributes.append(newattr)

newDomain = orange.Domain(attributes, data.domain.classVar)
newData = orange.ExampleTable(newDomain, data)
for ex in newData[:10]:
    print ex

# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import orange

data = orange.ExampleTable("iris")

domstat = orange.DomainBasicAttrStat(data)
newattrs = []
for attr in data.domain.attributes:
    attr_c = orange.FloatVariable(attr.name+"_n")
    attr_c.getValueFrom = orange.ClassifierFromVar(whichVar = attr)
    transformer = orange.NormalizeContinuous()
    attr_c.getValueFrom.transformer = transformer
    transformer.average = domstat[attr].avg
    transformer.span = domstat[attr].dev/2
    newattrs.append(attr_c)

newDomain = orange.Domain(newattrs, data.domain.classVar)
newData = orange.ExampleTable(newDomain, data)
for ex in newData[:5]:
    print ex
print "\n\n"

# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import orange

data = orange.ExampleTable("lenses")

age = data.domain["age"]

age_c = orange.FloatVariable("age_c")
age_c.getValueFrom = orange.ClassifierFromVar(whichVar = age)
age_c.getValueFrom.transformer = orange.Ordinal2Continuous()

age_cn = orange.FloatVariable("age_cn")
age_cn.getValueFrom = orange.ClassifierFromVar(whichVar = age)
age_cn.getValueFrom.transformer = orange.Ordinal2Continuous()
age_cn.getValueFrom.transformer.factor = 0.5

newDomain = orange.Domain([age, age_c, age_cn], data.domain.classVar)
newData = orange.ExampleTable(newDomain, data)
for ex in newData:
    print ex
print "\n\n"

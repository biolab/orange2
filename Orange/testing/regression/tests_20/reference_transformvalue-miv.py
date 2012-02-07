# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import orange

data = orange.ExampleTable("lenses")

age = data.domain["age"]

age_b = orange.EnumVariable("age_c", values = ['young', 'old'])
age_b.getValueFrom = orange.ClassifierFromVar(whichVar = age)
age_b.getValueFrom.transformer = orange.MapIntValue()
age_b.getValueFrom.transformer.mapping = [0, 1, 1]

newDomain = orange.Domain([age_b, age], data.domain.classVar)
newData = orange.ExampleTable(newDomain, data)
for ex in newData:
    print ex
print "\n\n"

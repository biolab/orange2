# Description: Shows how to use value transformers
# Category:    preprocessing
# Classes:     TransformValue, Continuous2Discrete, Discrete2Continuous, MapIntValue
# Uses:        
# Referenced:

import Orange.data
import Orange.feature

lenses = Orange.data.Table("lenses")
age = lenses.domain["age"]

age_c = Orange.feature.Continuous("age_c")
age_c.getValueFrom = Orange.classification.ClassifierFromVar(whichVar = age)
age_c.getValueFrom.transformer = Orange.data.utils.Ordinal2Continuous()

age_cn = Orange.feature.Continuous("age_cn")
age_cn.getValueFrom = Orange.classification.ClassifierFromVar(whichVar = age)
age_cn.getValueFrom.transformer = Orange.data.utils.Ordinal2Continuous()
age_cn.getValueFrom.transformer.factor = 0.5

newDomain = Orange.data.Domain([age, age_c, age_cn], lenses.domain.classVar)
newData = Orange.data.Table(newDomain, lenses)
for ex in newData:
    print ex
print "\n\n"

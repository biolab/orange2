import Orange

data = Orange.data.Table("lenses")
age = data.domain["age"]

age_b = Orange.feature.Discrete("age_c", values = ['young', 'old'])
age_b.getValueFrom = Orange.core.ClassifierFromVar(whichVar = age)
age_b.getValueFrom.transformer = Orange.data.utils.MapIntValue()
age_b.getValueFrom.transformer.mapping = [0, 1, 1]

newDomain = Orange.data.Domain([age_b, age], data.domain.classVar)
newData = Orange.data.Table(newDomain, data)
for ex in newData:
    print ex
print "\n\n"

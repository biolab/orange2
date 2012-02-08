import Orange

lenses = Orange.data.Table("lenses")
age = lenses.domain["age"]

age_b = Orange.feature.Discrete("age_c", values = ['young', 'old'])
age_b.getValueFrom = Orange.classification.ClassifierFromVar(whichVar = age)
age_b.getValueFrom.transformer = Orange.data.utils.MapIntValue()
age_b.getValueFrom.transformer.mapping = [0, 1, 1]

newDomain = Orange.data.Domain([age_b, age], lenses.domain.classVar)
newData = Orange.data.Table(newDomain, lenses)
for ex in newData:
    print ex
print "\n\n"

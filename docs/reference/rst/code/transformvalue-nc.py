import Orange

data = Orange.data.Table("iris")

domstat = Orange.statistics.basic.Domain(data)
newattrs = []
for attr in data.domain.features:
    attr_c = Orange.feature.Continous(attr.name+"_n")
    attr_c.getValueFrom = Orange.core.ClassifierFromVar(whichVar = attr)
    transformer = Orange.data.utils.NormalizeContinuous()
    attr_c.getValueFrom.transformer = transformer
    transformer.average = domstat[attr].avg
    transformer.span = domstat[attr].dev
    newattrs.append(attr_c)

newDomain = Orange.data.Domain(newattrs, data.domain.classVar)
newData = Orange.data.Table(newDomain, data)
for ex in newData[:5]:
    print ex
print "\n\n"

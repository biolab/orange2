import Orange

iris = Orange.data.Table("iris")

domstat = Orange.statistics.basic.Domain(iris)
newattrs = []
for attr in iris.domain.features:
    attr_c = Orange.feature.Continuous(attr.name + "_n")
    attr_c.getValueFrom = Orange.classification.ClassifierFromVar(whichVar=attr)
    transformer = Orange.data.utils.NormalizeContinuous()
    attr_c.getValueFrom.transformer = transformer
    transformer.average = domstat[attr].avg
    transformer.span = domstat[attr].dev
    newattrs.append(attr_c)

newDomain = Orange.data.Domain(newattrs, iris.domain.classVar)
newData = Orange.data.Table(newDomain, iris)
for ex in newData[:5]:
    print ex
print "\n\n"

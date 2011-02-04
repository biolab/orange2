import Orange
import distributions

myData = Orange.data.Table("iris.tab")
bas = distributions.DomainBasicAttrStat(myData) 

print "%20s %5s %5s %5s" % ("feature", "min", "max", "avg")
for a in bas:
    if a:
        print "%20s %5.3f %5.3f %5.3f" % (a.variable.name, a.min, a.max, a.avg)

print bas["sepal length"].avg
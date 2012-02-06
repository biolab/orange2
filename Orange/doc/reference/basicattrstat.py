# Description: Shows how to compute and print out the basic attribute statistics
# Category:    statistics
# Classes:     DomainBasicAttrStat, BasicAttrStat
# Uses:        iris
# Referenced:  basicstat.htm

import orange
data = orange.ExampleTable("iris")
bas = orange.DomainBasicAttrStat(data)

print "%20s  %5s  %5s  %5s" % ("attribute", "min", "max", "avg")
for a in bas:
    if a:
        print "%20s  %5.3f  %5.3f  %5.3f" % (a.variable.name, a.min, a.max, a.avg)

print bas["sepal length"].avg        
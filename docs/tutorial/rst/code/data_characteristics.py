# Description: Read data, output class values and attribute names, and show class distribution
# Category:    description
# Uses:        adult_sample.tab
# Referenced:  basic_exploration.htm

import orange
data = orange.ExampleTable("adult_sample.tab")
print "Classes:", len(data.domain.classVar.values)
print "Attributes:", len(data.domain.attributes), ",",

# count number of continuous and discrete attributes
ncont = 0; ndisc = 0
for a in data.domain.attributes:
    if a.varType == orange.VarTypes.Discrete:
        ndisc = ndisc + 1
    else:
        ncont = ncont + 1
print ncont, "continuous,", ndisc, "discrete"

# obtain class distribution
c = [0] * len(data.domain.classVar.values)
for e in data:
    c[int(e.getclass())] += 1
print "Instances: ", len(data), "total",
for i in range(len(data.domain.classVar.values)):
    print ",", c[i], "with class", data.domain.classVar.values[i],
print

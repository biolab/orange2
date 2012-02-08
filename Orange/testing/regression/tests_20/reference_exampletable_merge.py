# xtest: RANDOM

import orange

data1 = orange.ExampleTable("merge1")
data2 = orange.ExampleTable("merge2", use=data1.domain)

merged = orange.ExampleTable([data1, data2])

print
print "Domain 1: ", data1.domain.features
print "Domain 2: ", data2.domain.features
print "Merged:   ", merged.domain.features
print
for i in range(len(data1)):
    print "   %s\n + %s\n-> %s\n" % (data1[i], data2[i], merged[i])

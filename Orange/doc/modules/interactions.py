# Description: Demonstrates the use of interactions
# Category:    interactions
# Classes:     Mutual_information
# Uses:        bridges.tab

import orange
import orngInteractions

data = orange.ExampleTable("bridges")

x = data.domain["PURPOSE"]
y = data.domain["MATERIAL"]
c = data.domain["TYPE"]

mutual = orngInteractions.Mutual_information(data)
print "H(%s) = %5.5f" % (x.name, orngInteractions._entropy(orngInteractions.p2f(orange.Distribution(x, data))))
print "H(%s) = %5.5f" % (y.name, orngInteractions._entropy(orngInteractions.p2f(orange.Distribution(y, data))))
print "H(%s,%s)= %5.5f" % (x.name, y.name, orngInteractions.joint_entropy(x, y, data))
print "I(%s;%s)= %5.5f" % (x.name, y.name, mutual(x, y))
print "H(%s|%s)= %5.5f" % (x.name, c.name, mutual(x, c))
#print "InfoGain = %5.5f" % orange.MeasureAttribute_info(x, data)
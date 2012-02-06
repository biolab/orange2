# Description: Shows how to add basket attribute to a tab-delimited file
# Category:    data input
# Classes:     ExampleTable
# Uses:        tab-basket
# Referenced:  tabdelimited.htm

import orange
d = orange.ExampleTable("tab-basket")
for i in d[:5]:
    print i
print

d.save("del.tab")

d2 = orange.ExampleTable("del.tab")
for i in d[:5]:
    print i

import os
os.remove("del.tab")

def sorted(metas):
    l = metas.values()
    l.sort(lambda x,y:cmp(x.name, y.name))
    return l

print "All metas: ", sorted(d.domain.getmetas())
print "Required metas: ", sorted(d.domain.getmetas(False))
print "Optional metas: ", sorted(d.domain.getmetas(True))

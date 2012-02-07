# Description: Shows how to filter examples in ExampleTable using method 'filter'
# Category:    basic classes, preprocessing, filtering
# Classes:     ExampleTable
# Uses:        lenses
# Referenced:  ExampleTable.htm

import orange

data = orange.ExampleTable("lenses")

young = data.filter(age="young")
print "Selection: age=young"
for ex in young:
    print ex
print


young = data.filter(age=["young", "presbyopic"], astigmatic="yes")
print "Selection: age=young or presbyopic, astigmatic=yes"
for ex in young:
    print ex
print
    
print "Selection: NOT age=young, astigmatic=yes"
young = data.filter({"age": "young", "astigmatic": "yes"}, negate=1)
for ex in young:
    print ex
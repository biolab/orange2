# Author:      B Zupan
# Version:     1.0
# Description: Illustrates some basic operations on association rules (selection, removal, printing)
# Category:    description
# Uses:        lung-cancer.tab

import orange, orngAssoc
  
tab = orange.ExampleTable("lung-cancer.tab")
rules = orngAssoc.build(tab, 0.75)

#print out the number of rules and clone them
print "#rules = ",len(rules)
ar = rules.clone()
print "#rules = ",len(ar)

# a simple print put of rules, by default sorted by support and confidence
print rules
# sort by coverage, lift and support
rules.sortByFields(['coverage','lift','support'])
# print out the rules with its measures
rules.printMeasures(['support','confidence','lift','leverage','strength','coverage'])
# sort by confidence only
rules.sortByConfidence()

# print the support of the first rule
print rules[0].support
# print the first rule and the specified measures with dump function
print rules[0].dump("measures", ['support'])
rules[0].printMeasures(['support','lift'])

# delete the first rule
del rules[0]

# an example of write function - the same as dump but writes to file
f = open("foo.txt", "wt")
#rules[0].write(f, "measures", ["support"])
f.close()

# filter by support...
r1 = rules.filterBySupport(0.76)
# ... and confidence
r2 = rules.filterByConfidence(0.85)

# you can of course define your own filter function, e.g. filter on lift
r1=rules.filter(lambda x: x.lift>1.1)
print r1[0].dump("measures", ['lift'])
# Description: Demonstrates the use of orngABCN2 rules
# Category:    classification, rules
# Classes:     ABCN2, ABCN2Ordered, ABCN2M
# Uses:        titanic.tab

import orange
import orngABCN2
import orngCN2
data = orange.ExampleTable("titanic.tab")

# create learner
learner = orngABCN2.ABCN2()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"


learner = orngABCN2.ABCN2Ordered()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"


learner = orngABCN2.ABCN2M()

cl = learner(data)
for r in cl.rules:
    print orngCN2.ruleToString(r)
print "*****"
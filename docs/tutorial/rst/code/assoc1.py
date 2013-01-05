import orngAssoc
import Orange

data = Orange.data.Table("imports-85")
data = Orange.data.Table("zoo")
#data = Orange.data.preprocess.Discretize(data, \
#  method=Orange.data.discretization.EqualFreq(numberOfIntervals=3))
# data = data.select(range(10))

rules = Orange.associate.AssociationRulesInducer(data, support=0.4)

print "%i rules with support higher than or equal to %5.3f found.\n" % (len(rules), 0.4)

orngAssoc.sort(rules, ["support", "confidence"])

orngAssoc.printRules(rules[:5], ["support", "confidence"])
print

del rules[:3]
orngAssoc.printRules(rules[:5], ["support", "confidence"])
print

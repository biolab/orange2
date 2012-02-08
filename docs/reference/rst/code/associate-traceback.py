import Orange

data = Orange.data.Table("lenses")
rules = Orange.associate.AssociationRulesInducer(data, support=0.3)

rule = rules[0]
print "Rule: ", rule, "\n"

print "Supporting data instances:"
for d in data:
    if rule.appliesBoth(d):
        print d
print

print "Contradicting data instances:"
for d in data:
    if rule.applies_left(d) and not rule.applies_right(d):
        print d
print
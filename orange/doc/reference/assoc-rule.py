import orange

data = orange.ExampleTable("lenses")

rules = orange.AssociationRulesInducer(data, support = 0.3, storeExamples = True)
rule = rules[0]

print
print "Rule: ", rule
print

print "Supporting examples:"
for example in data:
    if rule.appliesBoth(example):
        print example
print

print "Contradicting examples:"
for example in data:
    if rule.appliesLeft(example) and not rule.appliesRight(example):
        print example
print

print rule
print "Match left: "
print "\n".join(str(rule.examples[i]) for i in rule.matchLeft)
print "\nMatch both: "
print "\n".join(str(rule.examples[i]) for i in rule.matchBoth)

inducer = orange.AssociationRulesInducer(support = 0.3, storeExamples = True)
itemsets = inducer.getItemsets(data)
print itemsets[8]
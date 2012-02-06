import orange

data = orange.ExampleTable("lenses")

rules = orange.AssociationRulesInducer(data, support = 0.3,
                                       storeExamples = True)
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

import orange

data = orange.ExampleTable("lenses")

print "\nAssociation rules"
rules = orange.AssociationRulesInducer(data, support = 0.3)
for r in rules:
    print "%5.3f  %5.3f  %s" % (r.support, r.confidence, r)

print "\nClassification rules"
rules = orange.AssociationRulesInducer(data, support = 0.3, classificationRules = 1)
for r in rules:
    print "%5.3f  %5.3f  %s" % (r.support, r.confidence, r)

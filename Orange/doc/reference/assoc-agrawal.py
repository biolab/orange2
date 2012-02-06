import orange

data = orange.ExampleTable("inquisition")
rules = orange.AssociationRulesSparseInducer(data,
            support = 0.5, storeExamples = True)

print "%5s   %5s" % ("supp", "conf")
for r in rules:
    print "%5.3f   %5.3f   %s" % (r.support, r.confidence, r)

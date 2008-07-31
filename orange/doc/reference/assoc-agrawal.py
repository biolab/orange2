import orange

data = orange.ExampleTable("inquisition")

rules = orange.AssociationRulesSparseInducer(data, support = 0.5, storeExamples = True)
print "%5s   %5s" % ("supp", "conf")
for r in rules:
    print "%5.3f   %5.3f   %s" % (r.support, r.confidence, r)

rule0 = rules[10]
print rule0
print "Match left: "
print [rule0.examples[i] for i in rule0.matchLeft]
print "\nMatch both: "
print [rule0.examples[i] for i in rule0.matchBoth]

inducer = orange.AssociationRulesSparseInducer(support = 0.5)
itemsets = inducer.getItemsets(data)
print itemsets[5]
import orange

data = orange.ExampleTable("inquisition")
rules = orange.AssociationRulesSparseInducer(data,
            support = 0.5, storeExamples = True)

rule0 = rules[10]

print "Rule:", rule0
print "Match left: "
print [rule0.examples[i] for i in rule0.matchLeft]
print "\nMatch both: "
print [rule0.examples[i] for i in rule0.matchBoth]

inducer = orange.AssociationRulesSparseInducer(support = 0.5)
itemsets = inducer.getItemsets(data)
print itemsets[5]

import Orange
data = Orange.data.Table("inquisition.basket")

rules = Orange.associate.AssociationRulesSparseInducer(data, support = 0.5)

print "%5s   %5s" % ("supp", "conf")
for r in rules:
    print "%5.3f   %5.3f   %s" % (r.support, r.confidence, r)

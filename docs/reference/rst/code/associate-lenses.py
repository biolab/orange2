import Orange

data = Orange.data.Table("lenses")
print "Association rules:"
rules = Orange.associate.AssociationRulesInducer(data, support=0.3)
for r in rules:
    print "%5.3f  %5.3f  %s" % (r.support, r.confidence, r)

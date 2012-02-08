import Orange
data = Orange.data.Table("market-basket.basket")

ind = Orange.associate.AssociationRulesSparseInducer(support=0.4, storeExamples = True)
itemsets = ind.get_itemsets(data)
for itemset, tids in itemsets[:5]:
    print "(%4.2f) %s" % (len(tids)/float(len(data)),
                          " ".join(data.domain[item].name for item in itemset))

import Orange

data = Orange.data.Table("lenses")
inducer = Orange.associate.AssociationRulesInducer(support = 0.3, store_examples = True)
itemsets = inducer.get_itemsets(data)
print itemsets[8]

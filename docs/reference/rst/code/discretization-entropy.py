import Orange

data = Orange.data.Table(Orange.data.Table("heart_disease.tab")[:100])
d_data = Orange.data.discretization.DiscretizeTable(data,
    method=Orange.feature.discretization.Entropy(forced=False))

old = set(data.domain.features)
new = set(x.get_value_from.variable if x.get_value_from else x for x in d_data.domain.features)
diff = old.difference(new)
print "Redundant features (%d of %d):" % (len(diff), len(data.domain.features))
print ", ".join(sorted(x.name for x in diff))

import Orange

average = lambda xs: sum(xs)/float(len(xs))

data = Orange.data.Table("iris")
targets = data.domain.class_var.values
print "%-15s %s" % ("Feature", " ".join("%15s" % c for c in targets))
for x in data.domain.features:
    dist = ["%15.2f" % average([d[x] for d in data if d.get_class()==c]) for c in targets]
    print "%-15s" % x.name, " ".join(dist)

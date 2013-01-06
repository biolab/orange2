import Orange

average = lambda xs: sum(xs)/float(len(xs))

data = Orange.data.Table("iris")
print "%-15s %s" % ("Feature", "Mean")
for x in data.domain.features:
    print "%-15s %.2f" % (x.name, average([d[x] for d in data]))

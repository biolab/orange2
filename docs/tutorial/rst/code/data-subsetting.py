import Orange

data = Orange.data.Table("iris.tab")
new_data = Orange.data.Table([d for d in data if d["petal length"]>3.0])
print "Subsetting from %d to %d instances." % (len(data), len(new_data))
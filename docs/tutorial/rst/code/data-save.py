import Orange
data = Orange.data.Table("lenses")
print "N1=%d" % len(data)
new_data = Orange.data.Table([d for d in data if d["prescription"]=="myope"])
print "N2=%d" %len(new_data)
new_data.save("lenses-subset.tab")

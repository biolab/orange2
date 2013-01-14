import Orange

data = Orange.data.Table("imports-85.tab")

print "Name of the first feature:", data.domain[0].name
name = 'fuel-type'
print "Values of feature '%s'" % name,
print data.domain[name].values
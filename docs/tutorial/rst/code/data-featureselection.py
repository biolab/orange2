import Orange

data = Orange.data.Table("iris.tab")
new_domain = Orange.data.Domain(data.domain.features[:2] + [data.domain.class_var])
new_data = Orange.data.Table(new_domain, data)

print data[0]
print new_data[0]

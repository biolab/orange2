import Orange

############ THIS IS WHAT YOU CAN DO WITH DISCRETE ATTRIBUTES

data = Orange.data.Table("lenses")

data[0][0] = "?"
data[1][0] = "?"
data[1][1] = "?"

fspec = Orange.data.filter.IsDefined(domain=data.domain)
print "\nCheck all attributes"
print [fspec(ex) for ex in data]

print "\nCheck all attributes (but with 'check' given)"
fspec.check = [1] * len(data.domain)
print [fspec(ex) for ex in data]

print "\nIgnore the first attribute"
fspec.check[0] = 0
print [fspec(ex) for ex in data]

print "\nIgnore the first attribute, check the second (and ignore the rest, list is too short)"
fspec.check = [0, 1]
data[0][3] = "?"
print [fspec(ex) for ex in data]

print "\nAdd the 'age' to the list of the checked"
fspec.check["age"] = 1
print [fspec(ex) for ex in data]

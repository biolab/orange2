import orange


############ THIS IS WHAT YOU CAN DO WITH DISCRETE ATTRIBUTES

data = orange.ExampleTable("lenses")

data[0][0] = "?"
data[1][0] = "?"
data[1][1] = "?"

fspec = orange.Filter_isDefined(domain=data.domain)
print "\nCheck all attributes"
print [fspec(ex) for ex in data]

print "\nCheck all attributes (but with 'check' given)"
fspec.check = [1] * len(data.domain)
print [fspec(ex) for ex in data]

print "\nIgnore the first attribute"
fspec.check[0] = 0
print [fspec(ex) for ex in data]

print "\nIgnore the first attribute, check the second (and ignore the rest, list is too short)"
data[0][3] = "?"
print [fspec(ex) for ex in data]

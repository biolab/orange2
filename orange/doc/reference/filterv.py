import orange

data = orange.ExampleTable("lenses")

############ THIS IS WHAT YOU CAN DO WITH DISCRETE ATTRIBUTES

print "\nYoung or presbyopic with astigmatism"
fya = orange.Filter_values(domain = data.domain)
fya["age"] = "young"
print "\nYoung examples\n"
for ex in fya(data):
    print ex

fya["age"] = "presbyopic"
print "\n\nPresbyopic examples\n"
for ex in fya(data):
    print ex


fya["age"] = ["presbyopic", "young"]
print "\n\nYoung and presbyopic examples\n"
for ex in fya(data):
    print ex

astigm = data.domain["astigmatic"]
fya["age"] = ["presbyopic", "young"]
fya[astigm] = "yes"
print "\n\nYoung and presbyopic examples that are astigmatic\n"
for ex in fya(data):
    print ex

fr = orange.Filter_values(domain = data.domain)
fr[3] = "reduced"

# Conjunction is not necessary here - we could still do this with a single filter
fcon = orange.Filter_conjunction([fya, fr])
print "\n\nYoung and presbyopic examples that are astigmatic and have reduced tear rate\n"
for ex in fcon(data):
    print ex

fcon = orange.Filter_disjunction([fya, fr])
print "\n\nYoung and presbyopic asticmatic examples and examples that have reduced tear rate\n"
for ex in fcon(data):
    print ex


############ THIS IS WHAT YOU CAN DO WITH CONTINUOUS ATTRIBUTES

data = orange.ExampleTable("iris")

fcont = orange.Filter_values(domain = data.domain)
fcont[0] = (orange.ValueFilter.Equal, 4.59999999999999) # This is to check that rounding errors don't hurt
print "\n\nThe first attribute equals 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.Less, 4.6)
print "\n\nThe first attribute is less than 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.LessEqual, 4.6)
print "\n\nThe first attribute is less than or equal to 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.Greater, 7.6)
print "\n\nThe first attribute is greater than 7.6"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.GreaterEqual, 7.6)
print "\n\nThe first attribute is greater than or equal to 7.6"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.Between, 4.6, 5.0)
print "\n\nThe first attribute is between to 4.5 and 5.0"
for ex in fcont(data):
    print ex

fcont[0] = (orange.ValueFilter.Outside, 4.6, 7.5)
print "\n\nThe first attribute is between to 4.5 and 5.0"
for ex in fcont(data):
    print ex


############ THIS IS WHAT YOU CAN DO WITH STRING ATTRIBUTES

data.domain.addmeta(orange.newmetaid(), orange.StringVariable("name"))
for ex in data:
    ex["name"] = str(ex.getclass())

fstr = orange.Filter_values(domain = data.domain)
fstr["name"] = "Iris-setosa"
print "\n\nSetosae"
d = fstr(data)
print "%i examples, starting with %s" % (len(d), d[0])

fstr["name"] = ["Iris-setosa", "Iris-virginica"]
print "\n\nSetosae and virginicae"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.Less, "Iris-versicolor")
print "\n\nLess than versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.LessEqual, "Iris-versicolor")
print "\n\nLess or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.Greater, "Iris-versicolor")
print "\n\nGreater than versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.GreaterEqual, "Iris-versicolor")
print "\n\nGreater or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.Between, "Iris-versicolor", "Iris-virginica")
print "\n\nGreater or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.Contains, "ers")
print "\n\nContains 'ers'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.NotContains, "ers")
print "\n\nDoesn't contains 'ers'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.BeginsWith, "Iris-ve")
print "\n\nBegins with 'Iris-ve'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.EndsWith, "olor")
print "\n\nBegins with 'Iris-ve'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (orange.Filter_values.EndsWith, "a"*50)
print "\n\nBegins with '%s'" % ("a"*50)
d = fstr(data)
if not len(d):
    print "<empty table>"
else:
    print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])


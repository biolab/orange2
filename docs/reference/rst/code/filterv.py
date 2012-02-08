import Orange


############ THIS IS WHAT YOU CAN DO WITH DISCRETE ATTRIBUTES

data = Orange.data.Table("lenses")

############ THIS IS WHAT YOU CAN DO WITH DISCRETE ATTRIBUTES

print "\nYoung or presbyopic with astigmatism"
fya = Orange.data.filter.Values(domain=data.domain)
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

fr = Orange.data.filter.Values(domain=data.domain)
fr[3] = "reduced"

# Conjunction is not necessary here - we could still do this with a single filter
fcon = Orange.data.filter.Conjunction([fya, fr])
print "\n\nYoung and presbyopic examples that are astigmatic and have reduced tear rate\n"
for ex in fcon(data):
    print ex

fcon = Orange.data.filter.Disjunction([fya, fr])
print "\n\nYoung and presbyopic asticmatic examples and examples that have reduced tear rate\n"
for ex in fcon(data):
    print ex

############ THIS IS WHAT YOU CAN DO WITH CONTINUOUS ATTRIBUTES

data = Orange.data.Table("iris.tab")

fcont = Orange.data.filter.Values(domain=data.domain)

fcont[0] = (Orange.data.filter.ValueFilter.Equal, 4.59999999999999) # This is
# to check that rounding errors don't hurt
print "\n\nThe first attribute equals 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.Less, 4.6)
print "\n\nThe first attribute is less than 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.LessEqual, 4.6)
print "\n\nThe first attribute is less than or equal to 4.6"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.Greater, 7.6)
print "\n\nThe first attribute is greater than 7.6"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.GreaterEqual, 7.6)
print "\n\nThe first attribute is greater than or equal to 7.6"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.Between, 4.6, 5.0)
print "\n\nThe first attribute is between to 4.5 and 5.0"
for ex in fcont(data):
    print ex

fcont[0] = (Orange.data.filter.ValueFilter.Outside, 4.6, 7.5)
print "\n\nThe first attribute is between to 4.5 and 5.0"
for ex in fcont(data):
    print ex


############ THIS IS WHAT YOU CAN DO WITH STRING ATTRIBUTES

data.domain.addmeta(
    Orange.feature.Descriptor.new_meta_id(),
    Orange.feature.String("name")
)
for ex in data:
    ex["name"] = str(ex.getclass())

fstr = Orange.data.filter.Values(domain=data.domain)
fstr["name"] = "Iris-setosa"
print "\n\nSetosae"
d = fstr(data)
print "%i examples, starting with %s" % (len(d), d[0])

fstr["name"] = ["Iris-setosa", "Iris-virginica"]
print "\n\nSetosae and virginicae"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = ["Iris-setosa", "Iris-viRGInica"]
fstr["name"].caseSensitive = 1
print "\n\nSetosae and viRGInicae (case sensitive)"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = ["Iris-setosa", "Iris-viRGInica"]
fstr["name"].caseSensitive = 0
print "\n\nSetosae and viRGInicae (case insensitive)"
d = fstr(data)

print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])
fstr["name"] = (Orange.data.filter.Values.Less, "Iris-versicolor")
print "\n\nLess than versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.LessEqual, "Iris-versicolor")
print "\n\nLess or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.Greater, "Iris-versicolor")
print "\n\nGreater than versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.GreaterEqual, "Iris-versicolor")
print "\n\nGreater or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.Between, "Iris-versicolor", "Iris-virginica")
print "\n\nGreater or equal versicolor"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.Contains, "ers")
print "\n\nContains 'ers'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.NotContains, "ers")
print "\n\nDoesn't contain 'ers'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.BeginsWith, "Iris-ve")
print "\n\nBegins with 'Iris-ve'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.EndsWith, "olor")
print "\n\nEnds with with 'olor'"
d = fstr(data)
print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.EndsWith, "a"*50)
print "\n\nBegins with '%s'" % ("a"*50)
d = fstr(data)
if not len(d):
    print "<empty table>"
else:
    print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr = Orange.data.filter.Values(domain=data.domain)
fstr["name"] = (Orange.data.filter.Values.BeginsWith, "Iris-VE")
fstr["name"].caseSensitive = 1
print "\n\nBegins with 'Iris-VE' (case sensitive)"
d = fstr(data)
if not len(d):
    print "<empty table>"
else:
    print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])

fstr["name"] = (Orange.data.filter.Values.BeginsWith, "Iris-VE")
fstr["name"].caseSensitive = 0
print "\n\nBegins with 'Iris-VE' (case insensitive)"
d = fstr(data)
if not len(d):
    print "<empty table>"
else:
    print "%i examples, starting with %s\n  finishing with %s" % (len(d), d[0], d[-1])



###### REFERENCES vs. COPIES OF EXAMPLES

data = Orange.data.Table("lenses")

print "\nYoung or presbyopic with astigmatism - as references"
fya = Orange.data.filter.Values(domain=data.domain)
fya["age"] = "young"
print "\nYoung examples\n"
d2 = fya(data, 1)
for ex in fya(d2):
    print ex

print "\nTesting whether this is really a reference"
d2[0][0] = "?"
print data[0]

print "\nTesting that we don't have references when not requested"
d2 = fya(data)
d2[1][0] = "?"
print data[1]

###### COUNTS OF EXAMPLES

data = Orange.data.Table("lenses")
fya = Orange.data.filter.Values(domain=data.domain)
fya["age"] = "young"
print "The data contains %i young fellows" % fya.count(data)

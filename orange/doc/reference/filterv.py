import orange

data = orange.ExampleTable("lenses")

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


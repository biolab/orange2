# Description: Read data and for each attribute report percent of instances with missing value
# Category:    description
# Uses:        adult_sample.tab
# Referenced:  basic_exploration.htm

import orange
data = orange.ExampleTable("adult_sample.tab")

natt = len(data.domain.attributes)
missing = [0.] * natt
for i in data:
    for j in range(natt):
        if i[j].isSpecial():
            missing[j] += 1
missing = map(lambda x, l=len(data):x / l * 100., missing)

print "Missing values per attribute:"
atts = data.domain.attributes
for i in range(natt):
    print "  %5.1f%s %s" % (missing[i], '%', atts[i].name)

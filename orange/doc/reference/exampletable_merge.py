import orange

data1 = orange.ExampleTable("merge1")
data2 = orange.ExampleTable("merge2", use = data1.domain)

merged = orange.ExampleTable([data1, data2])

print
print "Domain 1: ", data1.domain
print "Domain 2: ", data2.domain
print "Merged:   ", merged.domain
print
for ex in merged:
    print ex
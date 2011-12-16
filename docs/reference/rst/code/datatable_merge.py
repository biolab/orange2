import Orange

data1 = Orange.data.Table("merge1")
data2 = Orange.data.Table("merge2")

merged = Orange.data.Table([data1, data2])

print "Domain 1: ", data1.domain
print "Domain 2: ", data2.domain
print "Merged:   ", merged.domain
print
for i in range(len(data1)):
    print "  ", data1[i]
    print " +", data2[i]
    print "->", merged[i]
    print
    

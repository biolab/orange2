import orange
data = orange.ExampleTable("unusedValues")

newattrs = [orange.RemoveUnusedValues(attr, data) for attr in data.domain.variables]

print
for attr in range(len(data.domain)):
    print data.domain[attr],
    if newattrs[attr] == data.domain[attr]:
        print "retained as is"
    elif newattrs[attr]:
        print "reduced, new values are", newattrs[attr].values
    else:
        print "removed"

filteredattrs = filter(bool, newattrs)
newdata = orange.ExampleTable(orange.Domain(filteredattrs), data)

print "\nOriginal example table"
for ex in data:
    print ex

print "\nReduced example table"
for ex in newdata:
    print ex


print "\nRemoval with 'removedOneValued=true'"
reducer = orange.RemoveUnusedValues(removeOneValued = 1)
newattrs = [reducer(attr, data) for attr in data.domain.variables]

print
for attr in range(len(data.domain)):
    print data.domain[attr],
    if newattrs[attr] == data.domain[attr]:
        print "retained as is"
    elif newattrs[attr]:
        print "reduced, new values are", newattrs[attr].values
    else:
        print "removed"


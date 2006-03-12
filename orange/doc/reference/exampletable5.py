# Description: Shows and tests differences between ExampleTable's methods select, selectref and selectlist, and filter, filterref and filterlist
# Category:    basic classes, preprocessing
# Classes:     ExampleTable
# Uses:        bridges
# Referenced:  ExampleTable.htm

import orange, gc

data = orange.ExampleTable("bridges")

mid = data.filter(LENGTH=(1000, 2000))
del mid
gc.collect()

def testnonref(mid):
    pd0 = data[0][1]
    mid[0][1] += 1
    if pd0 != data[0][1]:
        raise "reference when there shouldn't be"

def testref(mid):
    pd0 = data[0][1]
    mid[0][1] += 1
    if pd0 == data[0][1]:
        raise "not reference when there should be"

filterany = orange.Filter_values()
filterany.domain = data.domain
filterany.conditions.append(orange.ValueFilter_continuous(position = data.domain.index("LENGTH"), min=-9999, max=9999, acceptSpecial=True))

# we sometime use LENGT=... and sometimes filterany
# the former cannot be given the 'acceptSpecial' flag, but we would
# still like to test the form of the call when we can
testnonref(data.filter(LENGTH=(-9999, 9999)))
testref(data.filterref(filterany))
testref(data.filterlist(filterany))

ll = [1]*len(data)
testnonref(data.select(ll))
testref(data.selectref(ll))
testref(data.selectlist(ll))

testnonref(data.getitems(range(10)))
testref(data.getitemsref(range(10)))

data = data.selectref(ll)
print gc.collect()

print data.ownsExamples

testnonref(data.filter(LENGTH=(-9999, 9999)))
testref(data.filterref(filterany))
testref(data.filterlist(filterany))

testnonref(data.select(ll))
testref(data.selectref(ll))
testref(data.selectlist(ll))

testnonref(data.getitems(range(10)))
testref(data.getitemsref(range(10)))

print "OK"
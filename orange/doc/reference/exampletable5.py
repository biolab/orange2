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

testnonref(data.filter(LENGTH=(-9999, 9999)))
testref(data.filterref(LENGTH=(-9999, 9999)))
testref(data.filterlist(LENGTH=(-9999, 9999)))

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
testref(data.filterref(LENGTH=(-9999, 9999)))
testref(data.filterlist(LENGTH=(-9999, 9999)))

testnonref(data.select(ll))
testref(data.selectref(ll))
testref(data.selectlist(ll))

testnonref(data.getitems(range(10)))
testref(data.getitemsref(range(10)))

print "OK"
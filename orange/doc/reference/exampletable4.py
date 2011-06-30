import orange

data = orange.ExampleTable("bridges")

mid = data.filter(LENGTH=(1000, 2000))
print "Selection: length between 1000 and 2000"
for ex in mid:
    print ex
print


mid = data.filter(LENGTH=(2000, 1000))
print "Selection: length not between 1000 and 2000"
for ex in mid:
    print ex
print

data.sort("LENGTH", "ERECTED")
print "Sorted"
for ex in data:
    print ex

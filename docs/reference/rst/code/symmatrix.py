import Orange.data

m = Orange.data.SymMatrix(4)
for i in range(4):
    for j in range(i+1):
        m[i, j] = (i+1)*(j+1)


print m
print

m.matrixType = m.Upper
print m
print

m.matrixType = m.UpperFilled
print m
print

m.matrixType = m.Lower
for row in m[:3]:
    print row

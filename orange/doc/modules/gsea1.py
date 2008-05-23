import orange, obiGsea

data = orange.ExampleTable("iris")

gen1 = dict([
    ("sepal",["sepal length", "sepal width"]), 
    ("petal",["petal length", "petal width", "petal color"])
    ])

res = obiGsea.runGSEA(data, minSize=2, geneSets=gen1)
print "%5s  %6s %6s %s" % ("LABEL", "NES", "P-VAL", "GENES")
for name,resu in res.items():
    print "%5s  %6.3f %6.3f %s" % (name, resu[1], resu[2], str(resu[6]))

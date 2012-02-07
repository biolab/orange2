# Description: Shows how to convert ExampleTable into matrices
# Category:    basic classes, preprocessing
# Classes:     ExampleTable
# Uses:        iris, heart_disease
# Referenced:  ExampleTable.htm

import orange

data = orange.ExampleTable("../datasets/iris")

for meth in [data.toNumeric, data.toNumarray, data.toNumpy]:
    try:
        a, c, w = meth()
        print type(a), type(c), type(w)
        print a.shape, c.shape
        print a[:5]
        print c[:5]
        print "\n\n"
    except:
        print "Call '%s' failed" % meth.__name__

a, c, w = data.toNumpy("1A/cw")
print type(a), type(w)
print a.shape
print a[130]
print "\n\n\n\n\n"

a, = data.toNumpy("ca1cc0")
print type(a)
print a.shape
print a[130]
print "\n\n\n\n\n"


data_h = orange.ExampleTable("../datasets/heart_disease")
try:
    a, c, w = data_h.toNumpy()
except:
    print "Converting heart_disease with toNumpy failed (as it should)"

for meth in [data_h.toNumericMA, data_h.toNumarrayMA, data_h.toNumpyMA]:
    try:
        a, c, w = meth()
        print type(a)
        print a[0]
    except:
        print "Call '%s' failed" % meth.__name__



for meth in [data.toNumeric, data.toNumarray, data.toNumpy]:
    try:
        a = meth("ac")[0]
        t2 = orange.ExampleTable(a)
        print t2.domain.attributes, t2.domain.classVar
        print t2[0]

        t3 = orange.ExampleTable(data.domain, a)
        print t3.domain.attributes, t3.domain.classVar
        print t3[0]

        columns = "sep length", "sep width", "pet length", "pet width"
        classValues = "setosa", "versicolor", "virginica"
        d4 = orange.Domain(map(orange.FloatVariable, columns),
                           orange.EnumVariable("type", values=classValues))
        t4 = orange.ExampleTable(d4, a)
        print t4.domain.attributes, t4.domain.classVar
        print t4[0]

        print
    except:
        print "Exception thrown for '%s'\n" % meth.__name__

zoo = orange.ExampleTable("../datasets/zoo")
zoo_s = orange.ExampleTable(orange.Domain(zoo.domain.attributes + zoo.domain.getmetas().values(), zoo.domain.classVar), zoo)
n = zoo_s.toNumpy()
print n[0]

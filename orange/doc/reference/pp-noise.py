# Description: Shows how to add attribute and class noise
# Category:    preprocessing, noise
# Classes:     Preprocessor, Preprocessor_addClassNoise, Preprocessor_addGaussianClassNoise, Preprocessor_addNoise, Preprocessor_addClassNoise
# Uses:        lenses
# Referenced:  preprocessing.htm

import orange
data = orange.ExampleTable("lenses")
age, prescr, astigm, tears, y = data.domain.variables

print "50% class noise (on lenses)\n"
data2 = orange.Preprocessor_addClassNoise(data, proportion = 0.5)
for i in range(len(data)):
    print data[i].getclass(), data2[i].getclass()

print "\n\nGaussian noise with deviation 10\n"
cdomain = orange.Domain([orange.FloatVariable()])
cdata = orange.ExampleTable(cdomain, [[100]]*20)
cdata2 = orange.Preprocessor_addGaussianClassNoise(cdata, deviation=10)
for i in cdata2:
    print i.getclass(),
print


print "\n\n30% noise in age, 50% in prescription and 20% elsewhere\n"
pp = orange.Preprocessor_addNoise()
pp.proportions[age]=0.3
pp.proportions[prescr]=0.5
pp.defaultProportion = 0.2
data2 = pp(data)
for i in range(len(data)):
    print "%s\n%s\n" % (data[i], data2[i])

print "\n\nGaussian noise with deviation 5.0 in all attributes except petal width\n"
iris = orange.ExampleTable("iris")
for attr in iris.domain.attributes:
    attr.numberOfDecimals = 3
    
pp = orange.Preprocessor_addGaussianNoise()
pp.deviations[iris.domain["petal width"]] = 0.0
pp.defaultDeviation = 1.0
data2 = pp(iris)
for i in range(10):
    print "%s\n%s\n" % (iris[i], data2[i])

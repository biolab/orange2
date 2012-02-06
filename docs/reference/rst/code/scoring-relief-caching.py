# Description: Shows why ReliefF needs to check the cached neighbours
# Category:    statistics
# Classes:     MeasureAttribute_relief
# Uses:        iris
# Referenced:  MeasureAttribute.htm

import orange
iris = orange.ExampleTable("iris")

r1 = orange.MeasureAttribute_relief()
r2 = orange.MeasureAttribute_relief(check_cached_data = False)

print "%.3f\t%.3f" % (r1(0, iris), r2(0, iris))
for ex in iris:
    ex[0] = 0
print "%.3f\t%.3f" % (r1(0, iris), r2(0, iris))

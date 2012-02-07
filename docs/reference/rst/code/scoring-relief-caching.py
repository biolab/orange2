# Description: Shows why ReliefF needs to check the cached neighbours
# Category:    statistics
# Classes:     MeasureAttribute_relief
# Uses:        iris
# Referenced:  MeasureAttribute.htm

import Orange
iris = Orange.data.Table("iris")

r1 = Orange.feature.scoring.Relief()
r2 = Orange.feature.scoring.Relief(check_cached_data = False)

print "%.3f\t%.3f" % (r1(0, iris), r2(0, iris))
for ex in iris:
    ex[0] = 0
print "%.3f\t%.3f" % (r1(0, iris), r2(0, iris))

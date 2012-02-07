# Description: Shows how to use probability estimators with measure of attribute quality
# Category:    attribute quality
# Classes:     MeasureAttribute, MeasureAttribute_info, ProbabilityEstimatorConstructor_m, ConditionalProbabilityEstimatorConstructor_ByRows
# Uses:        lenses
# Referenced:  MeasureAttribute.htm

import orange
data = orange.ExampleTable("lenses")

ms = (0, 2, 5, 10, 20)
measures = []
for m in ms:
    meas = orange.MeasureAttribute_info()
    meas.estimatorConstructor = orange.ProbabilityEstimatorConstructor_m(m = m)
    meas.conditionalEstimatorConstructor = orange.ConditionalProbabilityEstimatorConstructor_ByRows()
    meas.conditionalEstimatorConstructor.estimatorConstructor = meas.estimatorConstructor
    measures.append(meas)

print "%15s\t%5i\t%5i\t%5i\t%5i\t%5i\t" % (("attr",) + ms)
for attr in data.domain.attributes:
    print "%15s\t%5.3f\t%5.3f\t%5.3f\t%5.3f\t%5.3f" % ((attr.name,) + tuple([meas(attr, data) for meas in measures]))
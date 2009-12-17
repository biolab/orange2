import orange

data = orange.ExampleTable("iris")
classifier = orange.LinearLearner(data)

for i, cls_name in enumerate(data.domain.classVar.values):
    print "Attribute weights for %s vs. rest classification:\n\t" % cls_name,
    for attr, w in  zip(data.domain.attributes, classifier.weights[i]):
        print "%s: %.3f " % (attr.name, w),
    print
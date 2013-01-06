import Orange

data = Orange.data.Table("housing")
learner = Orange.regression.linear.LinearRegressionLearner()
model = learner(data)

print "pred obs"
for d in data[:3]:
    print "%.1f %.1f" % (model(d), d.get_class())

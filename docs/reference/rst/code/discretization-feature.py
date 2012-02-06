import Orange
data = Orange.data.Table("iris.tab")

idisc = Orange.feature.discretization.IntervalDiscretizer(points = [4.8, 5.0])
sep_l = idisc.construct_variable(data.domain["sepal length"])
new_data = data.select([data.domain["sepal length"], sep_l, data.domain.classVar])
for e in new_data[:5]:
    print e
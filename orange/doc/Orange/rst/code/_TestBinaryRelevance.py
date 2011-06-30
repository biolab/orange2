import Orange

data = Orange.data.Table("./data/multdata.tab")

classifier = Orange.multilabel.BinaryRelevanceLearner(data)

for e in data:
    c,p = classifier(e,Orange.classification.Classifier.GetBoth)
    print c,p
    #prints [<orange.Value 'Sports'='1'>, <orange.Value 'Politics'='1'>] <1.000, 0.000, 0.000, 1.000>
    #prints [<orange.Value 'SCience'='1'>, <orange.Value 'Politics'='1'>] <0.000, 0.000, 1.000, 1.000>
    #prints [<orange.Value 'Sports'='1'>] <1.000, 0.000, 0.000, 0.000>
    #prints [<orange.Value 'Religion'='1'>, <orange.Value 'SCience'='1'>] <0.000, 1.000, 1.000, 0.000>
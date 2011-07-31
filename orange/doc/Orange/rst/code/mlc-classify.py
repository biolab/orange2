import Orange

data = Orange.data.Table("multidata.tab")

classifier = Orange.multilabel.BinaryRelevanceLearner(data)

for e in data:
    c,p = classifier(e,Orange.classification.Classifier.GetBoth)
    print c,p
    #prints [<orange.Value 'Sports'='1'>, <orange.Value 'Politics'='1'>] <1.000, 0.000, 0.000, 1.000>
    #prints [<orange.Value 'SCience'='1'>, <orange.Value 'Politics'='1'>] <0.000, 0.000, 1.000, 1.000>
    #prints [<orange.Value 'Sports'='1'>] <1.000, 0.000, 0.000, 0.000>
    #prints [<orange.Value 'Religion'='1'>, <orange.Value 'SCience'='1'>] <0.000, 1.000, 1.000, 0.000>
    
powerset_cliassifer = Orange.multilabel.LabelPowersetLearner(data)
for e in data:
    c,p = powerset_cliassifer(e,Orange.classification.Classifier.GetBoth)
    print c,p
    #prints [<orange.Value 'Sports'='1'>, <orange.Value 'Politics'='1'>] <1.000, 0.000, 0.000, 1.000>
    #prints [<orange.Value 'SCience'='1'>, <orange.Value 'Politics'='1'>] <0.000, 0.000, 1.000, 1.000>
    #prints [<orange.Value 'Sports'='1'>] <1.000, 0.000, 0.000, 0.000>
    #prints [<orange.Value 'Religion'='1'>, <orange.Value 'SCience'='1'>] <0.000, 1.000, 1.000, 0.000>

mlknn_cliassifer = Orange.multilabel.MLkNNLearner(data,k=1)
for e in data:
    c,p = mlknn_cliassifer(e,Orange.classification.Classifier.GetBoth)
    print c,p
    
mmp_cliassifer = Orange.multilabel.MMPLearner(data)
for e in data:
    c,p = mmp_cliassifer(e,Orange.classification.Classifier.GetBoth)
    print c,p
import Orange

def bool_str(str):
    if str == '1':
        return "true"
    else:
        return "false"

data = Orange.data.Table("emotions.tab")

classifier = Orange.multilabel.BinaryRelevanceLearner(data)

for i in range(len(data)):
    c,p = classifier(data[i],Orange.classification.Classifier.GetBoth)
    print i,"\tBipartion: [",
    for j in range(len(c)):
        print bool_str( c[j].value ),
        if j <> len(c)-1:
            print ', ',
    
    print "] Confidences: [",
    for j in range(len(p)):
        print p[j],
        if j <> len(c)-1:
            print ', ',
    print
    #prints [<orange.Value 'Sports'='1'>, <orange.Value 'Politics'='1'>] <1.000, 0.000, 0.000, 1.000>
    #prints [<orange.Value 'SCience'='1'>, <orange.Value 'Politics'='1'>] <0.000, 0.000, 1.000, 1.000>
    #prints [<orange.Value 'Sports'='1'>] <1.000, 0.000, 0.000, 0.000>
    #prints [<orange.Value 'Religion'='1'>, <orange.Value 'SCience'='1'>] <0.000, 1.000, 1.000, 0.000>

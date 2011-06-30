import Orange
import Orange.multilabel.label as label

data = Orange.data.Table("multidata.tab")

#test getlabels
for e in data: 
    print label.get_labels(data,e)

#test getNumLabels
print label.get_num_labels(data)

#test getlabelIndices
for id in label.get_label_indices(data):
    print data.domain[id].name,
print

#test removeIndices
sub_domain = label.remove_indices(data,[2,3,4])
print sub_domain

sub_domain = Orange.data.Domain(sub_domain,sub_domain[0])
print sub_domain.classVar
data2 = data.translate(sub_domain)

for e in data2:
    print e

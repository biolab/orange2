import Orange
import Orange.multilabel.label as label

data = Orange.data.Table("emotions.tab")

#test getlabels
for e in data: 
    print label.get_labels(data,e)
#print [<orange.Value 'Sports'='1'>, <orange.Value 'Religion'='0'>, <orange.Value 'Science'='0'>, <orange.Value 'Politics'='1'>]
#print [<orange.Value 'Sports'='0'>, <orange.Value 'Religion'='0'>, <orange.Value 'Science'='1'>, <orange.Value 'Politics'='1'>]
#print [<orange.Value 'Sports'='1'>, <orange.Value 'Religion'='0'>, <orange.Value 'Science'='0'>, <orange.Value 'Politics'='0'>]
#print [<orange.Value 'Sports'='0'>, <orange.Value 'Religion'='1'>, <orange.Value 'Science'='1'>, <orange.Value 'Politics'='0'>]

#test get_lable_bitstream
for e in data:
    print label.get_label_bitstream(data,e)
print
#1001
#0011
#1000
#0110

#test getNumLabels
print label.get_num_labels(data)
#print 4

#test getlabelIndices
for id in label.get_label_indices(data):
    print data.domain[id].name,
print
# print Sports Religion Science Politics

#test removeIndices
sub_domain = label.remove_indices(data,[2,3,4])
print sub_domain
#print [EnumVariable 'Feature', EnumVariable 'Sports']

sub_domain = Orange.data.Domain(sub_domain,sub_domain[0])
print sub_domain.classVar
#print EnumVariable 'Feature'

data2 = data.translate(sub_domain)

for e in data2:
    print e
#['1', '1', '1']
#['2', '0', '2']
#['3', '1', '3']
#['4', '0', '4']

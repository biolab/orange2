import Orange
import xml.dom.minidom
from xml.dom.minidom import Node

def trans_mulan_data(xml_name,arff_name):
    """ transform the mulan data format to Tab file
    
        :param xml: a text file in XML format, specifying the labels and any hierarchical relationship among them. 
        see 'Mulan data format <http://mulan.sourceforge.net/format.html>'_
        :type xml: string
        
        :param arff: a text file in the 'ARFF format of Weka <http://weka.wikispaces.com/ARFF>'_.
        :type arff: string
        
        :rtype: :class:`Orange.data.Table`
    """
    
    #load XML file
    doc = xml.dom.minidom.parse(xml_name)
    
    labels = []
    for node in doc.getElementsByTagName("label"):
        labels.append( node.getAttribute("name").__str__() )
        
    #load ARFF file
    arff_table = Orange.data.Table(arff_name)
    domain = arff_table.domain
    
    #remove class tag
    domain = Orange.data.Domain(domain,False)
    
    for i, var in enumerate(domain.variables):
        if var.name in labels:
            domain[i].attributes["label"] = 1
    
    table = arff_table.translate(domain)
    
    return table

##############################################################################
# Test the code, run from DOS prompt
# assume the data file is in proper directory

if __name__=="__main__":
    table = trans_mulan_data("../../doc/datasets/emotions.xml","../../doc/datasets/emotions.arff")
    
    for i in range(10):
        print table[i]
    
    table.save("emotions.tab")
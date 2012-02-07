import Orange
import xml.dom.minidom
from xml.dom.minidom import Node

def trans_mulan_data(xml_name,arff_name, create_on_new = Orange.feature.MakeStatus.Incompatible, **kwargs):
    """ Transform the mulan data format to Tab file.
    
        :param xml: a text file in XML format, specifying the labels and any hierarchical relationship among them. 
        see 'Mulan data format <http://mulan.sourceforge.net/format.html>'_
        :type xml: string
        
        :param arff: a text file in the 'ARFF format of Weka <http://weka.wikispaces.com/ARFF>'_.
        :type arff: string
        
        :rtype: :class:`Orange.data.Table`
    """
    
    #load XML file
    doc = xml.dom.minidom.parse(xml_name)
    
    labels = [str(node.getAttribute("name"))
              for node in doc.getElementsByTagName("label")]
        
    #load ARFF file
    arff_table = Orange.data.io.loadARFF_Weka(arff_name,create_on_new)
    domain = arff_table.domain
    
    #remove class tag
    features = [v for v in domain.variables if v.name not in labels]
    class_vars = [v for v in domain.variables if v.name in labels]
    domain = Orange.data.Domain(features, None, class_vars = class_vars)
    
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
